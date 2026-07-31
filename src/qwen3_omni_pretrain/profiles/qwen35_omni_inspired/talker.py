from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from qwen3_omni_pretrain.architecture.manifest import ProfileManifest
from qwen3_omni_pretrain.profiles.qwen3_omni_reference.codec_streamer import (
    CodecOverlapState,
    ReferenceCodecStreamer,
    WaveformChunk,
)
from qwen3_omni_pretrain.profiles.qwen35_omni_inspired.configuration import (
    PREDECESSOR_CODEC_PROVENANCE,
)


_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


@dataclass(frozen=True)
class Qwen35TalkerOutput:
    main_code_logits: torch.Tensor
    loss: torch.Tensor | None
    hidden_states: torch.Tensor
    manifest: ProfileManifest

    def __post_init__(self) -> None:
        if self.main_code_logits.ndim != 3:
            raise ValueError("main_code_logits must have shape [B,T,V]")
        if self.hidden_states.ndim != 3 or self.hidden_states.shape[:2] != (
            self.main_code_logits.shape[:2]
        ):
            raise ValueError("Talker hidden states must share logits B/T axes")
        if not isinstance(self.manifest, ProfileManifest):
            raise TypeError("manifest must be ProfileManifest")


class Qwen35InspiredTalker(nn.Module):
    """Small public-Hybrid-MoE Talker under an explicit paper-inspired label."""

    def __init__(
        self,
        *,
        thinker_hidden_size: int,
        text_vocab_size: int,
        hidden_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        num_experts: int,
        num_experts_per_token: int,
        expert_intermediate_size: int,
        shared_intermediate_size: int,
        manifest: ProfileManifest,
        main_codebook_size: int = 3072,
    ) -> None:
        super().__init__()
        if not isinstance(manifest, ProfileManifest):
            raise TypeError("manifest must be ProfileManifest")
        if manifest.assumptions.count(PREDECESSOR_CODEC_PROVENANCE) != 1:
            raise ValueError("Talker manifest must mark predecessor-codec-proxy once")
        for name, value in (
            ("thinker_hidden_size", thinker_hidden_size),
            ("text_vocab_size", text_vocab_size),
            ("hidden_size", hidden_size),
            ("num_hidden_layers", num_hidden_layers),
            ("num_attention_heads", num_attention_heads),
            ("num_key_value_heads", num_key_value_heads),
            ("num_experts", num_experts),
            ("num_experts_per_token", num_experts_per_token),
            ("expert_intermediate_size", expert_intermediate_size),
            ("shared_intermediate_size", shared_intermediate_size),
            ("main_codebook_size", main_codebook_size),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if num_experts_per_token >= num_experts:
            raise ValueError("Talker top_k must be below num_experts")
        if hidden_size % num_attention_heads:
            raise ValueError("Talker hidden_size must divide attention heads")
        if num_hidden_layers < 4 or num_hidden_layers % 4:
            raise ValueError("Talker layer count must preserve complete 3:1 blocks")
        try:
            from transformers import Qwen3_5MoeTextConfig, Qwen3_5MoeTextModel
        except ImportError as exc:  # pragma: no cover - dependency guard.
            raise RuntimeError(
                "Qwen35InspiredTalker requires transformers==5.2.0"
            ) from exc
        head_dim = hidden_size // num_attention_heads
        kv_heads = min(num_key_value_heads, num_attention_heads)
        if num_attention_heads % kv_heads:
            raise ValueError(
                "Talker key/value head count must divide attention heads"
            )
        public_config = Qwen3_5MoeTextConfig(
            vocab_size=text_vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=kv_heads,
            head_dim=head_dim,
            linear_num_key_heads=num_attention_heads,
            linear_num_value_heads=num_attention_heads,
            linear_key_head_dim=head_dim,
            linear_value_head_dim=head_dim,
            linear_conv_kernel_dim=4,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_token,
            moe_intermediate_size=expert_intermediate_size,
            shared_expert_intermediate_size=shared_intermediate_size,
            layer_types=[
                "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
                for index in range(num_hidden_layers)
            ],
        )
        self.decoder = Qwen3_5MoeTextModel(public_config)
        self.thinker_projector = nn.Linear(thinker_hidden_size, hidden_size, bias=False)
        self.main_code_embedding = nn.Embedding(main_codebook_size, hidden_size)
        self.system_prompt_embedding = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.main_code_head = nn.Linear(hidden_size, main_codebook_size, bias=False)
        self.main_codebook_size = main_codebook_size
        self.manifest = manifest

    def forward(
        self,
        *,
        thinker_hidden: torch.Tensor,
        text_input_ids: torch.LongTensor,
        main_codes: torch.LongTensor,
        labels: torch.LongTensor | None = None,
        codec_prompt: torch.Tensor | None = None,
    ) -> Qwen35TalkerOutput:
        if not isinstance(thinker_hidden, torch.Tensor) or thinker_hidden.ndim != 3:
            raise ValueError("thinker_hidden must have shape [B,T,H]")
        if not thinker_hidden.is_floating_point():
            raise TypeError("thinker_hidden must have a floating dtype")
        if not isinstance(text_input_ids, torch.Tensor) or text_input_ids.dtype is not torch.long:
            raise TypeError("text_input_ids must have dtype torch.long")
        if text_input_ids.ndim != 2 or text_input_ids.shape[0] != thinker_hidden.shape[0]:
            raise ValueError("text_input_ids must have shape [B,T]")
        if not isinstance(main_codes, torch.Tensor) or main_codes.dtype is not torch.long:
            raise TypeError("main_codes must have dtype torch.long")
        if main_codes.ndim != 2 or main_codes.shape[0] != thinker_hidden.shape[0]:
            raise ValueError("main_codes must have shape [B,Tcode]")
        if bool(((main_codes < 0) | (main_codes >= self.main_codebook_size)).any().item()):
            raise ValueError("main_codes are outside the Talker codebook")
        batch_size = thinker_hidden.shape[0]
        text_hidden = self.decoder.embed_tokens(text_input_ids)
        pieces = [
            self.system_prompt_embedding.expand(batch_size, -1, -1),
            self.thinker_projector(thinker_hidden),
            text_hidden,
        ]
        if codec_prompt is not None:
            if (
                not isinstance(codec_prompt, torch.Tensor)
                or codec_prompt.ndim != 3
                or codec_prompt.shape[0] != batch_size
                or codec_prompt.shape[2] != text_hidden.shape[2]
            ):
                raise ValueError("codec_prompt must have shape [B,T,H_talker]")
            pieces.append(codec_prompt)
        pieces.append(self.main_code_embedding(main_codes))
        inputs_embeds = torch.cat(pieces, dim=1)
        outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=torch.ones(
                inputs_embeds.shape[:2],
                dtype=torch.bool,
                device=inputs_embeds.device,
            ),
            use_cache=False,
            return_dict=True,
        )
        code_hidden = outputs.last_hidden_state[:, -main_codes.shape[1] :]
        logits = self.main_code_head(code_hidden)
        loss = None
        if labels is not None:
            if not isinstance(labels, torch.Tensor) or labels.dtype is not torch.long:
                raise TypeError("Talker labels must have dtype torch.long")
            if labels.shape != main_codes.shape:
                raise ValueError("Talker labels must match main_codes shape")
            invalid = (labels != -100) & (
                (labels < 0) | (labels >= self.main_codebook_size)
            )
            if bool(invalid.any().item()):
                raise ValueError("Talker labels are outside the main codebook")
            valid_count = int((labels != -100).sum().item())
            if valid_count:
                loss = F.cross_entropy(
                    logits.reshape(-1, self.main_codebook_size).float(),
                    labels.reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                ) / valid_count
        return Qwen35TalkerOutput(logits, loss, code_hidden, self.manifest)


class PredecessorMTPProxy(nn.Module):
    def __init__(
        self,
        *,
        conditioning_size: int,
        hidden_size: int = 32,
        main_codebook_size: int = 3072,
        residual_codebook_size: int = 2048,
        residual_codebooks: int = 15,
    ) -> None:
        super().__init__()
        self.main_embedding = nn.Embedding(main_codebook_size, hidden_size)
        self.conditioning_projection = nn.Linear(conditioning_size, hidden_size, bias=False)
        self.heads = nn.ModuleList(
            nn.Linear(hidden_size, residual_codebook_size, bias=False)
            for _ in range(residual_codebooks)
        )
        self.main_codebook_size = main_codebook_size
        self.residual_codebook_size = residual_codebook_size

    def forward(
        self,
        *,
        main_codes: torch.LongTensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        if main_codes.ndim != 2 or main_codes.dtype is not torch.long:
            raise ValueError("main_codes must have shape [B,T] and dtype long")
        if conditioning.ndim != 3 or conditioning.shape[:2] != main_codes.shape:
            raise ValueError("conditioning must have shape [B,T,H]")
        hidden = torch.tanh(
            self.main_embedding(main_codes)
            + self.conditioning_projection(conditioning)
        )
        return torch.stack([head(hidden) for head in self.heads], dim=1)


class PrototypeCode2Wav(nn.Module):
    """Deterministic trainable proxy decoder; it is not an official codec."""

    def __init__(
        self,
        *,
        hidden_size: int = 16,
        codebook_size: int = 3072,
        samples_per_frame: int = 16,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(codebook_size, hidden_size)
        self.waveform_head = nn.Linear(hidden_size, samples_per_frame)
        self.samples_per_frame = samples_per_frame

    def chunked_decode(self, codes: torch.LongTensor) -> torch.Tensor:
        embedded = self.embedding(codes).mean(dim=1)
        frames = torch.tanh(self.waveform_head(embedded))
        return frames.flatten(1)


class PredecessorCodecProxy(nn.Module):
    def __init__(
        self,
        *,
        mtp_proxy: PredecessorMTPProxy,
        decoder: nn.Module,
        codec_streamer: ReferenceCodecStreamer,
        main_codebook_size: int = 3072,
        residual_codebook_size: int = 2048,
    ) -> None:
        super().__init__()
        if not isinstance(mtp_proxy, PredecessorMTPProxy):
            raise TypeError("mtp_proxy must be PredecessorMTPProxy")
        if not isinstance(decoder, nn.Module):
            raise TypeError("decoder must be an nn.Module")
        if not isinstance(codec_streamer, ReferenceCodecStreamer):
            raise TypeError("codec_streamer must be ReferenceCodecStreamer")
        if codec_streamer.decoder is not decoder:
            raise ValueError("codec streamer and proxy must share one decoder")
        self.mtp_proxy = mtp_proxy
        self.decoder = decoder
        self.codec_streamer = codec_streamer
        self.main_codebook_size = main_codebook_size
        self.residual_codebook_size = residual_codebook_size

    def _validate_codes(self, codes: torch.Tensor) -> torch.LongTensor:
        if not isinstance(codes, torch.Tensor) or codes.dtype not in _INTEGER_DTYPES:
            raise TypeError("codec codes must have an integer dtype")
        if codes.ndim != 3 or codes.shape[0] <= 0 or codes.shape[1] != 16:
            raise ValueError("codec codes must have shape [B,16,T]")
        if codes.shape[2] <= 0:
            raise ValueError("codec codes must contain frames")
        if bool(((codes[:, 0] < 0) | (codes[:, 0] >= self.main_codebook_size)).any().item()):
            raise ValueError("main codec code is outside its vocabulary")
        if bool(
            ((codes[:, 1:] < 0) | (codes[:, 1:] >= self.residual_codebook_size))
            .any()
            .item()
        ):
            raise ValueError("residual codec code is outside its vocabulary")
        return codes.to(torch.long)

    def predict_residual_codes(
        self,
        *,
        main_codes: torch.LongTensor,
        conditioning: torch.Tensor,
    ) -> torch.LongTensor:
        logits = self.mtp_proxy(main_codes=main_codes, conditioning=conditioning)
        return logits.argmax(dim=-1)

    def decode_codes(self, codes: torch.LongTensor) -> torch.Tensor:
        codes = self._validate_codes(codes)
        waveform = self.decoder.chunked_decode(codes)
        if waveform.ndim == 3 and waveform.shape[1] == 1:
            waveform = waveform[:, 0]
        if waveform.ndim != 2:
            raise ValueError("proxy decoder must return [B,samples]")
        return waveform

    def decode_code_chunk(
        self,
        *,
        codes: torch.LongTensor,
        state: CodecOverlapState,
        request_id: str,
        final: bool,
    ) -> WaveformChunk:
        codes = self._validate_codes(codes)
        return self.codec_streamer.push(
            codes=codes,
            state=state,
            request_id=request_id,
            final=final,
        )


__all__ = [
    "PredecessorCodecProxy",
    "PredecessorMTPProxy",
    "PrototypeCode2Wav",
    "Qwen35InspiredTalker",
    "Qwen35TalkerOutput",
]
