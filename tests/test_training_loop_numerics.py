import pytest
import torch
from torch.utils.data import DataLoader

from qwen3_omni_pretrain.training.loop import NonFiniteTrainingError, train_one_epoch


class FiniteLossInfiniteLogitsModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        loss = self.weight.square()
        logits = torch.full((x.shape[0], 2), float("inf"), device=x.device)
        return {"loss": loss, "logits": logits}


class MetadataRejectingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.seen_x = None

    def forward(self, x):
        self.seen_x = x.detach().clone()
        return {"loss": self.weight.square(), "logits": x * self.weight}


def test_nonfinite_logits_raise_before_optimizer_step():
    model = FiniteLossInfiniteLogitsModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = model.weight.detach().clone()
    dataloader = DataLoader(
        [{"x": torch.tensor([1.0]), "_sample_ids": ["sample-7"]}],
        batch_size=None,
    )

    with pytest.raises(NonFiniteTrainingError, match="sample-7"):
        train_one_epoch(model, dataloader, optimizer, None, torch.device("cpu"))

    assert torch.equal(model.weight.detach(), before)


def test_metadata_keys_are_not_forwarded_to_model():
    model = MetadataRejectingModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    dataloader = DataLoader(
        [{"x": torch.tensor([2.0]), "_sample_ids": ["ok"]}],
        batch_size=None,
    )

    loss = train_one_epoch(
        model, dataloader, optimizer, None, torch.device("cpu")
    )

    assert loss == pytest.approx(1.0)
    assert model.seen_x.tolist() == [2.0]


class FiniteOutputInfiniteGradientModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        loss = torch.sqrt(self.weight)
        return {"loss": loss, "logits": torch.zeros((x.shape[0], 2))}


def test_nonfinite_gradient_raises_before_optimizer_step():
    model = FiniteOutputInfiniteGradientModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    dataloader = DataLoader(
        [{"x": torch.tensor([1.0]), "_sample_ids": ["grad-bad"]}],
        batch_size=None,
    )

    with pytest.raises(NonFiniteTrainingError, match="gradient"):
        train_one_epoch(model, dataloader, optimizer, None, torch.device("cpu"))

    assert model.weight.item() == 0.0


class NonFiniteMoeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from qwen3_omni_pretrain.models.qwen3_omni_moe.modules.moe import (
            Qwen3OmniMoeMLP,
        )

        self.moe = Qwen3OmniMoeMLP(
            hidden_size=4,
            intermediate_size=8,
            num_experts=2,
            num_experts_per_tok=1,
            use_shared_expert=False,
        )
        with torch.no_grad():
            self.moe.gate.weight.fill_(float("nan"))

    def forward(self, x):
        logits, aux_loss = self.moe(x)
        return {
            "loss": logits.sum() + aux_loss,
            "logits": logits,
            "aux_loss": aux_loss,
        }


def test_nonfinite_moe_state_is_caught_at_synchronized_output_boundary():
    model = NonFiniteMoeModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    dataloader = DataLoader(
        [
            {
                "x": torch.ones(1, 1, 4),
                "_sample_ids": ["moe-bad"],
            }
        ],
        batch_size=None,
    )

    with pytest.raises(NonFiniteTrainingError, match="moe-bad"):
        train_one_epoch(
            model, dataloader, optimizer, None, torch.device("cpu")
        )


class SanitizingNonFiniteMoeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from qwen3_omni_pretrain.models.qwen3_omni_moe.modules.moe import (
            Qwen3OmniMoeMLP,
        )

        self.moe = Qwen3OmniMoeMLP(
            hidden_size=4,
            intermediate_size=8,
            num_experts=2,
            num_experts_per_tok=1,
            use_shared_expert=False,
        )
        with torch.no_grad():
            self.moe.gate.weight.fill_(float("nan"))

    def forward(self, x):
        logits, aux_loss = self.moe(x)
        sanitized_logits = torch.nan_to_num(logits)
        sanitized_aux_loss = torch.nan_to_num(aux_loss)
        return {
            "loss": sanitized_logits.sum() + sanitized_aux_loss,
            "logits": sanitized_logits,
            "aux_loss": sanitized_aux_loss,
        }


def test_module_diagnostic_catches_nonfinite_moe_state_after_output_sanitization():
    model = SanitizingNonFiniteMoeModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    dataloader = DataLoader(
        [
            {
                "x": torch.ones(1, 1, 4),
                "_sample_ids": ["moe-sanitized"],
            }
        ],
        batch_size=None,
    )

    with pytest.raises(NonFiniteTrainingError, match="MoE router probabilities"):
        train_one_epoch(
            model, dataloader, optimizer, None, torch.device("cpu")
        )


class DiagnosticOnlyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._nonfinite_diagnostic = torch.tensor(True)
        self._nonfinite_diagnostic_reason = "synthetic internal state is non-finite"


class FiniteOutputWithBadInternalDiagnostic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.diagnostic = DiagnosticOnlyModule()

    def forward(self, x):
        logits = x * self.weight
        return {"loss": logits.sum(), "logits": logits}


def test_internal_module_diagnostic_is_checked_after_forward():
    model = FiniteOutputWithBadInternalDiagnostic()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    dataloader = DataLoader(
        [{"x": torch.tensor([1.0]), "_sample_ids": ["internal-bad"]}],
        batch_size=None,
    )

    with pytest.raises(NonFiniteTrainingError, match="synthetic internal state"):
        train_one_epoch(
            model, dataloader, optimizer, None, torch.device("cpu")
        )
