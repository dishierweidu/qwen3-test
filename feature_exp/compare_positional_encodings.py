# compare_positional_encodings.py
# pip install torch
import math
import random
import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Utils
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def causal_mask(T: int, device):
    # [T, T] with -inf above diagonal
    m = torch.full((T, T), float("-inf"), device=device)
    return torch.triu(m, diagonal=1)

class IndirectIndexGen:
    """
    2D-grid Indirect Indexing:
      - Key segment and Value segment share same time_id (0..M-1) -> position collision
      - mod_id distinguishes segments (0=KEY, 1=VAL, 2=special)

    Sequence:
      [BOS][OFF_k][KEY_q][SEP]
        KEY_SEG (len=M)
      [SEP]
        VAL_SEG (len=M)
      [Q]

    Label:
      VAL_SEG[p_q + k]
    """
    def __init__(self, vocab_noise=256, n_values=64, n_keys=32, k_max=16, M=64):
        self.vocab_noise = vocab_noise
        self.n_values = n_values
        self.n_keys = n_keys
        self.k_max = k_max
        self.M = M

        base = vocab_noise
        self.BOS = base + 0
        self.SEP = base + 1
        self.Q   = base + 2

        self.OFF0 = base + 16                 # OFF tokens: OFF0..OFF(2*k_max)
        self.KEY0 = base + 128                # KEY tokens: KEY0..KEY(n_keys-1)
        self.VAL0 = base + 256                # VAL tokens: VAL0..VAL(n_values-1)

        self.vocab_size = base + 512 + 512

    def _sample_one(self):
        M = self.M
        # sample offset k in [-k_max, k_max]
        k = random.randint(-self.k_max, self.k_max)

        # place keys in KEY_SEG at unique positions
        key_positions = random.sample(range(M), self.n_keys)  # unique
        # choose query key q
        q = random.randint(0, self.n_keys - 1)
        # key token id
        key_token_q = self.KEY0 + q
        # its position p_q in [0..M-1]
        p_q = key_positions[q]

        # ensure p_q + k is in range; resample k a few times if needed
        for _ in range(50):
            if 0 <= p_q + k < M:
                break
            k = random.randint(-self.k_max, self.k_max)
        else:
            k = max(-p_q, min(k, M - 1 - p_q))

        # build key segment
        key_seg = [random.randint(0, self.vocab_noise - 1) for _ in range(M)]
        for i, p in enumerate(key_positions):
            key_seg[p] = self.KEY0 + i

        # build value segment
        val_seg = [self.VAL0 + random.randint(0, self.n_values - 1) for _ in range(M)]
        label = val_seg[p_q + k]

        # build full token seq
        off_token = self.OFF0 + (k + self.k_max)  # shift to [0..2*k_max]
        tokens = [self.BOS, off_token, key_token_q, self.SEP] + key_seg + [self.SEP] + val_seg + [self.Q]

        # IMPORTANT: 2D ids
        # prefix specials: time_id negative; mod_id=2
        time_id = [-3, -2, -1, -1]
        mod_id  = [ 2,  2,  2,  2]

        # key segment: time_id 0..M-1, mod_id=0
        time_id += list(range(M))
        mod_id  += [0] * M

        # sep: special
        time_id += [-1]
        mod_id  += [2]

        # value segment: time_id 0..M-1 AGAIN (collision!), mod_id=1
        time_id += list(range(M))
        mod_id  += [1] * M

        # Q: special
        time_id += [M]
        mod_id  += [2]

        return tokens, time_id, mod_id, label

    def sample_batch(self, B, device):
        xs, ts, ms, ys = [], [], [], []
        for _ in range(B):
            x, t, m, y = self._sample_one()
            xs.append(x); ts.append(t); ms.append(m); ys.append(y)
        x = torch.tensor(xs, device=device, dtype=torch.long)
        t = torch.tensor(ts, device=device, dtype=torch.long)
        m = torch.tensor(ms, device=device, dtype=torch.long)
        y = torch.tensor(ys, device=device, dtype=torch.long)
        return x, t, m, y
    
class IndirectStringGen:
    def __init__(self, vocab_noise=0, alphabet=64, L=40, k_max=16):
        self.alphabet = alphabet
        self.L = L
        self.k_max = k_max

        self.BOS = 0
        self.SEP = 1
        self.Q   = 2
        self.CHAR0  = 3
        self.SHIFT0 = self.CHAR0 + alphabet  # 紧接字符
        self.vocab_size = self.SHIFT0 + (2*k_max + 1)

    def _sample_one(self):
        perm = random.sample(range(self.alphabet), self.L)
        src = [self.CHAR0 + c for c in perm]
        p = random.randint(0, self.L - 1)
        src_char = src[p]

        # sample d in [-k_max, k_max], keep in range
        for _ in range(100):
            d = random.randint(-self.k_max, self.k_max)
            if d == 0:
                continue  # forbid copy shortcut
            if 0 <= p + d < self.L:
                break
        target = src[p + d]
        shift_tok = self.SHIFT0 + (d + self.k_max)

        tokens = [self.BOS] + src + [self.SEP, src_char, shift_tok, self.Q]
        time_id = list(range(len(tokens)))
        mod_id  = [0] * len(tokens)
        return tokens, time_id, mod_id, target

    def sample_batch(self, B, device):
        xs, ts, ms, ys = [], [], [], []
        for _ in range(B):
            x, t, m, y = self._sample_one()
            xs.append(x); ts.append(t); ms.append(m); ys.append(y)
        return (
            torch.tensor(xs, device=device, dtype=torch.long),
            torch.tensor(ts, device=device, dtype=torch.long),
            torch.tensor(ms, device=device, dtype=torch.long),
            torch.tensor(ys, device=device, dtype=torch.long),
        )


# -------------------------
# Toy multimodal generator
# -------------------------
@dataclass
class ToyCfg:
    T: int = 64
    vocab_noise: int = 256
    n_values: int = 32
    offset_min: int = -8
    offset_max: int = 8
    delay_min: int = 0
    delay_max: int = 6
    stride_choices = (1, 2, 4)   # mmWave lower rate
    p_decoy: float = 0.15       # add decoy anchors/values
    layout: str = "interleave"  # "interleave" or "segment"


class MultiModalToyGen:
    """
    Sequence structure (decoder-only):
      [BOS][OFF_k][DLY_d][STR_s]  <then tokens>  [Q]
    Label = value_id that sits at B_{t0 + k + d}

    Tokens are interleaved by time:
      time t: A_t, B_t
    Each token has:
      time_id (real time)
      mod_id  (0 for A, 1 for B, 2 for specials)
    """
    def __init__(self, cfg: ToyCfg):
        self.cfg = cfg

        # Special token ids
        self.BOS = cfg.vocab_noise + cfg.n_values + 0
        self.Q   = cfg.vocab_noise + cfg.n_values + 1
        self.ANCHOR = cfg.vocab_noise + cfg.n_values + 2
        self.ANCHOR_D = cfg.vocab_noise + cfg.n_values + 3
        self.B_MISS = cfg.vocab_noise + cfg.n_values + 4

        # OFF / DLY / STR tokens are embedded as unique ids
        self.OFF0 = cfg.vocab_noise + cfg.n_values + 16
        self.DLY0 = cfg.vocab_noise + cfg.n_values + 64
        self.STR0 = cfg.vocab_noise + cfg.n_values + 96

        self.vocab_size = cfg.vocab_noise + cfg.n_values + 128

    def _sample_one(self):
        cfg = self.cfg
        # choose anchor time
        t0 = random.randint(2, cfg.T - 3)
        k = random.randint(cfg.offset_min, cfg.offset_max)
        d = random.randint(cfg.delay_min, cfg.delay_max)
        stride = random.choice(cfg.stride_choices)

        # ensure target time in range and available under stride (for B)
        # we mark missing B at times not divisible by stride
        # so we force target time to be divisible by stride
        for _ in range(50):
            t_star = t0 + k + d
            if 0 <= t_star < cfg.T and (t_star % stride == 0):
                break
            # resample offset/delay a bit if invalid
            k = random.randint(cfg.offset_min, cfg.offset_max)
            d = random.randint(cfg.delay_min, cfg.delay_max)
        else:
            # fallback
            t_star = max(0, min(cfg.T - 1, t0))
            t_star = t_star - (t_star % stride)

        value_id = random.randint(0, cfg.n_values - 1)

        # build A and B streams over time
        A = [random.randint(0, cfg.vocab_noise - 1) for _ in range(cfg.T)]
        B = [random.randint(0, cfg.vocab_noise - 1) for _ in range(cfg.T)]

        A[t0] = self.ANCHOR
        # put target value token into B at t_star
        B[t_star] = cfg.vocab_noise + value_id

        # apply stride missing for B
        for t in range(cfg.T):
            if t % stride != 0:
                B[t] = self.B_MISS

        # decoys: extra anchors and values
        if random.random() < cfg.p_decoy:
            td = random.randint(0, cfg.T - 1)
            if td != t0:
                A[td] = self.ANCHOR_D
        if random.random() < cfg.p_decoy:
            td = random.randint(0, cfg.T - 1)
            if td != t_star and (td % stride == 0):
                B[td] = cfg.vocab_noise + random.randint(0, cfg.n_values - 1)

        # pack into tokens with meta
        prefix = [
            self.BOS,
            self.OFF0 + (k - cfg.offset_min),  # shift to >=0
            self.DLY0 + (d - cfg.delay_min),
            self.STR0 + {1:0, 2:1, 4:2}.get(stride, 0),
        ]
        tokens = []
        time_id = []
        mod_id = []

        # prefix meta: specials
        for i in range(len(prefix)):
            tokens.append(prefix[i])
            time_id.append(-10 + i)  # special negative time positions
            mod_id.append(2)

        if cfg.layout == "interleave":
            for t in range(cfg.T):
                # A_t
                tokens.append(A[t]); time_id.append(t); mod_id.append(0)
                # B_t
                tokens.append(B[t]); time_id.append(t); mod_id.append(1)
        elif cfg.layout == "segment":
            # A then B (same time_id but separated in sequence index)
            for t in range(cfg.T):
                tokens.append(A[t]); time_id.append(t); mod_id.append(0)
            for t in range(cfg.T):
                tokens.append(B[t]); time_id.append(t); mod_id.append(1)
        else:
            raise ValueError("layout must be interleave/segment")

        # query token at end
        tokens.append(self.Q); time_id.append(cfg.T); mod_id.append(2)

        # label is VALUE token id
        label_token = cfg.vocab_noise + value_id

        return tokens, time_id, mod_id, label_token

    def sample_batch(self, B, device):
        xs, ts, ms, ys = [], [], [], []
        for _ in range(B):
            x, t, m, y = self._sample_one()
            xs.append(x); ts.append(t); ms.append(m); ys.append(y)
        x = torch.tensor(xs, device=device, dtype=torch.long)
        t = torch.tensor(ts, device=device, dtype=torch.long)
        m = torch.tensor(ms, device=device, dtype=torch.long)
        y = torch.tensor(ys, device=device, dtype=torch.long)
        return x, t, m, y


# -------------------------
# Positional encodings
# -------------------------
def build_inv_freq_rope(dim_pair_count: int, base=10000.0, device="cpu"):
    # Standard RoPE: inv_freq = base^{-2i/d}
    i = torch.arange(0, dim_pair_count, device=device, dtype=torch.float32)
    return base ** (-2.0 * i / (2.0 * dim_pair_count))  # denom = d_head


def apply_rope(x, pos, inv_freq):
    """
    x: [B,H,T,D] where D even
    pos: [B,T] (can repeat across tokens)
    inv_freq: [D/2]
    """
    B,H,T,D = x.shape
    assert D % 2 == 0
    x = x.view(B,H,T,D//2,2)
    # angle: [B,1,T,D/2]
    ang = pos[:, None, :, None].to(x.dtype) * inv_freq[None, None, None, :]
    cos = ang.cos()
    sin = ang.sin()
    x1 = x[...,0]
    x2 = x[...,1]
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    y = torch.stack([y1,y2], dim=-1).view(B,H,T,D)
    return y


def apply_rope_partial(x, pos, base=10000.0, rotary_pct=0.5):
    """
    Apply RoPE on a fraction of dimensions to keep some plain content dims for easy matching.
    """
    B,H,T,D = x.shape
    D_rot = int(D * rotary_pct)
    D_rot = D_rot - (D_rot % 2)  # ensure even
    if D_rot <= 0:
        return x
    if D_rot >= D:
        inv = build_inv_freq_rope(D // 2, base=base, device=x.device)
        return apply_rope(x, pos, inv)

    inv = build_inv_freq_rope(D_rot // 2, base=base, device=x.device)
    x_rot = apply_rope(x[..., :D_rot], pos, inv)
    return torch.cat([x_rot, x[..., D_rot:]], dim=-1)


class PoPEPhaseBias(nn.Module):
    """
    Learnable but bounded delta in [-2pi, 0], as described in the paper. :contentReference[oaicite:9]{index=9}
    """
    def __init__(self, H, D, init="zero"):
        super().__init__()
        if init == "zero":
            p = torch.zeros(H, D)
        elif init == "uniform":
            p = torch.rand(H, D)  # map to [-2pi,0] via transform below
        else:
            raise ValueError("init must be zero/uniform")
        self.param = nn.Parameter(p)

    def forward(self):
        # map R -> [-2pi, 0] using sigmoid
        return -2.0 * math.pi * torch.sigmoid(self.param)


def pope_xy(mu, pos, freq, delta=None):
    """
    mu: [B,H,T,D] >=0
    pos: [B,T]
    freq: [D]  (per-dim frequency)
    delta: [H,D] optional (bias added to KEY phase)
    returns x,y: [B,H,T,D]
    """
    ang = pos[:, None, :, None].to(mu.dtype) * freq[None, None, None, :]
    if delta is not None:
        ang = ang + delta[None, :, None, :]
    return mu * ang.cos(), mu * ang.sin()


# 3D rotation (SO(3)) via Rodrigues
def rodrigues_rotate(v, axis, ang):
    """
    v: [B,H,T,G,3]
    axis: [G,3] (unit)
    ang: [B,1,T,G,1]
    """
    # broadcast
    a = axis[None,None,None,:,:].to(v.dtype)  # [1,1,1,G,3]
    cos = ang.cos()
    sin = ang.sin()
    # v_rot = v*cos + (a x v)*sin + a*(a·v)*(1-cos)
    axv = torch.cross(a, v, dim=-1)
    adv = (a * v).sum(dim=-1, keepdim=True)
    return v * cos + axv * sin + a * adv * (1.0 - cos)


def make_axes(G, device):
    # deterministic pseudo-random unit axes
    axes = []
    for g in range(G):
        x = math.sin(g + 1.0)
        y = math.cos(1.7 * (g + 1.0))
        z = math.sin(2.3 * (g + 1.0))
        v = torch.tensor([x,y,z], device=device, dtype=torch.float32)
        v = v / (v.norm() + 1e-9)
        axes.append(v)
    return torch.stack(axes, dim=0)  # [G,3]


# -------------------------
# Tiny decoder-only Transformer
# -------------------------
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.w


class MHA(nn.Module):
    def __init__(self, d_model, n_heads, pe_mode, rope_axes=False, device="cpu"):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.pe_mode = pe_mode
        self.rope_axes = rope_axes

        self.qkv = nn.Linear(d_model, 3*d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

        # RoPE freqs
        if "rope" in pe_mode:
            self.inv_freq_time = build_inv_freq_rope(self.d_head//2, base=10000.0, device=device)
            if "2axis" in pe_mode:
                self.inv_freq_mod  = build_inv_freq_rope((self.d_head//2)//2, base=10.0, device=device)

        # PoPE freqs (per-dim)
        if "pope" in pe_mode:
            # per-dim freq: 10000^{-i/d} (matches paper form up to base convention) :contentReference[oaicite:10]{index=10}
            i = torch.arange(0, self.d_head, device=device, dtype=torch.float32)
            self.freq_time = (10000.0 ** (-i / self.d_head))
            if "2axis" in pe_mode:
                j = torch.arange(0, self.d_head//2, device=device, dtype=torch.float32)
                self.freq_mod = (10.0 ** (-j / (self.d_head//2))) * math.pi  # scaled

            # learnable bounded delta (bias)
            self.delta = PoPEPhaseBias(self.n_heads, self.d_head, init="zero")

        # 3D rotation setup
        if "rot3d" in pe_mode:
            G = self.d_head // 3
            self.G = G
            self.axes_time = make_axes(G, device=device)
            self.axes_mod  = make_axes(G, device=device).roll(shifts=1, dims=0)
            g = torch.arange(0, G, device=device, dtype=torch.float32)
            self.freq3d_time = (10000.0 ** (-g / G))
            self.freq3d_mod  = (10.0 ** (-g / G)) * (math.pi/2)

    def forward(self, x, time_id, mod_id, attn_mask):
        """
        x: [B,T,d_model]
        time_id, mod_id: [B,T]
        """
        B,T,_ = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.d_head).permute(0, 3, 1, 2, 4)  # [B,H,T,3,D]
        q = qkv[:, :, :, 0, :]  # [B,H,T,D]
        k = qkv[:, :, :, 1, :]  # [B,H,T,D]
        v = qkv[:, :, :, 2, :]  # [B,H,T,D]

        if self.pe_mode == "rope_1d":
            # partial rotary: leave some dims unrotated for content matching, rotate the rest for relative shift
            Hc = self.n_heads // 2  # 前一半 head：纯内容（不加 RoPE）
            # 后一半 head：纯位置（加 RoPE）
            q = torch.cat([q[:, :Hc], apply_rope(q[:, Hc:], time_id, self.inv_freq_time)], dim=1)
            k = torch.cat([k[:, :Hc], apply_rope(k[:, Hc:], time_id, self.inv_freq_time)], dim=1)


            scores = torch.einsum("bhtd,bhsd->bhts", q, k) / math.sqrt(self.d_head)

        elif self.pe_mode == "pope_1d":
            # PoPE: magnitude=softplus, phase=pos*freq, key phase adds bounded delta :contentReference[oaicite:11]{index=11}
            mu_q = F.softplus(q)
            mu_k = F.softplus(k)
            delta = self.delta()  # [H,D] in [-2pi,0]

            xq, yq = pope_xy(mu_q, time_id, self.freq_time, delta=None)
            xk, yk = pope_xy(mu_k, time_id, self.freq_time, delta=delta)

            scores = (torch.einsum("bhtd,bhsd->bhts", xq, xk) +
                      torch.einsum("bhtd,bhsd->bhts", yq, yk)) / math.sqrt(self.d_head)

        elif self.pe_mode == "rope_2axis":
            # split dims into two halves: time + modality
            assert self.d_head % 4 == 0, "d_head must be divisible by 4 for rope_2axis"
            d_half = self.d_head // 2
            qt, qm = q[..., :d_half], q[..., d_half:]
            kt, km = k[..., :d_half], k[..., d_half:]

            inv_t = build_inv_freq_rope(d_half//2, base=10000.0, device=q.device)
            inv_m = build_inv_freq_rope((d_half//2), base=10.0, device=q.device)

            qt = apply_rope(qt, time_id, inv_t)
            kt = apply_rope(kt, time_id, inv_t)

            qm = apply_rope(qm, mod_id, inv_m)
            km = apply_rope(km, mod_id, inv_m)

            q2 = torch.cat([qt,qm], dim=-1)
            k2 = torch.cat([kt,km], dim=-1)
            scores = torch.einsum("bhtd,bhsd->bhts", q2, k2) / math.sqrt(self.d_head)

        elif self.pe_mode == "pope_2axis":
            # split dims into two halves: time + modality (PoPE-style, per-dim complex)
            assert self.d_head % 2 == 0
            d_half = self.d_head // 2

            mu_q = F.softplus(q)
            mu_k = F.softplus(k)
            delta = self.delta()

            mu_qt, mu_qm = mu_q[..., :d_half], mu_q[..., d_half:]
            mu_kt, mu_km = mu_k[..., :d_half], mu_k[..., d_half:]

            # time half
            it = torch.arange(0, d_half, device=q.device, dtype=torch.float32)
            freq_t = (10000.0 ** (-it / d_half))
            xqt, yqt = pope_xy(mu_qt, time_id, freq_t, delta=None)
            d_t = delta[:, :d_half]
            xkt, ykt = pope_xy(mu_kt, time_id, freq_t, delta=d_t)

            # mod half
            im = torch.arange(0, d_half, device=q.device, dtype=torch.float32)
            freq_m = (10.0 ** (-im / d_half)) * math.pi
            xqm, yqm = pope_xy(mu_qm, mod_id, freq_m, delta=None)
            d_m = delta[:, d_half:]
            xkm, ykm = pope_xy(mu_km, mod_id, freq_m, delta=d_m)

            xq = torch.cat([xqt,xqm], dim=-1)
            yq = torch.cat([yqt,yqm], dim=-1)
            xk = torch.cat([xkt,xkm], dim=-1)
            yk = torch.cat([ykt,ykm], dim=-1)

            scores = (torch.einsum("bhtd,bhsd->bhts", xq, xk) +
                      torch.einsum("bhtd,bhsd->bhts", yq, yk)) / math.sqrt(self.d_head)

        elif self.pe_mode == "rot3d":
            # rotate in 3D groups; compose time rotation then mod rotation
            G = self.G
            D3 = G * 3
            q0, qR = q[..., :D3], q[..., D3:]
            k0, kR = k[..., :D3], k[..., D3:]

            qv = q0.view(B, self.n_heads, T, G, 3)
            kv = k0.view(B, self.n_heads, T, G, 3)

            ang_t = (time_id[:,None,:,None,None].to(q.dtype) *
                     self.freq3d_time[None,None,None,:,None])
            qv = rodrigues_rotate(qv, self.axes_time, ang_t)
            kv = rodrigues_rotate(kv, self.axes_time, ang_t)

            ang_m = (mod_id[:,None,:,None,None].to(q.dtype) *
                     self.freq3d_mod[None,None,None,:,None])
            qv = rodrigues_rotate(qv, self.axes_mod, ang_m)
            kv = rodrigues_rotate(kv, self.axes_mod, ang_m)

            q_rot = torch.cat([qv.reshape(B,self.n_heads,T,D3), qR], dim=-1)
            k_rot = torch.cat([kv.reshape(B,self.n_heads,T,D3), kR], dim=-1)

            scores = torch.einsum("bhtd,bhsd->bhts", q_rot, k_rot) / math.sqrt(self.d_head)
        else:
            raise ValueError(f"unknown pe_mode: {self.pe_mode}")

        scores = scores + attn_mask[None,None,:,:]
        w = F.softmax(scores, dim=-1)
        out = torch.einsum("bhts,bhsd->bhtd", w, v).transpose(1,2).contiguous().view(B,T,self.d_model)
        return self.out(out)


class Block(nn.Module):
    def __init__(self, d_model, n_heads, pe_mode, device):
        super().__init__()
        self.n1 = RMSNorm(d_model)
        self.attn = MHA(d_model, n_heads, pe_mode, device=device)
        self.n2 = RMSNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Linear(4*d_model, d_model),
        )
    def forward(self, x, time_id, mod_id, mask):
        x = x + self.attn(self.n1(x), time_id, mod_id, mask)
        x = x + self.mlp(self.n2(x))
        return x


class TinyDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=192, n_heads=6, n_layers=4, pe_mode="rope_1d", device="cpu"):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, pe_mode, device) for _ in range(n_layers)])
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # tie output projection to input embeddings to help pointer-style tasks
        self.lm_head.weight = self.emb.weight

    def forward(self, tok, time_id, mod_id):
        B,T = tok.shape
        x = self.emb(tok)
        mask = causal_mask(T, tok.device)
        for blk in self.blocks:
            x = blk(x, time_id, mod_id, mask)
        x = self.norm(x)
        logits = self.lm_head(x)  # [B,T,V]
        return logits


# -------------------------
# Train / Eval
# -------------------------
def mask_logits_indirect_str(logits, gen, task):
    if task == "indirect_str":
        mask = torch.full_like(logits, float("-inf"))
        mask[:, gen.CHAR0 : gen.CHAR0 + gen.alphabet] = 0.0
        logits = logits + mask
    return logits


@torch.no_grad()
def eval_acc(model, gen, steps, batch_size, device, task):
    model.eval()
    correct = 0
    total = 0
    for _ in range(steps):
        x,t,m,y = gen.sample_batch(batch_size, device)
        logits = model(x,t,m)[:,-1,:]  # last token = Q
        logits = mask_logits_indirect_str(logits, gen, task)

        pred = logits.argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total


def train(args):
    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    set_seed(args.seed)
    cfg = ToyCfg(T=args.T, layout=args.layout)

    if args.task == "crossmodal":
        gen = MultiModalToyGen(cfg)
    elif args.task == "indirect_str":
        gen = IndirectStringGen(vocab_noise=0, alphabet=args.alphabet, L=args.L, k_max=args.Kmax)
    else:
        gen = IndirectIndexGen(vocab_noise=256, n_values=args.M, n_keys=args.Nkeys, k_max=args.Kmax, M=args.ArrLen)


    model = TinyDecoder(
        vocab_size=gen.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        pe_mode=args.pe,
        device=device
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.95))

    for step in range(1, args.train_steps + 1):
        model.train()
        x,t,m,y = gen.sample_batch(args.batch, device)
        logits = model(x,t,m)[:,-1,:]
        logits = mask_logits_indirect_str(logits, gen, args.task)
        loss = F.cross_entropy(logits, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % args.log_every == 0:
            acc = (logits.argmax(dim=-1) == y).float().mean().item()
            print(f"step {step:5d} | loss {loss.item():.4f} | train_acc {acc*100:.2f}%")

        if step % args.eval_every == 0:
            test_acc = eval_acc(model, gen, steps=50, batch_size=args.batch, device=device, task=args.task)
            tag = f"T={cfg.T}" if args.task == "crossmodal" else (f"L={args.L}" if args.task == "indirect_str" else f"M={args.ArrLen}")
            print(f"[eval@{tag}] acc {test_acc*100:.2f}%")

    # length extrapolation tests
    if args.extrapolate:
        if args.task == "crossmodal":
            for T2 in [args.T*2, args.T*4]:
                cfg2 = ToyCfg(T=T2, layout=args.layout)
                gen2 = MultiModalToyGen(cfg2)
                acc2 = eval_acc(model, gen2, steps=50, batch_size=args.batch, device=device, task=args.task)
                print(f"[extrapolate@T={T2}] acc {acc2*100:.2f}%")
        elif args.task == "indirect_str":
            for L2 in [args.L * 2, args.L * 4]:
                if L2 > args.alphabet:
                    raise ValueError("Need alphabet >= max extrapolate L when requiring unique chars.")
                gen2 = IndirectStringGen(vocab_noise=0, alphabet=args.alphabet, L=L2, k_max=args.Kmax)
                acc2 = eval_acc(model, gen2, steps=50, batch_size=args.batch, device=device, task=args.task)
                print(f"[extrapolate@L={L2}] acc {acc2*100:.2f}%")
        else:
            for M2 in [args.M*2, args.M*4]:
                gen2 = IndirectIndexGen(vocab_noise=256, n_values=args.M, n_keys=args.Nkeys, k_max=args.Kmax)
                acc2 = eval_acc(model, gen2, steps=50, batch_size=args.batch, device=device, task=args.task)
                print(f"[extrapolate@M={M2}] acc {acc2*100:.2f}%")

    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pe", type=str, default="rope_1d",
                   choices=["rope_1d","pope_1d","rope_2axis","pope_2axis","rot3d"])
    p.add_argument("--layout", type=str, default="interleave", choices=["interleave","segment"])
    p.add_argument("--T", type=int, default=64)
    p.add_argument("--d_model", type=int, default=192)
    p.add_argument("--n_heads", type=int, default=6)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--train_steps", type=int, default=2000)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--eval_every", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--extrapolate", action="store_true")
    p.add_argument("--task", type=str, default="crossmodal", choices=["crossmodal","indirect","indirect_str"])
    p.add_argument("--M", type=int, default=64)      # indirect: n_values
    p.add_argument("--Nkeys", type=int, default=32)  # indirect: n_keys
    p.add_argument("--Kmax", type=int, default=16)   # indirect: max offset
    p.add_argument("--ArrLen", type=int, default=64)
    p.add_argument("--L", type=int, default=64)
    p.add_argument("--alphabet", type=int, default=52)


    args = p.parse_args()
    raise SystemExit(train(args))


if __name__ == "__main__":
    main()
