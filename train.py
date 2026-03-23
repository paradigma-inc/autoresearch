"""
Autoresearch pretraining script. Single-GPU, single-file.
Cherry-picked and simplified from nanochat.
Usage: uv run train.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import math
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from kernels import get_kernel
cap = torch.cuda.get_device_capability()
# varunneal's FA3 is Hopper only; all other GPUs use PyTorch SDPA.
fa3 = get_kernel("varunneal/flash-attention-3").flash_attn_interface if cap == (9, 0) else None

from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb

# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"
    attnres_mode: str = "baseline"
    attnres_use_rmsnorm: bool = True


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (alternating, last always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class AttnResRMSNorm(nn.Module):
    def __init__(self, ndim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.eps = eps

    def forward(self, x):
        inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * inv_rms * self.weight


class DepthMixer(nn.Module):
    def __init__(self, config, num_queries):
        super().__init__()
        self.use_rmsnorm = config.attnres_use_rmsnorm
        self.queries = nn.Parameter(torch.zeros(num_queries, config.n_embd))
        self.key_norms = (
            nn.ModuleList([AttnResRMSNorm(config.n_embd) for _ in range(num_queries)])
            if self.use_rmsnorm
            else nn.ModuleList()
        )

    def _normalize_keys(self, values, query_idx):
        if not self.use_rmsnorm:
            return values
        return self.key_norms[query_idx](values)

    def aggregate(self, values, query_idx):
        stacked = torch.stack(values, dim=0)
        keys = self._normalize_keys(stacked, query_idx)
        query = self.queries[query_idx]
        logits = torch.einsum("d,nbtd->nbt", query, keys)
        weights = logits.softmax(dim=0)
        return torch.einsum("nbt,nbtd->btd", weights, stacked)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.use_fa3 = cap == (9, 0)
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        if self.use_fa3:
            y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            if self.n_kv_head == self.n_head:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
            else:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0, enable_gqa=True)
            y = y.transpose(1, 2)
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def attn_residual(self, x, ve, cos_sin, window_size):
        return self.attn(norm(x), ve, cos_sin, window_size)

    def mlp_residual(self, x):
        return self.mlp(norm(x))

    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.attn_residual(x, ve, cos_sin, window_size)
        x = x + self.mlp_residual(x)
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.attnres_mode not in ("baseline", "full"):
            raise ValueError(f"Unsupported attnres mode: {config.attnres_mode}")
        self.use_attnres = config.attnres_mode == "full"
        self.num_sublayers = config.n_layer * 2
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        self.depth_mixer = DepthMixer(config, self.num_sublayers + 1) if self.use_attnres else None
        # Value embeddings
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        # Rotary embeddings
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        # Transformer blocks
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        # Value embeddings
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        # Gate weights init to zero (sigmoid(0)=0.5, scaled by 2 -> 1.0 = neutral)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # Cast embeddings to bf16
        self.transformer.wte.to(dtype=torch.bfloat16)
        for ve in self.value_embeds.values():
            ve.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def estimate_flops(self):
        """Estimated FLOPs per token (forward + backward)."""
        nparams = sum(p.numel() for p in self.parameters())
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel())
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        return 6 * (nparams - nparams_exclude) + attn_flops

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        attnres = sum(p.numel() for p in self.depth_mixer.parameters()) if self.depth_mixer is not None else 0
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + attnres + scalars
        return {
            'wte': wte, 'value_embeds': value_embeds, 'lm_head': lm_head,
            'transformer_matrices': transformer_matrices, 'attnres': attnres,
            'scalars': scalars, 'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                        weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.config.n_embd
        matrix_params = list(self.transformer.h.parameters())
        attnres_params = list(self.depth_mixer.parameters()) if self.depth_mixer is not None else []
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == (len(matrix_params) + len(embedding_params) +
            len(lm_head_params) + len(value_embeds_params) + len(resid_params) +
            len(x0_params) + len(attnres_params))
        # Scale LR ∝ 1/√dmodel (tuned at 768 dim)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")
        def classify_shape(shape):
            rows, cols = shape
            aspect = max(rows, cols) / min(rows, cols)
            return "square" if aspect <= 1.5 else "rect"
        param_groups = [
            dict(kind='adamw', subkind='lm_head', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', subkind='token_embed', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', subkind='value_embed', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', subkind='resid', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', subkind='x0', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        if attnres_params:
            param_groups.append(dict(
                kind='adamw', subkind='attnres', params=attnres_params, lr=scalar_lr * 0.1,
                betas=adam_betas, eps=1e-10, weight_decay=0.0,
            ))
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', shape_class=classify_shape(shape), params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, reduction='mean'):
        B, T = idx.size()
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        if not self.use_attnres:
            for i, block in enumerate(self.transformer.h):
                x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
                ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
                x = block(x, ve, cos_sin, self.window_sizes[i])
        else:
            residual_outputs = []
            query_idx = 0
            for i, block in enumerate(self.transformer.h):
                ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None

                hidden = self.depth_mixer.aggregate([x0] + residual_outputs, query_idx)
                hidden = self.resid_lambdas[i] * hidden + self.x0_lambdas[i] * x0
                attn_out = block.attn_residual(hidden, ve, cos_sin, self.window_sizes[i])
                residual_outputs.append(attn_out)
                query_idx += 1

                hidden = self.depth_mixer.aggregate([x0] + residual_outputs, query_idx)
                hidden = self.resid_lambdas[i] * hidden + self.x0_lambdas[i] * x0
                mlp_out = block.mlp_residual(hidden)
                residual_outputs.append(mlp_out)
                query_idx += 1

            x = self.depth_mixer.aggregate([x0] + residual_outputs, query_idx)
        x = norm(x)

        softcap = 15
        logits = self.lm_head(x)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1, reduction=reduction)
            return loss
        return logits

# ---------------------------------------------------------------------------
# Optimizer (MuonAdamW, single GPU only)
# ---------------------------------------------------------------------------

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

def rre_extrapolate(iterates, regularization=1e-6):
    k = len(iterates) - 1
    if k < 2:
        return iterates[-1].clone()

    orig_dtype = iterates[0].dtype
    iterates_f32 = [x.float() for x in iterates]
    diffs = [iterates_f32[i + 1] - iterates_f32[i] for i in range(k)]
    U = torch.stack(diffs)
    G = U @ U.T
    reg = regularization * torch.trace(G) / k + 1e-10
    G = G + reg * torch.eye(k, device=G.device, dtype=G.dtype)
    ones = torch.ones(k, dtype=G.dtype, device=G.device)

    try:
        c = torch.linalg.solve(G, ones)
        c = c / c.sum()
    except Exception:
        return iterates[-1].clone()

    gamma = torch.flip(torch.cumsum(torch.flip(c, [0]), 0), [0])
    x_star = iterates_f32[-1] - (gamma @ U)
    return x_star.to(orig_dtype)


class LateMuonRRE:
    def __init__(self, muon_params, extrap_every, num_checkpoints, late_start_progress,
                 reset_state_after_extrap=True, per_layer=False, regularization=1e-6):
        self.muon_params = list(muon_params)
        self.extrap_every = extrap_every
        self.num_checkpoints = num_checkpoints
        self.late_start_progress = late_start_progress
        self.reset_state_after_extrap = reset_state_after_extrap
        self.per_layer = per_layer
        self.regularization = regularization
        self.checkpoints = [[] for _ in self.muon_params] if self.per_layer else []

    def _flatten_params(self):
        return torch.cat([p.data.flatten() for p in self.muon_params])

    def _unflatten_params(self, flat_params):
        offset = 0
        for p in self.muon_params:
            numel = p.numel()
            p.data.copy_(flat_params[offset:offset + numel].view_as(p))
            offset += numel

    @torch.no_grad()
    def maybe_step(self, step, progress, optimizer):
        if self.per_layer:
            for i, p in enumerate(self.muon_params):
                self.checkpoints[i].append(p.data.flatten().detach().clone())
                if len(self.checkpoints[i]) > self.num_checkpoints:
                    self.checkpoints[i].pop(0)
            if progress < self.late_start_progress:
                return False
            if step % self.extrap_every != 0 or len(self.checkpoints[0]) < self.num_checkpoints:
                return False
            for i, p in enumerate(self.muon_params):
                extrapolated = rre_extrapolate(self.checkpoints[i], regularization=self.regularization)
                p.data.copy_(extrapolated.view_as(p))
                self.checkpoints[i] = [extrapolated.clone()]
            if self.reset_state_after_extrap:
                optimizer.reset_muon_state()
            return True

        current_params = self._flatten_params().detach().clone()
        self.checkpoints.append(current_params)
        if len(self.checkpoints) > self.num_checkpoints:
            self.checkpoints.pop(0)
        if progress < self.late_start_progress:
            return False
        if step % self.extrap_every != 0 or len(self.checkpoints) < self.num_checkpoints:
            return False
        extrapolated = rre_extrapolate(self.checkpoints, regularization=self.regularization)
        self._unflatten_params(extrapolated)
        self.checkpoints = [extrapolated.clone()]
        if self.reset_state_after_extrap:
            optimizer.reset_muon_state()
        return True

@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)

@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    # Polar express orthogonalization
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X
    # NorMuon variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    # Cautious weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class MuonAdamW(torch.optim.Optimizer):
    """Combined optimizer: Muon for 2D matrix params, AdamW for others."""

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._schedule_progress = 0.0
        self._freshness_adam_applied = False
        self._freshness_rect_applied = False
        self._freshness_square_applied = False
        self._late_handoff_applied = False
        muon_params = []
        for group in self.param_groups:
            if group["kind"] == "muon":
                muon_params.extend(group["params"])
        self._late_rre = LateMuonRRE(
            muon_params=muon_params,
            extrap_every=RRE_EVERY,
            num_checkpoints=RRE_NUM_CHECKPOINTS,
            late_start_progress=RRE_LATE_START,
            reset_state_after_extrap=RRE_RESET_STATE,
            per_layer=RRE_PER_LAYER,
            regularization=RRE_REGULARIZATION,
        )

    def set_schedule_progress(self, progress):
        self._schedule_progress = progress

    def _apply_reset_ladder(self, group, state):
        progress = self._schedule_progress
        if group["shape_class"] == "square":
            if state.get("reset_ladder_stage", 0) < 1 and progress >= SQUARE_RESET_START:
                state["momentum_buffer"].mul_(SQUARE_MUON_MOMENTUM_SHRINK)
                state["second_momentum_buffer"].mul_(SQUARE_MUON_SECOND_SHRINK)
                state["reset_ladder_stage"] = 1
        else:
            if state.get("reset_ladder_stage", 0) < 2 and progress >= RECT_RESET_START:
                state["momentum_buffer"].mul_(RECT_MUON_MOMENTUM_SHRINK)
                state["second_momentum_buffer"].mul_(RECT_MUON_SECOND_SHRINK)
                state["reset_ladder_stage"] = 2

    def maybe_apply_freshness_ladder(self, progress):
        if not self._freshness_adam_applied and progress >= FRESHNESS_ADAM_START:
            for group in self.param_groups:
                if group["kind"] != "adamw" or group["subkind"] not in ("lm_head", "token_embed", "value_embed"):
                    continue
                if group["subkind"] == "lm_head":
                    exp_avg_shrink = FRESHNESS_HEAD_EXP_AVG_SHRINK
                    exp_avg_sq_shrink = FRESHNESS_HEAD_EXP_AVG_SQ_SHRINK
                else:
                    exp_avg_shrink = FRESHNESS_EMBED_EXP_AVG_SHRINK
                    exp_avg_sq_shrink = FRESHNESS_EMBED_EXP_AVG_SQ_SHRINK
                for p in group["params"]:
                    state = self.state.get(p)
                    if not state:
                        continue
                    if "exp_avg" in state:
                        state["exp_avg"].mul_(exp_avg_shrink)
                    if "exp_avg_sq" in state:
                        state["exp_avg_sq"].mul_(exp_avg_sq_shrink)
            self._freshness_adam_applied = True

        if not self._freshness_rect_applied and progress >= FRESHNESS_RECT_MUON_START:
            for group in self.param_groups:
                if group["kind"] != "muon" or group["shape_class"] != "rect":
                    continue
                params = group["params"]
                if not params:
                    continue
                state = self.state.get(params[0])
                if not state:
                    continue
                if "momentum_buffer" in state:
                    state["momentum_buffer"].mul_(FRESHNESS_RECT_MUON_MOMENTUM_SHRINK)
                if "second_momentum_buffer" in state:
                    state["second_momentum_buffer"].mul_(FRESHNESS_RECT_MUON_SECOND_SHRINK)
            self._freshness_rect_applied = True

        if not self._freshness_square_applied and progress >= FRESHNESS_SQUARE_MUON_START:
            for group in self.param_groups:
                if group["kind"] != "muon" or group["shape_class"] != "square":
                    continue
                params = group["params"]
                if not params:
                    continue
                state = self.state.get(params[0])
                if not state:
                    continue
                if "momentum_buffer" in state:
                    state["momentum_buffer"].mul_(FRESHNESS_SQUARE_MUON_MOMENTUM_SHRINK)
                if "second_momentum_buffer" in state:
                    state["second_momentum_buffer"].mul_(FRESHNESS_SQUARE_MUON_SECOND_SHRINK)
            self._freshness_square_applied = True

    def maybe_apply_late_handoff(self, progress):
        if self._late_handoff_applied or progress < LATE_ACCUM_START:
            return
        for group in self.param_groups:
            if group["kind"] == "muon":
                params = group["params"]
                if not params:
                    continue
                state = self.state.get(params[0])
                if not state:
                    continue
                if group["shape_class"] == "square":
                    momentum_shrink = LATE_SQUARE_MUON_MOMENTUM_SHRINK
                    second_shrink = LATE_SQUARE_MUON_SECOND_SHRINK
                else:
                    momentum_shrink = LATE_RECT_MUON_MOMENTUM_SHRINK
                    second_shrink = LATE_RECT_MUON_SECOND_SHRINK
                if "momentum_buffer" in state:
                    state["momentum_buffer"].mul_(momentum_shrink)
                if "second_momentum_buffer" in state:
                    state["second_momentum_buffer"].mul_(second_shrink)
                continue

            if group["subkind"] in ("resid", "x0"):
                exp_avg_shrink = LATE_SCALAR_EXP_AVG_SHRINK
                exp_avg_sq_shrink = LATE_SCALAR_EXP_AVG_SQ_SHRINK
            elif group["subkind"] == "lm_head":
                exp_avg_shrink = LATE_HEAD_EXP_AVG_SHRINK
                exp_avg_sq_shrink = LATE_HEAD_EXP_AVG_SQ_SHRINK
            else:
                exp_avg_shrink = LATE_EMBED_EXP_AVG_SHRINK
                exp_avg_sq_shrink = LATE_EMBED_EXP_AVG_SQ_SHRINK

            for p in group["params"]:
                state = self.state.get(p)
                if not state:
                    continue
                if "exp_avg" in state:
                    state["exp_avg"].mul_(exp_avg_shrink)
                if "exp_avg_sq" in state:
                    state["exp_avg_sq"].mul_(exp_avg_sq_shrink)
        self._late_handoff_applied = True

    def _step_adamw(self, group):
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            adamw_step_fused(p, grad, state['exp_avg'], state['exp_avg_sq'],
                            self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                            self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)

    def _step_muon(self, group):
        params = group['params']
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        self._apply_reset_ladder(group, state)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        muon_step_fused(stacked_grads, stacked_params,
                        state["momentum_buffer"], state["second_momentum_buffer"],
                        self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t,
                        self._muon_beta2_t, group["ns_steps"], red_dim)
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    def reset_muon_state(self):
        for group in self.param_groups:
            if group["kind"] != "muon":
                continue
            params = group["params"]
            if not params:
                continue
            state = self.state.get(params[0])
            if not state:
                continue
            if "momentum_buffer" in state:
                state["momentum_buffer"].zero_()
            if "second_momentum_buffer" in state:
                state["second_momentum_buffer"].zero_()

    def maybe_apply_late_rre(self, step, progress):
        return self._late_rre.maybe_step(step, progress, self)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
ASPECT_RATIO = 64       # model_dim = depth * ASPECT_RATIO
HEAD_DIM = 128          # target head dimension for attention
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context
ATTNRES_MODE = "full"   # baseline or full Attention Residuals depth mixing
ATTNRES_USE_RMSNORM = True

# Optimization
TOTAL_BATCH_SIZE = 196608 # moderate batch line: more optimizer steps in 5 minutes
EMBEDDING_LR = 0.6      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
MATRIX_LR = 0.04        # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
# Keep the early 039 schedule, then add a staged Muon reset ladder and a cautious tail rebound.
SWITCHBACK_RECAP_START = 0.68
SWITCHBACK_RECAP_END = 0.84
SQUARE_RESET_START = 0.44
SQUARE_RESET_PEAK = 0.47
SQUARE_RESET_END = 0.50
RECT_RESET_START = 0.62
RECT_RESET_PEAK = 0.66
RECT_RESET_END = 0.70
SQUARE_MUON_MOMENTUM_SHRINK = 0.42
SQUARE_MUON_SECOND_SHRINK = 0.60
RECT_MUON_MOMENTUM_SHRINK = 0.72
RECT_MUON_SECOND_SHRINK = 0.84
SCALAR_QUIET_1_START = 0.43
SCALAR_QUIET_1_PEAK = 0.47
SCALAR_QUIET_1_END = 0.52
SCALAR_QUIET_2_START = 0.60
SCALAR_QUIET_2_PEAK = 0.66
SCALAR_QUIET_2_END = 0.70
RESID_QUIET_1_FLOOR = 0.00
RESID_QUIET_2_FLOOR = 0.18
X0_QUIET_1_FLOOR = 0.10
X0_QUIET_2_FLOOR = 0.28
FINAL_LR_FRAC = 0.125   # fallback final LR fraction for uncategorized Adam groups
HEAD_FINAL_LR_FRAC = 0.08
TOKEN_EMBED_FINAL_LR_FRAC = 0.03
VALUE_EMBED_FINAL_LR_FRAC = 0.05
RESID_FINAL_LR_FRAC = 0.10
X0_FINAL_LR_FRAC = 0.18
MUON_SQUARE_FINAL_LR_FRAC = 0.24
MUON_RECT_FINAL_LR_FRAC = 0.16
MUON_SQUARE_FINAL_MOMENTUM = 0.90
MUON_RECT_FINAL_MOMENTUM = 0.93
MUON_SQUARE_FINAL_BETA2 = 0.89
MUON_RECT_FINAL_BETA2 = 0.92
WEIGHT_DECAY_FLOOR = 0.04
WEIGHT_DECAY_REBOUND = 0.06
WEIGHT_DECAY_REBOUND_START = 0.76
WEIGHT_DECAY_REBOUND_PEAK = 0.90
WEIGHT_DECAY_REBOUND_END = 1.00
LATE_ACCUM_START = 0.82
LATE_ACCUM_STEP_DELTA = 1
LATE_SQUARE_MUON_MOMENTUM_SHRINK = 0.58
LATE_SQUARE_MUON_SECOND_SHRINK = 0.78
LATE_RECT_MUON_MOMENTUM_SHRINK = 0.70
LATE_RECT_MUON_SECOND_SHRINK = 0.86
LATE_HEAD_EXP_AVG_SHRINK = 0.55
LATE_HEAD_EXP_AVG_SQ_SHRINK = 0.84
LATE_EMBED_EXP_AVG_SHRINK = 0.65
LATE_EMBED_EXP_AVG_SQ_SHRINK = 0.88
LATE_SCALAR_EXP_AVG_SHRINK = 0.25
LATE_SCALAR_EXP_AVG_SQ_SHRINK = 0.72
FRESHNESS_ADAM_START = 0.72
FRESHNESS_RECT_MUON_START = 0.84
FRESHNESS_SQUARE_MUON_START = 0.90
FRESHNESS_HEAD_EXP_AVG_SHRINK = 0.74
FRESHNESS_HEAD_EXP_AVG_SQ_SHRINK = 0.90
FRESHNESS_EMBED_EXP_AVG_SHRINK = 0.84
FRESHNESS_EMBED_EXP_AVG_SQ_SHRINK = 0.94
FRESHNESS_RECT_MUON_MOMENTUM_SHRINK = 0.90
FRESHNESS_RECT_MUON_SECOND_SHRINK = 0.95
FRESHNESS_SQUARE_MUON_MOMENTUM_SHRINK = 0.92
FRESHNESS_SQUARE_MUON_SECOND_SHRINK = 0.96
OUTPUT_CLEANUP_START = 0.90
OUTPUT_CLEANUP_PEAK = 0.96
OUTPUT_CLEANUP_END = 1.00
LM_HEAD_CLEANUP_BOOST = 0.12
TOKEN_EMBED_CLEANUP_BOOST = 0.16
SCALAR_CLEANUP_BOOST = 0.08
TOKEN_EMBED_COOL_START = 0.22
TOKEN_EMBED_COOL_MID = 0.40
TOKEN_EMBED_COOL_END = 0.84
TOKEN_EMBED_COOL_FINAL_FRAC = 0.46
VALUE_EMBED_COOL_START = 0.26
VALUE_EMBED_COOL_MID = 0.44
VALUE_EMBED_COOL_END = 0.72
VALUE_EMBED_COOL_MID_FRAC = 0.92
VALUE_EMBED_LATE_HEAT_START = 0.70
VALUE_EMBED_LATE_HEAT_END = 1.00
VALUE_EMBED_LATE_HEAT_BOOST = 0.10
GEOMETRY_MUON_HEAT_START = 0.58
GEOMETRY_MUON_RECT_BOOST = 0.05
GEOMETRY_MUON_SQUARE_BOOST = 0.08
RRE_PER_LAYER = True
RRE_EVERY = 24
RRE_NUM_CHECKPOINTS = 4
RRE_LATE_START = 0.68
RRE_RESET_STATE = True
RRE_REGULARIZATION = 1e-6
EXTRAP_COOLDOWN_START = 0.70
EXTRAP_FINAL_LR_SCALE = 0.58
EXTRAP_FINAL_MUON_MOMENTUM_DROP = 0.04

# Model size
DEPTH = 8               # number of transformer layers
DEVICE_BATCH_SIZE = 32   # per-device batch size (reduce if OOM)

# ---------------------------------------------------------------------------
# Setup: tokenizer, model, optimizer, dataloader
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
H100_BF16_PEAK_FLOPS = 989.5e12

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")

def build_model_config(depth):
    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
        attnres_mode=ATTNRES_MODE,
        attnres_use_rmsnorm=ATTNRES_USE_RMSNORM,
    )

config = build_model_config(DEPTH)
print(f"Model config: {asdict(config)}")

with torch.device("meta"):
    model = GPT(config)
model.to_empty(device=device)
model.init_weights()

param_counts = model.num_scaling_params()
print("Parameter counts:")
for key, value in param_counts.items():
    print(f"  {key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
base_grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd
tail_grad_accum_steps = base_grad_accum_steps + LATE_ACCUM_STEP_DELTA

optimizer = model.setup_optimizer(
    unembedding_lr=UNEMBEDDING_LR,
    embedding_lr=EMBEDDING_LR,
    scalar_lr=SCALAR_LR,
    adam_betas=ADAM_BETAS,
    matrix_lr=MATRIX_LR,
    weight_decay=WEIGHT_DECAY,
)

model = torch.compile(model, dynamic=False)

train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
x, y, epoch = next(train_loader)  # prefetch first batch

print(f"Time budget: {TIME_BUDGET}s")
print(f"Base gradient accumulation steps: {base_grad_accum_steps}")
print(f"Tail gradient accumulation steps: {tail_grad_accum_steps} (starts at progress {LATE_ACCUM_START:.2f})")

# Schedules (all based on progress = training_time / TIME_BUDGET)

def lerp(a, b, t):
    return a + (b - a) * t

def phase_mix(progress, start, end):
    if end <= start:
        return 1.0
    return max(0.0, min(1.0, (progress - start) / (end - start)))

def cosine_bell(progress, start, peak, end):
    if progress <= start or progress >= end:
        return 0.0
    if progress <= peak:
        return 0.5 * (1.0 - math.cos(math.pi * phase_mix(progress, start, peak)))
    return 0.5 * (1.0 + math.cos(math.pi * phase_mix(progress, peak, end)))

def quiet_window_scale(progress, start, peak, end, floor):
    return lerp(1.0, floor, cosine_bell(progress, start, peak, end))

def get_adam_final_lr_frac(subkind):
    if subkind == 'lm_head':
        return HEAD_FINAL_LR_FRAC
    if subkind == 'token_embed':
        return TOKEN_EMBED_FINAL_LR_FRAC
    if subkind == 'value_embed':
        return VALUE_EMBED_FINAL_LR_FRAC
    if subkind == 'resid':
        return RESID_FINAL_LR_FRAC
    if subkind == 'x0':
        return X0_FINAL_LR_FRAC
    return FINAL_LR_FRAC

def get_muon_final_lr_frac(shape_class):
    if shape_class == 'square':
        return MUON_SQUARE_FINAL_LR_FRAC
    return MUON_RECT_FINAL_LR_FRAC

def cooldown_hold_scale(progress, start, mid, end, floor):
    if progress < start:
        return 1.0
    if progress < mid:
        return lerp(1.0, floor, phase_mix(progress, start, mid))
    return lerp(floor, floor, phase_mix(progress, mid, end))

def cleanup_wedge_scale(progress, boost):
    return 1.0 + boost * cosine_bell(progress, OUTPUT_CLEANUP_START, OUTPUT_CLEANUP_PEAK, OUTPUT_CLEANUP_END)

def get_scalar_quiet_scale(subkind, progress):
    quiet_1 = quiet_window_scale(
        progress, SCALAR_QUIET_1_START, SCALAR_QUIET_1_PEAK, SCALAR_QUIET_1_END,
        RESID_QUIET_1_FLOOR if subkind == 'resid' else X0_QUIET_1_FLOOR,
    )
    quiet_2 = quiet_window_scale(
        progress, SCALAR_QUIET_2_START, SCALAR_QUIET_2_PEAK, SCALAR_QUIET_2_END,
        RESID_QUIET_2_FLOOR if subkind == 'resid' else X0_QUIET_2_FLOOR,
    )
    return min(quiet_1, quiet_2)

def get_output_handoff_scale(subkind, progress):
    if subkind == 'lm_head':
        cool = cooldown_hold_scale(progress, 0.26, 0.46, 0.86, 0.56)
        return cool * cleanup_wedge_scale(progress, LM_HEAD_CLEANUP_BOOST)
    if subkind == 'token_embed':
        cool = cooldown_hold_scale(
            progress, TOKEN_EMBED_COOL_START, TOKEN_EMBED_COOL_MID, TOKEN_EMBED_COOL_END,
            TOKEN_EMBED_COOL_FINAL_FRAC,
        )
        return cool * cleanup_wedge_scale(progress, TOKEN_EMBED_CLEANUP_BOOST)
    if subkind in ('resid', 'x0'):
        return get_scalar_quiet_scale(subkind, progress) * cleanup_wedge_scale(progress, SCALAR_CLEANUP_BOOST)
    return 1.0

def get_value_geometry_scale(progress):
    cool = cooldown_hold_scale(
        progress, VALUE_EMBED_COOL_START, VALUE_EMBED_COOL_MID, VALUE_EMBED_COOL_END,
        VALUE_EMBED_COOL_MID_FRAC,
    )
    late_heat = 1.0
    if progress >= VALUE_EMBED_LATE_HEAT_START:
        late_heat = lerp(
            1.0,
            1.0 + VALUE_EMBED_LATE_HEAT_BOOST,
            phase_mix(progress, VALUE_EMBED_LATE_HEAT_START, VALUE_EMBED_LATE_HEAT_END),
        )
    return cool * late_heat

def get_muon_geometry_scale(shape_class, progress):
    boost = GEOMETRY_MUON_SQUARE_BOOST if shape_class == 'square' else GEOMETRY_MUON_RECT_BOOST
    if progress < GEOMETRY_MUON_HEAT_START:
        return 1.0
    return lerp(1.0, 1.0 + boost, phase_mix(progress, GEOMETRY_MUON_HEAT_START, 1.0))

def get_geometry_bank_scale(group, progress):
    if group["kind"] == "muon":
        return get_muon_geometry_scale(group["shape_class"], progress)
    if group.get("subkind") == 'value_embed':
        return get_value_geometry_scale(progress)
    return 1.0

def get_adam_switchback_lr_profile(subkind):
    if subkind == 'lm_head':
        return 0.68, 1.05, 0.40, get_adam_final_lr_frac(subkind)
    if subkind == 'token_embed':
        return 0.55, 0.42, 0.24, get_adam_final_lr_frac(subkind)
    if subkind == 'value_embed':
        return 0.58, 0.48, 0.28, get_adam_final_lr_frac(subkind)
    if subkind == 'resid':
        return 0.84, 1.10, 0.52, get_adam_final_lr_frac(subkind)
    if subkind == 'x0':
        return 0.92, 1.18, 0.58, get_adam_final_lr_frac(subkind)
    return 0.60, 0.85, 0.35, get_adam_final_lr_frac(subkind)

def get_muon_switchback_lr_profile(shape_class):
    if shape_class == 'square':
        return 0.98, 0.54, 0.72, get_muon_final_lr_frac(shape_class)
    return 0.92, 0.60, 0.74, get_muon_final_lr_frac(shape_class)

def get_group_lr_multiplier(group, progress):
    if group['kind'] == 'muon':
        start_lr_frac, recap_lr_frac, recovery_lr_frac, final_lr_frac = get_muon_switchback_lr_profile(group['shape_class'])
    else:
        start_lr_frac, recap_lr_frac, recovery_lr_frac, final_lr_frac = get_adam_switchback_lr_profile(group['subkind'])

    if progress < SWITCHBACK_RECAP_START:
        base = lerp(start_lr_frac, recap_lr_frac, phase_mix(progress, 0.0, SWITCHBACK_RECAP_START))
    elif progress < SWITCHBACK_RECAP_END:
        base = lerp(recap_lr_frac, recovery_lr_frac, phase_mix(progress, SWITCHBACK_RECAP_START, SWITCHBACK_RECAP_END))
    else:
        base = lerp(recovery_lr_frac, final_lr_frac, phase_mix(progress, SWITCHBACK_RECAP_END, 1.0))
    if group["kind"] == "muon" or group.get("subkind") == "value_embed":
        return base * get_geometry_bank_scale(group, progress)
    if group["kind"] == "adamw" and group["subkind"] in ("lm_head", "token_embed", "resid", "x0"):
        return base * get_output_handoff_scale(group["subkind"], progress)
    return base

def get_muon_momentum(step, progress, shape_class):
    warm_momentum = lerp(0.86, 0.95, min(step / 300, 1))
    if progress < SWITCHBACK_RECAP_START:
        return lerp(warm_momentum, 0.95, phase_mix(progress, 0.0, SWITCHBACK_RECAP_START))
    if progress < SWITCHBACK_RECAP_END:
        return lerp(0.95, 0.88, phase_mix(progress, SWITCHBACK_RECAP_START, SWITCHBACK_RECAP_END))
    final_momentum = MUON_SQUARE_FINAL_MOMENTUM if shape_class == 'square' else MUON_RECT_FINAL_MOMENTUM
    return lerp(0.88, final_momentum, phase_mix(progress, SWITCHBACK_RECAP_END, 1.0))

def get_muon_beta2(progress, shape_class):
    if progress < SWITCHBACK_RECAP_START:
        return 0.95
    if progress < SWITCHBACK_RECAP_END:
        return lerp(0.95, 0.91, phase_mix(progress, SWITCHBACK_RECAP_START, SWITCHBACK_RECAP_END))
    final_beta2 = MUON_SQUARE_FINAL_BETA2 if shape_class == 'square' else MUON_RECT_FINAL_BETA2
    return lerp(0.91, final_beta2, phase_mix(progress, SWITCHBACK_RECAP_END, 1.0))

def get_weight_decay(progress):
    base = lerp(WEIGHT_DECAY, WEIGHT_DECAY_FLOOR, phase_mix(progress, 0.0, WEIGHT_DECAY_REBOUND_START))
    rebound = WEIGHT_DECAY_REBOUND * cosine_bell(
        progress, WEIGHT_DECAY_REBOUND_START, WEIGHT_DECAY_REBOUND_PEAK, WEIGHT_DECAY_REBOUND_END
    )
    return base + rebound

def get_extrap_cooldown_scale(progress):
    if progress < EXTRAP_COOLDOWN_START:
        return 1.0
    return lerp(1.0, EXTRAP_FINAL_LR_SCALE, phase_mix(progress, EXTRAP_COOLDOWN_START, 1.0))

def get_extrap_momentum_cooldown(progress):
    if progress < EXTRAP_COOLDOWN_START:
        return 0.0
    return lerp(0.0, EXTRAP_FINAL_MUON_MOMENTUM_DROP, phase_mix(progress, EXTRAP_COOLDOWN_START, 1.0))

def get_grad_accum_steps(progress):
    if progress < LATE_ACCUM_START:
        return base_grad_accum_steps
    return tail_grad_accum_steps

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
total_tokens = 0
measured_tokens = 0
step = 0

while True:
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    current_grad_accum_steps = get_grad_accum_steps(progress)
    current_total_batch_size = current_grad_accum_steps * tokens_per_fwdbwd
    optimizer.set_schedule_progress(progress)
    optimizer.maybe_apply_freshness_ladder(progress)
    optimizer.maybe_apply_late_handoff(progress)
    torch.cuda.synchronize()
    t0 = time.time()
    for micro_step in range(current_grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / current_grad_accum_steps
        loss.backward()
        x, y, epoch = next(train_loader)

    # Progress and schedules
    muon_weight_decay = get_weight_decay(progress)
    extrap_lr_scale = get_extrap_cooldown_scale(progress)
    report_lrm = None
    optimizer.set_schedule_progress(progress)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * get_group_lr_multiplier(group, progress) * extrap_lr_scale
        if group['kind'] == 'muon':
            group["momentum"] = max(
                0.80,
                get_muon_momentum(step, progress, group["shape_class"]) - get_extrap_momentum_cooldown(progress),
            )
            group["beta2"] = get_muon_beta2(progress, group["shape_class"])
            group["weight_decay"] = muon_weight_decay
            if report_lrm is None and group["shape_class"] == "square":
                report_lrm = group["lr"] / group["initial_lr"]
    if report_lrm is None:
        report_lrm = 0.0
    optimizer.step()
    did_extrap = optimizer.maybe_apply_late_rre(step + 1, progress)
    model.zero_grad(set_to_none=True)

    train_loss_f = train_loss.item()

    # Fast fail: abort if loss is exploding or NaN
    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("FAIL")
        exit(1)

    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0

    if step > 10:
        total_training_time += dt
        measured_tokens += current_total_batch_size
    total_tokens += current_total_batch_size

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(current_total_batch_size / dt)
    mfu = 100 * num_flops_per_token * current_total_batch_size / dt / H100_BF16_PEAK_FLOPS
    remaining = max(0, TIME_BUDGET - total_training_time)

    print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {report_lrm:.2f} | accum: {current_grad_accum_steps} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.1f}% | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)
    if did_extrap:
        print(f"\nlate_rre step {step + 1} progress {progress:.3f} per_layer {RRE_PER_LAYER} every {RRE_EVERY}", flush=True)

    # GC management (Python's GC causes ~500ms stalls)
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1

    # Time's up — but only stop after warmup steps so we don't count compilation
    if step > 10 and total_training_time >= TIME_BUDGET:
        break

print()  # newline after \r training log

# Final eval
model.eval()
with autocast_ctx:
    val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)

# Final summary
t_end = time.time()
startup_time = t_start_training - t_start
steady_state_mfu = 100 * num_flops_per_token * measured_tokens / total_training_time / H100_BF16_PEAK_FLOPS if total_training_time > 0 else 0
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"mfu_percent:      {steady_state_mfu:.2f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {DEPTH}")
