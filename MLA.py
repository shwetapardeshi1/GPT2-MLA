# MLA.py
from __future__ import annotations
import math, os, argparse
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    GPT2LMHeadModel, GPT2Config, AutoTokenizer, Trainer, TrainingArguments
)

# -----------------------------
# Logging Setup
# -----------------------------
def setup_logging(log_file: str = "mla_debug.log"):
    """Set up logging to both console and file"""
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else ".", exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),  # 'w' mode overwrites the file each run
            logging.StreamHandler()  # This will still print to console
        ]
    )
    
    # Get the logger
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Debug output will be saved to: {log_file}")
    return logger

# -----------------------------
# Helpers
# -----------------------------
import inspect
from transformers import TrainingArguments

def build_training_args(args):
    base = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",   # will be stripped if unsupported
        logging_dir=f"{args.output_dir}/logs",
        report_to="none",
        save_safetensors=True,
    )

    # optional mixed precision flags (strip if unsupported)
    if torch.cuda.is_available():
        major_cc = torch.cuda.get_device_capability(0)[0]
        base.update({
            "fp16": major_cc < 8,
            "bf16": major_cc >= 8,
        })

    # keep only kwargs accepted by this install of TrainingArguments
    sig = inspect.signature(TrainingArguments.__init__)
    allowed = set(sig.parameters) - {"self", "kwargs"}
    filtered = {k: v for k, v in base.items() if k in allowed}

    return TrainingArguments(**filtered)


def to_txt(path: str, array_2d):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for row in array_2d:
            f.write(",".join(str(x) for x in row) + "\n")

def quantize_to_int64(x: np.ndarray, nl_scale: int, nl_ell: int = 64) -> np.ndarray:
    scale = 1 << nl_scale
    q = np.round(x * scale).astype(object)
    if nl_ell == 64:
        mod = 1 << 64
        flat = [int(v) % mod for v in q.ravel()]
    else:
        mask = (1 << nl_ell) - 1
        flat = [int(v) & mask for v in q.ravel()]
    return np.array(flat, dtype=np.uint64).reshape(x.shape)

# -----------------------------
# MLA core (no RMSNorm)
# -----------------------------
@dataclass
class MLAConfig:
    hidden_size: int
    num_heads: int
    q_rank: int
    k_rank: int
    v_rank: int
    head_dim: int
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0
    use_bias: bool = False

class MLAAttention(nn.Module):
    def __init__(self, cfg: MLAConfig):
        super().__init__()
        d, H, d_h = cfg.hidden_size, cfg.num_heads, cfg.head_dim
        assert d == H * d_h, f"hidden_size ({d}) must equal num_heads*head_dim ({H}*{d_h})"
        self.cfg = cfg
        
        logging.info(f"\n=== MLA ATTENTION INITIALIZATION ===")
        logging.info(f"Hidden size (d) = {d}, Num heads (H) = {H}, Head dim (d_h) = {d_h}")
        logging.info(f"Q rank = {cfg.q_rank}, K rank = {cfg.k_rank}, V rank = {cfg.v_rank}")
        logging.info(f"Total parameters: Q={d*cfg.q_rank + cfg.q_rank*d}, K={d*cfg.k_rank + cfg.k_rank*d}, V={d*cfg.v_rank + cfg.v_rank*d}")

        # Down projections (d -> ranks)
        self.q_down = nn.Linear(d, cfg.q_rank, bias=cfg.use_bias)
        self.k_down = nn.Linear(d, cfg.k_rank, bias=cfg.use_bias)
        self.v_down = nn.Linear(d, cfg.v_rank, bias=cfg.use_bias)
        logging.info(f"Down projection layers created:")
        logging.info(f"  q_down: {d} -> {cfg.q_rank} (weight: {self.q_down.weight.shape})")
        logging.info(f"  k_down: {d} -> {cfg.k_rank} (weight: {self.k_down.weight.shape})")
        logging.info(f"  v_down: {d} -> {cfg.v_rank} (weight: {self.v_down.weight.shape})")

        # Up projections (ranks -> d)
        self.q_up = nn.Linear(cfg.q_rank, d, bias=False)
        self.k_up = nn.Linear(cfg.k_rank, d, bias=False)
        self.v_up = nn.Linear(cfg.v_rank, d, bias=False)
        logging.info(f"Up projection layers created:")
        logging.info(f"  q_up: {cfg.q_rank} -> {d} (weight: {self.q_up.weight.shape})")
        logging.info(f"  k_up: {cfg.k_rank} -> {d} (weight: {self.k_up.weight.shape})")
        logging.info(f"  v_up: {cfg.v_rank} -> {d} (weight: {self.v_up.weight.shape})")

        # Output
        self.out_proj = nn.Linear(d, d, bias=True)
        self.attn_drop = nn.Dropout(cfg.attn_dropout)
        self.resid_drop = nn.Dropout(cfg.resid_dropout)
        logging.info(f"Output projection: {d} -> {d} (weight: {self.out_proj.weight.shape})")
        logging.info(f"Dropout: attention={cfg.attn_dropout}, residual={cfg.resid_dropout}")

        # Cache for absorbed score matrix at eval
        self._M_cache = None
        logging.info(f"=== END MLA ATTENTION INITIALIZATION ===\n")

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self._M_cache = None  # invalidate cache when returning to train mode
        return self

    def _heads_view(self, weight: torch.Tensor, rank: int) -> torch.Tensor:
        """
        Reshape a (d, r) matrix into per-head form (H, r, d_h).
        """
        H, d_h, r = self.cfg.num_heads, self.cfg.head_dim, rank
        logging.info(f"  _heads_view: input weight shape={weight.shape}, H={H}, d_h={d_h}, r={rank}")
        result = weight.view(H, d_h, r).transpose(1, 2).contiguous()
        logging.info(f"  _heads_view: output shape={result.shape}")
        return result

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        B, T, d = x.shape
        H, d_h = self.cfg.num_heads, self.cfg.head_dim
        
        logging.info(f"\n=== MLA ATTENTION FORWARD PASS ===")
        logging.info(f"Input x: shape={x.shape}, dtype={x.dtype}, device={x.device}")
        logging.info(f"Batch size (B)={B}, Sequence length (T)={T}, Hidden dim (d)={d}")
        logging.info(f"Number of heads (H)={H}, Head dimension (d_h)={d_h}")
        logging.info(f"Q rank={self.cfg.q_rank}, K rank={self.cfg.k_rank}, V rank={self.cfg.v_rank}")

        # ---- Down project Q/K/V ----
        logging.info(f"\n--- Down Projection Q/K/V ---")
        Qd = self.q_down(x)        # (B, T, r_q)
        Kd_new = self.k_down(x)    # (B, T, r_k)
        Vd_new = self.v_down(x)    # (B, T, r_v)
        logging.info(f"Qd: shape={Qd.shape}, dtype={Qd.dtype}")
        logging.info(f"Kd_new: shape={Kd_new.shape}, dtype={Kd_new.dtype}")
        logging.info(f"Vd_new: shape={Vd_new.shape}, dtype={Vd_new.dtype}")

        if past_kv is not None:
            logging.info(f"Past KV provided - Kd: shape={past_kv[0].shape}, Vd: shape={past_kv[1].shape}")
            Kd = torch.cat([past_kv[0], Kd_new], dim=1)  # (B, S, r_k)
            Vd = torch.cat([past_kv[1], Vd_new], dim=1)  # (B, S, r_v)
            logging.info(f"Concatenated - Kd: shape={Kd.shape}, Vd: shape={Vd.shape}")
        else:
            Kd, Vd = Kd_new, Vd_new
            logging.info(f"No past KV - using new: Kd: shape={Kd.shape}, Vd: shape={Vd.shape}")
            
        S = Kd.size(1)  # key length (past + current)
        logging.info(f"Total key length (S) = {S}")

        # ---- Build absorbed score matrix M = Uq * Uk^T / sqrt(d_h) ----
        logging.info(f"\n--- Building Absorbed Score Matrix M ---")
        if (not self.training) and (self._M_cache is not None):
            M = self._M_cache
            logging.info(f"Using cached M matrix: shape={M.shape}")
        else:
            logging.info(f"Computing M matrix from scratch...")
            Uq_h = self._heads_view(self.q_up.weight, self.cfg.q_rank)  # (H, r_q, d_h)
            Uk_h = self._heads_view(self.k_up.weight, self.cfg.k_rank)  # (H, r_k, d_h)
            logging.info(f"Uq_h (q_up weight reshaped): shape={Uq_h.shape}")
            logging.info(f"Uk_h (k_up weight reshaped): shape={Uk_h.shape}")
            
            M = torch.einsum("hqd,hkd->hqk", Uq_h, Uk_h) * (1.0 / math.sqrt(d_h))  # (H, r_q, r_k)
            logging.info(f"M matrix (Uq_h @ Uk_h^T / sqrt(d_h)): shape={M.shape}")
            logging.info(f"M matrix stats: min={M.min().item():.6f}, max={M.max().item():.6f}, mean={M.mean().item():.6f}")
            
            if not self.training:
                self._M_cache = M
                logging.info(f"Cached M matrix for future use")

        # ---- Logits in compressed-Q space: (Qd M) Kd^T (fused) ----
        print(f"\n--- Computing Logits ---")
        print(f"Qd: shape={Qd.shape}, M: shape={M.shape}, Kd: shape={Kd.shape}")
        # Qd: (B,T,r_q), M: (H,r_q,r_k), Kd: (B,S,r_k) -> logits: (B,H,T,S)
        logits = torch.einsum("btq,hqk,bsk->bhts", Qd, M, Kd)
        print(f"Logits computed: shape={logits.shape}")
        print(f"Logits stats: min={logits.min().item():.6f}, max={logits.max().item():.6f}, mean={logits.mean().item():.6f}")

        # ---- Masks ----
        print(f"\n--- Applying Masks ---")
        # causal mask with past offset
        P = S - T  # tokens from past cache
        i = torch.arange(T, device=x.device)[:, None]       # (T,1)
        j = torch.arange(S, device=x.device)[None, :]       # (1,S)
        future_mask = j > (P + i)                           # (T,S)
        print(f"Past length (P) = {P}")
        print(f"Future mask: shape={future_mask.shape}")
        print(f"Future mask: {future_mask.sum().item()} masked positions out of {future_mask.numel()}")
        
        logits = logits.masked_fill(future_mask[None, None, :, :], float("-inf"))
        print(f"After future mask - logits stats: min={logits.min().item():.6f}, max={logits.max().item():.6f}")

        # key padding mask: attention_mask: (B,S) with 1=keep, 0=mask
        if attention_mask is not None:
            key_pad = (attention_mask == 0).to(torch.bool)  # (B,S)
            print(f"Attention mask: shape={attention_mask.shape}")
            print(f"Key padding mask: shape={key_pad.shape}, {key_pad.sum().item()} masked positions")
            logits = logits.masked_fill(key_pad[:, None, None, :], float("-inf"))
            print(f"After attention mask - logits stats: min={logits.min().item():.6f}, max={logits.max().item():.6f}")
        else:
            print(f"No attention mask provided")

        # ---- Softmax (fp32 for stability) ----
        print(f"\n--- Softmax and Attention Weights ---")
        attn = F.softmax(logits.float(), dim=-1)
        attn = torch.nan_to_num(attn)            # guard fully-masked rows
        attn = self.attn_drop(attn).to(x.dtype)
        print(f"Attention weights: shape={attn.shape}")
        print(f"Attention weights stats: min={attn.min().item():.6f}, max={attn.max().item():.6f}, sum per seq={attn.sum(dim=-1).mean().item():.6f}")

        # ---- Values path ----
        print(f"\n--- Values Path ---")
        Uv_h = self._heads_view(self.v_up.weight, self.cfg.v_rank)  # (H, r_v, d_h)
        print(f"Uv_h (v_up weight reshaped): shape={Uv_h.shape}")
        
        V_up = torch.einsum("bsr,hrd->bhsd", Vd, Uv_h)               # (B, H, S, d_h)
        print(f"V_up (Vd @ Uv_h): shape={V_up.shape}")
        print(f"V_up stats: min={V_up.min().item():.6f}, max={V_up.max().item():.6f}, mean={V_up.mean().item():.6f}")
        
        ctx = torch.einsum("bhts,bhsd->bhtd", attn, V_up)            # (B, H, T, d_h)
        print(f"Context (attn @ V_up): shape={ctx.shape}")
        print(f"Context stats: min={ctx.min().item():.6f}, max={ctx.max().item():.6f}, mean={ctx.mean().item():.6f}")
        
        ctx = ctx.transpose(1, 2).contiguous().view(B, T, H * d_h)   # (B, T, d)
        print(f"Context reshaped: shape={ctx.shape}")

        out = self.out_proj(ctx)
        out = self.resid_drop(out)
        print(f"Output: shape={out.shape}, dtype={out.dtype}")
        print(f"Output stats: min={out.min().item():.6f}, max={out.max().item():.6f}, mean={out.mean().item():.6f}")

        present = (Kd, Vd) if use_cache else None
        if use_cache:
            print(f"Returning KV cache: Kd shape={Kd.shape}, Vd shape={Vd.shape}")
        
        print(f"=== END MLA ATTENTION FORWARD PASS ===\n")
        return out, present


class MLP(nn.Module):
    def __init__(self, hidden_size: int, multiple_of: int = 4):
        super().__init__()
        inner = multiple_of * hidden_size
        self.fc_in = nn.Linear(hidden_size, inner)
        self.act = nn.GELU()
        self.fc_out = nn.Linear(inner, hidden_size)
    def forward(self, x):
        return self.fc_out(self.act(self.fc_in(x)))

class MLAWrappedBlock(nn.Module):
    def __init__(self, gpt2_cfg: GPT2Config, mla_cfg: MLAConfig):
        super().__init__()
        d = gpt2_cfg.hidden_size
        self.ln_1 = nn.LayerNorm(d, eps=gpt2_cfg.layer_norm_epsilon)
        self.attn = MLAAttention(mla_cfg)
        self.ln_2 = nn.LayerNorm(d, eps=gpt2_cfg.layer_norm_epsilon)
        self.mlp = MLP(d, multiple_of=4)
    def forward(self, x, attention_mask=None, past_kv=None, use_cache=False):
        h, present = self.attn(self.ln_1(x), attention_mask=attention_mask, past_kv=past_kv, use_cache=use_cache)
        x = x + h
        x = x + self.mlp(self.ln_2(x))
        return x, present

class TransformerWithMLA(nn.Module):
    def __init__(self, gpt2_cfg: GPT2Config, mla_cfg: MLAConfig):
        super().__init__()
        self.gpt2_cfg = gpt2_cfg
        self.mla_cfg = mla_cfg
        self.wte = nn.Embedding(gpt2_cfg.vocab_size, gpt2_cfg.hidden_size)
        self.wpe = nn.Embedding(gpt2_cfg.n_positions, gpt2_cfg.hidden_size)
        self.drop = nn.Dropout(gpt2_cfg.embd_pdrop)
        self.blocks = nn.ModuleList([MLAWrappedBlock(gpt2_cfg, mla_cfg) for _ in range(gpt2_cfg.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(gpt2_cfg.hidden_size, eps=gpt2_cfg.layer_norm_epsilon)
    def forward(self, input_ids, attention_mask=None, use_cache=False, past_key_values=None):
        B, T = input_ids.shape
        device = input_ids.device
        pos = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        x = self.wte(input_ids) + self.wpe(pos)
        x = self.drop(x)
        presents = [] if use_cache else None
        if past_key_values is None:
            past_key_values = (None,) * len(self.blocks)
        for block, past in zip(self.blocks, past_key_values):
            x, present = block(x, attention_mask=attention_mask, past_kv=past, use_cache=use_cache)
            if use_cache:
                presents.append(present)
        x = self.ln_f(x)
        return x, (tuple(presents) if use_cache else None)

class GPT2MLAForCausalLM(nn.Module):
    def __init__(self, gpt2_cfg: GPT2Config, mla_cfg: MLAConfig, tied_lm_head: bool = True):
        super().__init__()
        self.transformer = TransformerWithMLA(gpt2_cfg, mla_cfg)
        self.lm_head = nn.Linear(gpt2_cfg.hidden_size, gpt2_cfg.vocab_size, bias=False)
        self.tied_lm_head = tied_lm_head
        if tied_lm_head:
            self.lm_head.weight = self.transformer.wte.weight  # tie like GPT-2

    def forward(self, input_ids, attention_mask=None, labels=None, use_cache=False, past_key_values=None):
        hidden, presents = self.transformer(
            input_ids, attention_mask=attention_mask, use_cache=use_cache, past_key_values=past_key_values
        )
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return {"loss": loss, "logits": logits, "past_key_values": presents}

    # --- Safetensors-friendly saving: clone tied weight in state_dict ---
    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        if self.tied_lm_head:
            # If lm_head.weight shares storage with wte.weight, clone to break aliasing
            if "lm_head.weight" in sd and "transformer.wte.weight" in sd:
                if sd["lm_head.weight"].data_ptr() == sd["transformer.wte.weight"].data_ptr():
                    sd["lm_head.weight"] = sd["lm_head.weight"].clone()
        return sd

# -----------------------------
# SVD init + hard weight copy from GPT-2
# -----------------------------
def svd_init_from_gpt2(reference: GPT2LMHeadModel,
                       mla_model: GPT2MLAForCausalLM,
                       rq: int, rk: int, rv: int):
    d = int(reference.config.hidden_size)
    rq = max(1, min(int(rq), d))
    rk = max(1, min(int(rk), d))
    rv = max(1, min(int(rv), d))
    device = next(reference.parameters()).device
    print(f"\n=== SVD INITIALIZATION FROM GPT2 ===")
    print(f"[SVD init] requested ranks q/k/v = {rq}/{rk}/{rv}, hidden={d}, device={device}")
    print(f"Reference model config: {reference.config}")
    print(f"MLA model config: {mla_model.transformer.mla_cfg}")

    # 1) token + position embeddings
    print(f"\n--- Copying Embeddings ---")
    with torch.no_grad():
        mla_model.transformer.wte.weight.copy_(reference.transformer.wte.weight.data)
        mla_model.transformer.wpe.weight.copy_(reference.transformer.wpe.weight.data)
        print(f"Token embeddings: {mla_model.transformer.wte.weight.shape}")
        print(f"Position embeddings: {mla_model.transformer.wpe.weight.shape}")

    # 2) per-layer: attention SVD + MLP + norms + c_proj
    for i, (ref_block, mla_block) in enumerate(zip(reference.transformer.h, mla_model.transformer.blocks)):
        # --- Attention Q/K/V from GPT-2 c_attn
        print(f"\n--- Layer {i}: Attention SVD ---")
        # HF GPT-2 uses Conv1D; weight is (d,3d) effectively
        W_qkv = ref_block.attn.c_attn.weight.detach().cpu().to(torch.float32).contiguous()  # (d,3d)
        d_slice = W_qkv.shape[1] // 3
        print(f"  W_qkv (GPT2 c_attn): shape={W_qkv.shape}, d_slice={d_slice}")
        
        W_q = W_qkv[:, 0*d_slice:1*d_slice].contiguous()  # (d,d)
        W_k = W_qkv[:, 1*d_slice:2*d_slice].contiguous()
        W_v = W_qkv[:, 2*d_slice:3*d_slice].contiguous()
        print(f"  W_q: shape={W_q.shape}, W_k: shape={W_k.shape}, W_v: shape={W_v.shape}")
        
        with torch.no_grad():
            print(f"  Computing SVD for Q...")
            Uq, Sq, Vhq = torch.linalg.svd(W_q, full_matrices=False)
            print(f"    Uq: shape={Uq.shape}, Sq: shape={Sq.shape}, Vhq: shape={Vhq.shape}")
            print(f"    Sq stats: min={Sq.min().item():.6f}, max={Sq.max().item():.6f}, mean={Sq.mean().item():.6f}")
            
            print(f"  Computing SVD for K...")
            Uk, Sk, Vhk = torch.linalg.svd(W_k, full_matrices=False)
            print(f"    Uk: shape={Uk.shape}, Sk: shape={Sk.shape}, Vhk: shape={Vhk.shape}")
            print(f"    Sk stats: min={Sk.min().item():.6f}, max={Sk.max().item():.6f}, mean={Sk.mean().item():.6f}")
            
            print(f"  Computing SVD for V...")
            Uv, Sv, Vhv = torch.linalg.svd(W_v, full_matrices=False)
            print(f"    Uv: shape={Uv.shape}, Sv: shape={Sv.shape}, Vhv: shape={Vhv.shape}")
            print(f"    Sv stats: min={Sv.min().item():.6f}, max={Sv.max().item():.6f}, mean={Sv.mean().item():.6f}")
            
            Vq = Vhq.transpose(-2, -1); Vk = Vhk.transpose(-2, -1); Vv = Vhv.transpose(-2, -1)
            print(f"  V matrices: Vq={Vq.shape}, Vk={Vk.shape}, Vv={Vv.shape}")
            
            # W ≈ (U sqrt(S))(V sqrt(S))^T
            print(f"  Computing low-rank approximations...")
            Aq = Uq[:, :rq] @ torch.diag(Sq[:rq].sqrt())   # (d,rq)
            Ak = Uk[:, :rk] @ torch.diag(Sk[:rk].sqrt())   # (d,rk)
            Av = Uv[:, :rv] @ torch.diag(Sv[:rv].sqrt())   # (d,rv)
            Bq = Vq[:, :rq] @ torch.diag(Sq[:rq].sqrt())   # (d,rq)
            Bk = Vk[:, :rk] @ torch.diag(Sk[:rk].sqrt())   # (d,rk)
            Bv = Vv[:, :rv] @ torch.diag(Sv[:rv].sqrt())   # (d,rv)
            
            print(f"  Low-rank matrices:")
            print(f"    Aq: shape={Aq.shape}, Ak: shape={Ak.shape}, Av: shape={Av.shape}")
            print(f"    Bq: shape={Bq.shape}, Bk: shape={Bk.shape}, Bv: shape={Bv.shape}")

            # copy into down (r,d) and up (d,r)
            print(f"  Copying weights to MLA layers...")
            mla_block.attn.q_down.weight.copy_(Aq.t())
            mla_block.attn.k_down.weight.copy_(Ak.t())
            mla_block.attn.v_down.weight.copy_(Av.t())
            mla_block.attn.q_up.weight.copy_(Bq)
            mla_block.attn.k_up.weight.copy_(Bk)
            mla_block.attn.v_up.weight.copy_(Bv)
            print(f"    Down weights: q_down={mla_block.attn.q_down.weight.shape}, k_down={mla_block.attn.k_down.weight.shape}, v_down={mla_block.attn.v_down.weight.shape}")
            print(f"    Up weights: q_up={mla_block.attn.q_up.weight.shape}, k_up={mla_block.attn.k_up.weight.shape}, v_up={mla_block.attn.v_up.weight.shape}")

            # output proj
            mla_block.attn.out_proj.weight.copy_(ref_block.attn.c_proj.weight.detach())
            if (ref_block.attn.c_proj.bias is not None) and (mla_block.attn.out_proj.bias is not None):
                mla_block.attn.out_proj.bias.copy_(ref_block.attn.c_proj.bias.detach())
            print(f"    Output projection: weight={mla_block.attn.out_proj.weight.shape}, bias={mla_block.attn.out_proj.bias.shape if mla_block.attn.out_proj.bias is not None else 'None'}")

            # --- MLP copy (Conv1D -> Linear)
            print(f"  Copying MLP weights...")
            mla_block.mlp.fc_in.weight.copy_(ref_block.mlp.c_fc.weight.detach().t())
            mla_block.mlp.fc_in.bias.copy_(ref_block.mlp.c_fc.bias.detach())
            mla_block.mlp.fc_out.weight.copy_(ref_block.mlp.c_proj.weight.detach().t())
            mla_block.mlp.fc_out.bias.copy_(ref_block.mlp.c_proj.bias.detach())
            print(f"    MLP fc_in: weight={mla_block.mlp.fc_in.weight.shape}, bias={mla_block.mlp.fc_in.bias.shape}")
            print(f"    MLP fc_out: weight={mla_block.mlp.fc_out.weight.shape}, bias={mla_block.mlp.fc_out.bias.shape}")

            # --- LayerNorms
            print(f"  Copying LayerNorm weights...")
            mla_block.ln_1.weight.copy_(ref_block.ln_1.weight.detach())
            mla_block.ln_1.bias.copy_(ref_block.ln_1.bias.detach())
            mla_block.ln_2.weight.copy_(ref_block.ln_2.weight.detach())
            mla_block.ln_2.bias.copy_(ref_block.ln_2.bias.detach())
            print(f"    LayerNorm 1: weight={mla_block.ln_1.weight.shape}, bias={mla_block.ln_1.bias.shape}")
            print(f"    LayerNorm 2: weight={mla_block.ln_2.weight.shape}, bias={mla_block.ln_2.bias.shape}")

        print(f"[SVD init] Layer {i}: ranks (q,k,v)=({rq},{rk},{rv})")

    # 3) final ln_f
    with torch.no_grad():
        mla_model.transformer.ln_f.weight.copy_(reference.transformer.ln_f.weight.detach())
        mla_model.transformer.ln_f.bias.copy_(reference.transformer.ln_f.bias.detach())

# -----------------------------
# Export (TXT)
# -----------------------------
def export_mla_weights(model: GPT2MLAForCausalLM, out_dir: str,
                       export_fused: bool = False,
                       quantize: bool = False, nl_scale: int = 12, nl_ell: int = 64):
    os.makedirs(out_dir, exist_ok=True)
    d = model.transformer.gpt2_cfg.hidden_size
    H = model.transformer.gpt2_cfg.num_attention_heads
    d_h = d // H
    for L, block in enumerate(model.transformer.blocks):
        attn = block.attn
        base = os.path.join(out_dir, f"layer_{L}")
        os.makedirs(base, exist_ok=True)
        # (d,r)
        Wq_d = attn.q_down.weight.detach().cpu().numpy().T
        Wk_d = attn.k_down.weight.detach().cpu().numpy().T
        Wv_d = attn.v_down.weight.detach().cpu().numpy().T
        def heads(weight, r):
            return weight.detach().cpu().numpy().reshape(H, d_h, r).transpose(0, 2, 1).copy()
        Wuq = heads(attn.q_up.weight, attn.cfg.q_rank)  # (H,rq,d_h)
        Wuk = heads(attn.k_up.weight, attn.cfg.k_rank)  # (H,rk,d_h)
        Wuv = heads(attn.v_up.weight, attn.cfg.v_rank)  # (H,rv,d_h)

        if quantize:
            Wq_d = quantize_to_int64(Wq_d, nl_scale, nl_ell)
            Wk_d = quantize_to_int64(Wk_d, nl_scale, nl_ell)
            Wv_d = quantize_to_int64(Wv_d, nl_scale, nl_ell)

        to_txt(os.path.join(base, "Wq_d.txt"), Wq_d)
        to_txt(os.path.join(base, "Wk_d.txt"), Wk_d)
        to_txt(os.path.join(base, "Wv_d.txt"), Wv_d)

        for h in range(H):
            Wuq_h, Wuk_h, Wuv_h = Wuq[h], Wuk[h], Wuv[h]
            if quantize:
                Wuq_h = quantize_to_int64(Wuq_h, nl_scale, nl_ell)
                Wuk_h = quantize_to_int64(Wuk_h, nl_scale, nl_ell)
                Wuv_h = quantize_to_int64(Wuv_h, nl_scale, nl_ell)
            to_txt(os.path.join(base, f"W_uq_head{h}.txt"), Wuq_h)
            to_txt(os.path.join(base, f"W_uk_head{h}.txt"), Wuk_h)
            to_txt(os.path.join(base, f"W_uv_head{h}.txt"), Wuv_h)

        if export_fused:
            for h in range(H):
                W_fused_k = Wuq[h] @ Wuk[h].T  # (rq,dk)@(rk,dk)^T -> (rq,rk)
                W_fused_v = Wuq[h] @ Wuv[h].T  # (rq,dk)@(rv,dk)^T -> (rq,rv)
                if quantize:
                    W_fused_k = quantize_to_int64(W_fused_k, nl_scale, nl_ell)
                    W_fused_v = quantize_to_int64(W_fused_v, nl_scale, nl_ell)
                to_txt(os.path.join(base, f"W_fused_k_head{h}.txt"), W_fused_k)
                to_txt(os.path.join(base, f"W_fused_v_head{h}.txt"), W_fused_v)
    print(f"[export] MLA weights written to {out_dir}")

# -----------------------------
# Data
# -----------------------------
def load_and_tokenize(dataset_id: str, subset: str, tokenizer, block_size: int):
    print(f"Loading dataset {dataset_id} with subset {subset}")
    if dataset_id == "wikitext":
        ds = load_dataset(path=dataset_id, name=subset)
        print(f"Loaded full wikitext dataset - no subsetting applied")
    elif dataset_id == "HuggingFaceTB/smollm-corpus":
        ds = load_dataset(dataset_id, subset)
    else:
        ds = load_dataset(dataset_id)
    print(f"Dataset columns: {ds['train'].column_names}")

    # choose text column
    text_col = None
    for c in ["text", "content", "body", "raw", "document"]:
        if c in ds["train"].column_names:
            text_col = c; break
    if text_col is None:
        if len(ds["train"].column_names) == 1:
            text_col = ds["train"].column_names[0]
        else:
            raise ValueError(f"Could not infer text column from: {ds['train'].column_names}")

    print(f"Using text column: {text_col}")
    print(f"Full dataset sizes - Train: {len(ds['train'])}, Validation: {len(ds.get('validation', ds.get('test', [])))}")

    def tok(batch):
        return tokenizer(batch[text_col])
    print("Tokenizing dataset...")
    tokenized = ds.map(tok, batched=True, remove_columns=[c for c in ds["train"].column_names if c != text_col])

    def group_texts(examples):
        concat = []
        for arr in examples["input_ids"]:
            concat.extend(arr)
        total_len = (len(concat) // block_size) * block_size
        chunks = [concat[i:i+block_size] for i in range(0, total_len, block_size)]
        attn = [[1]*block_size for _ in range(len(chunks))]
        return {"input_ids": chunks, "labels": [ids[:] for ids in chunks], "attention_mask": attn}
    print("Grouping texts into blocks...")
    lm_ds = tokenized.map(group_texts, batched=True, remove_columns=tokenized["train"].column_names)
    print(f"Final dataset sizes - Train: {len(lm_ds['train'])}, Validation: {len(lm_ds.get('validation', lm_ds.get('test', [])))}")
    return lm_ds

# -----------------------------
# Main
# -----------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", type=str, default="gpt2")
    p.add_argument("--dataset-id", type=str, default="wikitext")
    p.add_argument("--subset", type=str, default="wikitext-2-raw-v1")
    p.add_argument("--output-dir", type=str, default="./gpt2-mla-out")
    p.add_argument("--log-file", type=str, default=None, help="Custom log file path (default: output_dir/mla_debug.log)")
    p.add_argument("--export-dir", type=str, default=None)
    p.add_argument("--export-fused", action="store_true")
    p.add_argument("--quantize", action="store_true")
    p.add_argument("--nl-scale", type=int, default=12)
    p.add_argument("--nl-ell", type=int, default=64)
    p.add_argument("--block-size", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--eval-batch-size", type=int, default=2)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--rank-q", type=int, default=64)
    p.add_argument("--rank-k", type=int, default=32)
    p.add_argument("--rank-v", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")

    args = p.parse_args()

    # Set up logging to save debug output to file
    if args.log_file:
        log_file = args.log_file
    else:
        log_file = os.path.join(args.output_dir, "mla_debug.log")
    logger = setup_logging(log_file)
    logger.info(f"Starting MLA training with arguments: {args}")

    torch.manual_seed(args.seed)
    # device
    device = torch.device("cuda" if (args.device=="auto" and torch.cuda.is_available()) else (args.device if args.device!="auto" else "cpu"))
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # tokenizer + reference gpt2
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    ref_gpt2 = GPT2LMHeadModel.from_pretrained(args.model_name).to(device)
    gcfg: GPT2Config = ref_gpt2.config

    # build MLA wrapper
    print(f"\n=== BUILDING MLA MODEL ===")
    head_dim = gcfg.hidden_size // gcfg.num_attention_heads
    print(f"GPT2 config: hidden_size={gcfg.hidden_size}, num_attention_heads={gcfg.num_attention_heads}, head_dim={head_dim}")
    print(f"Requested ranks: q={args.rank_q}, k={args.rank_k}, v={args.rank_v}")
    
    mla_cfg = MLAConfig(
        hidden_size=gcfg.hidden_size,
        num_heads=gcfg.num_attention_heads,
        head_dim=head_dim,
        q_rank=args.rank_q,
        k_rank=args.rank_k,
        v_rank=args.rank_v,
        attn_dropout=gcfg.attn_pdrop,
        resid_dropout=gcfg.resid_pdrop,
        use_bias=False,
    )
    print(f"MLA config created: {mla_cfg}")
    
    model = GPT2MLAForCausalLM(gcfg, mla_cfg, tied_lm_head=True).to(device)
    print(f"MLA model created with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    print(f"MLA model created and moved to device: {device}")
    print(f"Total model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"=== END BUILDING MLA MODEL ===\n")

    # SVD-init + copy all GPT-2 weights
    svd_init_from_gpt2(ref_gpt2, model, args.rank_q, args.rank_k, args.rank_v)

    # data
    print("Loading and tokenizing dataset...", flush=True)
    ds = load_and_tokenize(args.dataset_id, args.subset, tokenizer, args.block_size)

    # trainer
    targs = build_training_args(args)
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds.get("train"),
        eval_dataset=ds.get("validation", None) or ds.get("test", None),
        tokenizer=tokenizer,
    )

    print("Starting training...", flush=True)
    print(f"Training with {len(ds.get('train', []))} training samples and {len(ds.get('validation', ds.get('test', [])))} validation samples")
    trainer.train()

    print("Evaluating...", flush=True)
    eval_metrics = trainer.evaluate()
    print(f"Evaluation metrics: {eval_metrics}")
    if (m := eval_metrics.get("eval_loss")) is not None:
        ppl = math.exp(m) if m < 50 else float("inf")
        print(f"Perplexity: {ppl:.4f}")

    # Save model + tokenizer to output_dir (safetensors-friendly)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved HF model to {args.output_dir}")

    # export MLA weights for C++ (optional)
    if args.export_dir is not None:
        export_mla_weights(
            model,
            out_dir=args.export_dir,
            export_fused=args.export_fused,
            quantize=args.quantize,
            nl_scale=args.nl_scale,
            nl_ell=args.nl_ell,
        )

if __name__ == "__main__":
    print("=== STARTING MLA TRAINING SCRIPT ===")
    main()
    print("=== MLA TRAINING SCRIPT COMPLETED ===")
