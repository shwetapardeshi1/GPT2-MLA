
import os, math, time, random, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from torch.optim import AdamW
from dataclasses import dataclass
from typing import Optional, Tuple
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Config


# ---- speed & eval helpers ----
from contextlib import contextmanager

def enable_fast_matmul():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

@contextmanager
def no_absorb_output(model: nn.Module):
    """
    Temporarily disable MLA output absorption (attn.absorb_output=False)
    ONLY during token-by-token decode to avoid O(H·S·d) work.
    No-op for vanilla GPT-2.
    """
    to_restore = []
    tr = getattr(model, "transformer", None)
    blocks = getattr(tr, "blocks", None) or getattr(tr, "h", None)
    if blocks:
        for blk in blocks:
            attn = getattr(blk, "attn", None)
            if attn is not None and hasattr(attn, "absorb_output"):
                to_restore.append((attn, attn.absorb_output))
                attn.absorb_output = False
    try:
        yield
    finally:
        for attn, prev in to_restore:
            attn.absorb_output = prev

def cuda_time(fn, *args, **kw):
    if torch.cuda.is_available():
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(); s.record(); r = fn(*args, **kw); e.record(); torch.cuda.synchronize()
        return r, (s.elapsed_time(e) / 1000.0)
    t0 = time.time(); r = fn(*args, **kw); return r, time.time()-t0


def enable_fast_matmul():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

def build_optimizer_and_scheduler(model, total_steps, lr, weight_decay, warmup_steps=1000):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5*(1.0 + math.cos(math.pi * progress)))  # cosine
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ==================== CONFIG ====================
MODEL_NAME       = "gpt2"
DATASET_NAME     = "HuggingFaceTB/smollm-corpus"
DATASET_CONFIG   = "fineweb-edu-dedup"
OUTPUT_DIR       = "./results/gpt2-smollm-fineweb-edu-dedup-mla-jkv-stream-64-32"

BLOCK_SIZE       = 1024
BATCH_SIZE       = 8
EPOCHS           = 3
LEARNING_RATE    = 5e-5
WEIGHT_DECAY     = 0.0
GRAD_CLIP        = 1.0
SEED             = 42

# MLA (Joint KV) ranks — same core block as finalized
RANK_Q           = 64
RANK_KV          = 32
USE_BIAS_QKV     = False

# Streaming controls
STREAM_VAL_DOCS  = 10_000
STREAM_TEST_DOCS = 10_000
TRAIN_SHUFFLE_BUF= 10_000
STEPS_PER_EPOCH  = 2000
NUM_WORKERS      = 0  # keep 0 for streaming

# Decode-time (KV cache) benchmark
BENCH_MAX_NEW    = 128
BENCH_CONTEXTS   = [128, 512, 1024]
# =================================================

def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def fmt_bytes(n):
    for u in ["B","KB","MB","GB","TB"]:
        if n < 1024: return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} PB"

def _pick_text_column(example: dict) -> str:
    for k in ("text","content","raw","document","code","completion"):
        if k in example and isinstance(example[k], str): return k
    for k,v in example.items():
        if isinstance(v, str): return k
    return "text"

# =============== MLA (JOINT KV) — unchanged core ===============
@dataclass
class MLAJointConfig:
    hidden_size: int
    num_heads: int
    head_dim: int
    q_rank: int
    kv_rank: int
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0
    use_bias: bool = False

class MLAAttentionJoint(nn.Module):
    """
    Joint-KV MLA block: caches a single latent C_kv during generation.
    Weights-only absorption for scores (M) always; output absorption (A) in eval only.
    """
    def __init__(self, cfg: MLAJointConfig):
        super().__init__()
        d, H, d_h = cfg.hidden_size, cfg.num_heads, cfg.head_dim
        assert d == H * d_h
        self.cfg = cfg

        # Down-projections
        self.q_down  = nn.Linear(d, cfg.q_rank, bias=cfg.use_bias)
        self.kv_down = nn.Linear(d, cfg.kv_rank, bias=cfg.use_bias)

        # Up-projections
        self.q_up = nn.Linear(cfg.q_rank, d, bias=False)
        self.k_up = nn.Linear(cfg.kv_rank, d, bias=False)
        self.v_up = nn.Linear(cfg.kv_rank, d, bias=False)

        self.out_proj   = nn.Linear(d, d, bias=True)
        self.attn_drop  = nn.Dropout(cfg.attn_dropout)
        self.resid_drop = nn.Dropout(cfg.resid_dropout)

        # eval caches (weights-only)
        self._M_cache = None       # (H, r_q, r_kv)
        self._A_cache = None       # (H, r_kv, d)
        self.absorb_output = True  # eval-only path uses _A_cache

    def train(self, mode=True):
        super().train(mode)
        if mode:
            self._M_cache = None
            self._A_cache = None
        return self

    def _heads_view(self, W: torch.Tensor, r: int) -> torch.Tensor:
        H, d_h = self.cfg.num_heads, self.cfg.head_dim
        return W.view(H, d_h, r).transpose(1, 2).contiguous()  # (H, r, d_h)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_c_kv: Optional[torch.Tensor] = None,   # (B,S_past,r_kv)
        use_cache: bool = False,
    ):
        import math
        B, T, d = x.shape
        H, d_h  = self.cfg.num_heads, self.cfg.head_dim

        # down-proj to latents
        Q_lat = self.q_down(x)      # (B,T,r_q)
        C_new = self.kv_down(x)     # (B,T,r_kv)
        C_kv  = torch.cat([past_c_kv, C_new], dim=1) if past_c_kv is not None else C_new  # (B,S,r_kv)
        S = C_kv.size(1)

        # absorbed score matrix M = Uq Uk^T / sqrt(d_h) (weights-only cache)
        if (not self.training) and (self._M_cache is not None):
            M = self._M_cache.to(x.device)
        else:
            Uq_h = self._heads_view(self.q_up.weight, self.cfg.q_rank)  # (H,r_q,d_h)
            Uk_h = self._heads_view(self.k_up.weight, self.cfg.kv_rank) # (H,r_kv,d_h)
            M = torch.einsum("hqd,hkd->hqk", Uq_h, Uk_h) * (1.0 / math.sqrt(d_h))
            if not self.training:
                self._M_cache = M.detach()

        # logits in latent space
        logits = torch.einsum("btq,hqk,bsk->bhts", Q_lat, M, C_kv)  # (B,H,T,S)

        # causal mask (accounts for past length)
        P = S - T
        i = torch.arange(T, device=x.device)[:, None]
        j = torch.arange(S, device=x.device)[None, :]
        future = j > (P + i)
        logits = logits.masked_fill(future[None,None,:,:], float("-inf"))

        # key padding mask
        if attention_mask is not None:
            key_pad = (attention_mask == 0)
            logits = logits.masked_fill(key_pad[:,None,None,:], float("-inf"))

        # attention
        attn = F.softmax(logits.float(), dim=-1)
        attn = torch.nan_to_num(attn).to(x.dtype)
        attn = self.attn_drop(attn)

        # values path
        if (not self.training) and self.absorb_output:
            # Output absorption (weights-only cache): A = Uv @ W_out (per head)
            if (self._A_cache is None) or (self._A_cache.device != x.device):
                Uv_h = self._heads_view(self.v_up.weight, self.cfg.kv_rank)  # (H,r_kv,d_h)
                W = self.out_proj.weight                                    # (d_out, d_in=d)
                d_out, d_in = W.shape; assert d_in == H * d_h
                B_h = W.view(d_out, H, d_h).permute(1, 2, 0).contiguous()   # (H,d_h,d_out)
                A = torch.einsum("hrd,hdf->hrf", Uv_h, B_h)                  # (H,r_kv,d_out)
                self._A_cache = A.detach().to(x.device)

            V_out = torch.einsum("bsr,hrf->bhsf", C_kv, self._A_cache)       # (B,H,S,d_out)
            ctx_o = torch.einsum("bhts,bhsf->bhtf", attn, V_out)             # (B,H,T,d_out)
            y     = ctx_o.sum(dim=1)                                         # (B,T,d_out)
            if self.out_proj.bias is not None:
                y = y + self.out_proj.bias
            out = self.resid_drop(y)
            present = (C_kv if use_cache else None)
            return out, present

        # Standard values path (train)
        Uv_h = self._heads_view(self.v_up.weight, self.cfg.kv_rank)          # (H,r_kv,d_h)
        V_up = torch.einsum("bsr,hrd->bhsd", C_kv, Uv_h)                      # (B,H,S,d_h)
        ctx  = torch.einsum("bhts,bhsd->bhtd", attn, V_up)                    # (B,H,T,d_h)
        ctx  = ctx.transpose(1,2).contiguous().view(B, T, H*d_h)              # (B,T,d)
        out  = self.out_proj(ctx)
        out  = self.resid_drop(out)
        present = (C_kv if use_cache else None)
        return out, present

class MLAJointBlock(nn.Module):
    def __init__(self, gpt2_cfg: GPT2Config, mla_cfg: MLAJointConfig):
        super().__init__()
        d = gpt2_cfg.hidden_size
        self.ln_1 = nn.LayerNorm(d, eps=gpt2_cfg.layer_norm_epsilon)
        self.attn = MLAAttentionJoint(mla_cfg)
        self.ln_2 = nn.LayerNorm(d, eps=gpt2_cfg.layer_norm_epsilon)
        # light MLP (kept identical)
        inner = 4 * d
        self.fc_in  = nn.Linear(d, inner)
        self.act    = nn.GELU()
        self.fc_out = nn.Linear(inner, d)

    def forward(self, x, attention_mask=None, past_c_kv=None, use_cache=False):
        h, present = self.attn(self.ln_1(x), attention_mask=attention_mask, past_c_kv=past_c_kv, use_cache=use_cache)
        x = x + h
        x = x + self.fc_out(self.act(self.fc_in(self.ln_2(x))))
        return x, present

class TransformerWithMLAJoint(nn.Module):
    def __init__(self, gpt2_cfg: GPT2Config, mla_cfg: MLAJointConfig):
        super().__init__()
        self.config = gpt2_cfg  # expose GPT2Config (for n_positions etc.)
        self.mla_cfg = mla_cfg
        self.wte  = nn.Embedding(gpt2_cfg.vocab_size, gpt2_cfg.hidden_size)
        self.wpe  = nn.Embedding(gpt2_cfg.n_positions, gpt2_cfg.hidden_size)
        self.drop = nn.Dropout(gpt2_cfg.embd_pdrop)
        self.blocks = nn.ModuleList([MLAJointBlock(gpt2_cfg, mla_cfg) for _ in range(gpt2_cfg.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(gpt2_cfg.hidden_size, eps=gpt2_cfg.layer_norm_epsilon)

    def forward(self, input_ids, attention_mask=None, use_cache=False, past_key_values=None):
        B, T = input_ids.shape
        device = input_ids.device
        pos = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        x = self.wte(input_ids) + self.wpe(pos)
        x = self.drop(x)
        presents = [] if use_cache else None
        if past_key_values is None: past_key_values = (None,) * len(self.blocks)
        for block, past_c in zip(self.blocks, past_key_values):
            x, present = block(x, attention_mask=attention_mask, past_c_kv=past_c, use_cache=use_cache)
            if use_cache: presents.append(present)
        x = self.ln_f(x)
        return x, (tuple(presents) if use_cache else None)

class GPT2MLAJointForCausalLM(nn.Module):
    """
    GPT-2 with MLA Joint-KV backbone + tied LM head.
    """
    def __init__(self, gpt2_cfg: GPT2Config, mla_cfg: MLAJointConfig, tied_lm_head: bool = True):
        super().__init__()
        self.transformer = TransformerWithMLAJoint(gpt2_cfg, mla_cfg)
        self.lm_head = nn.Linear(gpt2_cfg.hidden_size, gpt2_cfg.vocab_size, bias=False)
        self.tied_lm_head = tied_lm_head
        if tied_lm_head:
            self.lm_head.weight = self.transformer.wte.weight
        # expose config for convenience (n_positions etc.)
        self.config = gpt2_cfg

    def forward(self, input_ids, attention_mask=None, labels=None, use_cache=False, past_key_values=None):
        hidden, presents = self.transformer(input_ids, attention_mask=attention_mask, use_cache=use_cache, past_key_values=past_key_values)
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return {"loss": loss, "logits": logits, "past_key_values": presents}

    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        if self.tied_lm_head and "lm_head.weight" in sd and "transformer.wte.weight" in sd:
            if sd["lm_head.weight"].data_ptr() == sd["transformer.wte.weight"].data_ptr():
                sd["lm_head.weight"] = sd["lm_head.weight"].clone()
        return sd
# ===============================================================

@torch.no_grad()
def svd_init_joint_from_gpt2(reference: GPT2LMHeadModel,
                             mla_backbone: TransformerWithMLAJoint,
                             rq: int, rkv: int):
    # embeddings
    mla_backbone.wte.weight.copy_(reference.transformer.wte.weight.data)
    mla_backbone.wpe.weight.copy_(reference.transformer.wpe.weight.data)

    for ref_block, mla_block in zip(reference.transformer.h, mla_backbone.blocks):
        W_qkv = ref_block.attn.c_attn.weight.detach().cpu().to(torch.float32).contiguous()  # (d,3d)
        d = W_qkv.shape[0]; d_slice = W_qkv.shape[1] // 3
        W_q = W_qkv[:, 0*d_slice:1*d_slice].contiguous()
        W_k = W_qkv[:, 1*d_slice:2*d_slice].contiguous()
        W_v = W_qkv[:, 2*d_slice:3*d_slice].contiguous()

        def svd_lowrank(W, r):
            U,S,Vh = torch.linalg.svd(W, full_matrices=False)
            r = max(1, min(int(r), W.shape[0]))
            A = U[:, :r] @ torch.diag(S[:r].sqrt())                       # (d,r)
            B = Vh.transpose(-2, -1)[:, :r] @ torch.diag(S[:r].sqrt())    # (d,r)
            return A,B

        Aq,Bq = svd_lowrank(W_q, rq)
        # joint KV: SVD on concat([K,V], dim=1)
        W_cat = torch.cat([W_k, W_v], dim=1)                              # (d,2d)
        Uk,Sk,Vh_cat = torch.linalg.svd(W_cat, full_matrices=False)
        rkv = max(1, min(int(rkv), d))
        A_kv = Uk[:, :rkv] @ torch.diag(Sk[:rkv].sqrt())                  # (d,rkv)
        B_cat = Vh_cat.transpose(-2, -1)[:, :rkv] @ torch.diag(Sk[:rkv].sqrt())  # (2d,rkv)
        Bk, Bv = B_cat[:d, :], B_cat[d:, :]

        attn = mla_block.attn
        attn.q_down.weight.copy_(Aq.t());  attn.q_up.weight.copy_(Bq)
        attn.kv_down.weight.copy_(A_kv.t())
        attn.k_up.weight.copy_(Bk);        attn.v_up.weight.copy_(Bv)

        # c_proj (Conv1D) -> Linear (transpose)
        attn.out_proj.weight.copy_(ref_block.attn.c_proj.weight.detach().t())
        if ref_block.attn.c_proj.bias is not None:
            attn.out_proj.bias.copy_(ref_block.attn.c_proj.bias.detach())

        # MLP
        mla_block.fc_in.weight.copy_(ref_block.mlp.c_fc.weight.detach().t())
        mla_block.fc_in.bias.copy_(ref_block.mlp.c_fc.bias.detach())
        mla_block.fc_out.weight.copy_(ref_block.mlp.c_proj.weight.detach().t())
        mla_block.fc_out.bias.copy_(ref_block.mlp.c_proj.bias.detach())

        # LayerNorms
        mla_block.ln_1.weight.copy_(ref_block.ln_1.weight.detach())
        mla_block.ln_1.bias.copy_(ref_block.ln_1.bias.detach())
        mla_block.ln_2.weight.copy_(ref_block.ln_2.weight.detach())
        mla_block.ln_2.bias.copy_(ref_block.ln_2.bias.detach())

    mla_backbone.ln_f.weight.copy_(reference.transformer.ln_f.weight.detach())
    mla_backbone.ln_f.bias.copy_(reference.transformer.ln_f.bias.detach())

# -------------------- Streaming dataset --------------------
class StreamingBlockDataset(IterableDataset):
    def __init__(self, hf_stream, tokenizer, text_col, block_size, max_docs=None):
        super().__init__()
        self.hf_stream = hf_stream
        self.tok = tokenizer
        self.text_col = text_col
        self.block = block_size
        self.max_docs = max_docs

    def __iter__(self):
        buf = []
        seen = 0
        for ex in self.hf_stream:
            txt = ex.get(self.text_col, "")
            if not isinstance(txt, str):
                continue
            ids = self.tok.encode(txt, add_special_tokens=False)
            if not ids: continue
            if buf:
                buf.append(self.tok.eos_token_id)
            buf.extend(ids)
            while len(buf) >= self.block:
                chunk = buf[:self.block]; buf = buf[self.block:]
                yield {
                    "input_ids": torch.tensor(chunk, dtype=torch.long),
                    "attention_mask": torch.ones(self.block, dtype=torch.long),
                    "labels": torch.tensor(chunk, dtype=torch.long),
                }
            seen += 1
            if self.max_docs is not None and seen >= self.max_docs:
                break

def build_streams_and_loaders(tokenizer, device):
    base = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train", streaming=True)
    probe = next(iter(base.take(1)))
    text_col = _pick_text_column(probe)

    val_stream  = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train", streaming=True).take(STREAM_VAL_DOCS)
    test_stream = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train", streaming=True).skip(STREAM_VAL_DOCS).take(STREAM_TEST_DOCS)
    train_stream= load_dataset(DATASET_NAME, DATASET_CONFIG, split="train", streaming=True).skip(STREAM_VAL_DOCS + STREAM_TEST_DOCS)
    if TRAIN_SHUFFLE_BUF and TRAIN_SHUFFLE_BUF > 0:
        train_stream = train_stream.shuffle(buffer_size=TRAIN_SHUFFLE_BUF, seed=SEED)

    train_ds = StreamingBlockDataset(train_stream, tokenizer, text_col, BLOCK_SIZE)
    val_ds   = StreamingBlockDataset(val_stream,   tokenizer, text_col, BLOCK_SIZE)
    test_ds  = StreamingBlockDataset(test_stream,  tokenizer, text_col, BLOCK_SIZE)

    pin_mem = (device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin_mem, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin_mem, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin_mem, num_workers=NUM_WORKERS)
    return train_loader, val_loader, test_loader

# -------------------- Eval & decode bench --------------------
@torch.no_grad()
def evaluate_perplexity(model, loader, device, max_steps=None):
    model.eval()
    total_loss, total_tokens, steps = 0.0, 0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
        loss = out["loss"]
        toks = batch["labels"].numel()
        total_loss += loss.item() * toks
        total_tokens += toks
        steps += 1
        if max_steps is not None and steps >= max_steps:
            break
    avg = (total_loss / total_tokens) if total_tokens > 0 else float("inf")
    ppl = math.exp(avg) if avg < 50 else float("inf")
    return {"nll": avg, "perplexity": ppl}

def build_prompt_tokens(tokenizer, text: str, target_len: int):
    ids = tokenizer.encode(text, add_special_tokens=False) or [tokenizer.eos_token_id]
    while len(ids) < target_len: ids = ids + ids
    return torch.tensor(ids[:target_len], dtype=torch.long)


@torch.no_grad()
def decode_benchmark(model, tokenizer, device, base_text, context_lengths, max_new_tokens=128):
    model.eval()
    res = []
    npos = int(getattr(model.config, "n_positions", 1024))

    def build_prompt_tokens(text, target_len):
        ids = tokenizer.encode(text, add_special_tokens=False) or [tokenizer.eos_token_id]
        while len(ids) < target_len:
            ids = ids + ids  # repeat to reach length
        return torch.tensor(ids[-target_len:], dtype=torch.long)

    for ctx in context_lengths:
        ctx = min(ctx, npos)
        prompt = build_prompt_tokens(base_text, ctx).unsqueeze(0).to(device)

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        def _prefill():
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                return model(input_ids=prompt, use_cache=True)
        out, prefill = cuda_time(_prefill)
        past = out["past_key_values"]

        next_tok = out["logits"][:, -1, :].argmax(dim=-1, keepdim=True)
        steps = max(0, max_new_tokens - 1)

        def _decode_loop():
            nonlocal next_tok, past
            for _ in range(steps):
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                    o = model(input_ids=next_tok, past_key_values=past, use_cache=True)
                past = o["past_key_values"]
                next_tok = o["logits"][:, -1, :].argmax(dim=-1, keepdim=True)

        with no_absorb_output(model):
            _, decode = cuda_time(_decode_loop)

        peak = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
        res.append({
            "context": ctx,
            "prefill_sec": prefill,
            "decode_sec": decode,
            "decode_tps": (steps / max(1e-9, decode)) if decode > 0 else 0.0,
            "peak_gpu_bytes": int(peak),
        })
    return res
# ----------------------------- Train -----------------------------
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'})")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    # streaming loaders
    t0 = time.time()
    train_loader, val_loader, test_loader = build_streams_and_loaders(tok, device)
    print(f"Streaming ready in {time.time()-t0:.2f}s | val_docs={STREAM_VAL_DOCS} test_docs={STREAM_TEST_DOCS}")

    # reference GPT-2 + MLA backbone
    ref = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    gcfg: GPT2Config = ref.config
    head_dim = gcfg.hidden_size // gcfg.num_attention_heads
    mla_cfg = MLAJointConfig(
        hidden_size=gcfg.hidden_size,
        num_heads=gcfg.num_attention_heads,
        head_dim=head_dim,
        q_rank=RANK_Q,
        kv_rank=RANK_KV,
        attn_dropout=gcfg.attn_pdrop,
        resid_dropout=gcfg.resid_pdrop,
        use_bias=USE_BIAS_QKV,
    )
    model = GPT2MLAJointForCausalLM(gcfg, mla_cfg, tied_lm_head=True).to(device)

    # SVD init from GPT-2 (copies emb/LN/MLP + low-rank factorization of c_attn)
    print("SVD-initializing MLA (joint KV) from pretrained GPT-2...")
    svd_init_joint_from_gpt2(ref, model.transformer, RANK_Q, RANK_KV)
    del ref

    total = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total/1e6:.2f}M")

    total_steps = EPOCHS * (STREAM_TRAIN_DOCS * (BLOCK_SIZE) // (BATCH_SIZE * BLOCK_SIZE)) if isinstance(STREAM_TRAIN_DOCS, int) else 100000
    optimizer, scheduler = build_optimizer_and_scheduler(model, total_steps, LEARNING_RATE, WEIGHT_DECAY, warmup_steps=1000)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    for epoch in range(1, EPOCHS + 1):
        # recreate stream each epoch to avoid exhaustion
        train_loader, val_loader, _ = build_streams_and_loaders(tok, device)

        model.train()
        if device.type == "cuda": torch.cuda.reset_peak_memory_stats()
        t1 = time.time()
        running_loss, running_tokens, steps = 0.0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for batch in pbar:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
                loss = out["loss"]
            scaler.scale(loss).backward()
            if GRAD_CLIP and GRAD_CLIP > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer); scaler.update(); scheduler.step()

            toks = batch["labels"].numel()
            running_loss += loss.item() * toks
            running_tokens += toks
            avg = running_loss / max(1, running_tokens)
            pbar.set_postfix(nll=avg, ppl=(math.exp(avg) if avg < 50 else float("inf")))

            steps += 1
            if steps >= STEPS_PER_EPOCH:
                break

        if device.type == "cuda": torch.cuda.synchronize()
        train_time = time.time() - t1
        peak = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0

        val = evaluate_perplexity(model, val_loader, device, max_steps=500)
        train_nll = running_loss / max(1, running_tokens)
        print(f"Epoch {epoch:02d} | Train NLL: {train_nll:.5f} | Train PPL: {math.exp(min(50,train_nll)):.3f} | "
              f"Val NLL: {val['nll']:.5f} | Val PPL: {val['perplexity']:.3f}")
        print(f"Timing: train {train_time:.2f}s (tok/s {running_tokens/max(1e-9,train_time):.0f}) | "
              f"GPU peak: {fmt_bytes(peak)}")

    # fresh test slice
    _, _, test_loader = build_streams_and_loaders(tok, device)
    test = evaluate_perplexity(model, test_loader, device, max_steps=500)
    print(f"\nTest NLL: {test['nll']:.5f} | Test Perplexity: {test['perplexity']:.3f}")

    # ====== decode-time benchmark with latent C_kv cache ======
    base_text = "The history of natural language processing begins in the 1950s."
    valid_ctx = [c for c in BENCH_CONTEXTS if c <= model.config.n_positions]
    bench = decode_benchmark(model, tok, device, base_text, valid_ctx, max_new_tokens=BENCH_MAX_NEW)
    print("\n=== Decode-time Latent C_kv Cache Benchmark (smollm-corpus, GPT-2 + MLA JKV, streaming) ===")
    for r in bench:
        print(f"Context={r['context']:4d} | prefill {r['prefill_sec']:.3f}s | "
              f"decode {r['decode_sec']:.3f}s | {r['decode_tps']:.1f} tok/s | "
              f"peak GPU {fmt_bytes(r['peak_gpu_bytes'])}")

    # save
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "pytorch_model.bin"))
    tok.save_pretrained(OUTPUT_DIR)
    with open(os.path.join(OUTPUT_DIR, "gpt2_config.json"), "w") as f:
        f.write(model.config.to_json_string())
    with open(os.path.join(OUTPUT_DIR, "mla_joint_config.txt"), "w") as f:
        f.write(str(mla_cfg))
    print(f"\nSaved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()