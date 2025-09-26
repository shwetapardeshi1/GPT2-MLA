# gpt2_mla_wt2_lm_jointkv.py
import os, math, time, itertools, json, random, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Config

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ==================== CONFIG ====================
MODEL_NAME     = "gpt2"                 # try "gpt2-medium" for better results
OUTPUT_DIR     = "./gpt2-mla-wt2-jkv"
BLOCK_SIZE     = 512
BATCH_SIZE     = 8
EPOCHS         = 3
LEARNING_RATE  = 1e-4
WEIGHT_DECAY   = 0.0
GRAD_CLIP      = 1.0
SEED           = 42
NUM_WORKERS    = 2

# MLA (joint KV) ranks
RANK_Q         = 128
RANK_KV        = 64
USE_BIAS_QKV   = False

# Generation micro-bench
GEN_PROMPT     = "The history of natural language processing"
GEN_MAX_NEW    = 128
GEN_TEMPERATURE= 1.0
# =================================================

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def fmt_bytes(n):
    for u in ["B","KB","MB","GB","TB"]:
        if n < 1024: return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} PB"

def report_device():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name()
        cap  = torch.cuda.get_device_capability()
        total= torch.cuda.get_device_properties(0).total_memory
        print(f"Device: cuda ({name}), cap={cap}, VRAM total={fmt_bytes(total)}")
        return torch.device("cuda")
    print("Device: cpu")
    return torch.device("cpu")

def try_psutil_rss():
    try:
        import psutil, os as _os
        return psutil.Process(_os.getpid()).memory_info().rss
    except Exception:
        return None

# ---------------- MLA (JOINT KV) ----------------
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
    MLA with shared latent C_kv for keys & values:
      x -> q_down -> Q_lat (B,T,rq)
      x -> kv_down -> C_kv  (B,T,rkv)   (this is what's cached)
      scores: Q_lat @ (Uq Uk^T) @ C_kv^T   with per-head absorbed M
      values: (two paths)
        - train: C_kv -> Uv -> V_up -> attn -> ctx -> out_proj
        - eval:  absorb Uv and out_proj per head into A_h, then
                 y = attn @ (C_kv @ A_h)  (skips materializing V_up)
    """
    def __init__(self, cfg: MLAJointConfig):
        super().__init__()
        d, H, d_h = cfg.hidden_size, cfg.num_heads, cfg.head_dim
        assert d == H * d_h, "hidden_size must equal num_heads * head_dim"
        self.cfg = cfg

        self.q_down  = nn.Linear(d, cfg.q_rank, bias=cfg.use_bias)
        self.kv_down = nn.Linear(d, cfg.kv_rank, bias=cfg.use_bias)

        self.q_up = nn.Linear(cfg.q_rank, d, bias=False)
        self.k_up = nn.Linear(cfg.kv_rank, d, bias=False)
        self.v_up = nn.Linear(cfg.kv_rank, d, bias=False)

        self.out_proj   = nn.Linear(d, d, bias=True)
        self.attn_drop  = nn.Dropout(cfg.attn_dropout)
        self.resid_drop = nn.Dropout(cfg.resid_dropout)

        # cached absorbed weights for eval
        self._M_cache = None     # (H, r_q, r_kv)
        self._A_cache = None     # (H, r_kv, d)
        self.absorb_output = True

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self._M_cache = None
            self._A_cache = None
        return self

    def _heads_view(self, W: torch.Tensor, rank: int) -> torch.Tensor:
        H, d_h, r = self.cfg.num_heads, self.cfg.head_dim, rank
        return W.view(H, d_h, r).transpose(1, 2).contiguous()  # (H, r, d_h)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_c_kv: Optional[torch.Tensor] = None,   # (B,S_past,r_kv)
        use_cache: bool = False,
    ):
        B, T, d = x.shape
        H, d_h  = self.cfg.num_heads, self.cfg.head_dim

        # Down-projects
        Q_lat  = self.q_down(x)          # (B,T,rq)
        C_new  = self.kv_down(x)         # (B,T,rkv)
        C_kv   = torch.cat([past_c_kv, C_new], dim=1) if past_c_kv is not None else C_new  # (B,S,rkv)
        S      = C_kv.size(1)

        # Absorbed score matrix M = Uq Uk^T / sqrt(d_h) (per head)
        if (not self.training) and (self._M_cache is not None):
            M = self._M_cache.to(x.device)
        else:
            Uq_h = self._heads_view(self.q_up.weight, self.cfg.q_rank)   # (H,rq,d_h)
            Uk_h = self._heads_view(self.k_up.weight, self.cfg.kv_rank)  # (H,rkv,d_h)
            M = torch.einsum("hqd,hkd->hqk", Uq_h, Uk_h) * (1.0 / math.sqrt(d_h))  # (H,rq,rkv)
            if not self.training:
                self._M_cache = M.detach()

        # Latent logits
        logits = torch.einsum("btq,hqk,bsk->bhts", Q_lat, M, C_kv)  # (B,H,T,S)

        # Causal mask (account for past)
        P = S - T
        i = torch.arange(T, device=x.device)[:, None]
        j = torch.arange(S, device=x.device)[None, :]
        future_mask = j > (P + i)
        logits = logits.masked_fill(future_mask[None, None, :, :], float("-inf"))

        # Key padding mask if provided (B,S) with 1 keep / 0 mask
        if attention_mask is not None:
            key_pad = (attention_mask == 0)
            logits = logits.masked_fill(key_pad[:, None, None, :], float("-inf"))

        attn = F.softmax(logits.float(), dim=-1)
        attn = torch.nan_to_num(attn).to(x.dtype)
        attn = self.attn_drop(attn)

        # Values path
        if (not self.training) and self.absorb_output:
            # Build absorbed output per head once on device
            if (self._A_cache is None) or (self._A_cache.device != x.device):
                Uv_h = self._heads_view(self.v_up.weight, self.cfg.kv_rank)  # (H,rkv,d_h)
                W = self.out_proj.weight  # (d_out, d_in=d)
                d_out, d_in = W.shape
                assert d_in == H * d_h
                B_h = W.view(d_out, H, d_h).permute(1, 2, 0).contiguous()    # (H,d_h,d_out)
                A_cache = torch.einsum("hrd,hdf->hrf", Uv_h, B_h)            # (H,rkv,d_out)
                self._A_cache = A_cache.detach().to(x.device)

            A_cache = self._A_cache
            V_out = torch.einsum("bsr,hrf->bhsf", C_kv, A_cache)         # (B,H,S,d_out)
            ctx_o = torch.einsum("bhts,bhsf->bhtf", attn, V_out)         # (B,H,T,d_out)
            y     = ctx_o.sum(dim=1)                                     # (B,T,d_out)
            if self.out_proj.bias is not None:
                y = y + self.out_proj.bias
            out = self.resid_drop(y)
            present = (C_kv if use_cache else None)
            return out, present

        # Training / non-absorbed path
        Uv_h = self._heads_view(self.v_up.weight, self.cfg.kv_rank)      # (H,rkv,d_h)
        V_up = torch.einsum("bsr,hrd->bhsd", C_kv, Uv_h)                  # (B,H,S,d_h)
        ctx  = torch.einsum("bhts,bhsd->bhtd", attn, V_up)                # (B,H,T,d_h)
        ctx  = ctx.transpose(1, 2).contiguous().view(B, T, H * d_h)       # (B,T,d)
        out  = self.out_proj(ctx)
        out  = self.resid_drop(out)
        present = (C_kv if use_cache else None)
        return out, present

class MLP(nn.Module):
    def __init__(self, hidden_size: int, multiple_of: int = 4):
        super().__init__()
        inner = multiple_of * hidden_size
        self.fc_in  = nn.Linear(hidden_size, inner)
        self.act    = nn.GELU()
        self.fc_out = nn.Linear(inner, hidden_size)
    def forward(self, x): return self.fc_out(self.act(self.fc_in(x)))

class MLAJointBlock(nn.Module):
    def __init__(self, gpt2_cfg: GPT2Config, mla_cfg: MLAJointConfig):
        super().__init__()
        d = gpt2_cfg.hidden_size
        self.ln_1 = nn.LayerNorm(d, eps=gpt2_cfg.layer_norm_epsilon)
        self.attn = MLAAttentionJoint(mla_cfg)
        self.ln_2 = nn.LayerNorm(d, eps=gpt2_cfg.layer_norm_epsilon)
        self.mlp  = MLP(d, multiple_of=4)
    def forward(self, x, attention_mask=None, past_c_kv=None, use_cache=False):
        h, present = self.attn(self.ln_1(x), attention_mask=attention_mask, past_c_kv=past_c_kv, use_cache=use_cache)
        x = x + h
        x = x + self.mlp(self.ln_2(x))
        return x, present

class TransformerWithMLAJoint(nn.Module):
    def __init__(self, gpt2_cfg: GPT2Config, mla_cfg: MLAJointConfig):
        super().__init__()
        self.gpt2_cfg = gpt2_cfg
        self.mla_cfg  = mla_cfg
        self.wte  = nn.Embedding(gpt2_cfg.vocab_size, gpt2_cfg.hidden_size)
        self.wpe  = nn.Embedding(gpt2_cfg.n_positions, gpt2_cfg.hidden_size)
        self.drop = nn.Dropout(gpt2_cfg.embd_pdrop)
        self.blocks = nn.ModuleList([MLAJointBlock(gpt2_cfg, mla_cfg) for _ in range(gpt2_cfg.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(gpt2_cfg.hidden_size, eps=gpt2_cfg.layer_norm_epsilon)

    def forward(self, input_ids, attention_mask=None, use_cache=False, past_key_values: Optional[Tuple[torch.Tensor,...]]=None):
        B, T = input_ids.shape
        device = input_ids.device
        pos = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        x = self.wte(input_ids) + self.wpe(pos)
        x = self.drop(x)
        presents = [] if use_cache else None
        if past_key_values is None:
            past_key_values = (None,) * len(self.blocks)
        for block, past_c in zip(self.blocks, past_key_values):
            x, present = block(x, attention_mask=attention_mask, past_c_kv=past_c, use_cache=use_cache)
            if use_cache: presents.append(present)
        x = self.ln_f(x)
        return x, (tuple(presents) if use_cache else None)

class GPT2MLAJointForCausalLM(nn.Module):
    def __init__(self, gpt2_cfg: GPT2Config, mla_cfg: MLAJointConfig, tie_weights: bool = True):
        super().__init__()
        self.transformer = TransformerWithMLAJoint(gpt2_cfg, mla_cfg)
        self.lm_head = nn.Linear(gpt2_cfg.hidden_size, gpt2_cfg.vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.transformer.wte.weight

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

    def state_dict(self, *a, **k):
        sd = super().state_dict(*a, **k)
        # if tied, ensure independent copy on save
        if "lm_head.weight" in sd and "transformer.wte.weight" in sd:
            if sd["lm_head.weight"].data_ptr() == sd["transformer.wte.weight"].data_ptr():
                sd["lm_head.weight"] = sd["lm_head.weight"].clone()
        return sd

# ----------- SVD init from pretrained GPT-2 -----------
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

        # Query low-rank SVD (W ≈ A B^T with symmetric split)
        Uq, Sq, Vhq = torch.linalg.svd(W_q, full_matrices=False)
        rq = max(1, min(int(rq), d))
        Aq = Uq[:, :rq] @ torch.diag(Sq[:rq].sqrt())                       # (d,rq)
        Bq = Vhq.transpose(-2, -1)[:, :rq] @ torch.diag(Sq[:rq].sqrt())    # (d,rq)

        attn = mla_block.attn
        attn.q_down.weight.copy_(Aq.t())    # (rq,d)
        attn.q_up.weight.copy_(Bq)          # (d,rq)

        # Joint KV SVD on concat [W_k | W_v]
        W_cat = torch.cat([W_k, W_v], dim=1)   # (d,2d)
        Uk, Sk, Vh_cat = torch.linalg.svd(W_cat, full_matrices=False)
        rkv = max(1, min(int(rkv), d))
        A_kv = Uk[:, :rkv] @ torch.diag(Sk[:rkv].sqrt())                              # (d,rkv)
        B_cat = Vh_cat.transpose(-2, -1)[:, :rkv] @ torch.diag(Sk[:rkv].sqrt())      # (2d,rkv)
        Bk = B_cat[:d, :]
        Bv = B_cat[d:, :]

        attn.kv_down.weight.copy_(A_kv.t())  # (rkv,d)
        attn.k_up.weight.copy_(Bk)           # (d,rkv)
        attn.v_up.weight.copy_(Bv)           # (d,rkv)

        # Out proj, MLP, LayerNorms
        attn.out_proj.weight.copy_(ref_block.attn.c_proj.weight.detach().t())
        if ref_block.attn.c_proj.bias is not None:
            attn.out_proj.bias.copy_(ref_block.attn.c_proj.bias.detach())

        mla_block.mlp.fc_in.weight.copy_(ref_block.mlp.c_fc.weight.detach().t())
        mla_block.mlp.fc_in.bias.copy_(ref_block.mlp.c_fc.bias.detach())
        mla_block.mlp.fc_out.weight.copy_(ref_block.mlp.c_proj.weight.detach().t())
        mla_block.mlp.fc_out.bias.copy_(ref_block.mlp.c_proj.bias.detach())

        mla_block.ln_1.weight.copy_(ref_block.ln_1.weight.detach())
        mla_block.ln_1.bias.copy_(ref_block.ln_1.bias.detach())
        mla_block.ln_2.weight.copy_(ref_block.ln_2.weight.detach())
        mla_block.ln_2.bias.copy_(ref_block.ln_2.bias.detach())

    mla_backbone.ln_f.weight.copy_(reference.transformer.ln_f.weight.detach())
    mla_backbone.ln_f.bias.copy_(reference.transformer.ln_f.bias.detach())

# ---------------- data: WikiText-2 into fixed blocks ----------------
def build_wt2(tokenizer, block_size=BLOCK_SIZE):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    def tok(batch): return tokenizer(batch["text"])
    tok_ds = ds.map(tok, batched=True, remove_columns=["text"])

    def group_texts(examples):
        ids = list(itertools.chain.from_iterable(examples["input_ids"]))
        n = (len(ids) // block_size) * block_size
        ids = ids[:n]
        if n == 0:
            return {"input_ids": [], "attention_mask": [], "labels": []}
        chunks = [ids[i:i+block_size] for i in range(0, n, block_size)]
        attn  = [[1]*block_size for _ in range(len(chunks))]
        return {"input_ids": chunks, "attention_mask": attn, "labels": [c[:] for c in chunks]}

    train = tok_ds["train"].map(group_texts, batched=True, remove_columns=tok_ds["train"].column_names)
    val   = tok_ds["validation"].map(group_texts, batched=True, remove_columns=tok_ds["validation"].column_names)
    test  = tok_ds["test"].map(group_texts, batched=True, remove_columns=tok_ds["test"].column_names)
    for split in (train, val, test):
        split.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return train, val, test

# ---------------- eval: NLL & PPL ----------------
@torch.no_grad()
def evaluate_lm(model, loader, device):
    model.eval()
    t0 = time.time()
    nll_sum, tok_count, n_batches = 0.0, 0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
        B, T = batch["input_ids"].shape
        tokens = B * (T - 1)
        nll_sum += out["loss"].item() * tokens
        tok_count += tokens
        n_batches += 1
    if device.type == "cuda": torch.cuda.synchronize()
    elapsed = time.time() - t0
    nll = nll_sum / max(1, tok_count)
    ppl = math.exp(min(50, nll))
    return {
        "nll": nll, "perplexity": ppl, "time_sec": elapsed,
        "latency_ms_per_batch": 1000.0 * elapsed / max(1, n_batches),
        "throughput_tokens_per_sec": tok_count / max(1e-9, elapsed),
    }

# --------- Greedy generation with C_kv cache (timed) ----------
@torch.no_grad()
def greedy_generate_with_cache(model, tok, device, prompt: str, max_new_tokens: int = 128, temperature: float = 1.0):
    model.eval()
    # prime
    ids = tok(prompt, return_tensors="pt")["input_ids"].to(device)
    attn = torch.ones_like(ids, device=device)
    out = model(input_ids=ids, attention_mask=attn, use_cache=True, past_key_values=None)
    past = out["past_key_values"]  # tuple per layer, each is C_kv (B,S,rkv)

    generated = [ids]
    t0 = time.time()
    for _ in range(max_new_tokens):
        last = generated[-1][:, -1:]  # (B,1)
        attn1 = torch.ones_like(last, device=device)
        out = model(input_ids=last, attention_mask=attn1, use_cache=True, past_key_values=past)
        logits = out["logits"][:, -1, :] / max(1e-6, temperature)
        next_id = torch.argmax(logits, dim=-1, keepdim=True)  # greedy
        generated.append(next_id)
        past = out["past_key_values"]
    if device.type == "cuda": torch.cuda.synchronize()
    elapsed = time.time() - t0

    full = torch.cat(generated, dim=1)
    text = tok.decode(full[0], skip_special_tokens=True)
    toks_gen = max_new_tokens
    return {"text": text, "time_sec": elapsed, "tok_per_s": toks_gen / max(1e-9, elapsed)}

# ---------------- training ----------------
def main():
    set_seed(SEED)
    device = report_device()
    rss0 = try_psutil_rss()
    if rss0 is not None:
        print(f"Process RSS (start): {fmt_bytes(rss0)}")

    # tokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # data
    print("Loading + tokenizing WikiText-2...")
    t_data0 = time.time()
    train_ds, val_ds, test_ds = build_wt2(tok, block_size=BLOCK_SIZE)
    t_data = time.time() - t_data0
    print(f"Data prep time: {t_data:.2f}s | Train blocks: {len(train_ds)}  Val blocks: {len(val_ds)}  Test blocks: {len(test_ds)}")

    pin = (device.type == "cuda")
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                               pin_memory=pin, num_workers=NUM_WORKERS)
    val_loader   = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                               pin_memory=pin, num_workers=NUM_WORKERS)
    test_loader  = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                                               pin_memory=pin, num_workers=NUM_WORKERS)

    # reference GPT-2 + MLA backbone
    ref_gpt2 = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    gcfg: GPT2Config = ref_gpt2.config
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
    model = GPT2MLAJointForCausalLM(gcfg, mla_cfg, tie_weights=True).to(device)

    print("SVD-initializing MLA (joint KV) from pretrained GPT-2...")
    svd_init_joint_from_gpt2(ref_gpt2, model.transformer, RANK_Q, RANK_KV)
    del ref_gpt2

    total = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total/1e6:.2f}M  (~{fmt_bytes(total*4)})")

    # optimizer + AMP
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    history = {"val": None, "test": None, "gen_bench": None, "epochs": []}

    # training epochs
    for epoch in range(1, EPOCHS + 1):
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        run_loss, run_tok, steps = 0.0, 0, 0

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
            scaler.step(optimizer); scaler.update()

            B, T = batch["input_ids"].shape
            tokens = B * (T - 1)
            run_loss += loss.item() * tokens
            run_tok  += tokens
            steps += 1
            avg_nll = run_loss / max(1, run_tok)
            pbar.set_postfix(nll=avg_nll, ppl=math.exp(min(50, avg_nll)))

        if device.type == "cuda": torch.cuda.synchronize()
        train_time = time.time() - t0
        peak = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
        train_nll = run_loss / max(1, run_tok)
        train_ppl = math.exp(min(50, train_nll))

        # validation (timed)
        val = evaluate_lm(model, val_loader, device)
        print(
            f"Epoch {epoch:02d} | Train NLL: {train_nll:.5f} | Train PPL: {train_ppl:.2f} | "
            f"Val NLL: {val['nll']:.5f} | Val PPL: {val['perplexity']:.2f}"
        )
        print(
            f"Timing: train {train_time:.2f}s "
            f"(tok/s {run_tok/max(1e-9,train_time):.0f}, batches/s {steps/max(1e-9,train_time):.2f}) | "
            f"val {val['time_sec']:.2f}s (tok/s {val['throughput_tokens_per_sec']:.0f})"
        )
        if device.type == "cuda":
            print(f"GPU peak alloc: {fmt_bytes(peak)}")
        rss = try_psutil_rss()
        if rss is not None:
            print(f"Process RSS: {fmt_bytes(rss)}")

        history["epochs"].append({
            "epoch": epoch,
            "train_nll": train_nll,
            "train_ppl": train_ppl,
            "val": val,
            "train_time_sec": train_time,
            "gpu_peak_alloc_bytes": int(peak) if device.type == "cuda" else None,
            "rss_bytes": int(rss) if rss is not None else None,
        })

    # final test (teacher-forcing; no cache)
    test = evaluate_lm(model, test_loader, device)
    print(f"\nTest NLL: {test['nll']:.5f} | Test PPL: {test['perplexity']:.2f} | "
          f"time {test['time_sec']:.2f}s (tok/s {test['throughput_tokens_per_sec']:.0f})")
    history["val"]  = history["epochs"][-1]["val"]
    history["test"] = test

    # generation micro-benchmark WITH C_kv cache
    bench = greedy_generate_with_cache(model, tok, device, GEN_PROMPT, GEN_MAX_NEW, GEN_TEMPERATURE)
    print(f"\n[GEN BENCH w/ cache] generated {GEN_MAX_NEW} new tokens in {bench['time_sec']:.2f}s "
          f"({bench['tok_per_s']:.1f} tok/s)")
    # Optionally print a snippet
    print("Sample generation (truncated):")
    print(bench["text"][:300] + " ...")
    history["gen_bench"] = {"tok": GEN_MAX_NEW, "time_sec": bench["time_sec"], "tok_per_s": bench["tok_per_s"]}

    # save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "pytorch_model.bin"))
    tok.save_pretrained(OUTPUT_DIR)
    with open(os.path.join(OUTPUT_DIR, "gpt2_config.json"), "w") as f:
        f.write(gcfg.to_json_string())
    with open(os.path.join(OUTPUT_DIR, "mla_joint_config.txt"), "w") as f:
        f.write(str(mla_cfg))
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nSaved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
