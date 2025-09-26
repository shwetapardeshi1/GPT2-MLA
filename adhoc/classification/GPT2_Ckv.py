# gpt2_mla_jointkv_sst2_manual.py
import os, random, math, time, json, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from dataclasses import dataclass
from typing import Optional, Tuple
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Config
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ==================== CONFIG ====================
MODEL_NAME       = "gpt2"          # try "gpt2-medium" for better results
OUTPUT_DIR       = "./gpt2-mla-jointkv-sst2"
MAX_LENGTH       = 128
BATCH_SIZE       = 8
EPOCHS           = 3
LEARNING_RATE    = 5e-5
WEIGHT_DECAY     = 0.0
GRAD_CLIP_NORM   = 1.0
SEED             = 42
NUM_WORKERS      = 2

# MLA ranks (JOINT KV)
RANK_Q           = 64
RANK_KV          = 48   # shared latent rank for keys and values
USE_BIAS_QKV     = False
# =================================================

# ---- utils ----
def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def fmt_bytes(n):
    for unit in ["B","KB","MB","GB","TB"]:
        if n < 1024: return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"

def report_device():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        name = torch.cuda.get_device_name(dev)
        cap = torch.cuda.get_device_capability(dev)
        total = torch.cuda.get_device_properties(dev).total_memory
        print(f"Device: {name} (cap {cap}), VRAM total={fmt_bytes(total)}")
    else:
        print("Device: CPU")

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def try_psutil_rss():
    try:
        import psutil, os as _os
        rss = psutil.Process(_os.getpid()).memory_info().rss
        return rss
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
    def __init__(self, cfg: MLAJointConfig):
        super().__init__()
        d, H, d_h = cfg.hidden_size, cfg.num_heads, cfg.head_dim
        assert d == H * d_h, "hidden_size must equal num_heads * head_dim"
        self.cfg = cfg

        # Down-projections
        self.q_down  = nn.Linear(d, cfg.q_rank, bias=cfg.use_bias)
        self.kv_down = nn.Linear(d, cfg.kv_rank, bias=cfg.use_bias)

        # Up-projections (to model dim)
        self.q_up = nn.Linear(cfg.q_rank, d, bias=False)
        self.k_up = nn.Linear(cfg.kv_rank, d, bias=False)
        self.v_up = nn.Linear(cfg.kv_rank, d, bias=False)

        # Output + drops
        self.out_proj   = nn.Linear(d, d, bias=True)
        self.attn_drop  = nn.Dropout(cfg.attn_dropout)
        self.resid_drop = nn.Dropout(cfg.resid_dropout)

        # Eval-time caches
        self._M_cache = None       # absorbed score matrix: (H, r_q, r_kv)
        self._A_cache = None       # absorbed output W_OV per head: (H, r_kv, d)
        self.absorb_output = True  # eval-only

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self._M_cache = None
            self._A_cache = None
        return self

    def _heads_view(self, weight: torch.Tensor, rank: int) -> torch.Tensor:
        # (d, r) -> (H, r, d_h)
        H, d_h, r = self.cfg.num_heads, self.cfg.head_dim, rank
        return weight.view(H, d_h, r).transpose(1, 2).contiguous()

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_c_kv: Optional[torch.Tensor] = None,   # (B, S_past, r_kv)
        use_cache: bool = False,
    ):
        import math
        B, T, d = x.shape
        H, d_h  = self.cfg.num_heads, self.cfg.head_dim

        # ---- Down-proj ----
        Q_lat  = self.q_down(x)              # (B, T, r_q)
        C_new  = self.kv_down(x)             # (B, T, r_kv)
        C_kv   = torch.cat([past_c_kv, C_new], dim=1) if past_c_kv is not None else C_new  # (B,S,r_kv)
        S      = C_kv.size(1)

        # ---- Absorbed score matrix M = Uq Uk^T / sqrt(d_h) ----
        if (not self.training) and (self._M_cache is not None):
            M = self._M_cache.to(x.device)
        else:
            Uq_h = self._heads_view(self.q_up.weight, self.cfg.q_rank)   # (H, r_q, d_h)
            Uk_h = self._heads_view(self.k_up.weight, self.cfg.kv_rank)  # (H, r_kv, d_h)
            M = torch.einsum("hqd,hkd->hqk", Uq_h, Uk_h) * (1.0 / math.sqrt(d_h))  # (H, r_q, r_kv)
            if not self.training:
                self._M_cache = M.detach()

        # ---- Logits in latent space ----
        logits = torch.einsum("btq,hqk,bsk->bhts", Q_lat, M, C_kv)  # (B,H,T,S)

        # ---- Masks ----
        P = S - T
        i = torch.arange(T, device=x.device)[:, None]
        j = torch.arange(S, device=x.device)[None, :]
        future_mask = j > (P + i)
        logits = logits.masked_fill(future_mask[None, None, :, :], float("-inf"))
        if attention_mask is not None:
            key_pad = (attention_mask == 0)
            logits = logits.masked_fill(key_pad[:, None, None, :], float("-inf"))

        # ---- Softmax ----
        attn = F.softmax(logits.float(), dim=-1)
        attn = torch.nan_to_num(attn)
        attn = self.attn_drop(attn).to(x.dtype)

        # =========================
        # Values path
        # =========================
        if (not self.training) and self.absorb_output:
            # -------- Eval-only OUTPUT ABSORPTION --------
            if (self._A_cache is None) or (self._A_cache.device != x.device):
                Uv_h = self._heads_view(self.v_up.weight, self.cfg.kv_rank)  # (H, r_kv, d_h)
                W = self.out_proj.weight    # (d_out, d_in=d)
                d_out, d_in = W.shape
                assert d_in == H * d_h, "out_proj input must equal H * d_h"
                B_h = W.view(d_out, H, d_h).permute(1, 2, 0).contiguous()  # (H, d_h, d_out)
                A_cache = torch.einsum("hrd,hdf->hrf", Uv_h, B_h)          # (H, r_kv, d_out)
                self._A_cache = A_cache.detach().to(x.device)

            A_cache = self._A_cache  # (H, r_kv, d_out)
            V_out = torch.einsum("bsr,hrf->bhsf", C_kv, A_cache)         # (B,H,S,d_out)
            ctx_o = torch.einsum("bhts,bhsf->bhtf", attn, V_out)         # (B,H,T,d_out)
            y     = ctx_o.sum(dim=1)                                     # (B,T,d_out)
            if self.out_proj.bias is not None:
                y = y + self.out_proj.bias
            out = self.resid_drop(y)
            present = (C_kv if use_cache else None)
            return out, present

        else:
            # -------- Training (or absorption disabled): standard path --------
            Uv_h = self._heads_view(self.v_up.weight, self.cfg.kv_rank)   # (H, r_kv, d_h)
            V_up = torch.einsum("bsr,hrd->bhsd", C_kv, Uv_h)              # (B,H,S,d_h)
            ctx  = torch.einsum("bhts,bhsd->bhtd", attn, V_up)            # (B,H,T,d_h)
            ctx  = ctx.transpose(1, 2).contiguous().view(B, T, H * d_h)   # (B,T,d)
            out = self.out_proj(ctx)
            out = self.resid_drop(out)
            present = (C_kv if use_cache else None)
            return out, present

# ---- Block / Transformer / Classifier ----
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

class GPT2MLAJointForSequenceClassification(nn.Module):
    def __init__(self, gpt2_cfg: GPT2Config, mla_cfg: MLAJointConfig, num_labels: int = 2, pool="mean"):
        super().__init__()
        assert pool in ("last","mean")
        self.transformer = TransformerWithMLAJoint(gpt2_cfg, mla_cfg)
        self.num_labels  = num_labels
        self.pool_type   = pool
        self.cls_drop    = nn.Dropout(0.1)
        self.score       = nn.Linear(gpt2_cfg.hidden_size, num_labels, bias=True)

    def _pool(self, hidden, attention_mask):
        if self.pool_type == "last":
            idx = attention_mask.long().sum(dim=1) - 1
            return hidden[torch.arange(hidden.size(0)), idx]
        else:
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def forward(self, input_ids, attention_mask=None):
        # NOTE: use_cache=False -> we are NOT saving KV cache in classifier
        hidden, _ = self.transformer(input_ids, attention_mask=attention_mask, use_cache=False, past_key_values=None)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        pooled = self._pool(hidden, attention_mask)
        logits = self.score(self.cls_drop(pooled))
        return logits

# ------------- SVD INIT (JOINT KV) -------------
@torch.no_grad()
def svd_init_joint_from_gpt2(reference: GPT2LMHeadModel,
                             mla_backbone: TransformerWithMLAJoint,
                             rq: int, rkv: int):
    mla_backbone.wte.weight.copy_(reference.transformer.wte.weight.data)
    mla_backbone.wpe.weight.copy_(reference.transformer.wpe.weight.data)

    for ref_block, mla_block in zip(reference.transformer.h, mla_backbone.blocks):
        W_qkv = ref_block.attn.c_attn.weight.detach().cpu().to(torch.float32).contiguous()
        d = W_qkv.shape[0]; d_slice = W_qkv.shape[1] // 3
        W_q = W_qkv[:, 0*d_slice:1*d_slice].contiguous()
        W_k = W_qkv[:, 1*d_slice:2*d_slice].contiguous()
        W_v = W_qkv[:, 2*d_slice:3*d_slice].contiguous()

        # Query low-rank
        Uq, Sq, Vhq = torch.linalg.svd(W_q, full_matrices=False)
        rq = max(1, min(int(rq), d))
        Aq = Uq[:, :rq] @ torch.diag(Sq[:rq].sqrt())                       # (d,rq)
        Bq = Vhq.transpose(-2, -1)[:, :rq] @ torch.diag(Sq[:rq].sqrt())    # (d,rq)

        attn = mla_block.attn
        attn.q_down.weight.copy_(Aq.t())   # (rq,d)
        attn.q_up.weight.copy_(Bq)         # (d,rq)

        # Joint KV SVD on concat
        W_cat = torch.cat([W_k, W_v], dim=1)   # (d, 2d)
        Uk, Sk, Vh_cat = torch.linalg.svd(W_cat, full_matrices=False)
        rkv = max(1, min(int(rkv), d))
        A_kv = Uk[:, :rkv] @ torch.diag(Sk[:rkv].sqrt())                              # (d, rkv)
        B_cat = Vh_cat.transpose(-2, -1)[:, :rkv] @ torch.diag(Sk[:rkv].sqrt())      # (2d, rkv)
        Bk = B_cat[:d, :]   # (d, rkv)
        Bv = B_cat[d:, :]   # (d, rkv)

        attn.kv_down.weight.copy_(A_kv.t())  # (rkv, d)
        attn.k_up.weight.copy_(Bk)           # (d, rkv)
        attn.v_up.weight.copy_(Bv)           # (d, rkv)

        # c_proj -> out_proj
        attn.out_proj.weight.copy_(ref_block.attn.c_proj.weight.detach().t())
        if ref_block.attn.c_proj.bias is not None:
            attn.out_proj.bias.copy_(ref_block.attn.c_proj.bias.detach())

        # MLP
        mla_block.mlp.fc_in.weight.copy_(ref_block.mlp.c_fc.weight.detach().t())
        mla_block.mlp.fc_in.bias.copy_(ref_block.mlp.c_fc.bias.detach())
        mla_block.mlp.fc_out.weight.copy_(ref_block.mlp.c_proj.weight.detach().t())
        mla_block.mlp.fc_out.bias.copy_(ref_block.mlp.c_proj.bias.detach())

        # LayerNorms
        mla_block.ln_1.weight.copy_(ref_block.ln_1.weight.detach())
        mla_block.ln_1.bias.copy_(ref_block.ln_1.bias.detach())
        mla_block.ln_2.weight.copy_(ref_block.ln_2.weight.detach())
        mla_block.ln_2.bias.copy_(ref_block.ln_2.bias.detach())

    mla_backbone.ln_f.weight.copy_(reference.transformer.ln_f.weight.detach())
    mla_backbone.ln_f.bias.copy_(reference.transformer.ln_f.bias.detach())

# ----------------- DATA & METRICS -----------------
class SST2Dataset(Dataset):
    def __init__(self, texts, labels):
        self.texts  = [str(t) for t in list(texts)]
        self.labels = torch.tensor(list(labels), dtype=torch.long)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.texts[idx], self.labels[idx]

def make_collate(tokenizer):
    def collate(batch):
        texts, labels = zip(*batch)
        enc = tokenizer(
            list(texts),
            truncation=True, padding=True, max_length=MAX_LENGTH,
            return_tensors="pt", return_attention_mask=True,
        )
        return enc, torch.stack(labels)
    return collate

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ce = nn.CrossEntropyLoss().to(device)
    all_preds, all_labels, total_loss = [], [], 0.0
    t0 = time.time()
    n_samples, n_batches = 0, 0
    for batch, labels in loader:
        batch  = {k: v.to(device) for k, v in batch.items()}
        labels = labels.to(device)
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        loss   = ce(logits, labels)
        total_loss += loss.item() * labels.size(0)
        all_preds.append(logits.argmax(1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        n_samples += labels.size(0); n_batches += 1
    if device.type == "cuda": torch.cuda.synchronize()
    elapsed = time.time() - t0
    y_pred = np.concatenate(all_preds); y_true = np.concatenate(all_labels)
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    return {
        "loss": total_loss/len(y_true),
        "accuracy": acc, "precision": p, "recall": r, "f1": f1,
        "time_sec": elapsed,
        "latency_ms_per_batch": 1000.0 * elapsed / max(1, n_batches),
        "throughput_samples_per_sec": n_samples / max(1e-9, elapsed),
    }

# ---------------------- TRAIN ----------------------
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_device()
    rss0 = try_psutil_rss()
    if rss0 is not None:
        print(f"Process RSS (start): {fmt_bytes(rss0)}")

    # Data + tokenizer
    t_data0 = time.time()
    ds  = load_dataset("glue", "sst2")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    train_set = SST2Dataset(ds["train"]["sentence"], ds["train"]["label"])
    val_set   = SST2Dataset(ds["validation"]["sentence"], ds["validation"]["label"])
    test_set  = SST2Dataset(ds["test"]["sentence"], [0]*len(ds["test"]))  # labels unknown in GLUE test; set dummy

    collate_fn = make_collate(tok)
    pin_mem = (device.type == "cuda")
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, pin_memory=pin_mem, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, pin_memory=pin_mem, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, pin_memory=pin_mem, num_workers=NUM_WORKERS)
    t_data = time.time() - t_data0
    print(f"Data prep time: {t_data:.2f}s")

    # Build MLA Joint backbone + head, SVD init from GPT-2
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
    model = GPT2MLAJointForSequenceClassification(gcfg, mla_cfg, num_labels=2, pool="mean")
    svd_init_joint_from_gpt2(ref_gpt2, model.transformer, RANK_Q, RANK_KV)
    del ref_gpt2
    model.to(device)

    total_params, trainable_params = count_params(model)
    approx_bytes = trainable_params * 4  # fp32 assumption for a rough size
    print(f"Model params: total={total_params/1e6:.2f}M, trainable={trainable_params/1e6:.2f}M, approx size={fmt_bytes(approx_bytes)}")

    # Optimizer + AMP
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    ce = nn.CrossEntropyLoss(label_smoothing=0.0).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    hist = {"train": [], "val": None, "test": None}

    # Train
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss, running_correct, running_total = 0.0, 0, 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for batch, labels in pbar:
            batch  = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                loss   = ce(logits, labels)

            scaler.scale(loss).backward()
            if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()

            running_loss   += loss.item() * labels.size(0)
            running_correct+= (logits.argmax(1) == labels).sum().item()
            running_total  += labels.size(0)
            pbar.set_postfix(loss=running_loss/running_total, acc=running_correct/running_total)

        if device.type == "cuda":
            torch.cuda.synchronize()
        train_time = time.time() - t0
        gpu_peak = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
        rss = try_psutil_rss()

        # Validation (timed)
        val_metrics = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:02d} | "
              f"Train Loss: {running_loss/running_total:.4f} | "
              f"Train Acc: {running_correct/running_total:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.4f} | "
              f"P: {val_metrics['precision']:.4f} R: {val_metrics['recall']:.4f} F1: {val_metrics['f1']:.4f}")
        print(f"Timing: train {train_time:.2f}s ({running_total/train_time:.1f} samples/s) | "
              f"val {val_metrics['time_sec']:.2f}s "
              f"(lat {val_metrics['latency_ms_per_batch']:.1f} ms/batch, "
              f"thr {val_metrics['throughput_samples_per_sec']:.1f} samp/s)")
        if device.type == "cuda":
            print(f"GPU peak alloc (epoch): {fmt_bytes(gpu_peak)}")
        if rss is not None:
            print(f"Process RSS (epoch end): {fmt_bytes(rss)}")

        hist["train"].append({
            "epoch": epoch,
            "train_loss": running_loss/running_total,
            "train_acc": running_correct/running_total,
            "val": val_metrics,
            "train_time_sec": train_time,
            "gpu_peak_alloc_bytes": int(gpu_peak) if device.type == "cuda" else None,
            "rss_bytes": int(rss) if rss is not None else None,
        })

    # Final validation (already ran above), then Test timing/metrics
    test_metrics = evaluate(model, test_loader, device)
    print(f"\nTEST | time {test_metrics['time_sec']:.2f}s | "
          f"lat {test_metrics['latency_ms_per_batch']:.1f} ms/batch | "
          f"thr {test_metrics['throughput_samples_per_sec']:.1f} samp/s")
    # GLUE SST-2 test has no labels; metrics reflect dummy labels. Use for timing only.
    hist["val"]  = hist["train"][-1]["val"]
    hist["test"] = test_metrics

    # Save weights + tokenizer + configs + metrics
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "pytorch_model.bin"))
    tok.save_pretrained(OUTPUT_DIR)
    with open(os.path.join(OUTPUT_DIR, "gpt2_config.json"), "w") as f:
        f.write(gcfg.to_json_string())
    with open(os.path.join(OUTPUT_DIR, "mla_joint_config.txt"), "w") as f:
        f.write(str(mla_cfg))
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(hist, f, indent=2)
    print(f"Saved model, tokenizer, and metrics to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
