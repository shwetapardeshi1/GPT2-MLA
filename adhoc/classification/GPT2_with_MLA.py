# gpt2_mla_sst2_manual.py
import os, random, math, numpy as np, torch
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
OUTPUT_DIR       = "./gpt2-mla-sst2-manual"
MAX_LENGTH       = 128
BATCH_SIZE       = 8
EPOCHS           = 3
LEARNING_RATE    = 5e-5
WEIGHT_DECAY     = 0.0
GRAD_CLIP_NORM   = 1.0
SEED             = 42
NUM_WORKERS      = 2

# MLA ranks
RANK_Q           = 64
RANK_K           = 32
RANK_V           = 32
USE_BIAS_QKV     = False
# =================================================

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# ---------- MLA CORE ----------
@dataclass
class MLAConfig:
    hidden_size: int
    num_heads: int
    head_dim: int
    q_rank: int
    k_rank: int
    v_rank: int
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0
    use_bias: bool = False

class MLAAttention(nn.Module):
    def __init__(self, cfg: MLAConfig):
        super().__init__()
        d, H, d_h = cfg.hidden_size, cfg.num_heads, cfg.head_dim
        assert d == H * d_h, "hidden_size must equal num_heads*head_dim"
        self.cfg = cfg
        # low-rank projections
        self.q_down = nn.Linear(d, cfg.q_rank, bias=cfg.use_bias)
        self.k_down = nn.Linear(d, cfg.k_rank, bias=cfg.use_bias)
        self.v_down = nn.Linear(d, cfg.v_rank, bias=cfg.use_bias)
        self.q_up   = nn.Linear(cfg.q_rank, d, bias=False)
        self.k_up   = nn.Linear(cfg.k_rank, d, bias=False)
        self.v_up   = nn.Linear(cfg.v_rank, d, bias=False)
        # out
        self.out_proj   = nn.Linear(d, d, bias=True)
        self.attn_drop  = nn.Dropout(cfg.attn_dropout)
        self.resid_drop = nn.Dropout(cfg.resid_dropout)
        self._M_cache   = None

    def train(self, mode=True):
        super().train(mode)
        if mode: self._M_cache = None
        return self

    def _heads_view(self, weight: torch.Tensor, rank: int) -> torch.Tensor:
        H, d_h, r = self.cfg.num_heads, self.cfg.head_dim, rank
        return weight.view(H, d_h, r).transpose(1, 2).contiguous()  # (H, r, d_h)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        B, T, d = x.shape
        H, d_h = self.cfg.num_heads, self.cfg.head_dim

        # latent Q/K/V
        Q_lat = self.q_down(x)   # (B,T,r_q)
        K_new = self.k_down(x)   # (B,T,r_k)
        V_new = self.v_down(x)   # (B,T,r_v)
        if past_kv is not None:
            K_lat = torch.cat([past_kv[0], K_new], dim=1)  # (B,S,r_k)
            V_lat = torch.cat([past_kv[1], V_new], dim=1)  # (B,S,r_v)
        else:
            K_lat, V_lat = K_new, V_new
        S = K_lat.size(1)

        # absorbed score matrix per head
        if (not self.training) and (self._M_cache is not None):
            M = self._M_cache
            if M.device != x.device:
                M = M.to(x.device); self._M_cache = M
        else:
            Uq_h = self._heads_view(self.q_up.weight, self.cfg.q_rank)  # (H,r_q,d_h)
            Uk_h = self._heads_view(self.k_up.weight, self.cfg.k_rank)  # (H,r_k,d_h)
            M = torch.einsum("hqd,hkd->hqk", Uq_h, Uk_h) * (1.0 / math.sqrt(d_h))  # (H,r_q,r_k)
            if not self.training: self._M_cache = M

        # logits in latent space
        logits = torch.einsum("btq,hqk,bsk->bhts", Q_lat, M, K_lat)  # (B,H,T,S)

        # causal mask with past offset
        P = S - T
        i = torch.arange(T, device=x.device)[:, None]
        j = torch.arange(S, device=x.device)[None, :]
        future_mask = j > (P + i)  # (T,S)
        logits = logits.masked_fill(future_mask[None, None, :, :], float("-inf"))

        # key padding mask: attention_mask (B,S), 1=keep, 0=mask
        if attention_mask is not None:
            key_pad = (attention_mask == 0)
            logits = logits.masked_fill(key_pad[:, None, None, :], float("-inf"))

        attn = F.softmax(logits.float(), dim=-1)
        attn = torch.nan_to_num(attn)
        attn = self.attn_drop(attn).to(x.dtype)

        # values path
        Uv_h = self._heads_view(self.v_up.weight, self.cfg.v_rank)   # (H,r_v,d_h)
        V_up = torch.einsum("bsr,hrd->bhsd", V_lat, Uv_h)            # (B,H,S,d_h)
        ctx  = torch.einsum("bhts,bhsd->bhtd", attn, V_up)           # (B,H,T,d_h)
        ctx  = ctx.transpose(1, 2).contiguous().view(B, T, H * d_h)  # (B,T,d)

        out = self.out_proj(ctx)
        out = self.resid_drop(out)
        present = (K_lat, V_lat) if use_cache else None
        return out, present

class MLP(nn.Module):
    def __init__(self, hidden_size: int, multiple_of: int = 4):
        super().__init__()
        inner = multiple_of * hidden_size
        self.fc_in  = nn.Linear(hidden_size, inner)
        self.act    = nn.GELU()
        self.fc_out = nn.Linear(inner, hidden_size)
    def forward(self, x): return self.fc_out(self.act(self.fc_in(x)))

class MLAWrappedBlock(nn.Module):
    def __init__(self, gpt2_cfg: GPT2Config, mla_cfg: MLAConfig):
        super().__init__()
        d = gpt2_cfg.hidden_size
        self.ln_1 = nn.LayerNorm(d, eps=gpt2_cfg.layer_norm_epsilon)
        self.attn = MLAAttention(mla_cfg)
        self.ln_2 = nn.LayerNorm(d, eps=gpt2_cfg.layer_norm_epsilon)
        self.mlp  = MLP(d, multiple_of=4)
    def forward(self, x, attention_mask=None, past_kv=None, use_cache=False):
        h, present = self.attn(self.ln_1(x), attention_mask=attention_mask, past_kv=past_kv, use_cache=use_cache)
        x = x + h
        x = x + self.mlp(self.ln_2(x))
        return x, present

class TransformerWithMLA(nn.Module):
    def __init__(self, gpt2_cfg: GPT2Config, mla_cfg: MLAConfig):
        super().__init__()
        self.gpt2_cfg = gpt2_cfg
        self.mla_cfg  = mla_cfg
        self.wte  = nn.Embedding(gpt2_cfg.vocab_size, gpt2_cfg.hidden_size)
        self.wpe  = nn.Embedding(gpt2_cfg.n_positions, gpt2_cfg.hidden_size)
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
        if past_key_values is None: past_key_values = (None,) * len(self.blocks)
        for block, past in zip(self.blocks, past_key_values):
            x, present = block(x, attention_mask=attention_mask, past_kv=past, use_cache=use_cache)
            if use_cache: presents.append(present)
        x = self.ln_f(x)
        return x, (tuple(presents) if use_cache else None)

class GPT2MLAForSequenceClassification(nn.Module):
    def __init__(self, gpt2_cfg: GPT2Config, mla_cfg: MLAConfig, num_labels: int = 2, pool="last"):
        super().__init__()
        assert pool in ("last","mean")
        self.transformer = TransformerWithMLA(gpt2_cfg, mla_cfg)
        self.num_labels  = num_labels
        self.pool_type   = pool
        self.score = nn.Linear(gpt2_cfg.hidden_size, num_labels, bias=True)

    def _pool(self, hidden, attention_mask):
        if self.pool_type == "last":
            idx = attention_mask.long().sum(dim=1) - 1         # last non-pad
            pooled = hidden[torch.arange(hidden.size(0)), idx]
        else:
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return pooled

    def forward(self, input_ids, attention_mask=None):
        hidden, _ = self.transformer(input_ids, attention_mask=attention_mask, use_cache=False, past_key_values=None)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        pooled = self._pool(hidden, attention_mask)
        logits = self.score(pooled)
        return logits

# ---------- SVD INIT FROM PRETRAINED GPT-2 ----------
@torch.no_grad()
def svd_init_from_gpt2(reference: GPT2LMHeadModel, mla_backbone: TransformerWithMLA, rq: int, rk: int, rv: int):
    mla_backbone.wte.weight.copy_(reference.transformer.wte.weight.data)
    mla_backbone.wpe.weight.copy_(reference.transformer.wpe.weight.data)

    for ref_block, mla_block in zip(reference.transformer.h, mla_backbone.blocks):
        # slice Q/K/V from GPT-2's c_attn (Conv1D with weight (d,3d))
        W_qkv = ref_block.attn.c_attn.weight.detach().cpu().to(torch.float32).contiguous()
        d = W_qkv.shape[0]; d_slice = W_qkv.shape[1] // 3
        W_q = W_qkv[:, 0*d_slice:1*d_slice].contiguous()
        W_k = W_qkv[:, 1*d_slice:2*d_slice].contiguous()
        W_v = W_qkv[:, 2*d_slice:3*d_slice].contiguous()

        def svd_split(W, r):
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            r = max(1, min(int(r), W.shape[0]))
            A = U[:, :r] @ torch.diag(S[:r].sqrt())                     # (d,r)
            B = Vh.transpose(-2, -1)[:, :r] @ torch.diag(S[:r].sqrt())  # (d,r)
            return A, B

        Aq, Bq = svd_split(W_q, rq)
        Ak, Bk = svd_split(W_k, rk)
        Av, Bv = svd_split(W_v, rv)

        attn = mla_block.attn
        attn.q_down.weight.copy_(Aq.t())  # (r,d)
        attn.k_down.weight.copy_(Ak.t())
        attn.v_down.weight.copy_(Av.t())
        attn.q_up.weight.copy_(Bq)        # (d,r)
        attn.k_up.weight.copy_(Bk)
        attn.v_up.weight.copy_(Bv)

        # out proj: Conv1D -> Linear (transpose)
        attn.out_proj.weight.copy_(ref_block.attn.c_proj.weight.detach().t())
        if ref_block.attn.c_proj.bias is not None:
            attn.out_proj.bias.copy_(ref_block.attn.c_proj.bias.detach())

        # MLP (Conv1D -> Linear)
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

# ---------- DATA (fast-tokenizer path) ----------
class SST2Dataset(Dataset):
    """Keep raw strings; tokenize in collate for speed with fast tokenizers."""
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
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
            return_attention_mask=True,
        )
        return enc, torch.stack(labels)
    return collate

# ---------- EVAL METRICS ----------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, total_loss = [], [], 0.0
    ce = nn.CrossEntropyLoss().to(device)
    for batch, labels in loader:
        batch  = {k: v.to(device) for k, v in batch.items()}
        labels = labels.to(device)
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        loss   = ce(logits, labels)
        total_loss += loss.item() * labels.size(0)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    y_pred = np.concatenate(all_preds); y_true = np.concatenate(all_labels)
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    return {"loss": total_loss/len(y_true), "accuracy": acc, "precision": p, "recall": r, "f1": f1}

# ---------- TRAIN ----------
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Data & tokenizer
    ds  = load_dataset("glue", "sst2")  # 'sentence', 'label'
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token  # GPT-2: reuse EOS as PAD

    train_set = SST2Dataset(ds["train"]["sentence"], ds["train"]["label"])
    val_set   = SST2Dataset(ds["validation"]["sentence"], ds["validation"]["label"])

    collate_fn   = make_collate(tok)
    pin_mem      = (device.type == "cuda")
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, pin_memory=pin_mem, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, pin_memory=pin_mem, num_workers=NUM_WORKERS)

    # Build GPT-2 config from reference and MLA backbone+head
    ref_gpt2 = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    gcfg: GPT2Config = ref_gpt2.config
    head_dim = gcfg.hidden_size // gcfg.num_attention_heads
    mla_cfg = MLAConfig(
        hidden_size=gcfg.hidden_size,
        num_heads=gcfg.num_attention_heads,
        head_dim=head_dim,
        q_rank=RANK_Q, k_rank=RANK_K, v_rank=RANK_V,
        attn_dropout=gcfg.attn_pdrop,
        resid_dropout=gcfg.resid_pdrop,
        use_bias=USE_BIAS_QKV,
    )
    model = GPT2MLAForSequenceClassification(gcfg, mla_cfg, num_labels=2, pool="last")

    # SVD init from pretrained GPT-2
    svd_init_from_gpt2(ref_gpt2, model.transformer, RANK_Q, RANK_K, RANK_V)
    del ref_gpt2
    model.to(device)

    # Optimizer + AMP
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    ce = nn.CrossEntropyLoss().to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # Train
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss, running_correct, running_total = 0.0, 0, 0
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

        # Eval
        metrics = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:02d} | "
              f"Train Loss: {running_loss/running_total:.4f} | "
              f"Train Acc: {running_correct/running_total:.4f} | "
              f"Val Loss: {metrics['loss']:.4f} | "
              f"Val Acc: {metrics['accuracy']:.4f} | "
              f"P: {metrics['precision']:.4f} R: {metrics['recall']:.4f} F1: {metrics['f1']:.4f}")

    # Save (state_dict + tokenizer + simple configs)
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "pytorch_model.bin"))
    tok.save_pretrained(OUTPUT_DIR)
    with open(os.path.join(OUTPUT_DIR, "gpt2_config.json"), "w") as f:
        f.write(gcfg.to_json_string())
    with open(os.path.join(OUTPUT_DIR, "mla_config.txt"), "w") as f:
        f.write(str(mla_cfg))
    print(f"Saved model + tokenizer to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
