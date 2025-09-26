# gpt2_sst2_manual_clean_timed.py
import os, time, random, numpy as np, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, GPT2ForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from collections import Counter

try:
    import psutil
except Exception:
    psutil = None

# ==================== CONFIG ====================
MODEL_NAME       = "gpt2"          # e.g., "gpt2-medium" for better results
OUTPUT_DIR       = "./gpt2-sst2-manual"
MAX_LENGTH       = 128
BATCH_SIZE       = 8
EPOCHS           = 3
LEARNING_RATE    = 5e-5
WEIGHT_DECAY     = 0.0
GRAD_CLIP_NORM   = 1.0
SEED             = 42
NUM_WORKERS      = 2              # set 0 if dataloader multiprocessing causes issues
WARMUP_BATCHES   = 5              # warm-up for inference timing
# =================================================

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    # (more-deterministic kernels; may slow down)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    # Ensures each DataLoader worker is deterministically seeded
    worker_seed = SEED + worker_id
    random.seed(worker_seed); np.random.seed(worker_seed)

class SST2Dataset(Dataset):
    """Stores raw strings + labels; we tokenize in collate (fast path)."""
    def __init__(self, texts, labels):
        self.texts  = [str(t) for t in list(texts)]
        self.labels = torch.tensor(list(labels), dtype=torch.long)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.texts[idx], self.labels[idx]

class SST2TestDataset(Dataset):
    """Test split has no labels."""
    def __init__(self, texts):
        self.texts = [str(t) for t in list(texts)]
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx): return self.texts[idx]

def make_collate(tokenizer):
    def collate(batch):
        if isinstance(batch[0], tuple):
            texts, labels = zip(*batch)
            enc = tokenizer(
                list(texts), truncation=True, padding=True,
                max_length=MAX_LENGTH, return_tensors="pt", return_attention_mask=True,
            )
            return enc, torch.stack(labels)
        else:
            texts = batch
            enc = tokenizer(
                list(texts), truncation=True, padding=True,
                max_length=MAX_LENGTH, return_tensors="pt", return_attention_mask=True,
            )
            return enc
    return collate

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, total_loss = [], [], 0.0
    ce = nn.CrossEntropyLoss().to(device)
    for batch, labels in loader:
        batch  = {k: v.to(device) for k, v in batch.items()}
        labels = labels.to(device)
        out = model(**batch)
        logits = out.logits
        loss = ce(logits, labels)
        total_loss += loss.item() * labels.size(0)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    y_pred = np.concatenate(all_preds); y_true = np.concatenate(all_labels)
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    return {"loss": total_loss/len(y_true), "accuracy": acc, "precision": p, "recall": r, "f1": f1}

@torch.no_grad()
def predict_and_time(model, loader, device, warmup_batches=WARMUP_BATCHES):
    """
    Run inference on a dataloader WITHOUT labels, measure latency/throughput, and return predictions.
    """
    model.eval()
    # Warm-up (helps stabilize CUDA clocks)
    it = iter(loader)
    for _ in range(min(warmup_batches, len(loader))):
        batch = next(it, None)
        if batch is None: break
        batch = {k: v.to(device) for k, v in batch.items()}
        _ = model(**batch)

    # Reset CUDA peak mem for clean inference stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    count_items = 0
    preds_list = []
    t0 = time.time()

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        preds = out.logits.argmax(dim=1)
        preds_list.append(preds.cpu().numpy())
        count_items += preds.size(0)

    t1 = time.time()
    total_time = t1 - t0
    throughput = count_items / max(total_time, 1e-9)

    gpu_peak_alloc = gpu_peak_rsvd = None
    if torch.cuda.is_available():
        gpu_peak_alloc = torch.cuda.max_memory_allocated(device) / (1024**2)
        gpu_peak_rsvd  = torch.cuda.max_memory_reserved(device) / (1024**2)

    preds = np.concatenate(preds_list) if preds_list else np.array([])
    return preds, {"samples": count_items, "seconds": total_time, "samples_per_sec": throughput,
                   "gpu_peak_alloc_mb": gpu_peak_alloc, "gpu_peak_reserved_mb": gpu_peak_rsvd}

def num_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return total, trainable, size_bytes / (1024**2)

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Data & tokenizer
    ds = load_dataset("glue", "sst2")  # train/validation/test; test has no labels
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token  # GPT-2: reuse EOS as PAD

    train_set = SST2Dataset(ds["train"]["sentence"], ds["train"]["label"])
    val_set   = SST2Dataset(ds["validation"]["sentence"], ds["validation"]["label"])
    test_set  = SST2TestDataset(ds["test"]["sentence"])

    collate_fn = make_collate(tok)
    g = torch.Generator()
    g.manual_seed(SEED)

    pin_mem = (device.type == "cuda")
    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, generator=g,
        collate_fn=collate_fn, pin_memory=pin_mem, num_workers=NUM_WORKERS, worker_init_fn=seed_worker
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, pin_memory=pin_mem, num_workers=NUM_WORKERS, worker_init_fn=seed_worker
    )
    test_loader = DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, pin_memory=pin_mem, num_workers=NUM_WORKERS, worker_init_fn=seed_worker
    )

    # Model
    model = GPT2ForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = tok.eos_token_id
    model.config.use_cache = False  # ensure NO KV cache usage
    model.to(device)

    total_p, trainable_p, model_mb = num_params(model)
    print(f"Model params: total={total_p/1e6:.2f}M, trainable={trainable_p/1e6:.2f}M, approx size={model_mb:.1f} MB")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(device)} (cap {torch.cuda.get_device_capability(device)})")
    if psutil is not None:
        mem = psutil.Process().memory_info().rss / (1024**2)
        print(f"Process RSS (start): {mem:.1f} MB")

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    ce = nn.CrossEntropyLoss().to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # Reset CUDA peak memory before training for clean measurement
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    train_t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss, running_correct, running_total = 0.0, 0, 0
        epoch_t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for batch, labels in pbar:
            batch  = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                out = model(**batch)
                logits = out.logits
                loss = ce(logits, labels)

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

        epoch_t1 = time.time()
        epoch_secs = epoch_t1 - epoch_t0

        # Validation each epoch
        metrics = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:02d} | "
              f"Train Loss: {running_loss/running_total:.4f} | "
              f"Train Acc: {running_correct/running_total:.4f} | "
              f"Val Loss: {metrics['loss']:.4f} | "
              f"Val Acc: {metrics['accuracy']:.4f} | "
              f"P: {metrics['precision']:.4f} R: {metrics['recall']:.4f} F1: {metrics['f1']:.4f} | "
              f"Epoch Time: {epoch_secs:.1f}s")

    train_t1 = time.time()
    total_train_secs = train_t1 - train_t0

    # Training resource stats
    gpu_peak_alloc = gpu_peak_rsvd = None
    if torch.cuda.is_available():
        gpu_peak_alloc = torch.cuda.max_memory_allocated(device) / (1024**2)
        gpu_peak_rsvd  = torch.cuda.max_memory_reserved(device) / (1024**2)
    if psutil is not None:
        mem = psutil.Process().memory_info().rss / (1024**2)
        print(f"Process RSS (post-train): {mem:.1f} MB")

    print(f"Total training time: {total_train_secs:.1f}s")
    if gpu_peak_alloc is not None:
        print(f"[TRAIN] GPU peak mem: allocated={gpu_peak_alloc:.1f} MB, reserved={gpu_peak_rsvd:.1f} MB")

    # -------- Test inference (no labels on GLUE test) --------
    preds, inf_stats = predict_and_time(model, test_loader, device)
    counts = Counter(preds.tolist())
    print(f"[TEST] samples={inf_stats['samples']}, time={inf_stats['seconds']:.3f}s, "
          f"throughput={inf_stats['samples_per_sec']:.1f} samples/s")
    if inf_stats['gpu_peak_alloc_mb'] is not None:
        print(f"[TEST] GPU peak mem: allocated={inf_stats['gpu_peak_alloc_mb']:.1f} MB, "
              f"reserved={inf_stats['gpu_peak_reserved_mb']:.1f} MB")
    print(f"[TEST] class counts: {dict(counts)}   (0=neg, 1=pos)")

    # Save predictions to CSV for apples-to-apples comparison with MLA runs
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    test_csv = os.path.join(OUTPUT_DIR, "test_predictions.csv")
    with open(test_csv, "w") as f:
        f.write("index,label\n")
        for i, p in enumerate(preds.tolist()):
            f.write(f"{i},{p}\n")
    print(f"Saved test predictions to {test_csv}")

    # Save model + tokenizer
    model.save_pretrained(OUTPUT_DIR)
    tok.save_pretrained(OUTPUT_DIR)
    print(f"Saved model + tokenizer to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
