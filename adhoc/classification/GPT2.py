# gpt2_sst2_manual_clean.py
import os, random, numpy as np, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, GPT2ForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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
NUM_WORKERS      = 2              # set 0 if dataloader multiprocessing is an issue
# =================================================

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

class SST2Dataset(Dataset):
    """Stores raw strings + labels; we tokenize in collate (fast path)."""
    def __init__(self, texts, labels):
        self.texts  = [str(t) for t in list(texts)]
        self.labels = torch.tensor(list(labels), dtype=torch.long)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.texts[idx], self.labels[idx]

def make_collate(tokenizer):
    def collate(batch):
        texts, labels = zip(*batch)  # tuple of strs, tuple of tensors
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

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, total_loss = [], [], 0.0
    ce = nn.CrossEntropyLoss().to(device)
    with torch.no_grad():
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

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) data & tokenizer
    ds = load_dataset("glue", "sst2")  # 'sentence', 'label'
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token  # GPT-2: reuse EOS as PAD

    train_set = SST2Dataset(ds["train"]["sentence"], ds["train"]["label"])
    val_set   = SST2Dataset(ds["validation"]["sentence"], ds["validation"]["label"])

    collate_fn = make_collate(tok)
    pin_mem = (device.type == "cuda")
    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, pin_memory=pin_mem, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, pin_memory=pin_mem, num_workers=NUM_WORKERS
    )

    # 2) model
    model = GPT2ForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = tok.eos_token_id
    model.config.use_cache = False
    model.to(device)

    # 3) optimizer + AMP
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    ce = nn.CrossEntropyLoss().to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # 4) training loop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss, running_correct, running_total = 0.0, 0, 0
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

        # eval
        metrics = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:02d} | "
              f"Train Loss: {running_loss/running_total:.4f} | "
              f"Train Acc: {running_correct/running_total:.4f} | "
              f"Val Loss: {metrics['loss']:.4f} | "
              f"Val Acc: {metrics['accuracy']:.4f} | "
              f"P: {metrics['precision']:.4f} R: {metrics['recall']:.4f} F1: {metrics['f1']:.4f}")

    # 5) save
    model.save_pretrained(OUTPUT_DIR)
    tok.save_pretrained(OUTPUT_DIR)
    print(f"Saved model + tokenizer to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
