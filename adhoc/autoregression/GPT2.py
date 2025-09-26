# GPT2_vanilla_with_cache.py
import os, math, time, random, numpy as np, torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, GPT2LMHeadModel

# Silence fork/parallelism warning from HF tokenizers
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ==================== CONFIG ====================
MODEL_NAME     = "gpt2"          # try "gpt2-medium" for better results
OUTPUT_DIR     = "./gpt2-wt2-vanilla-lm"
BLOCK_SIZE     = 512             # context length for training chunks
BATCH_SIZE     = 2               # adjust to your GPU
EPOCHS         = 3
LEARNING_RATE  = 5e-5
WEIGHT_DECAY   = 0.0
GRAD_CLIP      = 1.0
SEED           = 42
NUM_WORKERS    = 2               # 0 if dataloader workers cause issues
# Inference (decode) benchmark
BENCH_MAX_NEW  = 128
BENCH_CONTEXTS = [128, 512, 1024]  # must be <= model.config.n_positions
# =================================================

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def prepare_data(tokenizer, block_size=BLOCK_SIZE):
    """
    Loads WikiText-2 (raw), tokenizes, and packs into fixed-length blocks.
    Returns torch-formatted datasets: train/validation/test with
    fields: input_ids, attention_mask (all ones), labels (=input_ids).
    """
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")

    def tok(batch):
        # Faster when passing lists directly to fast tokenizer
        return tokenizer(batch["text"])

    tokenized = ds.map(tok, batched=True, remove_columns=["text"])

    def group_texts(examples):
        # flatten then chunk into block_size
        ids = [id_ for seq in examples["input_ids"] for id_ in seq]
        n = (len(ids) // block_size) * block_size
        if n == 0:
            return {"input_ids": [], "attention_mask": [], "labels": []}
        ids = ids[:n]
        chunks = [ids[i:i+block_size] for i in range(0, n, block_size)]
        attn  = [[1]*block_size for _ in range(len(chunks))]
        return {"input_ids": chunks, "attention_mask": attn, "labels": [c[:] for c in chunks]}

    lm_ds = tokenized.map(group_texts, batched=True, remove_columns=tokenized["train"].column_names)
    lm_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return lm_ds

@torch.no_grad()
def evaluate_perplexity(model, loader, device):
    """
    Average NLL over tokens (model computes shifted CE when labels are provided).
    """
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"])
        loss = out.loss  # mean over tokens in batch
        tokens = batch["labels"].numel()
        total_loss += loss.item() * tokens
        total_tokens += tokens
    avg_nll = total_loss / max(1, total_tokens)
    ppl = math.exp(avg_nll) if avg_nll < 50 else float("inf")
    return {"nll": avg_nll, "perplexity": ppl}

def build_prompt_tokens(tokenizer, base_text: str, target_len: int):
    """
    Tokenize base_text and repeat it to reach (or exceed) target_len, then trim to target_len.
    """
    ids = tokenizer.encode(base_text, add_special_tokens=False)
    if len(ids) == 0:
        ids = [tokenizer.eos_token_id]
    while len(ids) < target_len:
        ids = ids + ids  # repeat
    return torch.tensor(ids[:target_len], dtype=torch.long)

@torch.no_grad()
def decode_benchmark(model, tokenizer, device, base_text, context_lengths, max_new_tokens=128):
    """
    Measures prefill time (feeding the prompt once with use_cache=True),
    decode time per token (autoregressive with KV cache), and peak GPU memory.
    Returns list of dicts per context length.
    """
    model.eval()
    results = []
    npos = model.config.n_positions

    for ctx in context_lengths:
        ctx = min(ctx, npos)  # safety
        prompt_ids = build_prompt_tokens(tokenizer, base_text, ctx).unsqueeze(0).to(device)

        # Measure
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # Prefill (build KV cache for the prompt)
        t0 = time.time()
        out = model(input_ids=prompt_ids, use_cache=True)
        past = out.past_key_values  # tuple of length num_layers; each has (k, v)
        if device.type == "cuda": torch.cuda.synchronize()
        prefill_time = time.time() - t0

        # Decode loop (token-by-token) using cached K/V
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # greedy to keep deterministic
        gen_ids = [next_token]

        t1 = time.time()
        for _ in range(max_new_tokens - 1):
            out = model(input_ids=next_token, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            gen_ids.append(next_token)
        if device.type == "cuda": torch.cuda.synchronize()
        decode_time = time.time() - t1

        peak_mem = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0

        results.append({
            "context": ctx,
            "prefill_sec": prefill_time,
            "decode_sec": decode_time,
            "decode_tps": (max_new_tokens / max(1e-9, decode_time)),
            "peak_gpu_bytes": int(peak_mem),
        })

        # Optional: get the generated text (not printed by default)
        # full_out = torch.cat([prompt_ids, torch.cat(gen_ids, dim=1)], dim=1)
        # text = tokenizer.decode(full_out[0].tolist())
    return results

def fmt_bytes(n):
    for u in ["B","KB","MB","GB","TB"]:
        if n < 1024: return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} PB"

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'})")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # tokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token  # GPT-2 has no PAD; reuse EOS

    # data
    t_data0 = time.time()
    ds = prepare_data(tok, BLOCK_SIZE)
    t_data = time.time() - t_data0
    print(f"Data prep time: {t_data:.2f}s | Train blocks: {len(ds['train'])}  Val blocks: {len(ds['validation'])}  Test blocks: {len(ds['test'])}")

    pin_mem = (device.type == "cuda")
    train_loader = DataLoader(ds["train"], batch_size=BATCH_SIZE, shuffle=True,
                              pin_memory=pin_mem, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(ds["validation"], batch_size=BATCH_SIZE, shuffle=False,
                              pin_memory=pin_mem, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(ds["test"], batch_size=BATCH_SIZE, shuffle=False,
                              pin_memory=pin_mem, num_workers=NUM_WORKERS)

    # model
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = tok.eos_token_id
    model.config.use_cache = False  # IMPORTANT: no cache during training/eval perplexity
    model.to(device)

    # optimizer + AMP
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # training loop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss, running_tokens = 0.0, 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for batch in pbar:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                out = model(input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["labels"])  # shifted CE inside
                loss = out.loss

            scaler.scale(loss).backward()
            if GRAD_CLIP and GRAD_CLIP > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            tokens = batch["labels"].numel()
            running_loss += loss.item() * tokens
            running_tokens += tokens
            avg_nll = running_loss / max(1, running_tokens)
            pbar.set_postfix(nll=avg_nll, ppl=(math.exp(avg_nll) if avg_nll < 50 else float("inf")))

        if device.type == "cuda": torch.cuda.synchronize()
        train_time = time.time() - t0
        peak_mem = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0

        # validation perplexity
        val_metrics = evaluate_perplexity(model, val_loader, device)
        print(f"Epoch {epoch:02d} | Train NLL: {running_loss/max(1,running_tokens):.5f} | "
              f"Train PPL: {math.exp(min(50, running_loss/max(1,running_tokens))):.3f} | "
              f"Val NLL: {val_metrics['nll']:.5f} | Val PPL: {val_metrics['perplexity']:.3f}")
        print(f"Timing: train {train_time:.2f}s "
              f"(tok/s {running_tokens/max(1e-9,train_time):.0f}) | "
              f"GPU peak alloc: {fmt_bytes(peak_mem)}")

    # final test perplexity
    test_metrics = evaluate_perplexity(model, test_loader, device)
    print(f"\nTest NLL: {test_metrics['nll']:.5f} | Test Perplexity: {test_metrics['perplexity']:.3f}")

    # ====== AUTOREGRESSIVE INFERENCE BENCHMARK (WITH KV CACHING) ======
    # Switch model to cache for generation
    model.config.use_cache = True
    base_text_for_prompt = "The history of natural language processing begins in the 1950s."
    # Ensure contexts don't exceed model n_positions
    valid_contexts = [c for c in BENCH_CONTEXTS if c <= model.config.n_positions]
    bench = decode_benchmark(
        model, tok, device,
        base_text=base_text_for_prompt,
        context_lengths=valid_contexts,
        max_new_tokens=BENCH_MAX_NEW
    )
    print("\n=== Decode-time KV Cache Benchmark (vanilla GPT-2) ===")
    for r in bench:
        print(f"Context={r['context']:4d} | prefill {r['prefill_sec']:.3f}s | "
              f"decode {r['decode_sec']:.3f}s | {r['decode_tps']:.1f} tok/s | "
              f"peak GPU {fmt_bytes(r['peak_gpu_bytes'])}")

    # save model + tokenizer
    model.save_pretrained(OUTPUT_DIR)
    tok.save_pretrained(OUTPUT_DIR)
    print(f"\nSaved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
