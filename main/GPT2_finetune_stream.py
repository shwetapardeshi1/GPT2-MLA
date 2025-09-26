# GPT2_smollm_vanilla_stream.py
import os, math, time, random, numpy as np, torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from torch.optim import AdamW
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, GPT2LMHeadModel

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ==================== CONFIG ====================
MODEL_NAME       = "gpt2"
DATASET_NAME     = "HuggingFaceTB/smollm-corpus"
DATASET_CONFIG   = "fineweb-edu-dedup"
OUTPUT_DIR       = "./results/gpt2-smollm-fineweb-edu-dedup-vanilla-stream"

BLOCK_SIZE       = 1024
BATCH_SIZE       = 8
EPOCHS           = 3
LEARNING_RATE    = 5e-5
WEIGHT_DECAY     = 0.0
GRAD_CLIP        = 1.0
SEED             = 42

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

@torch.no_grad()
def decode_benchmark(model, tokenizer, device, base_text, context_lengths, max_new_tokens=128):
    model.eval()
    results = []
    npos = int(getattr(model.config, "n_positions", 1024))

    def build_prompt_tokens(text, target_len):
        ids = tokenizer.encode(text, add_special_tokens=False) or [tokenizer.eos_token_id]
        while len(ids) < target_len:
            ids = ids + ids
        return torch.tensor(ids[:target_len], dtype=torch.long)

    for ctx in context_lengths:
        # cap context to model limit
        ctx = min(int(ctx), npos)

        # how many tokens can we safely generate?
        remain = max(0, npos - ctx)
        eff_new = min(int(max_new_tokens), remain)

        prompt_ids = build_prompt_tokens(base_text, ctx).unsqueeze(0).to(device)

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # prefill (build KV cache for the prompt)
        t0 = time.time()
        out = model(input_ids=prompt_ids, use_cache=True)
        past = out.past_key_values
        if device.type == "cuda":
            torch.cuda.synchronize()
        prefill_time = time.time() - t0

        decode_time = 0.0
        if eff_new > 0:
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

            t1 = time.time()
            # we already have 1 token (next_token); generate eff_new-1 more steps
            for _ in range(eff_new - 1):
                out = model(input_ids=next_token, past_key_values=past, use_cache=True)
                past = out.past_key_values
                next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            decode_time = time.time() - t1

        peak_mem = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
        results.append({
            "context": ctx,
            "prefill_sec": prefill_time,
            "decode_sec": decode_time,
            "decode_tps": (eff_new / max(1e-9, decode_time)) if eff_new > 0 else float("nan"),
            "peak_gpu_bytes": int(peak_mem),
        })

    return results

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
            if not ids:
                continue
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
    # ensure no KV cache for loss computation
    if hasattr(model, "config"):
        try: model.config.use_cache = False
        except: pass

    total_loss, total_tokens, steps = 0.0, 0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
        loss = out.loss
        toks = batch["labels"].numel()
        total_loss += float(loss) * toks
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
    npos = model.config.n_positions
    for ctx in context_lengths:
        ctx = min(ctx, npos)
        prompt = build_prompt_tokens(tokenizer, base_text, ctx).unsqueeze(0).to(device)

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # prefill (build KV caches for the prompt)
        t0 = time.time()
        out = model(input_ids=prompt, use_cache=True)
        past = out.past_key_values
        if device.type == "cuda": torch.cuda.synchronize()
        prefill = time.time() - t0

        # token-by-token decode with cached KV
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        t1 = time.time()
        for _ in range(max_new_tokens - 1):
            out = model(input_ids=next_tok, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        if device.type == "cuda": torch.cuda.synchronize()
        decode = time.time() - t1

        peak = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
        res.append({
            "context": ctx,
            "prefill_sec": prefill,
            "decode_sec": decode,
            "decode_tps": (max_new_tokens / max(1e-9, decode)),
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

    # vanilla GPT-2 (no MLA)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = tok.eos_token_id
    model.config.use_cache = False  # OFF for training/perplexity
    model.to(device)

    total = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total/1e6:.2f}M")

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
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
                out = model(input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["labels"])
                loss = out.loss
            scaler.scale(loss).backward()
            if GRAD_CLIP and GRAD_CLIP > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer); scaler.update()

            toks = batch["labels"].numel()
            running_loss += float(loss) * toks
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

    # ====== decode-time benchmark with HF KV cache ======
    model.config.use_cache = True
    base_text = "The history of natural language processing begins in the 1950s."
    valid_ctx = [
    c for c in BENCH_CONTEXTS
    if (c < model.config.n_positions) and (c + BENCH_MAX_NEW) <= model.config.n_positions
    ]

    bench = decode_benchmark(model, tok, device, base_text, valid_ctx, max_new_tokens=BENCH_MAX_NEW)
    print("\n=== Decode-time KV Cache Benchmark (smollm-corpus, vanilla GPT-2, streaming) ===")
    for r in bench:
        print(f"Context={r['context']:4d} | prefill {r['prefill_sec']:.3f}s | "
              f"decode {r['decode_sec']:.3f}s | {r['decode_tps']:.1f} tok/s | "
              f"peak GPU {fmt_bytes(r['peak_gpu_bytes'])}")

    # save
    model.save_pretrained(OUTPUT_DIR)
    tok.save_pretrained(OUTPUT_DIR)
    print(f"\nSaved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
