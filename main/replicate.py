# replicate_gpt2_base.py
# Evaluate GPT-2 base (12-layer) on GPT-2 paper-style benchmarks with streaming and total time reporting.

import os, math, time, json, random, numpy as np, torch
from typing import Iterable, List, Dict, Any, Optional, Tuple
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, GPT2LMHeadModel

# ======================== CONFIG ========================
MODEL_ID           = "gpt2"   # base, 12 layers
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"
SEED               = 42

# evaluation chunking
BLOCK_SIZE         = 1024     # must be <= model.config.n_positions (1024 for gpt2 base)
BATCH_SIZE_PPL     = 8        # reduce if you hit OOM
NUM_WORKERS        = 0

# place caches on a roomy path (no big writes; streaming used for datasets)
CACHE_DIR          = "/scr/shweta/hf_datasets_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ.setdefault("HF_HOME", CACHE_DIR)
os.environ.setdefault("HF_DATASETS_CACHE", CACHE_DIR)
os.environ.setdefault("TRANSFORMERS_CACHE", CACHE_DIR)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

OUT_JSON           = "replicate_gpt2_base_results.json"
# ========================================================

def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def trim_ctx(ids: List[int], max_len: int) -> List[int]:
    return ids[-max_len:] if len(ids) > max_len else ids

def model_forward_loss(model, input_ids: torch.Tensor, labels: torch.Tensor) -> float:
    out = model(input_ids=input_ids, labels=labels)
    return float(out.loss)

# ----------------- LAMBADA (streaming) -----------------
@torch.no_grad()
def eval_lambada(model, tok) -> Dict[str, Any]:
    """
    - Accuracy: greedy-generate the final word and compare.
    - Target-only perplexity: loss over target tokens only.
    - Full perplexity: loss over the entire passage (for reference).
    Uses streaming so nothing big is written to disk.
    """
    model.eval()
    try: model.config.use_cache = False
    except Exception: pass

    ds = load_dataset("lambada", split="test", streaming=True)

    npos = model.config.n_positions
    def split_ctx_target(text: str) -> Tuple[str, str]:
        text = text.rstrip()
        i = text.rfind(" ")
        if i < 0: return "", text
        return text[:i], text[i+1:]

    correct, total = 0, 0
    tgt_loss_sum, tgt_tok_sum = 0.0, 0
    full_loss_sum, full_tok_sum = 0.0, 0

    t0 = time.time()
    for ex in ds:
        text = ex["text"]
        ctx_txt, tgt_txt = split_ctx_target(text)

        # tokenization
        raw_ctx_ids = tok.encode(ctx_txt, add_special_tokens=False)
        tgt_ids     = tok.encode((" " if ctx_txt else "") + tgt_txt, add_special_tokens=False)
        if len(tgt_ids) == 0:
            continue

        # ---------- Accuracy (greedy on target) ----------
        # Make room so ctx_len + len(tgt_ids) <= n_positions
        allowed_ctx = max(1, npos - len(tgt_ids))
        ctx_ids = trim_ctx(raw_ctx_ids, allowed_ctx)

        # prefill
        x = torch.tensor([ctx_ids], dtype=torch.long, device=DEVICE)
        out  = model(input_ids=x, use_cache=True)
        past = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        gen = [next_tok]
        for _ in range(len(tgt_ids) - 1):
            out = model(input_ids=next_tok, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            gen.append(next_tok)
        pred_ids = torch.cat(gen, dim=1)[0].tolist()
        if tok.decode(pred_ids).strip() == tgt_txt.strip():
            correct += 1

        # ---------- Target-only PPL ----------
        # Fit context+target in window by trimming context only
        ctx_ids_for_loss = trim_ctx(raw_ctx_ids, max(1, npos - len(tgt_ids)))
        inp = torch.tensor([ctx_ids_for_loss + tgt_ids], dtype=torch.long, device=DEVICE)
        lab = torch.tensor([[-100]*len(ctx_ids_for_loss) + tgt_ids], dtype=torch.long, device=DEVICE)
        tgt_loss_sum += model_forward_loss(model, inp, lab) * len(tgt_ids)
        tgt_tok_sum  += len(tgt_ids)

        # ---------- Full passage PPL (reference) ----------
        full_ids = tok.encode(text, add_special_tokens=False)
        if len(full_ids) >= 2:
            finp = torch.tensor([trim_ctx(full_ids, npos)], dtype=torch.long, device=DEVICE)
            flab = finp.clone()
            flab[:, 0] = -100  # ignore the very first token
            full_loss_sum += model_forward_loss(model, finp, flab) * (finp.size(1) - 1)
            full_tok_sum  += (finp.size(1) - 1)

        total += 1

    sec = time.time() - t0
    acc = correct / max(1, total)
    tgt_nll  = tgt_loss_sum / max(1, tgt_tok_sum)
    full_nll = full_loss_sum / max(1, full_tok_sum) if full_tok_sum > 0 else float("inf")
    return {
        "acc": acc,
        "target_ppl": math.exp(min(50, tgt_nll)),
        "full_ppl": math.exp(min(50, full_nll)) if full_tok_sum > 0 else float("inf"),
        "n": total,
        "sec": sec,
    }

# --------------- CBT (CN/NE) accuracy -----------------
def _score_candidate_ll(model, tok, context: str, question: str, candidate: str) -> float:
    text = (context or "") + "\n\n" + question.replace("XXXXX", candidate)
    ids  = tok.encode(text, add_special_tokens=False)
    x = torch.tensor([trim_ctx(ids, model.config.n_positions)], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        out = model(input_ids=x, labels=x)
        nll = float(out.loss) * (x.size(1) - 1)
    return -nll  # higher is better

@torch.no_grad()
def eval_cbt(model, tok, variant="CN") -> Dict[str, Any]:
    # streaming test split
    ds = load_dataset("cbt", variant, split="test", streaming=True)
    correct, total = 0, 0
    t0 = time.time()
    for ex in ds:
        ctx = ex.get("context", ex.get("article", ""))
        q   = ex["question"]
        cands = ex.get("candidates") or ex.get("options") or []
        gold  = ex["answer"]
        if not cands:
            continue
        scores = [_score_candidate_ll(model, tok, ctx, q, c) for c in cands]
        pred = cands[int(torch.tensor(scores).argmax().item())]
        if pred == gold:
            correct += 1
        total += 1
    return {"acc": correct / max(1, total), "n": total, "sec": time.time() - t0}

# --------- Generic streaming perplexity (WT2/WT103) ----------
class BlockStream(torch.utils.data.IterableDataset):
    def __init__(self, hf_id: str, cfg: Optional[str], split: str, text_col: str,
                 tok: AutoTokenizer, block_size: int):
        super().__init__()
        self.hf_id, self.cfg, self.split = hf_id, cfg, split
        self.text_col = text_col
        self.tok = tok
        self.block = block_size

    def _text_iter(self) -> Iterable[str]:
        ds = load_dataset(self.hf_id, self.cfg, split=self.split, streaming=True, cache_dir=CACHE_DIR)
        for ex in ds:
            if self.text_col in ex and isinstance(ex[self.text_col], str):
                yield ex[self.text_col]

    def __iter__(self):
        buf: List[int] = []
        for text in self._text_iter():
            ids = self.tok.encode(text, add_special_tokens=False)
            if not ids:
                continue
            buf.extend(ids)
            while len(buf) >= self.block:
                chunk = buf[:self.block]; buf = buf[self.block:]
                ids_t = torch.tensor(chunk, dtype=torch.long)
                yield {"input_ids": ids_t, "labels": ids_t}

@torch.no_grad()
def eval_stream_ppl(model, loader) -> Dict[str, Any]:
    model.eval()
    try: model.config.use_cache = False
    except Exception: pass
    tok_sum, loss_sum = 0, 0.0
    t0 = time.time()
    for batch in tqdm(loader, leave=False):
        input_ids = batch["input_ids"].to(DEVICE)
        labels    = batch["labels"].to(DEVICE)
        out = model(input_ids=input_ids, labels=labels)
        T = input_ids.size(1)
        tok_sum  += input_ids.size(0) * (T - 1)
        loss_sum += float(out.loss) * (input_ids.size(0) * (T - 1))
    sec = time.time() - t0
    nll = loss_sum / max(1, tok_sum)
    ppl = math.exp(min(50, nll))
    return {"nll": nll, "ppl": ppl, "tok": tok_sum, "sec": sec}

# ------------------------------ main ------------------------------
def main():
    set_seed(SEED)
    device = torch.device(DEVICE)
    print(f"Device: {device}")

    # Tokenizer (GPT-2 base)
    tok = AutoTokenizer.from_pretrained("gpt2", cache_dir=CACHE_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Model (GPT-2 base)
    model = GPT2LMHeadModel.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    model.to(device).eval()

    results: Dict[str, Any] = {}

    # -------- LAMBADA --------
    print("\nLAMBADA (accuracy + target/full perplexity):")
    r = eval_lambada(model, tok)
    print(f"  Acc {100*r['acc']:.2f}% | Target-PPL {r['target_ppl']:.2f} | Full-PPL {r['full_ppl']:.2f} | "
          f"n={r['n']} | total {r['sec']:.1f}s")
    results["LAMBADA"] = r

    # -------- CBT-CN --------
    print("\nCBT-CN (accuracy):")
    r = eval_cbt(model, tok, variant="CN")
    print(f"  Acc {100*r['acc']:.2f}% | n={r['n']} | total {r['sec']:.1f}s")
    results["CBT-CN"] = r

    # -------- CBT-NE --------
    print("\nCBT-NE (accuracy):")
    r = eval_cbt(model, tok, variant="NE")
    print(f"  Acc {100*r['acc']:.2f}% | n={r['n']} | total {r['sec']:.1f}s")
    results["CBT-NE"] = r

    # -------- WikiText-2 --------
    print("\nWikiText2 (perplexity):")
    bs = BlockStream("wikitext", "wikitext-2-raw-v1", "test", "text", tok, BLOCK_SIZE)
    dl = torch.utils.data.DataLoader(bs, batch_size=BATCH_SIZE_PPL, num_workers=NUM_WORKERS)
    r = eval_stream_ppl(model, dl)
    print(f"  NLL {r['nll']:.5f} | PPL {r['ppl']:.2f} | {r['tok']} tok | total {r['sec']:.1f}s")
    results["WikiText2"] = r

    # -------- WikiText-103 --------
    print("\nWikiText103 (perplexity):")
    bs = BlockStream("wikitext", "wikitext-103-raw-v1", "test", "text", tok, BLOCK_SIZE)
    dl = torch.utils.data.DataLoader(bs, batch_size=BATCH_SIZE_PPL, num_workers=NUM_WORKERS)
    r = eval_stream_ppl(model, dl)
    print(f"  NLL {r['nll']:.5f} | PPL {r['ppl']:.2f} | {r['tok']} tok | total {r['sec']:.1f}s")
    results["WikiText103"] = r

    # -------- Historical script datasets (skipped cleanly) --------
    skip_notes = {
        "PTB": "HF script dataset often disabled (ptb_text_only.py)",
        "enwiki8": "HF script dataset often disabled (enwik8.py)",
        "text8": "Not currently available on Hub under 'text8'",
        "1BW": "HF script dataset often disabled (lm1b.py)",
    }
    for k, note in skip_notes.items():
        print(f"\n{k}: [skipped] {note}")
        results[k] = {"skipped": True, "note": note}

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {OUT_JSON}")

if __name__ == "__main__":
    main()
