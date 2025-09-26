# eval_gpt2_smollm.py
import os, math, time, json, torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional, Iterable
from dataclasses import dataclass
from torch.utils.data import DataLoader, IterableDataset
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Config


@torch.no_grad()
def rolling_ppl(model: nn.Module, tokenizer: AutoTokenizer, texts, window: int = 1024, stride: int = 512) -> Dict[str, Any]:
    """Compute perplexity with rolling window (stride) over a text iterable."""
    model.eval()
    if hasattr(model, "config"):
        try: model.config.use_cache = False
        except Exception: pass
    nll_sum = 0.0
    tok_count = 0
    t0 = time.time()
    for text in tqdm(texts, leave=False):
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) < 2: continue
        i = 0
        while i < len(ids):
            chunk = ids[max(0, i - window): i]
            target = ids[max(0, i - window + 1): i + 1]
            if len(target) == 0:
                i += stride; continue
            x = torch.tensor([chunk[-window:]], dtype=torch.long, device=DEVICE)
            y = torch.tensor([[-100]*(len(chunk)-len(target)) + target[-window:]], dtype=torch.long, device=DEVICE)
            out = model(input_ids=x, labels=y)
            nll_sum += float(mget(out, "loss")) * (len(target))
            tok_count += len(target)
            i += stride
    if torch.cuda.is_available(): torch.cuda.synchronize()
    sec = time.time() - t0
    return {"nll": (nll_sum / max(1, tok_count)), "ppl": math.exp(min(50, nll_sum/max(1, tok_count))), "sec": sec}



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


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ======================= CONFIG (NO ARGS) =======================
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
HF_GPT2         = "gpt2"  # base (12-layer) GPT-2
FT_GPT2_DIR     = "./results/gpt2-smollm-vanilla-stream"      # change if you saved elsewhere
FT_MLA_DIR      = "./results/gpt2-smollm-cosmopedia-v2-mla-jkv-stream"  # change if you saved elsewhere

# batching for perplexity datasets
BLOCK_SIZE      = 512     # <= n_positions (1024 for gpt2)
BATCH_SIZE      = 8
NUM_WORKERS     = 0
OUT_JSON        = "eval_results.json"

# What to run
RUN_LAMBADA     = True
RUN_CBT_CN      = True
RUN_CBT_NE      = True
RUN_WT2         = True
RUN_WT103       = True

# ===============================================================

# ---- small helper so code works for both HF ModelOutput and dicts (MLA) ----
def mget(out, key):
    if hasattr(out, key):
        return getattr(out, key)
    return out[key]

# ---- try to import your MLA classes from either filename you use ----
MLA_IMPORT_OK = False
try:
    from GPT2_MLA_finetune_stream import (
        MLAJointConfig as _MLAJointConfig,
        GPT2MLAJointForCausalLM as _GPT2MLAJointForCausalLM,
    )
    MLA_IMPORT_OK = True
except Exception:
    try:
        from GPT2_smollm_MLA_stream import (
            MLAJointConfig as _MLAJointConfig,
            GPT2MLAJointForCausalLM as _GPT2MLAJointForCausalLM,
    )
        MLA_IMPORT_OK = True
    except Exception:
        pass

# ---------- MLA loader (reads config/text files saved by your train script) ----------
def load_mla_from_dir(ft_dir: str, device: torch.device):
    if not MLA_IMPORT_OK:
        raise RuntimeError("Could not import MLA classes. Ensure the MLA finetune file is in PYTHONPATH.")

    gcfg = GPT2Config.from_json_file(os.path.join(ft_dir, "gpt2_config.json"))
    # parse dataclass-like text file
    with open(os.path.join(ft_dir, "mla_joint_config.txt")) as f:
        txt = f.read()
    fields = {}
    for part in txt.replace("MLAJointConfig(", "").replace(")", "").split(","):
        if "=" in part:
            k, v = [s.strip() for s in part.split("=")]
            if k in {"hidden_size", "num_heads", "head_dim", "q_rank", "kv_rank"}:
                fields[k] = int(v)
            elif k in {"attn_dropout", "resid_dropout"}:
                fields[k] = float(v)
            elif k == "use_bias":
                fields[k] = v.lower() == "true"
    mcfg = _MLAJointConfig(**fields)
    model = _GPT2MLAJointForCausalLM(gcfg, mcfg)
    sd = torch.load(os.path.join(ft_dir, "pytorch_model.bin"), map_location=device)
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval()

# ---------- generic streaming block packer for WT2/WT103 ----------
def make_block_stream(dataset_name: str, config: Optional[str], split: str,
                      tokenizer: AutoTokenizer, block_size: int) -> IterableDataset:
    stream = load_dataset(dataset_name, config, split=split, streaming=True)

    def gen():
        buf = []
        for ex in stream:
            # choose first string field
            if "text" in ex and isinstance(ex["text"], str):
                text = ex["text"]
            else:
                k = next((k for k, v in ex.items() if isinstance(v, str)), None)
                if k is None:
                    continue
                text = ex[k]
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if not ids:
                continue
            buf.extend(ids)
            while len(buf) >= block_size:
                chunk = buf[:block_size]
                buf = buf[block_size:]
                yield {
                    "input_ids": torch.tensor(chunk, dtype=torch.long),
                    "labels":    torch.tensor(chunk, dtype=torch.long),
                }

    class _Iter(IterableDataset):
        def __iter__(self):
            return gen()

    return _Iter()

@torch.no_grad()
def eval_stream_ppl(model: nn.Module, loader: DataLoader) -> Dict[str, Any]:
    model.eval()
    # teacher-forced loss → disable cache
    if hasattr(model, "config"):
        try: model.config.use_cache = False
        except Exception: pass

    tok_sum, loss_sum = 0, 0.0
    t0 = time.time()
    for batch in tqdm(loader, leave=False):
        x = batch["input_ids"].to(DEVICE)
        # ignore first token when computing loss (shifted CE)
        labels = x.clone(); labels[:, 0] = -100
        out = model(input_ids=x, labels=labels)
        loss = mget(out, "loss")
        tok_sum  += x.size(0) * (x.size(1) - 1)
        loss_sum += float(loss) * (x.size(0) * (x.size(1) - 1))
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    sec = time.time() - t0
    nll = loss_sum / max(1, tok_sum)
    ppl = math.exp(min(50, nll))
    return {"nll": nll, "ppl": ppl, "sec": sec}

# ---------- LAMBADA: accuracy + target-only PPL + full-pass PPL ----------
@torch.no_grad()
def eval_lambada(model: nn.Module, tok: AutoTokenizer) -> Dict[str, Any]:
    model.eval()
    npos = model.config.n_positions
    ds = load_dataset("EleutherAI/lambada_openai", split="test", streaming=True)

    def split_ctx_target(text: str) -> Tuple[str, str]:
        text = text.rstrip()
        i = text.rfind(" ")
        if i < 0:
            return "", text
        return text[:i], text[i + 1 :]

    correct = total = 0
    tgt_loss_sum = tgt_tok_sum = 0.0
    full_loss_sum = full_tok_sum = 0.0

    t0 = time.time()
    for ex in tqdm(ds, leave=False):
        text = ex["text"]
        ctx_txt, tgt_txt = split_ctx_target(text)
        # tokenization
        raw_ctx = tok.encode(ctx_txt, add_special_tokens=False)
        tgt_ids = tok.encode((" " if ctx_txt else "") + tgt_txt, add_special_tokens=False)
        if len(tgt_ids) == 0:
            continue

        # --- greedy generation accuracy over target tokens (with KV cache) ---
        allowed_ctx = max(1, npos - len(tgt_ids))
        ctx_ids = raw_ctx[-allowed_ctx:] if len(raw_ctx) > allowed_ctx else raw_ctx
        x = torch.tensor([ctx_ids], dtype=torch.long, device=DEVICE)
        out  = model(input_ids=x, use_cache=True)
        past = mget(out, "past_key_values")
        next_tok = mget(out, "logits")[:, -1, :].argmax(dim=-1, keepdim=True)
        gen = [next_tok]
        for _ in range(len(tgt_ids) - 1):
            out = model(input_ids=next_tok, past_key_values=past, use_cache=True)
            past = mget(out, "past_key_values")
            next_tok = mget(out, "logits")[:, -1, :].argmax(dim=-1, keepdim=True)
            gen.append(next_tok)
        pred_ids = torch.cat(gen, dim=1)[0].tolist()
        if tok.decode(pred_ids).strip() == tgt_txt.strip():
            correct += 1

        # --- target-only perplexity (mask context) ---
        inp = torch.tensor([ (raw_ctx[-allowed_ctx:] if len(raw_ctx) > allowed_ctx else raw_ctx) + tgt_ids ],
                           dtype=torch.long, device=DEVICE)
        lab = torch.tensor([ [-100]* (inp.size(1)-len(tgt_ids)) + tgt_ids ],
                           dtype=torch.long, device=DEVICE)
        out = model(input_ids=inp, labels=lab)
        tgt_loss_sum += float(mget(out, "loss")) * len(tgt_ids)
        tgt_tok_sum  += len(tgt_ids)

        # --- full passage perplexity (reference) ---
        full_ids = tok.encode(text, add_special_tokens=False)
        if len(full_ids) >= 2:
            finp = torch.tensor([full_ids[-npos:]], dtype=torch.long, device=DEVICE)
            flab = finp.clone(); flab[:, 0] = -100
            out  = model(input_ids=finp, labels=flab)
            full_loss_sum += float(mget(out, "loss")) * (finp.size(1) - 1)
            full_tok_sum  += (finp.size(1) - 1)

        total += 1

    torch.cuda.synchronize() if torch.cuda.is_available() else None
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

# ---------- CBT multiple-choice accuracy (CN/NE) ----------
def _as_text(v):
    """Join list-of-sentences or return string; else empty."""
    if isinstance(v, list):
        return " ".join([s for s in v if isinstance(s, str)])
    return v if isinstance(v, str) else ""

def _extract_cands(ex):
    cands = ex.get("candidates", None)
    if cands is None:
        cands = ex.get("options", None) or ex.get("choices", None)
    if isinstance(cands, str):
        # common CBT formatting: pipe-separated
        cands = [c.strip() for c in cands.split("|") if c.strip()]
    elif isinstance(cands, list):
        cands = [str(c).strip() for c in cands if str(c).strip()]
    else:
        cands = []
    return cands

def _extract_answer(ex):
    ans = ex.get("answer", None)
    if ans is None:
        ans = ex.get("answers", None) or ex.get("target", None)
    if isinstance(ans, list):
        ans = ans[0] if ans else ""
    return (str(ans).strip() if ans is not None else "")

def _make_question_template(qtxt: str) -> str:
    """
    Normalize different blank markers to a single {BLANK} token
    so we can fill with each candidate.
    """
    q = qtxt or ""
    if "@placeholder" in q:
        return q.replace("@placeholder", "{BLANK}")
    if "XXXXX" in q:
        return q.replace("XXXXX", "{BLANK}")
    if "_____" in q:
        return q.replace("_____", "{BLANK}")
    # if no explicit blank, just append one at the end
    return (q + " {BLANK}").strip()

def _score_text_nll(model: nn.Module, tok: AutoTokenizer, text: str) -> float:
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) < 2:
        return 0.0
    # respect context window
    ids = ids[-model.config.n_positions:]
    x = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        # standard shifted CE: mask first token
        labels = x.clone()
        labels[:, 0] = -100
        out = model(input_ids=x, labels=labels)
        # loss is mean over (T-1) tokens → multiply back to get total nll
        nll = float(mget(out, "loss")) * (x.size(1) - 1)
    return nll

@torch.no_grad()
def eval_cbt(model: nn.Module, tok: AutoTokenizer, variant: str) -> dict:
    """
    CBT-CN / CBT-NE accuracy with total wall-clock time.
    variant: "CN" or "NE"
    """
    model.eval()
    ds = load_dataset("cbt", variant, split="test", streaming=True)

    correct = 0
    total = 0
    t0 = time.time()

    for ex in tqdm(ds, leave=False):
        passage = _as_text(ex.get("story", None)) or _as_text(ex.get("passage", None)) or ""
        qtxt    = _as_text(ex.get("question", "")) or ""
        tmpl    = _make_question_template(qtxt)

        cands   = _extract_cands(ex)
        answer  = _extract_answer(ex)
        if not cands or not answer:
            continue

        # score each candidate with teacher-forced NLL
        best_c, best_score = None, float("inf")
        prefix = (passage + "\n\n") if passage else ""
        for c in cands:
            text = prefix + tmpl.replace("{BLANK}", c)
            nll = _score_text_nll(model, tok, text)
            if nll < best_score:
                best_score, best_c = nll, c

        if (best_c or "").strip() == answer.strip():
            correct += 1
        total += 1

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    sec = time.time() - t0
    acc = correct / max(1, total)
    return {"acc": acc, "n": total, "sec": sec}

# ============================== main ==============================
def main():
    device = torch.device(DEVICE)
    print(f"Device: {device}")

    tok = AutoTokenizer.from_pretrained(HF_GPT2)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Build model set
    TRY_COMPILE=False
    models = {}
    print("Loading models...")
    models["pretrained-gpt2"] = GPT2LMHeadModel.from_pretrained(HF_GPT2).to(device).eval()
    if TRY_COMPILE and hasattr(torch, "compile"): 
        try: models["pretrained-gpt2"]=torch.compile(models["pretrained-gpt2"], mode="reduce-overhead")
        except Exception as e: print(f"[warn] torch.compile failed for pretrained-gpt2: {e}")

    if os.path.isdir(FT_GPT2_DIR):
        try:
            models["ft-gpt2"] = GPT2LMHeadModel.from_pretrained(FT_GPT2_DIR).to(device).eval()
            if TRY_COMPILE and hasattr(torch, "compile"): 
                try: models["ft-gpt2"]=torch.compile(models["ft-gpt2"], mode="reduce-overhead")
                except Exception as e: print(f"[warn] torch.compile failed for ft-gpt2: {e}")
        except Exception as e:
            print(f"[warn] Failed to load ft-gpt2 from {FT_GPT2_DIR}: {e}")
    else:
        print(f"[warn] ft_gpt2_dir not found: {FT_GPT2_DIR} (skipping)")

    if os.path.isdir(FT_MLA_DIR):
        try:
            models["ft-mla"] = load_mla_from_dir(FT_MLA_DIR, device)
            if TRY_COMPILE and hasattr(torch, "compile"): 
                try: models["ft-mla"]=torch.compile(models["ft-mla"], mode="reduce-overhead")
                except Exception as e: print(f"[warn] torch.compile failed for ft-mla: {e}")
        except Exception as e:
            print(f"[warn] Failed to load ft-mla from {FT_MLA_DIR}: {e}")
    else:
        print(f"[warn] ft_mla_dir not found: {FT_MLA_DIR} (skipping)")

    results = {}

    # -------- LAMBADA --------
    if RUN_LAMBADA:
        print("\n=== LAMBADA (accuracy + target/full perplexity) ===")
        for tag, m in models.items():
            r = eval_lambada(m, tok)
            print(f"{tag:>16s} | Acc {r['acc']*100:5.2f}% | Target-PPL {r['target_ppl']:.2f} | "
                  f"Full-PPL {r['full_ppl']:.2f} | n={r['n']} | total {r['sec']:.1f}s")
            results.setdefault("LAMBADA", {})[tag] = r

    # -------- CBT-CN --------
    if RUN_CBT_CN:
        print("\n=== CBT-CN (accuracy) ===")
        for tag, m in models.items():
            r = eval_cbt(m, tok, "CN")
            print(f"{tag:>16s} | Acc {r['acc']*100:5.2f}% | n={r['n']} | total {r['sec']:.1f}s")
            results.setdefault("CBT-CN", {})[tag] = r

    # -------- CBT-NE --------
    if RUN_CBT_NE:
        print("\n=== CBT-NE (accuracy) ===")
        for tag, m in models.items():
            r = eval_cbt(m, tok, "NE")
            print(f"{tag:>16s} | Acc {r['acc']*100:5.2f}% | n={r['n']} | total {r['sec']:.1f}s")
            results.setdefault("CBT-NE", {})[tag] = r

    # -------- WikiText-2 perplexity --------
    if RUN_WT2:
        print("\n=== WikiText2 (perplexity) ===")
        wt2 = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", streaming=False)["text"]
        for tag, m in models.items():
            loader = DataLoader(ds_iter, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                                pin_memory=(device.type == "cuda"))
            r = rolling_ppl(m, tok, wt2, window=BLOCK_SIZE, stride=BLOCK_SIZE//2)
            print(f"{tag:>16s} | NLL {r['nll']:.5f} | PPL {r['ppl']:.2f} | total {r['sec']:.1f}s")
            results.setdefault("WikiText2", {})[tag] = r

    # -------- WikiText-103 perplexity --------
    if RUN_WT103:
        print("\n=== WikiText103 (perplexity) ===")
        wt103 = load_dataset("wikitext", "wikitext-103-raw-v1", split="test", streaming=False)["text"]
        for tag, m in models.items():
            loader = DataLoader(ds_iter, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                                pin_memory=(device.type == "cuda"))
            r = rolling_ppl(m, tok, wt2, window=BLOCK_SIZE, stride=BLOCK_SIZE//2)
            print(f"{tag:>16s} | NLL {r['nll']:.5f} | PPL {r['ppl']:.2f} | total {r['sec']:.1f}s")
            results.setdefault("WikiText103", {})[tag] = r

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved full results to {OUT_JSON}")

if __name__ == "__main__":
    main()
