#!/usr/bin/env python3
"""
-----------------
- Data format supported (per line, JSON):
    {"original": "...", "perturbed": "...", "selected_word": "...", "target_attribute": "..."}
  or
    {"source": "...", "target": "...", "selected_word": "...", "target_attribute": "..."}

- Before & after training: evaluate pairwise preference accuracy on EVAL_PATH.
  We report both sum-logprob ("official-style") and mean-per-token (length-normalized) variants.

- Run modes:
    * full (default): no sub-sampling; evaluate on the entire validation set
    * fast: small subset + limited steps for quick sanity checks

- Resuming:
    Set --enable-resume and optionally --resume-dir to continue from a checkpoint.

This script contains NO hardcoded local usernames/paths or model cache paths.
"""

import os
import json
import time
import random
from pathlib import Path
from typing import List, Optional

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from tqdm import tqdm
import argparse


# -----------------------------
# Defaults (overridable by CLI)
# -----------------------------
DEFAULT_MODEL_ID = os.environ.get("BASE_MODEL_ID", "xxx")
DEFAULT_DATA_PATH = os.environ.get("PANDA_TRAIN", "xxx")
DEFAULT_EVAL_PATH = os.environ.get("PANDA_EVAL",  "xxx")
DEFAULT_OUT_DIR   = os.environ.get("OUTPUT_DIR",  "xxx")

MAX_LEN    = 512
BATCH_SIZE = 3
GRAD_ACCUM = 16
EPOCHS     = 1
LR         = 2e-4
WARMUP     = 0.03

# LoRA configuration (for Mistral-like archs)
LORA_R         = 8
LORA_ALPHA     = 16
LORA_DROPOUT   = 0.05
TARGET_MODULES = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

# System prompt wrapper
SYS = "You are generating respectful stories and sentences."
def wrap_inst(user_text: str) -> str:
    return f"<s>[INST] <<SYS>>\n{SYS}\n<</SYS>>\n{user_text.strip()} [/INST]"


def build_panda_prompt(original: str, sel_word: str, target_attr: str) -> str:
    """Construct the PANDA re-write instruction."""
    return (
        "Rewrite the text by replacing demographic references so that "
        f"'{sel_word}' becomes '{target_attr}'. Preserve the original meaning and grammar.\n\n"
        f"Text: {original.strip()}"
    )


# -----------------------------
# Data loading & parsing
# -----------------------------
def try_parse_panda_pair(ex: dict) -> Optional[dict]:
    """
    Convert a raw PANDA example into a pair {prompt, chosen, rejected}.
    Returns None if required keys are missing.
    """
    # canonical PANDA fields
    if all(k in ex for k in ["original","perturbed","selected_word","target_attribute"]):
        p  = build_panda_prompt(ex["original"], ex["selected_word"], ex["target_attribute"])
        ch = ex["perturbed"].strip()
        rj = ex["original"].strip()
        return {"prompt": p, "chosen": ch, "rejected": rj}

    # alternate names occasionally used in derived datasets
    if ("source" in ex) and ("target" in ex):
        sel = ex.get("selected_word","X")
        tgt = ex.get("target_attribute","Y")
        p  = build_panda_prompt(ex["source"], sel, tgt)
        ch = ex["target"].strip()
        rj = ex["source"].strip()
        return {"prompt": p, "chosen": ch, "rejected": rj}

    return None


def load_pairs(jsonl_path: str) -> List[dict]:
    pairs, n = [], 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            n += 1
            try:
                ex = json.loads(ln)
            except Exception:
                continue
            rec = try_parse_panda_pair(ex)
            if rec:
                pairs.append(rec)
    if not pairs:
        raise ValueError(
            f"Failed to parse any PANDA pairs from: {jsonl_path}. "
            "Expected fields: original/perturbed/selected_word/target_attribute (or source/target variants)."
        )
    print(f"[INFO] Loaded {n} lines; parsed {len(pairs)} PANDA pairs from {jsonl_path}")
    return pairs


def set_seed(s: int):
    random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# -----------------------------
# Evaluation utilities
# -----------------------------
@torch.inference_mode()
def seq_logprob(model, tok, prompt_wrapped: str, response: str):
    """
    Compute token-level log-prob for the completion tokens conditioned on the prompt.
    Returns (sum_logprob, num_tokens).
    """
    p_ids = tok(
        prompt_wrapped, return_tensors="pt", truncation=True, max_length=MAX_LEN
    ).to(model.device)
    full_ids = tok(
        prompt_wrapped + " " + response, return_tensors="pt",
        truncation=True, max_length=MAX_LEN
    ).to(model.device)

    start = p_ids["input_ids"].shape[1]
    input_ids = full_ids["input_ids"]

    out = model(input_ids)
    logits = out.logits[:, start-1:-1, :]
    target = input_ids[:, start:]
    logprobs = torch.log_softmax(logits, dim=-1)
    tok_lp = logprobs.gather(2, target.unsqueeze(-1)).squeeze(-1)

    lp_sum = float(tok_lp.sum().item())
    lp_len = int(target.numel())
    return lp_sum, lp_len


def eval_pairwise_pref_acc(
    model, tok, pairs: List[dict], sample_limit: Optional[int] = None
) -> dict:
    """
    Pairwise preference accuracy:
      - *_sum: based on sum logprob (official-style).
      - *_avg: based on mean per-token logprob (length-normalized).
    If sample_limit is None, evaluates the full list.
    """
    use = pairs if sample_limit is None else pairs[:sample_limit]
    model.eval()

    correct_sum = 0
    correct_avg = 0
    margins_sum, margins_avg = [], []

    for ex in tqdm(use, desc="Eval(pref-acc)"):
        pr = wrap_inst(ex["prompt"])

        ch_sum, ch_len = seq_logprob(model, tok, pr, ex["chosen"])
        rj_sum, rj_len = seq_logprob(model, tok, pr, ex["rejected"])

        # sum version
        if ch_sum > rj_sum:
            correct_sum += 1
        margins_sum.append(ch_sum - rj_sum)

        # mean-per-token version
        ch_avg = ch_sum / max(1, ch_len)
        rj_avg = rj_sum / max(1, rj_len)
        if ch_avg > rj_avg:
            correct_avg += 1
        margins_avg.append(ch_avg - rj_avg)

    n = len(use) if use else 0

    def _stats(xs):
        if not xs:
            return (0.0, 0.0, 0.0)
        return (
            round(float(sum(xs) / len(xs)), 6),
            round(float(min(xs)), 6),
            round(float(max(xs)), 6),
        )

    avg_m_sum, min_m_sum, max_m_sum = _stats(margins_sum)
    avg_m_avg, min_m_avg, max_m_avg = _stats(margins_avg)

    return {
        "n": n,
        "pref_accuracy_sum": round(correct_sum / n, 6) if n else 0.0,
        "avg_margin_sum": avg_m_sum,
        "min_margin_sum": min_m_sum,
        "max_margin_sum": max_m_sum,
        "pref_accuracy_avg": round(correct_avg / n, 6) if n else 0.0,
        "avg_margin_avg": avg_m_avg,
        "min_margin_avg": min_m_avg,
        "max_margin_avg": max_m_avg,
    }


def pretty_print_eval(tag, res):
    print(f"\n[{tag}] n={res['n']}")
    print(
        f"  (sum)  pref-acc_sum={res['pref_accuracy_sum']:.4f} | "
        f"avg_margin_sum={res['avg_margin_sum']:.3f} "
        f"[{res['min_margin_sum']:.3f}, {res['max_margin_sum']:.3f}]"
    )
    print(
        f"  (mean) pref-acc_avg={res['pref_accuracy_avg']:.4f} | "
        f"avg_margin_avg={res['avg_margin_avg']:.5f} "
        f"[{res['min_margin_avg']:.5f}, {res['max_margin_avg']:.5f}]"
    )


def latest_checkpoint_dir(root: str) -> Optional[str]:
    p = Path(root)
    if not p.exists():
        return None
    cks = [d for d in p.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
    if not cks:
        return None
    cks.sort(key=lambda x: int(x.name.split("-")[-1]))
    return str(cks[-1])


# -----------------------------
# Main
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="SFT + LoRA on PANDA; full-set eval by default.")
    ap.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID,
                    help="Base HF model id (e.g., mistralai/Mistral-7B-Instruct-v0.2).")
    ap.add_argument("--train-jsonl", type=str, default=DEFAULT_DATA_PATH,
                    help="PANDA training JSONL path.")
    ap.add_argument("--eval-jsonl", type=str, default=DEFAULT_EVAL_PATH,
                    help="PANDA eval JSONL path.")
    ap.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR,
                    help="Output directory for checkpoints and logs.")
    ap.add_argument("--run-mode", type=str, choices=["full", "fast"], default=os.environ.get("RUN_MODE","full"),
                    help="full: full dataset & steps; fast: small subset for sanity checks.")
    ap.add_argument("--enable-resume", action="store_true",
                    help="Enable resume-from-checkpoint if found.")
    ap.add_argument("--resume-dir", type=str, default="",
                    help="Explicit checkpoint dir to resume from; if empty, auto-pick the latest in out-dir.")
    return ap.parse_args()


def main():
    args = parse_args()

    # Run-mode presets
    PRESETS = {
        "fast": {  # connectivity test mode
            "TRAIN_LIMIT": 100,
            "EVAL_LIMIT":  10,
            "MAX_STEPS":   300,
            "LOG_STEPS":   50,
            "SAVE_STEPS":  200,
            "SAVE_LIMIT":  3,
            "NUM_PROC":    8,
            "NUM_WORKERS": 2,
        },
        "full": {  # default: no sub-sampling; evaluate entire eval set
            "TRAIN_LIMIT": None,
            "EVAL_LIMIT":  None,   # <-- full eval set
            "MAX_STEPS":   None,
            "LOG_STEPS":   20,
            "SAVE_STEPS":  1000,
            "SAVE_LIMIT":  2,
            "NUM_PROC":    8,
            "NUM_WORKERS": 2,
        },
    }
    CFG = PRESETS[args.run_mode]

    start = time.time()
    out_dir = Path(args.out_dir).expanduser().absolute()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output dir: {out_dir}")
    print(f"[INFO] Run mode:   {args.run_mode}")
    print(f"[INFO] Resume:     {'enabled' if args.enable_resume else 'disabled'}")


    # Load data
    train_pairs = load_pairs(args.train_jsonl)
    eval_pairs  = load_pairs(args.eval_jsonl)

    # Optional train sub-sampling (fast mode)
    if CFG["TRAIN_LIMIT"]:
        train_pairs = train_pairs[:min(CFG["TRAIN_LIMIT"], len(train_pairs))]

    eval_limit = CFG["EVAL_LIMIT"]  # None => evaluate full set

    ds_train = Dataset.from_list(train_pairs)

    # dtype
    can_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    dtype = torch.bfloat16 if can_bf16 else torch.float16

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.model_max_length = MAX_LEN
    tok.padding_side = "right"

    # Try flash-attn if available
    attn_impl = None
    if torch.cuda.is_available():
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except Exception:
            pass

    # Base model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation=attn_impl,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # LoRA
    lora_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Pre-train evaluation (FULL eval set by default)
    print("\n[Pre-Train Evaluation]")
    pre = eval_pairwise_pref_acc(model, tok, eval_pairs, sample_limit=eval_limit)
    pretty_print_eval("Pre-Train", pre)
    with open(out_dir / "eval_before.json", "w", encoding="utf-8") as f:
        json.dump(pre, f, ensure_ascii=False, indent=2)

    # Prepare SFT data (concatenate instruction + chosen continuation)
    def sft_format(batch):
        return {"text": [wrap_inst(p) + " " + ch.strip()
                         for p, ch in zip(batch["prompt"], batch["chosen"])]}

    ds_proc = ds_train.map(
        sft_format,
        batched=True,
        remove_columns=ds_train.column_names,
        num_proc=CFG["NUM_PROC"]
    )

    # TRL SFT configuration
    targs = SFTConfig(
        output_dir=str(out_dir),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP,
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),

        logging_steps=CFG["LOG_STEPS"],
        save_steps=CFG["SAVE_STEPS"],
        save_total_limit=CFG["SAVE_LIMIT"],
        report_to=[],

        dataset_text_field="text",
        packing=False,
        max_length=MAX_LEN,
        dataset_num_proc=CFG["NUM_PROC"],
        dataloader_num_workers=CFG["NUM_WORKERS"],
        group_by_length=False,

        max_steps=CFG["MAX_STEPS"] if CFG["MAX_STEPS"] is not None else -1,
    )

    trainer = SFTTrainer(
        model=model,
        args=targs,
        train_dataset=ds_proc,
        processing_class=tok,
    )

    # Train
    if args.enable_resume:
        resume_ckpt = args.resume_dir or latest_checkpoint_dir(str(out_dir))
        if resume_ckpt:
            print(f"[INFO] Resuming from: {resume_ckpt}")
            trainer.train(resume_from_checkpoint=resume_ckpt)
        else:
            print("[INFO] No checkpoint found; starting fresh.")
            trainer.train()
    else:
        trainer.train()

    # Save LoRA-only adapter checkpoint to out_dir
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    print(f"\n[INFO] LoRA-SFT finished: {out_dir}")

    # Post-train evaluation (FULL eval set by default)
    print("\n[Post-Train Evaluation]")
    post = eval_pairwise_pref_acc(model, tok, eval_pairs, sample_limit=eval_limit)
    pretty_print_eval("Post-Train", post)
    with open(out_dir / "eval_after.json", "w", encoding="utf-8") as f:
        json.dump(post, f, ensure_ascii=False, indent=2)

    # Summary deltas
    print("\n=== Preference Accuracy (chosen > rejected) ===")
    print(
        f"(sum)  Before: {pre['pref_accuracy_sum']:.4f} | After: {post['pref_accuracy_sum']:.4f} "
        f"| Δ={post['pref_accuracy_sum']-pre['pref_accuracy_sum']:.4f}"
    )
    print(
        f"(mean) Before: {pre['pref_accuracy_avg']:.4f} | After: {post['pref_accuracy_avg']:.4f} "
        f"| Δ={post['pref_accuracy_avg']-pre['pref_accuracy_avg']:.4f}"
    )

    print(f"\nTotal time: {(time.time()-start)/60:.2f} min")


if __name__ == "__main__":
    main()
