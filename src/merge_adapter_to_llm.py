#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge a trained LoRA adapter into a base Causal LM and save a standalone merged model.

Key features:
- All paths are CLI arguments (no hard-coded absolute paths).
- English-only logs/comments for anonymous release.
- Optional tiny forward-probe to sanity-check the merge.
- Works with PEFT LoRA adapters (adapter_config.json + adapter weights).
- Saves tokenizer alongside the merged model.

Example:
  python merge_lora_adapter.py \
    --base-model mistralai/Mistral-7B-Instruct-v0.2 \
    --adapter-dir ./runs/mistral7b_panda_lora \
    --out-dir ./models/mistral7b_panda_merged \
    --dtype float16 --device-map auto --local-files-only

Options:
  --no-probe           # skip the tiny forward sanity check
  --safe-serialize     # try to save with safetensors when available
"""

import os
import json
import argparse
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def check_adapter_dir(path: str):
    cfg = os.path.join(path, "adapter_config.json")
    has_weights = any(
        os.path.exists(os.path.join(path, fn))
        for fn in ["adapter_model.safetensors", "pytorch_model.bin"]
    )
    if not os.path.exists(cfg) or not has_weights:
        raise FileNotFoundError(
            f"[ERROR] Adapter directory incomplete: {path}\n"
            f"Expect adapter_config.json and adapter_model.safetensors/pytorch_model.bin."
        )


@torch.inference_mode()
def tiny_probe_diff(model_a, model_b, tok, device_a: str, device_b: str) -> float:
    """
    A tiny forward sanity check (not a task metric):
    compares the mean |Δlogits| of last token on a dummy input.
    """
    text = "Hello world! " * 64
    x = tok(text, return_tensors="pt", truncation=True, max_length=256)
    xa = {k: v.to(device_a) for k, v in x.items()}
    xb = {k: v.to(device_b) for k, v in x.items()}

    ya = model_a(**xa).logits[0, -1].float()
    yb = model_b(**xb).logits[0, -1].float()
    return (ya - yb).abs().mean().item()


def parse_args():
    ap = argparse.ArgumentParser(description="Merge LoRA adapter into base model.")
    ap.add_argument("--base-model", required=True,
                    help="Base model id or local path (e.g., mistralai/Mistral-7B-Instruct-v0.2)")
    ap.add_argument("--adapter-dir", required=True,
                    help="Directory containing LoRA adapter (adapter_config.json + weights)")
    ap.add_argument("--out-dir", required=True,
                    help="Output directory for merged model")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"],
                    help="Torch dtype for loading models")
    ap.add_argument("--device-map", default="auto",
                    help='Device map for loading (e.g., "auto", "cpu", "cuda:0")')
    ap.add_argument("--local-files-only", action="store_true",
                    help="Load models/tokenizer only from local cache")
    ap.add_argument("--no-probe", action="store_true",
                    help="Skip tiny forward sanity check")
    ap.add_argument("--safe-serialize", action="store_true",
                    help="Attempt to save with safetensors when available")
    return ap.parse_args()


def str_to_dtype(name: str):
    name = name.lower()
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    check_adapter_dir(args.adapter_dir)

    dtype = str_to_dtype(args.dtype)

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(
        args.base_model,
        local_files_only=args.local_files_only,
        use_fast=True
    )
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    # Base model
    print(f"[INFO] Loading base model: {args.base_model}")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map=args.device_map,
        torch_dtype=dtype,
        local_files_only=args.local_files_only
    ).eval()

    # Load LoRA adapter via PEFT
    from peft import PeftModel
    print(f"[INFO] Loading LoRA adapter: {args.adapter_dir}")
    peft_model = PeftModel.from_pretrained(
        base,
        args.adapter_dir,
        is_trainable=False
    ).eval()

    # Activate first adapter if multiple exist
    names = list(getattr(peft_model, "peft_config", {}).keys())
    active = names[0] if names else "default"
    if hasattr(peft_model, "set_adapter"):
        peft_model.set_adapter(active)

    # Log adapter config (best-effort)
    cfg = getattr(peft_model, "peft_config", {}).get(active, None)
    if cfg is not None:
        r = getattr(cfg, "r", None) or getattr(cfg, "lora_r", None)
        alpha = getattr(cfg, "lora_alpha", None)
        targets = getattr(cfg, "target_modules", None)
        print(f"[INFO] LoRA config: r={r}, alpha={alpha}, targets={targets}")

    # Merge -> standalone
    print("[INFO] Merging LoRA into base (merge_and_unload)...")
    merged = peft_model.merge_and_unload().eval()

    # Optional tiny sanity probe
    if not args.no_probe:
        try:
            # Derive device strings
            dev_a = next(merged.parameters()).device
            dev_b = next(base.parameters()).device
            diff = tiny_probe_diff(merged, base, tok, device_a=str(dev_a), device_b=str(dev_b))
            print(f"[PROBE] mean|Δlogits| (merged vs base) = {diff:.6e}")
        except Exception as e:
            print(f"[PROBE] Skipped due to error: {e}")

    # Save merged model + tokenizer
    print(f"[INFO] Saving merged model to: {out_dir}")
    # Move to CPU to avoid GPU OOM during serialization (best-effort)
    try:
        merged_cpu = merged.to("cpu")
    except Exception:
        merged_cpu = merged

    save_kwargs = {}
    if args.safe-serialize:
        # Newer transformers will automatically pick safetensors when available;
        # we keep a flag to emphasize intent (backward-friendly).
        save_kwargs["safe_serialization"] = True

    merged_cpu.save_pretrained(out_dir, **save_kwargs)
    tok.save_pretrained(out_dir)

    # Write a small meta for reproducibility
    meta = {
        "base_model": args.base_model,
        "adapter_dir": args.adapter_dir,
        "dtype": str(dtype),
        "device_map": args.device_map,
    }
    with open(out_dir / "merge_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[DONE] Merge complete. You can load it via AutoModelForCausalLM.from_pretrained(OUT_DIR).")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
