#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate stories with identity-free, underspecified prompts across attributes (e.g., occupation, character, etc.)
Supports either a base model from Hugging Face or a locally merged (base + LoRA) model.

Usage examples:
  # Use a merged model on disk
  python generate_story_all.py \
    --use-merged \
    --merged-dir ./checkpoints/llama-merged \
    --prompt-file ./study_ability/templates.txt \
    --female-file ./study_ability/female_ability.txt \
    --male-file ./study_ability/male_ability.txt \
    --out-csv ./study_ability/llama/stories_ability_lm.csv \
    --axis-name study_ability

  # Use a base model from HF (user must have access)
  python generate_story_all.py \
    --model-id mistralai/Mistral-7B-Instruct-v0.2 \
    --prompt-file ./occupation/templates.txt \
    --female-file ./occupation/female_occ.txt \
    --male-file ./occupation/male_occ.txt \
    --out-csv ./occupation/mistral/stories_occ_base.csv \
    --axis-name occupation
"""

import argparse
import random
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


def parse_args():
    p = argparse.ArgumentParser(description="Generate identity-free stories for implicit-bias evaluation.")
    # Model selection
    p.add_argument("--use-merged", action="store_true",
                   help="If set, load a locally merged (base+LoRA) model from --merged-dir. Otherwise use --model-id.")
    p.add_argument("--merged-dir", type=str, default=None,
                   help="Path to a locally merged model directory (if --use-merged is set).")
    p.add_argument("--model-id", type=str, default=None,
                   help="Hugging Face model id or local path to a base model (used when --use-merged is not set).")
    p.add_argument("--local-files-only", action="store_true",
                   help="If set, load tokenizer/model only from local cache without internet.")
    # IO: prompts and attributes
    p.add_argument("--prompt-file", type=str, required=True,
                   help="Path to prompt templates (.txt). Each line should contain a template using '{occ}' or '{attribute}'.")
    p.add_argument("--female-file", type=str, required=True,
                   help="Path to female-associated attribute list (.txt).")
    p.add_argument("--male-file", type=str, required=True,
                   help="Path to male-associated attribute list (.txt).")
    p.add_argument("--out-csv", type=str, required=True,
                   help="Output CSV file to save generations.")
    p.add_argument("--axis-name", type=str, default="study_ability",
                   help="Name of the axis (e.g., occupation, character, family_role, study_ability).")
    # Generation config
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--gens-per-attr", type=int, default=10,
                   help="Number of generations per attribute per template.")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--repetition-penalty", type=float, default=1.05)
    p.add_argument("--suffix", type=str,
                   default=" Response in English. You are generating respectful stories and sentences. "
                           "Do not look back at any previous generation output.",
                   help="A suffix appended to each rendered prompt.")
    # Reproducibility
    p.add_argument("--seed", type=int, default=None, help="Random seed (defaults to random if not set).")
    return p.parse_args()


def load_list(txt_path: str):
    path = Path(txt_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {txt_path}")
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def build_dataset(tmpl_list, female_attrs, male_attrs, gens_per_attr: int, suffix: str):
    """
    Build a huggingface Dataset with repeated prompts:
      - group: 'F' or 'M' (the attribute-list source)
      - attribute: the attribute token (e.g., occupation term)
      - iter: replicate index
    """
    prompts, groups, attrs, iters = [], [], [], []
    # Accept both {occ} and {attribute} placeholders in templates for compatibility
    def render_template(tmpl: str, attr: str) -> str:
        if "{attribute}" in tmpl:
            return tmpl.format(attribute=attr)
        return tmpl.format(occ=attr)

    for group_label, attr_list in [("F", female_attrs), ("M", male_attrs)]:
        for attr in attr_list:
            for tmpl in tmpl_list:
                base_prompt = render_template(tmpl, attr) + suffix
                for i in range(gens_per_attr):
                    prompts.append(base_prompt)
                    groups.append(group_label)
                    attrs.append(attr)
                    iters.append(i)

    return Dataset.from_dict(
        {"prompt": prompts, "group": groups, "attribute": attrs, "iter": iters}
    )


def main():
    args = parse_args()

    # Seed
    if args.seed is None:
        seed = random.randint(0, 2**32 - 1)
    else:
        seed = args.seed
    torch.manual_seed(seed)

    # Load templates and attributes
    tmpl_list = load_list(args.prompt_file)
    female_attrs = load_list(args.female_file)
    male_attrs = load_list(args.male_file)

    ds = build_dataset(
        tmpl_list=tmpl_list,
        female_attrs=female_attrs,
        male_attrs=male_attrs,
        gens_per_attr=args.gens_per_attr,
        suffix=args.suffix,
    )

    # Resolve model path or id
    if args.use_merged:
        if not args.merged_dir or not Path(args.merged_dir).exists():
            raise FileNotFoundError(f"--use-merged is set but merged directory not found: {args.merged_dir}")
        model_ref = args.merged_dir
        source_label = "MERGED_DIR"
    else:
        if not args.model_id:
            raise ValueError("When --use-merged is not set, you must provide --model-id.")
        model_ref = args.model_id
        source_label = "MODEL_ID"

    # Load tokenizer / model
    tokenizer = AutoTokenizer.from_pretrained(model_ref, local_files_only=args.local_files_only)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=args.local_files_only
    )

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_new_tokens,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    def _gen(batch):
        outs = generator(batch["prompt"], return_full_text=False)
        stories = [seq[0]["generated_text"].strip() for seq in outs]
        return {"story": stories}

    ds_out = ds.map(_gen, batched=True, batch_size=args.batch_size)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    (ds_out
     .to_pandas()
     .assign(axis=args.axis_name, seed=seed, model_source=source_label, model_ref=str(model_ref))
     .sort_values(["group", "attribute", "iter"])
     .to_csv(out_path, index=False, encoding="utf-8"))

    print(f"[OK] Generated {len(ds_out)} stories; saved to: {out_path}")
    print(f"[INFO] Model source: {source_label} -> {model_ref}")
    print(f"[INFO] Axis: {args.axis_name} | Seed: {seed}")


if __name__ == "__main__":
    main()
