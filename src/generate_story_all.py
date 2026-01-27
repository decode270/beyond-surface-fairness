import random
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ========= botton：Used to determine whether to load the merged model. =========
USE_MERGED = True  # False: bas model；True:MERGED_DIR with lora
MERGED_DIR = "xxxx"  #

BATCH_SIZE     = 8
N_GEN_PER_OCC  = 10
MAX_NEW_TOKENS = 128
TEMP, TOP_P    = 0.9, 0.8
REP_PENALTY    = 1.1

PROMPT_FILE = "xxx.txt"
F_FILE      = "xxx.txt"
M_FILE      = "xxx.txt"
OUT_CSV     = "xxx.csv"

SUFFIX = " Response in English. You are generating respectful stories and sentences.Do not look back at any previous generation output."

def load_list(txt_path):
    return [ln.strip() for ln in Path(txt_path).read_text(encoding="utf-8").splitlines() if ln.strip()]

def build_dataset(tmpl_list, f_occ, m_occ):
    prompts, groups, occs, iters = [], [], [], []
    for g, occ_list in [("F", f_occ), ("M", m_occ)]:  
        for occ in occ_list:
            for tmpl in tmpl_list:
                base_prompt = tmpl.format(occ=occ) + SUFFIX
                for i in range(N_GEN_PER_OCC):
                    prompts.append(base_prompt)
                    groups.append(g)
                    occs.append(occ)
                    iters.append(i)
    return Dataset.from_dict(
        {"prompt": prompts, "group": groups, "occupation": occs, "iter": iters}
    )


def main():
    tmpl_list = load_list(PROMPT_FILE)
    occ_f = load_list(F_FILE)
    occ_m = load_list(M_FILE)
    ds = build_dataset(tmpl_list, occ_f, occ_m)

    # ——botton——
    model_path = MERGED_DIR if (USE_MERGED and Path(MERGED_DIR).exists()) else MODEL_ID
    if USE_MERGED and not Path(MERGED_DIR).exists():
        raise FileNotFoundError(f"USE_MERGED=True，but cannot find MERGED_DIR: {MERGED_DIR}")

    
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token

    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True
    )

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        batch_size=BATCH_SIZE,
        do_sample=True,
        temperature=TEMP,
        top_p=TOP_P,
        repetition_penalty=REP_PENALTY,
        max_new_tokens=MAX_NEW_TOKENS,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    def _gen(batch):
        outs = generator(batch["prompt"], return_full_text=False)
        stories = [seq[0]["generated_text"].strip() for seq in outs]
        return {"story": stories}

    ds_out = ds.map(_gen, batched=True, batch_size=BATCH_SIZE)

    (ds_out
     .to_pandas()
     .sort_values(["group", "occupation", "iter"])
     .to_csv(OUT_CSV, index=False, encoding="utf-8"))

    print(f" Complete Generation，Sum {len(ds_out)} items，and saved to {OUT_CSV}")
    print(f" Model Sourced from：{'MERGED_DIR' if USE_MERGED else 'MODEL_ID'} -> {model_path}")

if __name__ == "__main__":
    torch.manual_seed(random.randint(0, 2**32 - 1))
    main()
