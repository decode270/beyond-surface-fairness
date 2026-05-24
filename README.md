
> Anonymous repository for research replication.

## Overview
This repository provides a reproducible pipeline to study gender implicit bias in LLMs in 4 axes (Occupation Family_role, study_ability, Character), followed by identity-free generation, metric computation, and visualization.

- Create templates for inducing implicit stereotypes and bias in LLMs.
- Generate stories from prompt templates.
- Supervised LoRA Fine-tune LLMs on PANDA pairs.
- Merge adapter into base model for inference
- Compute bias metrics (FS/SHR)
- Plot results (FS/SHR) for bidirectional bias.

---

## Folder-by-Folder Guide (what is inside)

### `configs/`
- `llama8b.yml`, `mistral7b.yml`  configuration stubs for model family / run presets (used to keep experiment knobs in one place).

### `data/`
Input prompt templates, and generated answers from Mistral 7B and Llama 8B grouped by 4 axes:

- `Character/`, `Family_role/`, `Occupation/`, `Study_ability/`
  - `template_attribution/`
    - `female.txt`, `male.txt` attribute/trait lists per axis.
    - `templates.txt` identity-free prompt templates with `{occ}` placeholders.
  - `llama/`, `mistral/`.
    - May contain intermediate XLSX artifacts organized by base model (`llama`, `mistral`) and debiased variants (`stf, ins, dpo`).  
    - Typical files:
      - `_*_*_*_analyze.xlsx` analysis spreadsheets with per-story tagging stats (Female/Male/Unknown counts, tags, first-hit, etc.).
  - `StereoSet_data`:
    -  `dev.json`, `test.json` orignal downloaded from StereoSet
  - `Training_data`:
    -  `Constructed_PANDA_DPO_train`: constructed from PANDA dataset for dpo format, which used for dpo_lora fine-tuning.
    -  `Huggingface_BiasDPO`: downloaded original data and cited_link shown in paper, which used for dpo_lora fine-tuing with constructed panda_dpo.
    -  PANDA data were used for SFT lora fine-tuning.

### `results/`
Official outputs and figures for the submission.

- `Visualization/`
  - `Character/`, `Family_role/`, `Occupation/`, `Study_ability`
    - `llama/`, `mistral/`
      - visualization for bidirectional bias by Stereotype-hit Rate [ln(SHR)] and distributional balance by Fairness Score [FS] , e.g., `MODEL_AXIS_VARIANT_FS/SHR.pdf`, (FS/SHR comparisons across models/axes).
    - Papers' Figures:  
      - Methodology's process illustrations included.
- `Metrics_results/`
  - `_*_*_*_metrics.xlsx` metrics spreadsheets with computed measurment (SHR/FS) for the diagnostic framework and built-in Excel formulas.
- `StereoSet_test`:
  -  tested results by ss,lms,icat, across debiased variants and baselines in stereoset benchmark.

### `src/`
Source code for training, merging, generation, analysis, and plotting.

- `generate_story_all.py` : generation of identity-free stories from templates; supports merged or base models.
- `analyze_metrics.py` metrics pipeline: token tagging + aggregation + SHR/FS computation; can also export Excel with formulas.
- `evaluation.py` and `stereoset_test.py` were used for ss, lms, icat testing built on stereoSet.
- `Visualization/`
  - `Vis_FS.R`  plots Fairness Score distributions (FS).
  - `Vis_SHR.R`  plots Stereotype-Hit-Rate comparisons (SHR).


### Root files
- `environment.yml`  Conda environment spec (Python, PyTorch, Transformers, TRL/PEFT and so on).
- `README.md`  this document.

## Environment
```bash
conda env create -f environment.yml
```
