
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

- `character/`, `family_role/`, `occupation/`, `study_ability/`
  - `template_attribution/`
    - `female.txt`, `male.txt` attribute/trait lists per axis.
    - `templates.txt` identity-free prompt templates with `{occ}` placeholders.
  - `llama/`, `mistral/`, `ft_llama/`, `ft_mistral/`
    - May contain intermediate XLSX artifacts organized by base model (`llama`, `mistral`) and fine-tuned variants (`ft_*`).  
    - Typical files:
      - `stories_*_*_analyze.xlsx` analysis spreadsheets with per-story tagging stats (Female/Male/Unknown counts, tags, first-hit, etc.).  

### `results/`
Official outputs and figures for the submission.

- `Graphs/`
  - `character/`, `family_role/`, `occupation/`, `study_ability`
    - `llama/`, `mistral/`  Visualization, e.g., `Rplot_fs.png`, `Rplot_shr.png` (FS/SHR comparisons across models/axes).
    - Papers' Figures:  
      - `bidirect.png`, `example_detection.png`, `process.png` diagrams illustrations included in the paper figures.
- `metrics_results/`
  - `stories_*_*_metrics.xlsx` metrics spreadsheets with computed fairness measures (SHR/FS) and built-in Excel formulas.

### `src/`
Source code for training, merging, generation, analysis, and plotting.

- `lora_panda.py` : LoRA SFT on PANDA dataset.
- `merge_adapter_to_llm.py` : merge a trained LoRA adapter into the base model and save a standalone merged model.
- `generate_story_all.py` : generation of identity-free stories from templates; supports merged or base models.
- `analyze_metrics.py` metrics pipeline: token tagging + aggregation + SHR/FS computation; can also export Excel with formulas.
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
