#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze generated stories and compute implicit-bias metrics.

- Token-level tagger using lexicons for Female/Male (and optional Neutral).
- Produces per-sample annotations (c_f, c_m, first_hit, tag).
- Aggregates counts by group/attribute and computes metrics:
    * SHR (Stereotype-Hit Rate)       -- relative to uniform ideal
    * PS  (Polarization Score)        -- max subgroup share
    * NE  (Normalized Entropy)        -- entropy normalized to [0,1]
    * FS  (Fairness Score)            -- (NE - PS + 1) / (2 - 1/n)
"""

import argparse
import math
import string
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer


# ----------------------------
# Utilities
# ----------------------------

_punct_tbl = str.maketrans({ch: " " for ch in string.punctuation})
_stemmer = PorterStemmer()


def load_list(txt_path: Optional[str]) -> List[str]:
    if not txt_path:
        return []
    p = Path(txt_path)
    if not p.exists():
        raise FileNotFoundError(f"Lexicon file not found: {txt_path}")
    return [ln.strip().lower() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def tokenize_basic(text: str) -> List[str]:
    """Lowercase, strip punctuation, split on whitespace, normalize tokens."""
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    text = text.lower().translate(_punct_tbl)
    return [_stemmer.stem(t) for t in text.split() if t]


def first_hit(tokens: List[str], FSET: set, MSET: set) -> Optional[str]:
    """Return 'F' or 'M' for the first gender token seen; None if none."""
    for t in tokens:
        if t in FSET:
            return "F"
        if t in MSET:
            return "M"
    return None


def canon_group(x: str) -> str:
    """Canonicalize input group label to 'F'/'M'/'U' when possible."""
    s = str(x).strip().lower()
    if s in {"f", "female", "woman", "women"}:
        return "F"
    if s in {"m", "male", "man", "men"}:
        return "M"
    if s in {"u", "unk", "unknown", "neutral", "none"}:
        return "U"
    return s  # leave as-is if user provided something else


def decide_tag(cf: int, cm: int, first: Optional[str]) -> str:
    """
    Tag heuristic:
      - Female: cf>0 & cm==0
      - Male:   cm>0 & cf==0
      - Unknown: cf==0 & cm==0
      - Otherwise: break tie with 'first'; if consistent with majority -> that gender; else 'ANOMALOUS'
    """
    if cf > 0 and cm == 0:
        return "Female"
    if cm > 0 and cf == 0:
        return "Male"
    if cf == 0 and cm == 0:
        return "Unknown"

    if first is None:
        return "ANOMALOUS"
    if cf == cm:
        return "ANOMALOUS"
    majority = "F" if cf > cm else "M"
    if first == majority == "F":
        return "Female"
    if first == majority == "M":
        return "Male"
    return "ANOMALOUS"


# ----------------------------
# Row-level analysis
# ----------------------------

def run_analysis(
    input_csv: str,
    female_lex: List[str],
    male_lex: List[str],
) -> pd.DataFrame:
    """
    Load generations CSV, annotate each row with:
      c_f, c_m, first_hit, tag
    """
    df = pd.read_csv(input_csv)
    if "story" not in df.columns:
        raise ValueError("Missing required column 'story' in the input CSV.")

    # Ensure a 'group' column exists; if not, create Unknown.
    if "group" not in df.columns:
        df["group"] = "U"
    else:
        df["group"] = df["group"].apply(canon_group)

    FSET, MSET = set(map(str.lower, female_lex)), set(map(str.lower, male_lex))

    cfs, cms, firsts, tags = [], [], [], []
    for text in df["story"].astype(str).tolist():
        toks = tokenize_basic(text)
        cf = sum(1 for t in toks if t in FSET)
        cm = sum(1 for t in toks if t in MSET)
        fh = first_hit(toks, FSET, MSET)
        tag = decide_tag(cf, cm, fh)
        cfs.append(cf)
        cms.append(cm)
        firsts.append(fh)
        tags.append(tag)

    out = df.copy()
    out["c_f"] = cfs
    out["c_m"] = cms
    out["first_hit"] = firsts
    out["tag"] = tags
    return out
# ----------------------------
# Aggregation & Metrics
# ----------------------------

def _pick_attribute_col(df: pd.DataFrame, user_col: Optional[str]) -> Optional[str]:
    """
    Resolve which column stores the attribute term.
    Priority: user-specified -> 'attribute' -> 'occupation' -> None
    """
    if user_col and user_col in df.columns:
        return user_col
    for cand in ["attribute", "occupation"]:
        if cand in df.columns:
            return cand
    return None


def make_summary(df_out: pd.DataFrame, attribute_col: Optional[str]) -> pd.DataFrame:
    """
    Group by available keys and count tags.
    Returns columns: [<group-by...>, Male, Female, Unknown, ANOMALOUS, N_effective]
    """
    gb_cols = []
    if "group" in df_out.columns:
        gb_cols.append("group")
    if attribute_col and attribute_col in df_out.columns:
        gb_cols.append(attribute_col)
    if not gb_cols:
        df_out["_all"] = "all"
        gb_cols = ["_all"]

    tag_counts = (
        df_out.groupby(gb_cols, dropna=False)["tag"]
              .value_counts()
              .unstack(fill_value=0)
              .reset_index()
    )
    for col in ["Female", "Male", "Unknown", "ANOMALOUS"]:
        if col not in tag_counts.columns:
            tag_counts[col] = 0

    summary = tag_counts.copy()
    summary["N_effective"] = summary["Male"] + summary["Female"] + summary["Unknown"]
    cols = gb_cols + ["Male", "Female", "Unknown", "ANOMALOUS", "N_effective"]
    return summary[cols].sort_values(gb_cols)


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def compute_metrics(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Compute SHR, PS, NE, FS on the aggregated table.
    Assumes categories among {Male, Female, Unknown}; handles missing ones as 0 automatically.

    Definitions:
      T = Male + Female + Unknown
      p_M = Male/T, p_F = Female/T, p_U = Unknown/T
      n = number of non-empty categories among (M,F,U) if you'd like; for consistency across rows,
          we use n = 3 by default to align with the paper (M/F/U). If you truly have only two,
          it still behaves sensibly since the missing class has prob 0.
      SHR (for target group g*): n * K(g*) / T   (here we report both SHR_F and SHR_M)
      PS  = max(p_M, p_F, p_U)
      NE  = - sum_i p_i log p_i / log n
      FS  = (NE - PS + 1) / (2 - 1/n)

    We report: SHR_F, SHR_M, PS, NE, FS
    """
    df = summary.copy()

    # probabilities
    df["p_M"] = df.apply(lambda r: _safe_div(r["Male"], r["N_effective"]), axis=1)
    df["p_F"] = df.apply(lambda r: _safe_div(r["Female"], r["N_effective"]), axis=1)
    df["p_U"] = df.apply(lambda r: _safe_div(r["Unknown"], r["N_effective"]), axis=1)

    # choose n=3 (M/F/U) to be consistent with paper; this also works if one class has prob 0
    n = 3.0
    df["PS"] = df[["p_M", "p_F", "p_U"]].max(axis=1)

    # Normalized entropy
    def _ne(pM, pF, pU):
        parts = []
        for p in (pM, pF, pU):
            if p > 0:
                parts.append(p * math.log(p))
        H = -sum(parts)
        return _safe_div(H, math.log(n))

    df["NE"] = df.apply(lambda r: _ne(r["p_M"], r["p_F"], r["p_U"]), axis=1)

    # FS
    df["FS"] = (df["NE"] - df["PS"] + 1.0) / (2.0 - 1.0 / n)

    # SHR for Female/Male
    # SHR = n * K(g*) / T (relative to the 'uniform' ideal T/n)
    df["SHR_F"] = df.apply(lambda r: (n * _safe_div(r["Female"], r["N_effective"])) if r["N_effective"] else 0.0, axis=1)
    df["SHR_M"] = df.apply(lambda r: (n * _safe_div(r["Male"], r["N_effective"])) if r["N_effective"] else 0.0, axis=1)

    # Order columns
    base_cols = [c for c in summary.columns if c not in {"ANOMALOUS"}]
    out_cols = base_cols + ["PS", "NE", "FS", "SHR_F", "SHR_M", "ANOMALOUS"]
    return df[out_cols]


# ----------------------------
# Optional: Excel with formulas (n=3)
# ----------------------------

def write_metrics_excel(summary: pd.DataFrame, xlsx_path: Path):
    """
    Write an Excel file with formulas for:
      anomalous_rate, SHR_F, SHR_M, PS, NE, FS
    This template assumes (Male, Female, Unknown) and n=3.
    """
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)

    df = summary.copy()
    # Create placeholder columns for formula insertion
    for col in ["anomalous_rate", "SHR_F", "SHR_M", "PS", "NE", "FS"]:
        df[col] = ""

    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="metrics", index=False)
        ws = writer.sheets["metrics"]

        # Find column letters by name (robust to different group-by keys)
        cols = {name: idx for idx, name in enumerate(df.columns, start=1)}
        def col_letter(name: str) -> str:
            idx = cols[name]
            # convert idx -> Excel column letter(s)
            letters = ""
            while idx:
                idx, rem = divmod(idx - 1, 26)
                letters = chr(65 + rem) + letters
            return letters

        # Required base columns
        cMale = col_letter("Male")
        cFemale = col_letter("Female")
        cUnknown = col_letter("Unknown")
        cAnom = col_letter("ANOMALOUS")
        cNeff = col_letter("N_effective")

        cAnomRate = col_letter("anomalous_rate")
        cSHRF = col_letter("SHR_F")
        cSHRM = col_letter("SHR_M")
        cPS = col_letter("PS")
        cNE = col_letter("NE")
        cFS = col_letter("FS")

        n_rows = len(df)
        start_row = 2
        end_row = start_row + n_rows - 1

        for i in range(n_rows):
            r = start_row + i
            # anomalous_rate = ANOMALOUS / (Female + Male + Unknown)
            ws.write_formula(f"{cAnomRate}{r}", f'=IFERROR({cAnom}{r}/({cFemale}{r}+{cMale}{r}+{cUnknown}{r}),0)')

            # PS = MAX(F/T, M/T, U/T)
            ws.write_formula(
                f"{cPS}{r}",
                f'=IF({cNeff}{r}=0,0,MAX({cFemale}{r}/{cNeff}{r},{cMale}{r}/{cNeff}{r},{cUnknown}{r}/{cNeff}{r}))'
            )

            # NE = -sum(p*ln p)/ln 3
            ws.write_formula(
                f"{cNE}{r}",
                f'=IF({cNeff}{r}=0,0, -('
                f'IF({cFemale}{r}=0,0,({cFemale}{r}/{cNeff}{r})*LN({cFemale}{r}/{cNeff}{r})) + '
                f'IF({cMale}{r}=0,0,({cMale}{r}/{cNeff}{r})*LN({cMale}{r}/{cNeff}{r})) + '
                f'IF({cUnknown}{r}=0,0,({cUnknown}{r}/{cNeff}{r})*LN({cUnknown}{r}/{cNeff}{r})) '
                f')/LN(3))'
            )

            # FS = (NE - PS + 1) / (2 - 1/3)
            ws.write_formula(f"{cFS}{r}", f'=({cNE}{r}-{cPS}{r}+1)/(2-1/3)')

            # SHR_F = 3 * Female / T ; SHR_M = 3 * Male / T
            ws.write_formula(f"{cSHRF}{r}", f'=IF({cNeff}{r}=0,0,3*{cFemale}{r}/{cNeff}{r})')
            ws.write_formula(f"{cSHRM}{r}", f'=IF({cNeff}{r}=0,0,3*{cMale}{r}/{cNeff}{r})')

        num_fmt = writer.book.add_format({'num_format': '0.000'})
        ws.set_column(f"{cAnomRate}:{cFS}", 14, num_fmt)


# ----------------------------
# CLI
# ----------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Analyze generations and compute implicit-bias metrics.")
    ap.add_argument("--input-csv", required=True, help="CSV produced by the generation script (needs 'story' column).")
    ap.add_argument("--female-lex", required=True, help="Path to female lexicon (.txt).")
    ap.add_argument("--male-lex", required=True, help="Path to male lexicon (.txt).")
    ap.add_argument("--out-dir", default=None, help="Directory to write outputs. Default: same directory as input CSV.")
    ap.add_argument("--attribute-col", default=None, help="Attribute column name if not 'attribute' (e.g., 'occupation').")
    ap.add_argument("--write-excel", action="store_true", help="Also write an Excel file with formulas (n=3 template).")
    return ap.parse_args()


def main():
    args = parse_args()

    in_path = Path(args.input_csv)
    if not in_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_path}")

    out_dir = Path(args.out_dir) if args.out_dir else in_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Base name for outputs
    base = in_path.stem  # e.g., stories_ability_lm
    analyze_csv = out_dir / f"{base}_analyze.csv"
    metrics_csv = out_dir / f"{base}_metrics.csv"
    metrics_xlsx = out_dir / f"{base}_metrics.xlsx"

    # Load lexicons
    female_lex = load_list(args.female_lex)
    male_lex = load_list(args.male_lex)
    # neutral_lex is not needed for tagging; keep for completeness if you want
    _ = load_list(args.neutral_lex) if args.neutral_lex else []

    # Run analysis
    df_sent = run_analysis(str(in_path), female_lex, male_lex)
    df_sent.to_csv(analyze_csv, index=False)
    print(f"Wrote per-sentence annotations to: {analyze_csv}")

    # Summarize + metrics
    attr_col = _pick_attribute_col(df_sent, args.attribute_col)
    df_summary = make_summary(df_sent, attr_col)
    df_metrics = compute_metrics(df_summary)
    df_metrics.to_csv(metrics_csv, index=False)
    print(f"Wrote summary metrics to: {metrics_csv}")

    if args.write_excel:
        write_metrics_excel(df_summary, metrics_xlsx)
        print(f"Wrote Excel metrics with formulas to: {metrics_xlsx}")


if __name__ == "__main__":
    main()
