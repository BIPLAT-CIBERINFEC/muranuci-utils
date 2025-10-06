#!/usr/bin/env python3
"""
evaluate_mags_qc.py
===================

Evaluate MAG quality from a single combined binning QC summary TSV
(`bin_summary.tsv`) that already includes CheckM and QUAST metrics. The
script assigns quality categories (MIMAG‑like) per bin, writes an output TSV
with a `QC_label` column, and prints a per‑sample summary.

Usage:
  python scripts/evaluate_mags_qc.py \
      --bin-summary bin_summary.tsv \
      --out MAGs_QC_evaluated.tsv

Notes:
- No need for separate inputs; it works with a single `bin_summary.tsv` from
  the binning QC pipeline.
- Classification thresholds are configurable through flags.
"""

import argparse
import re
from typing import Optional

import pandas as pd

def extract_sample_info(bin_id: str) -> dict:
    """Extract assembler, binner, sample, id and DAStool_evaluation from bin_id string."""
    if not isinstance(bin_id, str):
        return {"assembler": None, "binner": None, "Sample": None, "id": None}
     # Split by dot
    parts_dot = bin_id.split(".")
    if len(parts_dot) == 1:
        left = parts_dot[0]
        bin_id_only = None
    else:
        left = parts_dot[0]
        bin_id_only = parts_dot[-1]
    # Split left part by hyphen
    parts = left.split("-")
    if len(parts) < 3:
        return {"assembler": None, "binner": None, "Sample": None, "id": None}
    assembler = parts[0]
    binner = parts[1]
    sample = "-".join(parts[2:])
    # Procesar binner y DAStool
    if isinstance(binner, str) and "Refined" in binner:
        binner_clean = binner.replace("Refined", "").strip()
        dastool = "Refined"
    else:
        binner_clean = binner
        dastool = "No refined"
    return {
        "assembler": assembler,
        "binner": binner,
        "Sample": sample,
        "id": bin_id_only,
        "DAStool_evaluation": dastool 
    }


def load_data(checkm_file: str, quast_file: str):
    checkm_df = pd.read_csv(checkm_file, sep="\t")
    quast_df = pd.read_csv(quast_file, sep="\t")
    checkm_df.columns = checkm_df.columns.str.strip()
    quast_df.columns = quast_df.columns.str.strip()
    return checkm_df, quast_df


def count_rrna_genes(rrna_string: Optional[str]) -> int:
    """Convert strings like '3 + 1 part' to total rRNA gene count.

    Returns 0 if the string is empty or malformed.
    """
    try:
        if pd.isna(rrna_string):
            return 0
        text = str(rrna_string).strip()
        if "+" not in text:
            # número simple
            return int(text.split()[0])
        main, part = text.split("+")
        return int(main.strip().split()[0]) + int(part.strip().split()[0])
    except Exception:
        return 0


# Ya no se requiere mapeo ni fusión: el TSV de entrada contiene todas las métricas


def assign_mimag_qc(
    row: pd.Series,
    hi_comp: float = 90.0,
    hi_cont: float = 5.0,
    hi_n50: int = 20000,
    med_comp: float = 50.0,
    med_cont: float = 10.0,
) -> str:
    """Assign a QC label using simple MIMAG‑like rules.

    High: completeness >= hi_comp and contamination <= hi_cont and N50 >= hi_n50
    Medium: completeness >= med_comp and contamination <= med_cont
    Low: completeness < med_comp and contamination <= med_cont
    Unclassified: all other cases
    """
    try:
        completeness = float(row.get("Completeness", float("nan")))
        contamination = float(row.get("Contamination", float("nan")))
    except Exception:
        return "Unclassified"
    n50 = float(row.get("N50", 0) or 0)

    if pd.notna(completeness) and pd.notna(contamination):
        if completeness >= hi_comp and contamination <= hi_cont and n50 >= hi_n50:
            return "High"
        if completeness >= med_comp and contamination <= med_cont:
            return "Medium"
        if completeness < med_comp and contamination <= med_cont:
            return "Low"
    return "Unclassified"


def print_quality_summary_by_sample(df: pd.DataFrame) -> None:
    print("\n📊 MAG quality summary by sample:\n")
    if not {"Sample", "QC_label"}.issubset(df.columns):
        print("Columnas requeridas no presentes para el resumen.")
        return
    summary = df.groupby(["Sample", "QC_label"]).size().unstack(fill_value=0)
    # Orden sugerido si existen
    for col in ["High", "Medium", "Low"]:
        if col not in summary.columns:
            summary[col] = 0
    summary = summary[["High", "Medium", "Low"]]
    print(summary.to_string())


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate MAGs from a single bin_summary.tsv and assign quality labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--bin-summary", required=True, help="Combined binning QC TSV (bin_summary.tsv)")
    p.add_argument("--out", default="MAGs_QC_evaluated.tsv", help="Output TSV path")
    p.add_argument("--high-completeness", type=float, default=90.0, help="Threshold for High completeness")
    p.add_argument("--high-contamination", type=float, default=5.0, help="Threshold for High contamination")
    p.add_argument("--high-n50", type=int, default=20000, help="Threshold for High N50")
    p.add_argument("--medium-completeness", type=float, default=50.0, help="Threshold for Medium completeness")
    p.add_argument("--medium-contamination", type=float, default=10.0, help="Threshold for Medium contamination")
    return p


def main():
    args = build_argparser().parse_args()

    # Load combined TSV
    df = pd.read_csv(args.bin_summary, sep="\t")
    df.columns = df.columns.str.strip()

    # Extract sample name from 'Bin Id'
    if "Bin Id" not in df.columns:
        if 'bin' in df.columns:
            print("Creando columna 'Bin Id' a partir de 'bin'.")
            df["Bin Id"] = df["bin"].astype(str).str.replace(r"\.fa$", "", regex=True) 
        else:   
            raise SystemExit("El TSV debe contener la columna 'Bin Id' o 'bin'.")
    df = df.copy()
    sample_info = df["Bin Id"].apply(extract_sample_info).apply(pd.Series)
    df = pd.concat([df, sample_info], axis=1)

    # rRNA total
    rrna_col = "# predicted rRNA genes"
    df["rRNA_total"] = df[rrna_col].apply(count_rrna_genes) if rrna_col in df.columns else 0

    # Ensure N50 exists
    if "N50" not in df.columns:
        df["N50"] = 0

    # Assign quality labels
    df["QC_label"] = df.apply(
        assign_mimag_qc,
        axis=1,
        hi_comp=args.high_completeness,
        hi_cont=args.high_contamination,
        hi_n50=args.high_n50,
        med_comp=args.medium_completeness,
        med_cont=args.medium_contamination,
    )

    # Save results
    df.to_csv(args.out, sep="\t", index=False)
    print(f"✅ Resultado guardado en '{args.out}'")

    # Per-sample summary
    print_quality_summary_by_sample(df)


if __name__ == "__main__":
    main()
