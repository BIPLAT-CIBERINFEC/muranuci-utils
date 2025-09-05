#!/usr/bin/env python3
"""
extract_bins_by_quality.py
==========================

Filter the evaluated MAGs TSV by a quality label (QC_label) and produce a
text file containing FASTA paths for the matching bins.

Usage:
  python scripts/extract_bins_by_quality.py \
      --tsv MAGs_QC_evaluated.tsv \
      --qc_label High \
      --output high_bins.txt \
      --fasta_dir /path/to/fastas \
      --ext .fa
"""

import argparse
import os
import pandas as pd


def detect_fasta_path(bin_id: str, fasta_dir: str | None, ext: str) -> str:
    """Return a candidate FASTA path for a given bin.

    If ``fasta_dir`` is None, returns "<bin_id><ext>" in the cwd. If the name
    contains a tool tag (MetaBAT2/MaxBin2/CONCOCT), normalize to
    "<Tool>Refined" for compatibility with common pipelines.
    """
    refined = str(bin_id)
    for tool in ["MetaBAT2", "MaxBin2", "CONCOCT"]:
        if tool in refined and f"{tool}Refined" not in refined:
            refined = refined.replace(tool, f"{tool}Refined")
            break
    fname = f"{refined}{ext}"
    return os.path.join(fasta_dir, fname) if fasta_dir else fname


def extract_bins_by_qc_label(tsv_file: str, qc_label: str, output_file: str, fasta_dir: str | None = None, ext: str = ".fa") -> int:
    df = pd.read_csv(tsv_file, sep="\t")
    df.columns = df.columns.str.strip()

    if "QC_label" not in df.columns or "Bin Id" not in df.columns:
        raise SystemExit("The TSV must contain 'QC_label' and 'Bin Id' columns.")

    mask = df["QC_label"].astype(str).str.lower() == qc_label.lower()
    filtered_df = df.loc[mask]
    if filtered_df.empty:
        print(f"⚠️ No bins found with category '{qc_label}'.")
        open(output_file, "w").close()
        return 0

    valid_bins: list[str] = []
    missing: list[str] = []
    for bin_id in filtered_df["Bin Id"].astype(str):
        fasta_path = detect_fasta_path(bin_id, fasta_dir=fasta_dir, ext=ext)
        if os.path.isfile(fasta_path):
            valid_bins.append(fasta_path)
        else:
            missing.append(fasta_path)

    with open(output_file, "w") as out:
        for name in valid_bins:
            out.write(name + "\n")

    print(f"✅ {len(valid_bins)} valid '{qc_label}' bins written to '{output_file}'")
    if missing:
        print(f"⚠️ {len(missing)} bins not found (first 10):")
        for m in missing[:10]:
            print(f"  - {m}")
    return len(valid_bins)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract a list of bins by quality category.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--tsv", required=True, help="Evaluation TSV (MAGs_QC_evaluated.tsv)")
    p.add_argument("--qc_label", required=True, help="Category: High, Medium, Low, etc.")
    p.add_argument("--output", default="filtered_bins.txt", help="Output file listing FASTA paths")
    p.add_argument("--fasta_dir", default=None, help="Directory where .fa/.fna files live")
    p.add_argument("--ext", default=".fa", help="FASTA extension (e.g., .fa, .fna)")
    return p


def main():
    args = build_argparser().parse_args()
    extract_bins_by_qc_label(
        tsv_file=args.tsv,
        qc_label=args.qc_label,
        output_file=args.output,
        fasta_dir=args.fasta_dir,
        ext=args.ext,
    )


if __name__ == "__main__":
    main()
