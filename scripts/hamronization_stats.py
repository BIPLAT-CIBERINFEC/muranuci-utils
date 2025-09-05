import os
import re
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    from matplotlib_venn import venn2, venn3
    HAS_VENN = True
except Exception:
    HAS_VENN = False

# -------------------------
# Configuration
# -------------------------
import argparse

parser = argparse.ArgumentParser(description="Exploratory analysis of HAMRONIZATION results (nf-core funscan).")
parser.add_argument("-i", "--input", required=False, help="HAMRONIZATION TSV file (wide table)")
parser.add_argument("-o", "--outdir", default="hamronization_out", help="Output directory (default: ./hamronization_out)")

# handle notebook args safely
try:
    args, _unknown = parser.parse_known_args()
    ham_tsv = args.input or os.environ.get("HAMRONIZATION_TSV", "").strip()
    out_dir = args.outdir
except SystemExit:
    ham_tsv = os.environ.get("HAMRONIZATION_TSV", "").strip()
    out_dir = os.environ.get("HAMRONIZATION_OUTDIR", "hamronization_out").strip()

if not ham_tsv:
    raise SystemExit("Set --input or HAMRONIZATION_TSV to point to your TSV file.")
os.makedirs(out_dir, exist_ok=True)

# -------------------------
# Load
# -------------------------
df = pd.read_csv(ham_tsv, sep="\t", dtype=str)

for col in ["sequence_identity", "coverage_percentage", "coverage_depth", "coverage_ratio",
            "input_gene_length", "input_protein_length", "reference_gene_length", "reference_protein_length"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

required_cols = ["input_file_name", "analysis_software_name", "gene_symbol"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise SystemExit(f"Missing required columns: {missing}")

if "reference_accession" not in df.columns:
    df["reference_accession"] = df.get("gene_name", np.nan)

# -------------------------
# Parse sample/bin using FIRST DOT rule
# -------------------------
BIN_REGEX = re.compile(r"^\d+(?:_[^.\s]+)?$")

def parse_sample_bin_firstdot(filename: str):
    if not isinstance(filename, str) or "." not in filename:
        return filename, np.nan
    parts = filename.split(".")
    sample = parts[0]
    candidate = parts[1] if len(parts) > 1 else None
    if candidate and BIN_REGEX.match(candidate):
        bin_id = candidate
    else:
        bin_id = np.nan
    return sample, bin_id

parsed = df["input_file_name"].apply(parse_sample_bin_firstdot)
df["sample"] = parsed.apply(lambda x: x[0])
df["bin_id"] = parsed.apply(lambda x: x[1])

df["tool"] = df["analysis_software_name"].astype(str).str.strip()

def make_hit_key(r):
    return f"{str(r.get('gene_symbol',''))}||{str(r.get('reference_accession',''))}"

df["_hit_key"] = df.apply(make_hit_key, axis=1)

# -------------------------
# Summaries
# -------------------------
n_rows = len(df)
n_samples = df["sample"].nunique()
n_sample_bins = df[["sample","bin_id"]].drop_duplicates().shape[0]
n_tools = df["tool"].nunique()
n_unique_hits = df["_hit_key"].nunique()

global_summary = pd.DataFrame({
    "metric": ["rows", "samples", "sample_bins", "tools", "unique_hits"],
    "value": [n_rows, n_samples, n_sample_bins, n_tools, n_unique_hits]
})
print("\n=== Global summary ===")
print(global_summary.to_string(index=False))
global_summary.to_csv(os.path.join(out_dir, "global_summary.csv"), index=False)

# By sample
by_sample = df.groupby("sample").agg(
    n_rows=("sample", "size"),
    n_bins=("bin_id", lambda x: pd.Series(x).nunique()),
    n_tools=("tool", "nunique"),
    n_unique_hits=("_hit_key", "nunique"),
    mean_identity=("sequence_identity", "mean"),
    mean_cov_perc=("coverage_percentage", "mean"),
    mean_cov_depth=("coverage_depth", "mean")
).reset_index().sort_values("n_unique_hits", ascending=False)

print("\n=== By sample summary (top 20) ===")
print(by_sample.head(20).to_string(index=False))
by_sample.to_csv(os.path.join(out_dir, "by_sample_summary.csv"), index=False)

# By sample+bin
by_sample_bin = df.groupby(["sample", "bin_id"]).agg(
    n_rows=("bin_id", "size"),
    n_tools=("tool", "nunique"),
    n_unique_hits=("_hit_key", "nunique"),
    mean_identity=("sequence_identity", "mean"),
    mean_cov_perc=("coverage_percentage", "mean"),
    mean_cov_depth=("coverage_depth", "mean")
).reset_index().sort_values(["sample", "bin_id", "n_unique_hits"], ascending=[True, True, False])

print("\n=== By sample+bin summary (top 20) ===")
print(by_sample_bin.head(20).to_string(index=False))
by_sample_bin.to_csv(os.path.join(out_dir, "by_sample_bin_summary.csv"), index=False)

# By tool — fix n_sample_bins via drop_duplicates
def _count_unique_sample_bins(g):
    return g[["sample","bin_id"]].drop_duplicates().shape[0]

by_tool = df.groupby("tool").apply(lambda g: pd.Series({
    "n_rows": g.shape[0],
    "n_samples": g["sample"].nunique(),
    "n_sample_bins": _count_unique_sample_bins(g),
    "n_unique_hits": g["_hit_key"].nunique(),
    "mean_identity": g["sequence_identity"].mean(),
    "mean_cov_perc": g["coverage_percentage"].mean(),
    "mean_cov_depth": g["coverage_depth"].mean(),
})).reset_index()

print("\n=== By tool summary ===")
print(by_tool.to_string(index=False))
by_tool.to_csv(os.path.join(out_dir, "by_tool_summary.csv"), index=False)

# By drug_class (if present) — include n_sample_bins safely
if "drug_class" in df.columns:
    def _agg_drug(g):
        return pd.Series({
            "n_rows": g.shape[0],
            "n_samples": g["sample"].nunique(),
            "n_sample_bins": _count_unique_sample_bins(g),
            "n_unique_hits": g["_hit_key"].nunique(),
        })
    by_drug = df.groupby("drug_class").apply(_agg_drug).reset_index()
    by_drug = by_drug.sort_values("n_unique_hits", ascending=False)
    print("\n=== By drug_class summary (top 30) ===")
    print(by_drug.head(30).to_string(index=False))
    by_drug.to_csv(os.path.join(out_dir, "by_drug_class_summary.csv"), index=False)

# Top genes
top_genes = df.groupby("gene_symbol").apply(lambda g: pd.Series({
    "n_rows": g.shape[0],
    "n_samples": g["sample"].nunique(),
    "n_sample_bins": _count_unique_sample_bins(g),
    "n_tools": g["tool"].nunique(),
    "n_unique_hits": g["_hit_key"].nunique(),
})).reset_index().sort_values(
    ["n_samples","n_unique_hits","n_rows"], ascending=False
)
print("\n=== Top genes (top 30) ===")
print(top_genes.head(30).to_string(index=False))
top_genes.to_csv(os.path.join(out_dir, "top_genes_summary.csv"), index=False)

# Redundancy per sample
per_sample_hit_tools = (
    df.groupby(["sample", "_hit_key"])["tool"].nunique().reset_index(name="n_tools_support")
)
redundancy = per_sample_hit_tools["n_tools_support"].value_counts().sort_index().reset_index()
redundancy.columns = ["n_tools_support", "count_of_hits"]
redundancy.to_csv(os.path.join(out_dir, "redundancy_tools_per_hit.csv"), index=False)
print("\n=== Redundancy of detections per sample (how many tools support the same hit) ===")
print(redundancy.to_string(index=False))

unique_hits_long = df.drop_duplicates(subset=["sample", "tool", "_hit_key"])[["sample", "bin_id", "tool", "_hit_key"]]
unique_hits_long.to_csv(os.path.join(out_dir, "unique_hits_long.csv"), index=False)

# -------------------------
# Venn/UpSet by sample (bins aggregated)
# -------------------------
def compute_tool_sets_for_sample(sample_df):
    sets = {}
    for tool, sub in sample_df.groupby("tool"):
        sets[tool] = set(sub["_hit_key"].unique())
    return sets

def plot_venn_or_upset(sets_dict, title, out_png):
    tools = list(sets_dict.keys())
    if len(tools) == 2 and HAS_VENN:
        a, b = tools[0], tools[1]
        plt.figure()
        venn2([sets_dict[a], sets_dict[b]], set_labels=(a, b))
    elif len(tools) == 3 and HAS_VENN:
        a, b, c = tools[0], tools[1], tools[2]
        plt.figure()
        venn3([sets_dict[a], sets_dict[b], sets_dict[c]], set_labels=(a, b, c))
    else:
        combos = []
        tools_sorted = sorted(tools)
        for r in range(1, len(tools_sorted)+1):
            for comb in itertools.combinations(tools_sorted, r):
                inter = set.intersection(*(sets_dict[t] for t in comb))
                combos.append((" & ".join(comb), len(inter)))
        upset_df = pd.DataFrame(combos, columns=["combination", "size"]).sort_values("size", ascending=False)
        plt.figure(figsize=(8, 5))
        plt.bar(range(len(upset_df)), upset_df["size"].values)
        plt.xticks(range(len(upset_df)), upset_df["combination"].values, rotation=90)
        plt.title(title + " (UpSet-like)")
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()
        return "upset"
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    return "venn"

# plot top-K samples by number of tools
K = 5
samples_to_plot = (
    df.groupby("sample")["tool"].nunique()
      .sort_values(ascending=False)
      .head(K).index.tolist()
)

generated_plots = []
for s in samples_to_plot:
    s_df = df[df["sample"] == s]
    tool_sets = compute_tool_sets_for_sample(s_df)
    if len(tool_sets) < 2:
        continue
    fig_type = plot_venn_or_upset(tool_sets, title=f"Tools overlap — {s} (bins aggregated)",
                                  out_png=os.path.join(out_dir, f"overlap_{s}.png"))
    generated_plots.append((s, fig_type))

pd.DataFrame(generated_plots, columns=["sample", "figure_type"]).to_csv(
    os.path.join(out_dir, "overlap_plots_generated.csv"), index=False
)

# Display a few tables interactively if possible
try:
    from caas_jupyter_tools import display_dataframe_to_user
    display_dataframe_to_user("Global summary", global_summary)
    display_dataframe_to_user("By sample summary", by_sample.head(200))
    display_dataframe_to_user("By sample+bin summary", by_sample_bin.head(200))
    display_dataframe_to_user("By tool summary", by_tool)
    if "drug_class" in df.columns:
        display_dataframe_to_user("By drug_class summary", by_drug.head(100))
    display_dataframe_to_user("Top genes", top_genes.head(100))
except Exception as e:
    print(f"(Note) Could not display interactive tables: {e}")