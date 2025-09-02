import os
import math
import itertools
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Try optional venn; if missing, we'll fallback automatically later
try:
    from matplotlib_venn import venn2, venn3
    HAS_VENN = True
except Exception:
    HAS_VENN = False

# -------------------------
# Parse arguments
# -------------------------
parser = argparse.ArgumentParser(
    description="Exploratory analysis of HAMRONIZATION results (nf-core funscan)."
)
parser.add_argument(
    "-i", "--input",
    required=True,
    help="HAMRONIZATION TSV file (wide table)"
)
parser.add_argument(
    "-o", "--outdir",
    default="hamronization_out",
    help="Output directory for summaries and plots (default: ./hamronization_out)"
)
args = parser.parse_args()

# Setup intput/output
ham_tsv = args.input
out_dir = args.outdir
os.makedirs(out_dir, exist_ok=True)

# -------------------------
# Load data
# -------------------------
if not ham_tsv:
    raise SystemExit("Set HAMRONIZATION_TSV env var or edit the notebook to point to your TSV file.")

df = pd.read_csv(ham_tsv, sep="\t", dtype=str)  # read as string to avoid dtype surprises
# coerce numeric where useful
for col in ["sequence_identity", "coverage_percentage", "coverage_depth", "coverage_ratio",
            "input_gene_length", "input_protein_length", "reference_gene_length", "reference_protein_length"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Basic sanitization of critical columns
required_cols = ["input_file_name", "analysis_software_name", "gene_symbol"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise SystemExit(f"Missing required columns: {missing}")

if "reference_accession" not in df.columns:
    # fallback to gene_name if present, else keep NaN
    df["reference_accession"] = df.get("reference_accession", df.get("gene_name", np.nan))

# Define a conservative "hit key" to identify equivalent detections across tools
def make_hit_key(r):
    g = str(r.get("gene_symbol", ""))
    acc = str(r.get("reference_accession", ""))
    return f"{g}||{acc}"

df["_hit_key"] = df.apply(make_hit_key, axis=1)

# Normalize tool, sample fields
df["tool"] = df["analysis_software_name"].astype(str).str.strip()
df["sample"] = df["input_file_name"].astype(str).str.strip()

# -------------------------
# Descriptive summaries
# -------------------------

# 1) Global counts
n_rows = len(df)
n_samples = df["sample"].nunique()
n_tools = df["tool"].nunique()
n_unique_hits = df["_hit_key"].nunique()

global_summary = pd.DataFrame({
    "metric": ["rows", "samples", "tools", "unique_hits"],
    "value": [n_rows, n_samples, n_tools, n_unique_hits]
})
print("\n=== Global summary ===")
print(global_summary.to_string(index=False))
global_summary.to_csv(os.path.join(out_dir, "global_summary.csv"), index=False)

# 2) By sample: counts and simple stats
by_sample = df.groupby("sample").agg(
    n_rows=("sample", "size"),
    n_tools=("tool", "nunique"),
    n_unique_hits=("_hit_key", "nunique"),
    mean_identity=("sequence_identity", "mean"),
    mean_cov_perc=("coverage_percentage", "mean"),
    mean_cov_depth=("coverage_depth", "mean")
).reset_index().sort_values("n_unique_hits", ascending=False)

print("\n=== By sample summary (top 20) ===")
print(by_sample.head(20).to_string(index=False))
by_sample.to_csv(os.path.join(out_dir, "by_sample_summary.csv"), index=False)

# 3) By tool: counts and coverage identity stats
by_tool = df.groupby("tool").agg(
    n_rows=("tool", "size"),
    n_samples=("sample", "nunique"),
    n_unique_hits=("_hit_key", "nunique"),
    mean_identity=("sequence_identity", "mean"),
    mean_cov_perc=("coverage_percentage", "mean"),
    mean_cov_depth=("coverage_depth", "mean")
).reset_index().sort_values("n_unique_hits", ascending=False)

print("\n=== By tool summary ===")
print(by_tool.to_string(index=False))
by_tool.to_csv(os.path.join(out_dir, "by_tool_summary.csv"), index=False)

# 4) By drug_class (if available)
if "drug_class" in df.columns:
    by_drug = df.groupby("drug_class").agg(
        n_rows=("drug_class", "size"),
        n_samples=("sample", "nunique"),
        n_unique_hits=("_hit_key", "nunique")
    ).reset_index().sort_values("n_unique_hits", ascending=False)
    print("\n=== By drug_class summary (top 30) ===")
    print(by_drug.head(30).to_string(index=False))
    by_drug.to_csv(os.path.join(out_dir, "by_drug_class_summary.csv"), index=False)

# 5) Top genes (gene_symbol)
top_genes = df.groupby("gene_symbol").agg(
    n_rows=("gene_symbol", "size"),
    n_samples=("sample", "nunique"),
    n_tools=("tool", "nunique"),
    n_unique_hits=("_hit_key", "nunique")
).reset_index().sort_values(["n_samples", "n_unique_hits", "n_rows"], ascending=False)

print("\n=== Top genes (top 30) ===")
print(top_genes.head(30).to_string(index=False))
top_genes.to_csv(os.path.join(out_dir, "top_genes_summary.csv"), index=False)

# 6) Redundancy: same hit_key seen by how many tools per sample
per_sample_hit_tools = (
    df.groupby(["sample", "_hit_key"])["tool"].nunique().reset_index(name="n_tools_support")
)
redundancy = per_sample_hit_tools["n_tools_support"].value_counts().sort_index().reset_index()
redundancy.columns = ["n_tools_support", "count_of_hits"]
redundancy.to_csv(os.path.join(out_dir, "redundancy_tools_per_hit.csv"), index=False)
print("\n=== Redundancy of detections per sample (how many tools support the same hit) ===")
print(redundancy.to_string(index=False))

# Save a long-format table of unique hits per sample-tool for downstream plots
unique_hits_long = df.drop_duplicates(subset=["sample", "tool", "_hit_key"])[["sample", "tool", "_hit_key"]]
unique_hits_long.to_csv(os.path.join(out_dir, "unique_hits_long.csv"), index=False)


# -------------------------
# Venn (or UpSet-like) — Tools overlap per sample
# -------------------------

def compute_tool_sets_for_sample(sample_df):
    sets = {}
    for tool, sub in sample_df.groupby("tool"):
        sets[tool] = set(sub["_hit_key"].unique())
    return sets

def plot_venn_or_upset(sets_dict, title, out_png):
    tools = list(sets_dict.keys())
    n = len(tools)
    if n == 2 and HAS_VENN:
        a, b = tools[0], tools[1]
        plt.figure()
        venn2([sets_dict[a], sets_dict[b]], set_labels=(a, b))
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()
        return "venn2"
    elif n == 3 and HAS_VENN:
        a, b, c = tools[0], tools[1], tools[2]
        plt.figure()
        venn3([sets_dict[a], sets_dict[b], sets_dict[c]], set_labels=(a, b, c))
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()
        return "venn3"
    else:
        # UpSet-like fallback: intersection sizes for all non-empty combinations
        combos = []
        for r in range(1, n+1):
            for comb in itertools.combinations(tools, r):
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

# Choose up to K samples with most tools to plot
K = 5
samples_order = by_sample.sort_values("n_tools", ascending=False)["sample"].tolist()
samples_to_plot = samples_order[:K]

generated_plots = []
for s in samples_to_plot:
    s_df = df[df["sample"] == s]
    tool_sets = compute_tool_sets_for_sample(s_df)
    if len(tool_sets) < 2:
        continue
    fig_type = plot_venn_or_upset(tool_sets, title=f"Tools overlap — {s}", 
                                  out_png=os.path.join(out_dir, f"overlap_{s}.png"))
    generated_plots.append((s, fig_type))

pd.DataFrame(generated_plots, columns=["sample", "figure_type"]).to_csv(
    os.path.join(out_dir, "overlap_plots_generated.csv"), index=False
)

print("\n=== Overlap figures generated ===")
if generated_plots:
    for s, t in generated_plots:
        print(f"- {s}: {t}")
else:
    print("No overlap plots generated (e.g., only one tool present or no eligible samples).")

# Display a couple of key tables here (interactive)
try:
    from caas_jupyter_tools import display_dataframe_to_user
    display_dataframe_to_user("Global summary", global_summary)
    display_dataframe_to_user("By sample summary", by_sample.head(50))
    display_dataframe_to_user("By tool summary", by_tool)
    if "drug_class" in df.columns:
        display_dataframe_to_user("By drug_class summary", by_drug.head(100))
    display_dataframe_to_user("Top genes", top_genes.head(100))
except Exception as e:
    print(f"(Note) Could not display interactive tables: {e}")
