#!/usr/bin/env python3
"""
hamronization_visuals.py
========================

This script generates a suite of exploratory visualisations from summary
statistics produced by the HAMRONIZATION analysis pipeline.  The input
directory should contain the CSV summaries produced by the existing
`hamronization_exploratory.py` script (e.g. ``global_summary.csv``,
``by_sample_summary.csv``, ``by_tool_summary.csv``, etc.).  The
generated figures are saved into a designated output directory.

Usage example::

    python hamronization_visuals.py \
        --input_dir /path/to/hamronization_stats/stats \
        --outdir /path/to/output

The script will read the available CSVs and produce several
stand‑alone charts including:

* Unique hits per sample (sorted bar chart)
* Number of bins per sample (sorted bar chart)
* Distribution of number of tools per sample (histogram)
* Total hits per tool (bar chart)
* Unique hits per tool (bar chart)
* Number of samples per tool (bar chart)
* Top drug classes by unique hits (horizontal bar chart)
* Top gene symbols by unique hits (horizontal bar chart)
* Redundancy of detections (bar chart of number of tools supporting each hit)
* Heatmap of unique hit counts per sample per tool (optional top 20 samples)

Figures are created using matplotlib only (no seaborn), with one plot
per figure and no explicit colour specifications, as per platform
requirements.
"""

import argparse
import os
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt


def load_summary_tables(base_dir: str):
    """Load all expected summary tables from ``base_dir``.

    Parameters
    ----------
    base_dir : str
        Directory containing the CSV summary files.

    Returns
    -------
    dict
        A dictionary keyed by table name containing the loaded DataFrames.
    """
    tables = {}
    files = {
        'global_summary': 'global_summary.csv',
        'by_sample_summary': 'by_sample_summary.csv',
        'by_tool_summary': 'by_tool_summary.csv',
        'by_drug_class_summary': 'by_drug_class_summary.csv',
        'top_genes_summary': 'top_genes_summary.csv',
        'redundancy_tools_per_hit': 'redundancy_tools_per_hit.csv',
        'unique_hits_long': 'unique_hits_long.csv',
        'by_sample_bin_summary': 'by_sample_bin_summary.csv',
    }
    for key, fname in files.items():
        path = os.path.join(base_dir, fname)
        if os.path.isfile(path):
            try:
                tables[key] = pd.read_csv(path)
            except Exception as e:
                raise RuntimeError(f"Failed to load {path}: {e}")
    return tables


def make_unique_hits_per_sample(df: pd.DataFrame, out_path: str):
    """Create a horizontal bar chart of unique hits per sample.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame as loaded from ``by_sample_summary.csv``.
    out_path : str
        Path to save the resulting PNG.
    """
    if 'sample' not in df.columns or 'n_unique_hits' not in df.columns:
        return
    # Sort samples by descending n_unique_hits
    plot_df = df.sort_values('n_unique_hits', ascending=False)
    plt.figure(figsize=(10, max(4, 0.25 * len(plot_df))))
    plt.barh(plot_df['sample'], plot_df['n_unique_hits'])
    plt.gca().invert_yaxis()
    plt.xlabel('Número de hits únicos')
    plt.title('Número de hits únicos por muestra')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def make_bins_per_sample(df: pd.DataFrame, out_path: str):
    """Create a horizontal bar chart of number of bins per sample.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame as loaded from ``by_sample_summary.csv``.
    out_path : str
        Path to save the resulting PNG.
    """
    if 'sample' not in df.columns or 'n_bins' not in df.columns:
        return
    plot_df = df.sort_values('n_bins', ascending=False)
    plt.figure(figsize=(10, max(4, 0.25 * len(plot_df))))
    plt.barh(plot_df['sample'], plot_df['n_bins'])
    plt.gca().invert_yaxis()
    plt.xlabel('Número de bins')
    plt.title('Número de bins por muestra')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def make_tools_per_sample_distribution(df: pd.DataFrame, out_path: str):
    """Create a histogram of number of tools contributing per sample.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame as loaded from ``by_sample_summary.csv``.
    out_path : str
        Path to save the resulting PNG.
    """
    if 'n_tools' not in df.columns:
        return
    # Drop NaNs and convert to int
    tools_counts = df['n_tools'].dropna().astype(int)
    max_tools = tools_counts.max() if not tools_counts.empty else 0
    bins = range(1, max_tools + 2)
    plt.figure(figsize=(6, 4))
    plt.hist(tools_counts, bins=bins, align='left')
    plt.xlabel('Número de herramientas')
    plt.ylabel('Número de muestras')
    plt.title('Distribución de herramientas por muestra')
    plt.xticks(bins)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def make_bar_chart(df: pd.DataFrame, category_col: str, value_col: str, title: str, xlabel: str, ylabel: str, out_path: str, horizontal: bool = False):
    """General helper to produce a bar chart from a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Table to plot.
    category_col : str
        Column to use for categories (x-axis or y-axis if horizontal).
    value_col : str
        Column to use for values (height or width).
    title : str
        Title of the chart.
    xlabel : str
        Label of the x-axis (or y-axis if horizontal).
    ylabel : str
        Label of the y-axis (or x-axis if horizontal).
    out_path : str
        Path to save the figure.
    horizontal : bool, optional
        If True, draw a horizontal bar chart; otherwise a vertical one.
    """
    plt.figure(figsize=(10, 6))
    # Sort for readability
    plot_df = df.sort_values(value_col, ascending=False)
    if horizontal:
        plt.barh(plot_df[category_col], plot_df[value_col])
        plt.gca().invert_yaxis()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    else:
        plt.bar(plot_df[category_col], plot_df[value_col])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(rotation=45, ha='right')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def make_heatmap_sample_tool(unique_hits: pd.DataFrame, out_path: str, top_n_samples: Optional[int] = 20):
    """Create a heatmap of counts of unique hits per sample per tool.

    Parameters
    ----------
    unique_hits : pandas.DataFrame
        DataFrame loaded from ``unique_hits_long.csv`` with columns ``sample``,
        ``tool``, etc.
    out_path : str
        Path to save the heatmap PNG.
    top_n_samples : int, optional
        Number of samples with most hits to include in the heatmap.  If
        None, include all samples.
    """
    if unique_hits is None or unique_hits.empty:
        return
    if not {'sample', 'tool'}.issubset(unique_hits.columns):
        return
    # Compute counts of unique hits per sample per tool
    # Each row in unique_hits is already a unique hit per sample-tool pair
    counts = unique_hits.groupby(['sample', 'tool']).size().unstack(fill_value=0)
    # Optionally filter to the top N samples by total hits
    if top_n_samples is not None and len(counts) > top_n_samples:
        total_counts = counts.sum(axis=1)
        counts = counts.loc[total_counts.sort_values(ascending=False).head(top_n_samples).index]
    # Sort tools by total hits across included samples
    counts = counts.loc[:, counts.sum(axis=0).sort_values(ascending=False).index]
    # Plot heatmap using imshow
    n_rows, n_cols = counts.shape
    # Define figure size proportionally to number of samples and tools
    fig_w = max(6, 0.6 * n_cols)
    fig_h = max(4, 0.3 * n_rows)
    plt.figure(figsize=(fig_w, fig_h))
    plt.imshow(counts.values, aspect='auto', interpolation='nearest')
    plt.colorbar(label='Número de hits únicos')
    plt.xticks(range(n_cols), counts.columns, rotation=45, ha='right')
    plt.yticks(range(n_rows), counts.index)
    plt.title('Heatmap de hits únicos por muestra y herramienta')
    plt.xlabel('Herramienta')
    plt.ylabel('Muestra')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate visualisations from HAMRONIZATION summary tables.')
    parser.add_argument('-i', '--input_dir', required=True, help='Directory containing summary CSVs.')
    parser.add_argument('-o', '--outdir', required=True, help='Directory to save generated figures.')
    parser.add_argument('--top_samples', type=int, default=20, help='Number of top samples to include in the heatmap.')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    tables = load_summary_tables(args.input_dir)

    # Generate charts from by_sample_summary.csv
    if 'by_sample_summary' in tables:
        by_sample = tables['by_sample_summary']
        # Unique hits per sample bar chart
        make_unique_hits_per_sample(by_sample, os.path.join(args.outdir, 'unique_hits_per_sample.png'))
        # Bins per sample bar chart
        make_bins_per_sample(by_sample, os.path.join(args.outdir, 'bins_per_sample.png'))
        # Distribution of number of tools per sample
        make_tools_per_sample_distribution(by_sample, os.path.join(args.outdir, 'tools_distribution_per_sample.png'))
    # Generate charts from by_tool_summary.csv
    if 'by_tool_summary' in tables:
        by_tool = tables['by_tool_summary']
        # Total hits per tool
        make_bar_chart(by_tool, category_col='tool', value_col='n_rows', title='Número total de hits por herramienta', xlabel='Herramienta', ylabel='Número de hits', out_path=os.path.join(args.outdir, 'total_hits_per_tool.png'))
        # Unique hits per tool
        if 'n_unique_hits' in by_tool.columns:
            make_bar_chart(by_tool, category_col='tool', value_col='n_unique_hits', title='Número de hits únicos por herramienta', xlabel='Herramienta', ylabel='Número de hits únicos', out_path=os.path.join(args.outdir, 'unique_hits_per_tool.png'))
        # Number of samples per tool
        if 'n_samples' in by_tool.columns:
            make_bar_chart(by_tool, category_col='tool', value_col='n_samples', title='Número de muestras con resultados por herramienta', xlabel='Herramienta', ylabel='Número de muestras', out_path=os.path.join(args.outdir, 'samples_per_tool.png'))
    # Generate charts from by_drug_class_summary.csv
    if 'by_drug_class_summary' in tables:
        by_drug = tables['by_drug_class_summary']
        # Limit to top 20 classes by n_unique_hits
        top_drug = by_drug.sort_values('n_unique_hits', ascending=False).head(20)
        make_bar_chart(top_drug, category_col='drug_class', value_col='n_unique_hits', title='Top 20 clases de antibióticos por número de hits únicos', xlabel='Número de hits únicos', ylabel='Clase de antibiótico', out_path=os.path.join(args.outdir, 'top20_drug_classes_unique_hits.png'), horizontal=True)
        # Also show number of samples per class (top 20 by n_samples)
        if 'n_samples' in by_drug.columns:
            top_samples_drug = by_drug.sort_values('n_samples', ascending=False).head(20)
            make_bar_chart(top_samples_drug, category_col='drug_class', value_col='n_samples', title='Top 20 clases de antibióticos por número de muestras', xlabel='Número de muestras', ylabel='Clase de antibiótico', out_path=os.path.join(args.outdir, 'top20_drug_classes_samples.png'), horizontal=True)
    # Generate charts from top_genes_summary.csv
    if 'top_genes_summary' in tables:
        top_genes = tables['top_genes_summary']
        # Limit to top 20 genes by n_unique_hits
        topg = top_genes.sort_values('n_unique_hits', ascending=False).head(20)
        make_bar_chart(topg, category_col='gene_symbol', value_col='n_unique_hits', title='Top 20 genes por número de hits únicos', xlabel='Número de hits únicos', ylabel='Gen', out_path=os.path.join(args.outdir, 'top20_genes_unique_hits.png'), horizontal=True)
    # Generate chart for redundancy
    if 'redundancy_tools_per_hit' in tables:
        red = tables['redundancy_tools_per_hit']
        if 'n_tools_support' in red.columns and 'count_of_hits' in red.columns:
            plt.figure(figsize=(6, 4))
            plt.bar(red['n_tools_support'].astype(str), red['count_of_hits'])
            plt.xlabel('Número de herramientas que soportan el hit')
            plt.ylabel('Número de hits')
            plt.title('Redundancia de las detecciones (apoyo por herramientas)')
            plt.tight_layout()
            plt.savefig(os.path.join(args.outdir, 'redundancy_of_hits.png'), dpi=300)
            plt.close()
    # Heatmap of unique hits per sample per tool
    if 'unique_hits_long' in tables:
        unique_hits_long = tables['unique_hits_long']
        make_heatmap_sample_tool(unique_hits_long, os.path.join(args.outdir, 'heatmap_hits_per_sample_tool.png'), top_n_samples=args.top_samples)


if __name__ == '__main__':
    main()