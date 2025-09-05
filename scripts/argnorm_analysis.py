#!/usr/bin/env python3
"""
argnorm_analysis.py

Load ARGNORM results from per-tool directories, normalize annotations, and
generate summaries and figures for descriptive resistome analysis. Outputs
(tables and PNGs) are saved to the folder specified with --outdir.

Requirements: pandas, matplotlib.
"""

import argparse
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
from typing import Optional, Tuple, List

def detect_delim(filename: str):
    """Return the delimiter based on file extension."""
    return '\t' if filename.endswith(".tsv") else ','

def parse_sample_bin(input_file_name: str) -> Tuple[str, Optional[str]]:
    """Extract sample and bin_id from common ARGNORM input_file_name strings.

    Examples:
    - HUSE002-03-M.007.tsv.amrfinderplus -> sample=HUSE002-03-M, bin_id=007
    - MEGAHIT-MaxBin2Refined-Community-2.003 -> sample=MEGAHIT-MaxBin2Refined-Community-2, bin_id=003
    - SAMPLE-001.tsv.amrfinderplus -> sample=SAMPLE-001, bin_id=None
    - SAMPLE-001.mapping.ARG.deeparg -> sample=SAMPLE-001, bin_id=None
    """
    parts = input_file_name.split('.')
    sample = parts[0] if parts else input_file_name
    bin_id = None
    if len(parts) >= 2 and re.match(r'^[0-9]+(?:[_-]\w+)?$', parts[1]):
        bin_id = parts[1]
    return sample, bin_id

def load_argnorm_results(base_dir: str):
    """Walk .tsv/.csv files under base_dir and concatenate into one DataFrame."""
    dfs = []
    for root, _, files in os.walk(base_dir):
        # Tool is the immediate subfolder name under base_dir
        tool = os.path.basename(root)
        for f in files:
            if f.endswith(".tsv") or f.endswith(".csv"):
                filepath = os.path.join(root, f)
                try:
                    df = pd.read_csv(filepath, sep=detect_delim(filepath), low_memory=False)
                except Exception:
                    continue
                df = df.assign(tool=tool)
                dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def clean_data(df: pd.DataFrame):
    """Add parsed columns (sample, bin_id) and hit_key; coerce numerics when possible."""
    if 'input_file_name' in df.columns:
        samples = df['input_file_name'].apply(parse_sample_bin)
        df['sample'] = samples.apply(lambda x: x[0])
        df['bin_id'] = samples.apply(lambda x: x[1])
    # Unique hit key (gene + accession)
    if {'gene_symbol','reference_accession'}.issubset(df.columns):
        df['_hit_key'] = df['gene_symbol'].astype(str) + '||' + df['reference_accession'].astype(str)
    # Try to convert numeric columns
    for col in ['coverage_percentage','coverage_depth','coverage_ratio','sequence_identity']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def explode_column(df: pd.DataFrame, col: str, seps: str = r'[;,/]', keep_extra: bool = True):
    """Explode a multi-valued column into rows splitting on provided separators.

    - seps: a regex character class of separators. Defaults to split on comma, semicolon, or slash.
    - keep_extra: when True, retain helpful columns like 'gene_symbol' and 'ARO' if present.
    - Returns a tidy DataFrame with the requested column lowercased and stripped.
    """
    if col not in df.columns:
        base_cols = [col, 'sample', 'bin_id', 'tool', '_hit_key']
        if keep_extra:
            base_cols += ['gene_symbol', 'ARO']
        return pd.DataFrame(columns=[c for c in base_cols if c])
    base_cols = [c for c in [col, 'sample', 'bin_id', 'tool', '_hit_key'] if c in df.columns]
    if keep_extra:
        for extra in ['gene_symbol', 'ARO']:
            if extra in df.columns:
                base_cols.append(extra)
    exploded = df[base_cols].copy()
    exploded[col] = exploded[col].astype(str)
    exploded = exploded[exploded[col].str.strip().astype(bool)]
    if exploded.empty:
        return exploded
    exploded[col] = exploded[col].str.replace('\n', ' ', regex=False)
    exploded = exploded.assign(**{col: exploded[col].str.split(seps, regex=True)})
    exploded = exploded.explode(col)
    exploded[col] = exploded[col].astype(str).str.strip().str.lower()
    exploded = exploded[exploded[col] != '']
    return exploded.reset_index(drop=True)

def infer_amr_gene_family(gene_name: Optional[str], gene_symbol: Optional[str] = None) -> Optional[str]:
    """Best-effort inference of AMR gene family from gene_name.

    Heuristics:
    - If 'family' is present, capture the substring ending in 'family'.
    - Else, capture patterns like 'class X ... beta-lactamase'.
    - Else, capture common AMR terms; fallback to gene_symbol.
    """
    if not gene_name or not isinstance(gene_name, str):
        return gene_symbol
    txt = gene_name.strip()
    m = re.search(r'\b([A-Za-z0-9/\-]+(?:\s+[A-Za-z0-9/\-]+)*)\s+family\b', txt)
    if m:
        return f"{m.group(1)} family"
    m = re.search(r'\b(class\s+[A-Z]\s+.*?beta\-lactamase)\b', txt, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    patterns = [
        r'\bribosomal protection protein\b',
        r'\befflux pump\b',
        r'\bphosphotransferase\b',
        r'\bnucleotidyltransferase\b',
        r'\bacetyltransferase\b',
        r'\bmethyltransferase\b',
        r'\bmetallo\-beta\-lactamase\b',
    ]
    for p in patterns:
        m = re.search(p, txt, flags=re.IGNORECASE)
        if m:
            return m.group(0)
    return gene_symbol

def compute_summaries(df: pd.DataFrame):
    """Compute global, per-sample, per-tool, by drug class, by ARO, by confers, and by mechanism summaries."""
    summaries = {}
    # global
    summaries['global_summary'] = pd.DataFrame({
        'metric':['rows','samples','sample_bins','tools','unique_hits','unique_aros'],
        'value':[len(df),
                 df['sample'].nunique() if 'sample' in df.columns else 0,
                 df.dropna(subset=['bin_id'])[['sample','bin_id']].drop_duplicates().shape[0],
                 df['tool'].nunique() if 'tool' in df.columns else 0,
                 df['_hit_key'].nunique() if '_hit_key' in df.columns else 0,
                 df['ARO'].nunique() if 'ARO' in df.columns else 0]
    })
    # by sample
    if 'sample' in df.columns:
        by_sample = df.groupby('sample').agg(
            n_rows=('sample','size'),
            n_bins=('bin_id', lambda x: x.nunique(dropna=True)),
            n_tools=('tool', lambda x: x.nunique(dropna=True)),
            n_unique_hits=('_hit_key', lambda x: x.nunique(dropna=True)),
            n_unique_aros=('ARO', lambda x: x.nunique(dropna=True) if 'ARO' in df.columns else None),
            mean_identity=('sequence_identity','mean') if 'sequence_identity' in df.columns else None,
            mean_cov_perc=('coverage_percentage','mean') if 'coverage_percentage' in df.columns else None,
            mean_cov_depth=('coverage_depth','mean') if 'coverage_depth' in df.columns else None
        ).reset_index()
        by_sample['n_bins'] = by_sample['n_bins'].fillna(0).astype(int)
        by_sample['n_tools'] = by_sample['n_tools'].fillna(0).astype(int)
        summaries['by_sample_summary'] = by_sample
    # by tool
    if 'tool' in df.columns:
        by_tool = df.groupby('tool').agg(
            n_rows=('tool','size'),
            n_samples=('sample', lambda x: x.nunique(dropna=True)),
            n_unique_hits=('_hit_key', lambda x: x.nunique(dropna=True)),
            n_unique_aros=('ARO', lambda x: x.nunique(dropna=True)),
            mean_identity=('sequence_identity','mean') if 'sequence_identity' in df.columns else None,
            mean_cov_perc=('coverage_percentage','mean') if 'coverage_percentage' in df.columns else None,
            mean_cov_depth=('coverage_depth','mean') if 'coverage_depth' in df.columns else None
        ).reset_index()
        summaries['by_tool_summary'] = by_tool
    # por clase de antibiótico
    if 'drug_class' in df.columns:
        # algunos registros tienen varias clases separadas por coma; explotamos
        df_dc = explode_column(df, 'drug_class')
        if not df_dc.empty:
            by_dc = (
                df_dc.groupby('drug_class')
                    .apply(lambda g: pd.Series({
                        'n_rows': g.shape[0],
                        'n_samples': g['sample'].nunique(dropna=True),
                        'n_sample_bins': g[['sample','bin_id']].drop_duplicates().shape[0],
                        'n_unique_hits': g['_hit_key'].nunique(dropna=True),
                        'n_unique_aros': g['ARO'].nunique(dropna=True) if 'ARO' in g.columns else g['_hit_key'].nunique(dropna=True)
                    }))
                    .reset_index()
                    .rename(columns={'drug_class': 'term'})
            )
            summaries['by_drug_class_summary'] = by_dc
    # por ARO
    if 'ARO' in df.columns:
        by_aro = df.groupby('ARO').agg(
            n_rows=('ARO','size'),
            n_samples=('sample', lambda x: x.nunique(dropna=True)),
            n_tools=('tool', lambda x: x.nunique(dropna=True)),
            n_unique_hits=('_hit_key', lambda x: x.nunique(dropna=True))
        ).reset_index()
        summaries['by_aro_summary'] = by_aro
    # top genes / top ARO
    if 'gene_symbol' in df.columns:
        top_genes = df.groupby('gene_symbol').agg(
            n_rows=('gene_symbol','size'),
            n_samples=('sample', lambda x: x.nunique(dropna=True)),
            n_tools=('tool', lambda x: x.nunique(dropna=True)),
            n_unique_hits=('_hit_key', lambda x: x.nunique(dropna=True))
        ).reset_index().sort_values('n_unique_hits', ascending=False)
        summaries['top_genes_summary'] = top_genes
    if 'ARO' in df.columns:
        top_aro = df.groupby('ARO').agg(
            n_rows=('ARO','size'),
            n_samples=('sample', lambda x: x.nunique(dropna=True)),
            n_tools=('tool', lambda x: x.nunique(dropna=True)),
            n_unique_hits=('_hit_key', lambda x: x.nunique(dropna=True))
        ).reset_index().sort_values('n_unique_hits', ascending=False)
        summaries['top_aro_summary'] = top_aro
    # por confers (antibióticos concretos)
    if 'confers_resistance_to' in df.columns:
        df_conf = explode_column(df, 'confers_resistance_to')
        if not df_conf.empty:
            by_conf = df_conf.groupby('confers_resistance_to').agg(
                n_rows=('confers_resistance_to','size'),
                n_samples=('sample', lambda x: x.nunique(dropna=True)),
                n_unique_hits=('_hit_key', lambda x: x.nunique(dropna=True)),
                n_unique_aros=('_hit_key', lambda x: x.nunique(dropna=True))
            ).reset_index().rename(columns={'confers_resistance_to':'term'})
            summaries['by_confers_summary'] = by_conf
    # por mecanismo
    if 'resistance_mechanism' in df.columns:
        by_mech = df.groupby('resistance_mechanism').agg(
            n_rows=('resistance_mechanism','size'),
            n_samples=('sample', lambda x: x.nunique(dropna=True)),
            n_unique_hits=('_hit_key', lambda x: x.nunique(dropna=True)),
            n_unique_aros=('_hit_key', lambda x: x.nunique(dropna=True))
        ).reset_index().rename(columns={'resistance_mechanism':'term'})
        summaries['by_mechanism_summary'] = by_mech
    return summaries

def save_summary_tables(summaries, out_dir):
    """Guarda todos los DataFrames en CSV."""
    for name, df in summaries.items():
        try:
            df.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
        except Exception:
            pass

def make_bar(df, cat_col, val_col, xlabel, ylabel, title, out_path, horizontal=False):
    """Dibuja gráfico de barras sencillo."""
    if df.empty:
        return
    plot_df = df.sort_values(val_col, ascending=False).copy()
    plt.figure(figsize=(10, max(4, 0.3 * len(plot_df))))
    if horizontal:
        plt.barh(plot_df[cat_col], plot_df[val_col])
        plt.gca().invert_yaxis()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    else:
        plt.bar(plot_df[cat_col], plot_df[val_col])
        plt.xticks(rotation=45, ha='right')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def make_hist(series, xlabel, ylabel, title, out_path):
    """Dibuja histograma discreto (por valores enteros)."""
    if series.empty:
        return
    series = series.dropna().astype(int)
    bins = range(series.min(), series.max()+2)
    plt.figure(figsize=(6,4))
    plt.hist(series, bins=bins, align='left')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(bins)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def make_heatmap(df, index_col, col_col, value_col, top_rows=20, top_cols=10, title="", xlabel="", ylabel="", out_path=""):
    """Genera heatmap con imshow a partir de un DataFrame ya pivoteado."""
    if df.empty:
        return
    pivot = df.pivot(index=index_col, columns=col_col, values=value_col).fillna(0)
    # filtrar top n filas y columnas
    if top_rows is not None and len(pivot) > top_rows:
        pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).head(top_rows).index]
    if top_cols is not None and len(pivot.columns) > top_cols:
        pivot = pivot.loc[:, pivot.sum(axis=0).sort_values(ascending=False).head(top_cols).index]
    n_rows, n_cols = pivot.shape
    fig_w = max(6, 0.6 * n_cols)
    fig_h = max(4, 0.3 * n_rows)
    plt.figure(figsize=(fig_w, fig_h))
    plt.imshow(pivot.values, aspect='auto', interpolation='nearest')
    plt.colorbar(label=value_col)
    plt.xticks(range(n_cols), pivot.columns, rotation=45, ha='right')
    plt.yticks(range(n_rows), pivot.index)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def stacked_bar_per_sample_by_drug(df: pd.DataFrame, out_path: str, tool_filter: Optional[str] = None, table_out: Optional[str] = None):
    """Plot 1: Stacked barplot per sample with drug classes (colours = classes).

    - Uses unique hits (by _hit_key) per sample × drug_class.
    - If tool_filter is provided, limit to that tool.
    - Saves the underlying pivot table if table_out is provided.
    """
    if df.empty:
        return
    work = df.copy()
    if tool_filter and 'tool' in work.columns:
        work = work[work['tool'].str.lower() == tool_filter.lower()]
    if work.empty or 'drug_class' not in work.columns:
        return
    dc = explode_column(work, 'drug_class')
    if dc.empty:
        return
    # unique hit per sample × drug_class
    dc = dc.dropna(subset=['_hit_key','sample'])
    dc = dc[['sample','_hit_key','drug_class']].drop_duplicates()
    counts = dc.groupby(['sample','drug_class']).size().unstack(fill_value=0)
    # reorder samples by total and classes by total
    counts = counts.loc[counts.sum(axis=1).sort_values(ascending=False).index]
    counts = counts.loc[:, counts.sum(axis=0).sort_values(ascending=False).index]
    # compute percentages per sample
    totals = counts.sum(axis=1).replace(0, 1)
    perc = counts.div(totals, axis=0) * 100.0
    # save tables (counts and percents)
    if table_out:
        try:
            counts.to_csv(table_out)
            base, ext = os.path.splitext(table_out)
            perc.to_csv(f"{base}.percent{ext}")
        except Exception:
            pass
    # plot stacked bars (percent)
    plt.figure(figsize=(max(8, 0.5 * len(perc)), 6))
    bottoms = [0] * len(counts)
    x = range(len(perc))
    for cls in perc.columns:
        vals = perc[cls].values
        plt.bar(x, vals, bottom=bottoms, label=cls)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    plt.xticks(list(x), perc.index, rotation=45, ha='right')
    plt.ylabel('Proportion of hits (%)')
    plt.ylim(0, 100)
    plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))
    title = 'Drug classes per sample'
    if tool_filter:
        title += f' (tool: {tool_filter})'
    plt.title(title)
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1), fontsize='small', ncol=1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def count_unique_genes(df: pd.DataFrame, by: List[str]) -> pd.DataFrame:
    """Helper to count unique gene symbols by provided keys."""
    if 'gene_symbol' not in df.columns:
        return pd.DataFrame(columns=by + ['n_unique_genes'])
    g = df.dropna(subset=['gene_symbol']).groupby(by)['gene_symbol'].nunique().reset_index(name='n_unique_genes')
    return g

def plot2_genes_per_sample_and_tool(df: pd.DataFrame, outdir: str):
    """Plot 2: Number of resistance genes per sample and per tool (two charts).

    - Per sample: unique gene_symbol across all tools.
    - Per tool: unique gene_symbol across all samples.
    - Saves companion CSV tables.
    """
    os.makedirs(outdir, exist_ok=True)
    per_sample = count_unique_genes(df, ['sample'])
    if not per_sample.empty:
        per_sample.to_csv(os.path.join(outdir, 'plot2_genes_per_sample.csv'), index=False)
        make_bar(per_sample, 'sample', 'n_unique_genes', 'Number of genes', 'Sample',
                 'Unique resistance genes per sample', os.path.join(outdir, 'plot2_genes_per_sample.png'), horizontal=True)
    per_tool = count_unique_genes(df, ['tool'])
    if not per_tool.empty:
        per_tool.to_csv(os.path.join(outdir, 'plot2_genes_per_tool.csv'), index=False)
        make_bar(per_tool, 'tool', 'n_unique_genes', 'Tool', 'Number of genes',
                 'Unique resistance genes per tool', os.path.join(outdir, 'plot2_genes_per_tool.png'))

def heatmap_top_features(df: pd.DataFrame, row_key: str, feature_col: str, out_path: str, top_features: int = 20, table_out: Optional[str] = None, cluster: bool = False):
    """Generic heatmap for top N features (ARG genes or ARO) per row_key (bin/sample).

    Counts unique hits (_hit_key) per row × feature, selects top features overall, and plots.
    Saves the pivot to CSV if table_out given.
    """
    if df.empty or feature_col not in df.columns or row_key not in df.columns:
        return
    work = df.dropna(subset=[row_key, feature_col, '_hit_key'])
    if work.empty:
        return
    # Unique hits per row × feature
    tidy = work[[row_key, feature_col, '_hit_key']].drop_duplicates()
    counts = tidy.groupby([row_key, feature_col]).size().reset_index(name='n')
    # Select top features globally
    top_feats = (
        counts.groupby(feature_col)['n']
              .sum()
              .sort_values(ascending=False)
              .head(top_features)
              .index
    )
    counts = counts[counts[feature_col].isin(top_feats)]
    # Pivot
    pivot = counts.pivot(index=row_key, columns=feature_col, values='n').fillna(0)
    # Order rows by total
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
    # Save table
    if table_out:
        try:
            pivot.to_csv(table_out)
        except Exception:
            pass
    # Plot heatmap (optionally with dendrogram clustering)
    n_rows, n_cols = pivot.shape
    if cluster and n_rows > 1 and n_cols > 1:
        try:
            import numpy as np
            from matplotlib import gridspec
            from scipy.cluster.hierarchy import linkage, dendrogram
            # Compute linkages
            row_link = linkage(pivot.values, method='average', metric='euclidean')
            col_link = linkage(pivot.values.T, method='average', metric='euclidean')
            # Figure layout
            fig_w = max(8, 0.6 * n_cols + 2)
            fig_h = max(6, 0.3 * n_rows + 2)
            fig = plt.figure(figsize=(fig_w, fig_h))
            gs = gridspec.GridSpec(nrows=2, ncols=2, width_ratios=[1.5, 6], height_ratios=[2, 6], wspace=0.05, hspace=0.05)
            ax_col = fig.add_subplot(gs[0, 1])
            ax_row = fig.add_subplot(gs[1, 0])
            ax_heat = fig.add_subplot(gs[1, 1])
            # Dendrograms
            col_den = dendrogram(col_link, ax=ax_col, orientation='top', no_labels=True, color_threshold=None)
            row_den = dendrogram(row_link, ax=ax_row, orientation='left', no_labels=True, color_threshold=None)
            ax_col.set_xticks([])
            ax_col.set_yticks([])
            ax_row.set_xticks([])
            ax_row.set_yticks([])
            # Reorder pivot
            row_ord = row_den['leaves']
            col_ord = col_den['leaves']
            pivot_ord = pivot.iloc[row_ord, col_ord]
            im = ax_heat.imshow(pivot_ord.values, aspect='auto', interpolation='nearest')
            ax_heat.set_xticks(range(len(pivot_ord.columns)))
            ax_heat.set_xticklabels(pivot_ord.columns, rotation=45, ha='right')
            ax_heat.set_yticks(range(len(pivot_ord.index)))
            ax_heat.set_yticklabels(pivot_ord.index)
            ax_heat.set_xlabel(feature_col)
            ax_heat.set_ylabel(row_key)
            ax_heat.set_title(f'Heatmap (clustered): {feature_col} by {row_key}')
            # Colorbar
            fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04, label='Unique hits')
            plt.tight_layout()
            plt.savefig(out_path, dpi=300)
            plt.close()
            return
        except Exception:
            # Fallback to simple heatmap if clustering not available
            pass
    # Simple heatmap (no clustering)
    fig_w = max(6, 0.6 * n_cols)
    fig_h = max(4, 0.3 * n_rows)
    plt.figure(figsize=(fig_w, fig_h))
    plt.imshow(pivot.values, aspect='auto', interpolation='nearest')
    plt.colorbar(label='Unique hits')
    plt.xticks(range(n_cols), pivot.columns, rotation=45, ha='right')
    plt.yticks(range(n_rows), pivot.index)
    plt.title(f'Heatmap: {feature_col} by {row_key}')
    plt.xlabel(feature_col)
    plt.ylabel(row_key)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot3_heatmaps_genes_and_aro(df: pd.DataFrame, outdir: str, top_n: int = 20, cluster: bool = False):
    """Plot 3: Heatmaps of resistance genes and ARO terms per bin and per sample.

    - Two entities: bins (without aggregation) and samples (grouped by sample).
    - Two features: gene_symbol and ARO.
    - Uses the top N most abundant feature types.
    """
    os.makedirs(outdir, exist_ok=True)
    # Genes by bin (rows: sample.bin if available, else bin_id)
    df_bin = df.dropna(subset=['bin_id']).copy()
    if not df_bin.empty:
        df_bin['sample_bin'] = df_bin['sample'].astype(str) + '.' + df_bin['bin_id'].astype(str)
        heatmap_top_features(df_bin, 'sample_bin', 'gene_symbol', os.path.join(outdir, 'plot3_heatmap_genes_by_bin.png'), top_features=top_n, table_out=os.path.join(outdir, 'plot3_table_genes_by_bin.csv'), cluster=cluster)
        if 'ARO' in df_bin.columns:
            heatmap_top_features(df_bin, 'sample_bin', 'ARO', os.path.join(outdir, 'plot3_heatmap_aro_by_bin.png'), top_features=top_n, table_out=os.path.join(outdir, 'plot3_table_aro_by_bin.csv'), cluster=cluster)
    # Genes by sample (aggregation)
    heatmap_top_features(df, 'sample', 'gene_symbol', os.path.join(outdir, 'plot3_heatmap_genes_by_sample.png'), top_features=top_n, table_out=os.path.join(outdir, 'plot3_table_genes_by_sample.csv'), cluster=cluster)
    if 'ARO' in df.columns:
        heatmap_top_features(df, 'sample', 'ARO', os.path.join(outdir, 'plot3_heatmap_aro_by_sample.png'), top_features=top_n, table_out=os.path.join(outdir, 'plot3_table_aro_by_sample.csv'), cluster=cluster)

def table1_top20_arg(df: pd.DataFrame, out_path: str, top_n: int = 20):
    """Table 1: Top 20 abundant ARGs with drug class, mechanism, and inferred AMR gene family."""
    if df.empty or 'gene_symbol' not in df.columns:
        return
    base = df.dropna(subset=['gene_symbol']).copy()
    base['_hit_key'] = base['_hit_key'].fillna(base['gene_symbol'])
    ranks = (base.groupby('gene_symbol')['_hit_key']
                 .nunique()
                 .reset_index(name='n_unique_hits')
                 .sort_values('n_unique_hits', ascending=False))
    top = ranks.head(top_n)
    genes = top['gene_symbol'].tolist()
    sub = base[base['gene_symbol'].isin(genes)].copy()
    # drug classes (explode and aggregate)
    dc = explode_column(sub, 'drug_class')
    if dc is None or dc.empty or 'gene_symbol' not in dc.columns:
        # Fallback: no drug_class exploded or gene_symbol missing; build minimal table
        try:
            top.assign(n_samples=top['gene_symbol'].map(samp_map).fillna(0).astype(int)).to_csv(out_path, index=False)
        except Exception:
            pass
        return
    drug_map = (dc.groupby('gene_symbol')['drug_class']
                  .apply(lambda s: ','.join(sorted(set([x for x in s if isinstance(x, str) and x]))))
                  .to_dict())
    # mechanisms
    mech_map = (sub.groupby('gene_symbol')['resistance_mechanism']
                    .apply(lambda s: ','.join(sorted(set([str(x).lower() for x in s.dropna().astype(str) if x]))))
                    .to_dict() if 'resistance_mechanism' in sub.columns else {})
    # amr gene family inferred from gene_name
    fam_series = sub.groupby('gene_symbol')['gene_name']\
                   .apply(lambda s: ','.join(sorted(set([infer_amr_gene_family(x, s.name) for x in s.dropna().astype(str)]))))
    fam_map = fam_series.to_dict()
    # samples support
    samp_map = sub.groupby('gene_symbol')['sample'].nunique().to_dict()
    # Build table
    out = top.copy()
    out['n_samples'] = out['gene_symbol'].map(samp_map).fillna(0).astype(int)
    out['drug_class'] = out['gene_symbol'].map(drug_map).fillna('')
    out['resistance_mechanism'] = out['gene_symbol'].map(mech_map).fillna('')
    out['amr_gene_family'] = out['gene_symbol'].map(fam_map).fillna('')
    try:
        out.to_csv(out_path, index=False)
    except Exception:
        pass

def generate_requested_plots_and_tables(df: pd.DataFrame, out_dir: str, top_n: int = 20, tool_for_drugplot: Optional[str] = None, cluster_heatmaps: bool = False):
    """Create the requested outputs (Plots 1–3 and Table 1)."""
    os.makedirs(out_dir, exist_ok=True)
    stacked_bar_per_sample_by_drug(
        df,
        out_path=os.path.join(out_dir, 'plot1_stacked_drugclass_per_sample.png'),
        tool_filter=tool_for_drugplot,
        table_out=os.path.join(out_dir, 'plot1_table_drugclass_per_sample.csv')
    )
    plot2_genes_per_sample_and_tool(df, outdir=out_dir)
    plot3_heatmaps_genes_and_aro(df, outdir=out_dir, top_n=top_n, cluster=cluster_heatmaps)
    table1_top20_arg(df, out_path=os.path.join(out_dir, 'table1_top20_arg.csv'), top_n=top_n)
def generate_figures(summaries, out_dir, top_terms=20):
    """Generate relevant figures from summaries."""
    # hits and AROs by sample
    if 'by_sample_summary' in summaries:
        df = summaries['by_sample_summary']
        make_bar(df, 'sample', 'n_unique_hits', 'Number of unique hits', 'Sample',
                 'Unique hits per sample (desc)', os.path.join(out_dir, 'unique_hits_per_sample.png'), horizontal=True)
        if 'n_unique_aros' in df.columns:
            make_bar(df, 'sample', 'n_unique_aros', 'Number of unique AROs', 'Sample',
                     'Unique AROs per sample (desc)', os.path.join(out_dir, 'unique_aros_per_sample.png'), horizontal=True)
        make_hist(df['n_tools'], 'Number of tools', 'Number of samples',
                  'Distribution of tools per sample', os.path.join(out_dir, 'tools_distribution_per_sample.png'))
    # by tool
    if 'by_tool_summary' in summaries:
        df = summaries['by_tool_summary']
        make_bar(df, 'tool', 'n_rows', 'Tool', 'Total hits',
                 'Total hits per tool', os.path.join(out_dir, 'total_hits_per_tool.png'))
        make_bar(df, 'tool', 'n_unique_hits', 'Tool', 'Unique hits',
                 'Unique hits per tool', os.path.join(out_dir, 'unique_hits_per_tool.png'))
        if 'n_unique_aros' in df.columns:
            make_bar(df, 'tool', 'n_unique_aros', 'Tool', 'Unique AROs',
                     'Unique AROs per tool', os.path.join(out_dir, 'unique_aros_per_tool.png'))
    # top ARO
    if 'top_aro_summary' in summaries:
        df = summaries['top_aro_summary'].head(top_terms)
        make_bar(df, 'ARO', 'n_unique_hits', 'ARO', 'Unique hits',
                 f'Top {top_terms} AROs by number of unique hits', os.path.join(out_dir, 'top_aro_unique_hits.png'), horizontal=True)
    # top genes
    if 'top_genes_summary' in summaries:
        df = summaries['top_genes_summary'].head(top_terms)
        make_bar(df, 'gene_symbol', 'n_unique_hits', 'Gene', 'Unique hits',
                 f'Top {top_terms} genes by number of unique hits', os.path.join(out_dir, 'top_genes_unique_hits.png'), horizontal=True)
    # antibiotic classes
    if 'by_drug_class_summary' in summaries:
        df = summaries['by_drug_class_summary'].sort_values('n_unique_hits', ascending=False).head(top_terms)
        make_bar(df, 'term', 'n_unique_hits', 'Drug class', 'Unique hits',
                 f'Top {top_terms} drug classes (unique hits)', os.path.join(out_dir, 'top_drug_class_hits.png'), horizontal=True)
    # confers
    if 'by_confers_summary' in summaries:
        df = summaries['by_confers_summary'].sort_values('n_unique_hits', ascending=False).head(top_terms)
        make_bar(df, 'term', 'n_unique_hits', 'Antibiotic', 'Unique hits',
                 f'Top {top_terms} conferred antibiotics (unique hits)', os.path.join(out_dir, 'top_confers_hits.png'), horizontal=True)
    # mechanisms
    if 'by_mechanism_summary' in summaries:
        df = summaries['by_mechanism_summary'].sort_values('n_unique_hits', ascending=False).head(top_terms)
        make_bar(df, 'term', 'n_unique_hits', 'Mechanism', 'Unique hits',
                 f'Top {top_terms} resistance mechanisms (unique hits)', os.path.join(out_dir, 'top_mechanisms_hits.png'), horizontal=True)
    # ARO heatmap by sample and tool (use unique_aros by sample/tool)
    if 'by_aro_summary' in summaries:
        # For the heatmap we need the sample × tool matrix of unique AROs.
        # This can be computed from the original DataFrame (df_full), but here
        # we assume summaries['_raw'] exists if needed.
        pass  # implement if required

def main():
    parser = argparse.ArgumentParser(description='ARGNORM results processing for resistance analysis (focused outputs).')
    parser.add_argument('-i','--input_dir', required=True, help='Base directory with per-tool folders and *.normalized.tsv files')
    parser.add_argument('-o','--outdir', required=True, help='Directory to save tables and figures')
    parser.add_argument('--top_terms', type=int, default=20, help='Top terms/features for charts (genes/ARO/classes)')
    parser.add_argument('--tool_for_drugplot', type=str, default=None, help='Filter Plot 1 (stacked drug classes) to a specific tool (e.g., abricate, amrfinderplus, deeparg)')
    parser.add_argument('--cluster_heatmaps', action='store_true', help='Cluster heatmaps with dendrograms (requires SciPy).')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Read and clean data
    df = load_argnorm_results(args.input_dir)
    if df.empty:
        print('No files found to analyze.')
        return
    df = clean_data(df)

    # Compute & save compact summaries (still useful as tables)
    summaries = compute_summaries(df)
    save_summary_tables(summaries, args.outdir)
    # Generate focused outputs requested
    generate_requested_plots_and_tables(df, args.outdir, top_n=args.top_terms, tool_for_drugplot=args.tool_for_drugplot, cluster_heatmaps=args.cluster_heatmaps)

    print('Analysis complete. Tables and figures saved to', args.outdir)

if __name__ == '__main__':
    main()
