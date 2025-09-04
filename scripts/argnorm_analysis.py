#!/usr/bin/env python3
"""
argnorm_analysis.py

Carga resultados de ARGNORM desde directorios por herramienta, normaliza las
anotaciones y genera resúmenes y figuras para un análisis descriptivo del
resistoma.  Los resultados (tablas y PNGs) se guardan en la carpeta
especificada con --outdir.

Requisitos: pandas, matplotlib.
"""

import argparse
import os
import re
import pandas as pd
import matplotlib.pyplot as plt

def detect_delim(filename: str):
    """Devuelve el delimitador apropiado según la extensión."""
    return '\t' if filename.endswith(".tsv") else ','

def parse_sample_bin(input_file_name: str):
    """Extrae sample y bin_id asumiendo 'sample.bin.*'."""
    parts = input_file_name.split('.')
    sample = parts[0]
    bin_id = None
    if len(parts) >= 2 and re.match(r'^[0-9]+(?:[_-]\w+)?$', parts[1]):
        bin_id = parts[1]
    return sample, bin_id

def load_argnorm_results(base_dir: str):
    """Recorre todos los archivos .tsv/.csv dentro de base_dir y concatena en un único DataFrame."""
    dfs = []
    for root, _, files in os.walk(base_dir):
        # herramienta es el nombre de la subcarpeta inmediata a base_dir
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
    """Añade columnas parsed (sample, bin_id) y hit_key; convierte numéricos cuando sea posible."""
    if 'input_file_name' in df.columns:
        samples = df['input_file_name'].apply(parse_sample_bin)
        df['sample'] = samples.apply(lambda x: x[0])
        df['bin_id'] = samples.apply(lambda x: x[1])
    # clave única del hit (gene + accession)
    if {'gene_symbol','reference_accession'}.issubset(df.columns):
        df['_hit_key'] = df['gene_symbol'].astype(str) + '||' + df['reference_accession'].astype(str)
    # intentar convertir numéricos
    for col in ['coverage_percentage','coverage_depth','coverage_ratio','sequence_identity']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def explode_column(df: pd.DataFrame, col: str):
    """Explota una columna coma-separada en filas individuales (elimina NaNs)."""
    if col not in df.columns:
        return pd.DataFrame(columns=[col,'sample','bin_id','tool','_hit_key'])
    exploded = df[[col,'sample','bin_id','tool','_hit_key']].dropna(subset=[col]).copy()
    exploded[col] = exploded[col].astype(str)
    exploded = exploded.assign(**{col: exploded[col].str.split(',')})
    return exploded.explode(col).assign(**{col: lambda x: x[col].str.strip().str.lower()}).reset_index(drop=True)

def compute_summaries(df: pd.DataFrame):
    """Calcula resúmenes global, por muestra, por herramienta, por clase de antibiótico, por ARO, por confers y por mecanismo."""
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
    # por muestra
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
    # por herramienta
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

def generate_figures(summaries, out_dir, top_terms=20):
    """Genera figuras relevantes a partir de los resúmenes."""
    # hits y AROs por muestra
    if 'by_sample_summary' in summaries:
        df = summaries['by_sample_summary']
        make_bar(df, 'sample', 'n_unique_hits', 'Número de hits únicos', 'Muestra',
                 'Hits únicos por muestra (desc)', os.path.join(out_dir, 'unique_hits_per_sample.png'), horizontal=True)
        if 'n_unique_aros' in df.columns:
            make_bar(df, 'sample', 'n_unique_aros', 'Número de AROs únicos', 'Muestra',
                     'AROs únicos por muestra (desc)', os.path.join(out_dir, 'unique_aros_per_sample.png'), horizontal=True)
        make_hist(df['n_tools'], 'Número de herramientas', 'Número de muestras',
                  'Distribución de herramientas por muestra', os.path.join(out_dir, 'tools_distribution_per_sample.png'))
    # por herramienta
    if 'by_tool_summary' in summaries:
        df = summaries['by_tool_summary']
        make_bar(df, 'tool', 'n_rows', 'Herramienta', 'Hits totales',
                 'Hits totales por herramienta', os.path.join(out_dir, 'total_hits_per_tool.png'))
        make_bar(df, 'tool', 'n_unique_hits', 'Herramienta', 'Hits únicos',
                 'Hits únicos por herramienta', os.path.join(out_dir, 'unique_hits_per_tool.png'))
        if 'n_unique_aros' in df.columns:
            make_bar(df, 'tool', 'n_unique_aros', 'Herramienta', 'AROs únicos',
                     'AROs únicos por herramienta', os.path.join(out_dir, 'unique_aros_per_tool.png'))
    # top ARO
    if 'top_aro_summary' in summaries:
        df = summaries['top_aro_summary'].head(top_terms)
        make_bar(df, 'ARO', 'n_unique_hits', 'ARO', 'Hits únicos',
                 f'Top {top_terms} AROs por nº de hits únicos', os.path.join(out_dir, 'top_aro_unique_hits.png'), horizontal=True)
    # top genes
    if 'top_genes_summary' in summaries:
        df = summaries['top_genes_summary'].head(top_terms)
        make_bar(df, 'gene_symbol', 'n_unique_hits', 'Gen', 'Hits únicos',
                 f'Top {top_terms} genes por nº de hits únicos', os.path.join(out_dir, 'top_genes_unique_hits.png'), horizontal=True)
    # clases de antibiótico
    if 'by_drug_class_summary' in summaries:
        df = summaries['by_drug_class_summary'].sort_values('n_unique_hits', ascending=False).head(top_terms)
        make_bar(df, 'term', 'n_unique_hits', 'Clase de antibiótico', 'Hits únicos',
                 f'Top {top_terms} clases de antibiótico (hits únicos)', os.path.join(out_dir, 'top_drug_class_hits.png'), horizontal=True)
    # confers
    if 'by_confers_summary' in summaries:
        df = summaries['by_confers_summary'].sort_values('n_unique_hits', ascending=False).head(top_terms)
        make_bar(df, 'term', 'n_unique_hits', 'Antibiótico', 'Hits únicos',
                 f'Top {top_terms} antibióticos conferidos (hits únicos)', os.path.join(out_dir, 'top_confers_hits.png'), horizontal=True)
    # mecanismos
    if 'by_mechanism_summary' in summaries:
        df = summaries['by_mechanism_summary'].sort_values('n_unique_hits', ascending=False).head(top_terms)
        make_bar(df, 'term', 'n_unique_hits', 'Mecanismo', 'Hits únicos',
                 f'Top {top_terms} mecanismos de resistencia (hits únicos)', os.path.join(out_dir, 'top_mechanisms_hits.png'), horizontal=True)
    # heatmap de ARO por muestra y herramienta (usar unique_aros por sample/tool)
    if 'by_aro_summary' in summaries:
        # Para el heatmap necesitamos la matriz muestra × herramienta de ARO únicos
        # Podemos calcularla a partir del DataFrame original (df_full), pero aquí
        # asumimos que summary tiene df_original en summaries['_raw']
        pass  # lo implementaremos si es necesario

def main():
    parser = argparse.ArgumentParser(description='Procesamiento de resultados ARGNORM para resistencias.')
    parser.add_argument('-i','--input_dir', required=True, help='Directorio base con carpetas por herramienta y ficheros *.normalized.tsv')
    parser.add_argument('-o','--outdir', required=True, help='Directorio donde guardar tablas y figuras')
    parser.add_argument('--top_samples', type=int, default=20, help='Número de muestras top para heatmaps (no implementado)')
    parser.add_argument('--top_terms', type=int, default=20, help='Número de términos top para gráficos (genes, AROs, clases)')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Leer y limpiar datos
    df = load_argnorm_results(args.input_dir)
    if df.empty:
        print('No se encontraron ficheros para analizar.')
        return
    df = clean_data(df)

    # Calcular resúmenes
    summaries = compute_summaries(df)
    # Guardar tablas
    save_summary_tables(summaries, args.outdir)
    # Generar figuras
    generate_figures(summaries, args.outdir, top_terms=args.top_terms)

    print('Análisis completado. Tablas y figuras guardadas en', args.outdir)

if __name__ == '__main__':
    main()
