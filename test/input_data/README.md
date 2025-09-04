This directory contains non-sensitive sample data to run and test the repository scripts without exposing real data.

**Structure and conventions:**

* **Folder structure:** kept the same as the real data so that paths work (e.g., `funscan_argnorm/amrfinderplus`, `funscan_argnorm/deeparg`, `mag_hamronization`).
* **Sample files:** small files are included with the same header/columns and format (`.tsv` with tab separator) as the originals.
* **Naming convention:** sample files use the prefix `SAMPLE-` or the suffix `.sample.tsv` to distinguish them from real data.
* **Privacy:** everything else under `test/input_data/` is ignored by Git to prevent uploading sensitive data.

**How to use:**

* For quick tests or CI, use the `SAMPLE-*.tsv` files and `hamronization_combined_report.sample.tsv` already included.
* For local tests with real data, copy your files into these same folders. Git is configured to ignore them.
* If your scripts expect specific filenames, adapt the paths to the `SAMPLE-*.tsv` files or create local (non-versioned) copies with the required names.

**Summary of included samples:**

* `funscan_argnorm/amrfinderplus/SAMPLE-001.normalized.tsv`
* `funscan_argnorm/deeparg/SAMPLE-001.ARG.normalized.tsv`
* `funscan_argnorm/deeparg/SAMPLE-002.potential.ARG.normalized.tsv`
* `mag_hamronization/hamronization_combined_report.sample.tsv`
