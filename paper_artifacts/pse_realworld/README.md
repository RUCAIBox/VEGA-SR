# PSE real-world paper artifacts

This directory contains the machine-readable outputs underlying
`PSE_REALWORLD_PAPER_RESULTS.md`.

- `all_results.csv`: 161 case-level records covering PSE, PSE-aligned VEGA-SR,
  PySR, Operon, and the deterministic EMPS Physics-LS ablation. Local absolute
  paths were mechanically replaced with `<REPO_ROOT>`; numerical fields and
  experiment traces were not changed.
- `summary.csv`: aggregate statistics used in the paper table.
- `status_counts.csv`: completion counts by method and dataset.
- `completion_audit.json`: completion and invariant audit for the 80 formal
  PSE/VEGA-SR cases.
- `REPORT.md`: concise PSE/VEGA-SR report with the sealed-test limitation stated
  explicitly.
- `SHA256SUMS`: integrity manifest for this release directory.

The canonical experiment protocol is `pse_realworld_vega_sr.yaml`. Rebuild the
combined outputs with:

```bash
python scripts/summarize_pse_realworld_comparison.py
python scripts/generate_pse_realworld_paper_md.py
```

Candidate ranking excludes test metrics from its explicit score. The current
VEGA-SR fitter nevertheless evaluates test-domain predictions while constructing
fit records and may reject a numerically invalid expression. Accordingly, these
artifacts do not support the stronger claim that the test split was completely
unaccessed during search.
