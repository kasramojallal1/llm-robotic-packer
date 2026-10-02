# Scripts that produce the paper's tables and figures

Every number in the results tables and the qualitative figure of the paper is generated from the run records in `results/`:

```bash
python paper/make_tables.py --roots results --out <tables-dir>          # LaTeX rows for Tables 5-10 and A1-A5
python paper/make_qual_figure.py --results results --out qualitative.pdf  # Figure 5 (final packings)
```

Conventions: utilization is mean ± sample standard deviation (ddof = 1) over the five sequences of a benchmark; paired tests are Wilcoxon signed-rank (paired t as a check) over the 20 shared sequences, with Holm-adjusted p-values over each table's family of comparisons.
