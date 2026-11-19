# Analysis code, revision R1

Python 3 (NumPy, SciPy, Matplotlib, statsmodels not required). Scripts were run in this order; input/output paths at the top of each script point to the working directories used during the revision and must be adapted (inputs are in `data/R1/`).

1. `data.py` (uses `build.py`): builds the analysis dataset (`ds.json`) from `reextraction_records.jsonl`, applying the eligibility exclusions, the overlap rule and the prespecified estimate-selection rule, and derives logit-AUC standard errors (CI; Hanley-McNeil with class counts; Hanley-McNeil with evaluation-set size and median prevalence).
2. `meta.py`: REML (and DerSimonian-Laird) random-effects models, Hartung-Knapp intervals, prediction intervals, within-study aggregation and the three-level model.
3. `run.py`, `run2.py`, `run3.py`: primary syntheses by contrast, sensitivity analyses, subgroups, multivariable meta-regression, small-study tests and bivariate model (results in `results.json`).
4. `probast.py`: PROBAST signalling-question judgements from the verified extraction and full-text term searches (decision rules in ESM 2).
5. `nums_r1.py`: descriptive counts; `figs.py` and `prisma.py`: Figures 1-4.
