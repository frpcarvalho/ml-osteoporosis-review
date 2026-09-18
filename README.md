# Machine Learning for the Classification of Osteoporosis and Low Bone Mineral Density in Adults — Data & Code

Data, analysis code, and search strategies for the systematic review and meta-analysis:

> Carvalho FR, Gavaia PJ. **Machine Learning for the Classification of Osteoporosis and Low Bone Mineral Density in Adults: A Systematic Review and Meta-Analysis.** (submitted, 2026)

## Contents

| Path | Description |
|---|---|
| `data/data_extraction.csv` | Full data-extraction table for the 133 update-round included studies (also as `.xlsx`) |
| `data/meta_v3_data.csv` | Analysis dataset for the meta-analysis (k = 136: 21 original-round + 115 update-round studies with poolable AUC) |
| `data/PROBAST_lite_provisional.xlsx` | Structured item-level risk-of-bias appraisal of the update-round studies |
| `code/meta_analysis.py` | Reproduces the pooled estimate, subgroups, leave-one-out, meta-regression, Egger's test, and Figures 2–4 |
| `search/search_strings.txt` | Complete reproducible search strategies for PubMed, Scopus, Web of Science, and IEEE Xplore |

## Reproducing the meta-analysis

```bash
pip install -r code/requirements.txt
python code/meta_analysis.py
```

Expected output: pooled AUC 0.884 (95% CI 0.873–0.894), τ² = 0.348, 95% prediction interval 0.703–0.961, I² = 98.8% (k = 136), subgroup and meta-regression estimates as reported in the manuscript, and Figures 2–4 written to `figures/`.

## Dataset columns (`meta_v3_data.csv`)

`study` (first author, year) · `rid` (screening record ID, update round) · `year` · `auc`, `auc_lo`, `auc_hi` (best-model AUC and 95% CI where reported) · `n` (total sample) · `data_type` (Imaging / Clinical-Lab / Multimodal) · `ml_category` (Deep Learning / Ensemble / Traditional ML) · `ext_val` (external validation, Yes/No) · `population` · `logit_auc`, `logit_se` (logit-transformed AUC and its standard error; SE from the reported CI where available, otherwise the Hanley–McNeil approximation) · `source` (original / update2026 search round)

## Methods summary

Two PRISMA 2020 search rounds (to August 2026); eligibility restricted to ML classification of prevalent osteoporosis or low BMD in adults against a densitometric reference standard; DerSimonian–Laird random-effects pooling of logit-AUC. Full methods in the manuscript.

## AI-assistance disclosure

An AI assistant (Claude, Anthropic) supported literature organisation, de-duplication, pre-screening, extraction drafting, and code preparation; all decisions and outputs were verified by the authors, who assume full responsibility.

## License

Code: MIT. Data: CC BY 4.0. Please cite the manuscript when reusing.
