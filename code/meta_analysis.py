"""Reproducible meta-analysis for:
"Machine Learning for the Classification of Osteoporosis and Low Bone Mineral
Density in Adults: A Systematic Review and Meta-Analysis" (Carvalho & Gavaia).

Reads  ../data/meta_v3_data.csv  and reproduces the pooled estimate, subgroup
analyses, leave-one-out sensitivity, meta-regression, Egger's test, and
Figures 2-4 of the manuscript.

Usage:  python meta_analysis.py
Requires: pandas, numpy, scipy, matplotlib  (see requirements.txt)
"""
import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "meta_v3_data.csv")
OUT = os.path.join(HERE, "..", "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 12, "font.family": "DejaVu Sans"})

df = pd.read_csv(DATA)
k = len(df)
y = df.logit_auc.values
se = df.logit_se.values

def dl_pool(y, se):
    """DerSimonian-Laird random-effects pooling on the logit scale."""
    w = 1 / se**2
    yf = np.sum(w*y) / np.sum(w)
    Q = np.sum(w * (y - yf)**2)
    dfree = len(y) - 1
    C = np.sum(w) - np.sum(w**2) / np.sum(w)
    tau2 = max(0, (Q - dfree) / C)
    wr = 1 / (se**2 + tau2)
    yb = np.sum(wr*y) / np.sum(wr)
    sp = np.sqrt(1 / np.sum(wr))
    I2 = max(0, (Q - dfree) / Q) * 100 if Q > 0 else 0
    return yb, sp, tau2, Q, I2, wr

expit = lambda x: 1 / (1 + np.exp(-x))

yb, sp, tau2, Q, I2, wr = dl_pool(y, se)
ci = (yb - 1.96*sp, yb + 1.96*sp)
tcrit = stats.t.ppf(0.975, k - 2)
pih = tcrit * np.sqrt(tau2 + sp**2)
pi = (yb - pih, yb + pih)
p_Q = 1 - stats.chi2.cdf(Q, k - 1)

print(f"k = {k}")
print(f"Pooled AUC = {expit(yb):.3f} (95% CI {expit(ci[0]):.3f}-{expit(ci[1]):.3f})")
print(f"tau2 = {tau2:.4f} | 95% PI = {expit(pi[0]):.3f}-{expit(pi[1]):.3f}")
print(f"Q = {Q:.1f} (df={k-1}, p={'<0.001' if p_Q < 0.001 else f'{p_Q:.3f}'}), I2 = {I2:.1f}%")

print("\n--- Subgroups ---")
sub = {}
for var in ["data_type", "ml_category", "ext_val", "population", "source"]:
    sub[var] = []
    for lev, g in df.groupby(var):
        if len(g) < 3:
            print(f"{var} = {lev}: k={len(g)} (not pooled)")
            continue
        b, s, t2, q, i2, _ = dl_pool(g.logit_auc.values, g.logit_se.values)
        print(f"{var} = {lev}: AUC {expit(b):.3f} ({expit(b-1.96*s):.3f}-{expit(b+1.96*s):.3f}), "
              f"k={len(g)}, I2={i2:.0f}%, tau2={t2:.3f}")
        sub[var].append((lev, expit(b), expit(b-1.96*s), expit(b+1.96*s), len(g)))

print("\n--- Leave-one-out ---")
loo = []
for i in range(k):
    m = np.ones(k, bool); m[i] = False
    b, s, *_ = dl_pool(y[m], se[m])
    loo.append(expit(b))
print(f"Pooled AUC omitting one study: {min(loo):.3f} - {max(loo):.3f}")

print("\n--- Meta-regression (WLS, random-effects weights) ---")
X = pd.DataFrame({
    "imaging": (df.data_type == "Imaging").astype(int),
    "deep": (df.ml_category == "Deep Learning").astype(int),
    "ensemble": (df.ml_category == "Ensemble").astype(int),
    "extval": (df.ext_val == "Yes").astype(int),
    "log_n": np.log10(df.n.fillna(df.n.median())),
    "update": (df.source == "update2026").astype(int),
})
Xd = np.column_stack([np.ones(k), X.values])
Wm = np.diag(wr)
beta = np.linalg.solve(Xd.T @ Wm @ Xd, Xd.T @ Wm @ y)
covb = np.linalg.inv(Xd.T @ Wm @ Xd)
seb = np.sqrt(np.diag(covb))
for nm, b, s in zip(["Intercept","Imaging","DeepLearning","Ensemble","ExternalVal","log10(n)","UpdateRound"], beta, seb):
    z = b / s
    pv = 2 * (1 - stats.norm.cdf(abs(z)))
    print(f"{nm}: beta={b:+.3f} (SE {s:.3f}), p={pv:.3f}")

print("\n--- Egger's test ---")
sl, it, r, p_eg, sei = stats.linregress(1/se, y/se)
print(f"intercept = {it:.2f}, p = {p_eg:.4f}")

# ---------------- Figures ----------------
dfo = df.sort_values("auc").reset_index(drop=True)
fig, ax = plt.subplots(figsize=(11, 20))
yp = np.arange(len(dfo))
lo = dfo.apply(lambda r: r.auc_lo if pd.notna(r.auc_lo) else expit(r.logit_auc - 1.96*r.logit_se), axis=1).clip(0.4, 1)
hi = dfo.apply(lambda r: r.auc_hi if pd.notna(r.auc_hi) else expit(r.logit_auc + 1.96*r.logit_se), axis=1).clip(0.4, 1)
colors = dfo.source.map({"original": "#7F7F7F", "update2026": "#2F5496"})
ax.errorbar(dfo.auc, yp, xerr=[np.maximum(dfo.auc-lo, 0), np.maximum(hi-dfo.auc, 0)],
            fmt="none", ecolor="#9BB0D3", capsize=2, lw=1.1)
ax.scatter(dfo.auc, yp, c=colors, s=28, zorder=3)
ax.axvline(expit(yb), color="#C00000", ls="--", lw=1.8, label=f"Pooled AUC = {expit(yb):.3f}")
ax.axvspan(expit(pi[0]), expit(pi[1]), color="#C00000", alpha=0.06,
           label=f"95% PI ({expit(pi[0]):.3f}-{expit(pi[1]):.3f})")
ax.set_yticks(yp); ax.set_yticklabels(dfo.study, fontsize=8)
ax.set_xlabel("AUC (95% CI)", fontsize=13)
ax.set_title(f"ML classification of osteoporosis / low BMD (k = {k})", fontsize=14, pad=12)
ax.legend(fontsize=12, loc="lower right"); ax.set_xlim(0.4, 1.0); ax.set_ylim(-1, len(dfo))
fig.tight_layout(); fig.savefig(os.path.join(OUT, "Fig2_forest.png"), dpi=300); plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 9))
rows = []
for var, lab in [("data_type","Data type"), ("ml_category","ML category"),
                 ("ext_val","External validation"), ("population","Population")]:
    for lev, a, l, h, kk in sub[var]:
        rows.append((f"{lab}: {lev} (k={kk})", a, l, h))
yp = np.arange(len(rows))[::-1]
for (labl, a, l, h), ypos in zip(rows, yp):
    ax.errorbar(a, ypos, xerr=[[a-l], [h-a]], fmt="o", color="#2F5496", ms=9, capsize=4, lw=2)
ax.axvline(expit(yb), color="#C00000", ls="--", lw=1.8)
ax.set_yticks(yp); ax.set_yticklabels([r[0] for r in rows], fontsize=12)
ax.set_xlabel("Pooled AUC (95% CI)", fontsize=13)
ax.set_title("Subgroup analyses", fontsize=14, pad=12)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "Fig3_subgroups.png"), dpi=300); plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 7))
ax.scatter(y, se, s=50, c=df.source.map({"original": "#7F7F7F", "update2026": "#2F5496"}), zorder=3)
sg = np.linspace(0.001, max(se)*1.1, 100)
ax.plot(yb - 1.96*sg, sg, "k--", lw=1.3); ax.plot(yb + 1.96*sg, sg, "k--", lw=1.3)
ax.axvline(yb, color="#C00000", lw=1.5); ax.invert_yaxis()
ax.set_xlabel("Logit AUC", fontsize=13); ax.set_ylabel("Standard error", fontsize=13)
ax.set_title(f"Funnel plot (k = {k}) - Egger p = {p_eg:.3f}", fontsize=14, pad=12)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "Fig4_funnel.png"), dpi=300); plt.close(fig)

print(f"\nFigures written to {os.path.abspath(OUT)}")
