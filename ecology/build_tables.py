"""Build per-model LaTeX result tables from glmm.csv outputs.

Reads ecology/outcomes/<model>/evals/stats/glmm.csv and writes a LaTeX longtable
per (model, split) showing per-epoch accuracy with 95% CIs and the
clean-vs-contaminated / exact-vs-contaminated odds ratios + p-values.
"""

import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent / "outcomes"

def _pick(stats_dir):
    """Prefer glmm.csv if it contains glmer rows; else glmm_glmer.csv."""
    main = stats_dir / "glmm.csv"
    alt = stats_dir / "glmm_glmer.csv"
    if main.exists() and "glmer" in main.read_text(encoding="utf-8"):
        return main
    if alt.exists():
        return alt
    return main

MODELS = [
    ("OLMo3-7B", _pick(ROOT / "outputs_olmo" / "evals" / "stats")),
    ("Qwen3-8B", _pick(ROOT / "outputs_qwen3" / "evals" / "stats")),
    ("Qwen3.5-9B", _pick(ROOT / "outputs_qwen35" / "evals" / "stats")),
]
OUT_DIR = ROOT / "tables"
MAX_EPOCH = 10


def stars(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return r"\sss"
    if p < 0.01:
        return r"\ss"
    if p < 0.05:
        return r"\s"
    return ""


def fmt_pct(m, lo, hi):
    if pd.isna(m):
        return "--"
    return f"{m*100:.1f} [{lo*100:.1f}, {hi*100:.1f}]"


def fmt_or(o, lo, hi):
    if pd.isna(o):
        return "--"
    return f"{o:.2f} [{lo:.2f}, {hi:.2f}]"


def fmt_p(p):
    if pd.isna(p):
        return "--"
    if p < 1e-3:
        return f"{p:.1e}"
    return f"{p:.3f}"


def build_table(label, df, split):
    sub = df[(df["split"] == split) & (df["epoch"] <= MAX_EPOCH)].sort_values("epoch")
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering\small")
    pretty = {"contaminated": "Contaminated", "clean": "Clean"}[split]
    lines.append(rf"\caption{{{label}: per-epoch accuracy and GLMM contrasts on the {pretty} test split. "
                 r"Means with 95\% bootstrap CIs. ORs are from a binomial mixed-effects logistic regression "
                 r"(\texttt{correct $\sim$ model + (1\,$\vert$\,sample\_id)}) fit with \texttt{lme4::glmer}. "
                 r"$^{*}p<0.05,\ ^{**}p<0.01,\ ^{***}p<0.001$.}")
    lines.append(rf"\label{{tab:{label.lower().replace('.','_').replace('-','_')}_{split}}}")
    lines.append(r"\begin{tabular}{r ccc cc cc}")
    lines.append(r"\toprule")
    lines.append(r"Ep & Clean acc.\ \% & Contam.\ acc.\ \% & Exact acc.\ \% "
                 r"& OR clean/contam & $p$ & OR exact/contam & $p$ \\")
    lines.append(r"\midrule")
    for _, r in sub.iterrows():
        lines.append(
            f"{int(r['epoch'])} & "
            f"{fmt_pct(r['clean_mean'], r['clean_ci_lo'], r['clean_ci_hi'])} & "
            f"{fmt_pct(r['contam_mean'], r['contam_ci_lo'], r['contam_ci_hi'])} & "
            f"{fmt_pct(r['exact_mean'], r['exact_ci_lo'], r['exact_ci_hi'])} & "
            f"{fmt_or(r['or_clean_vs_contam'], r['or_cc_lo'], r['or_cc_hi'])} & "
            f"{fmt_p(r['p_clean_vs_contam'])}{stars(r['p_clean_vs_contam'])} & "
            f"{fmt_or(r['or_exact_vs_contam'], r['or_ec_lo'], r['or_ec_hi'])} & "
            f"{fmt_p(r['p_exact_vs_contam'])}{stars(r['p_exact_vs_contam'])} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for label, path in MODELS:
        df = pd.read_csv(path)
        for split in ("contaminated", "clean"):
            tex = build_table(label, df, split)
            fname = f"{label.lower().replace('.','_').replace('-','_')}_{split}.tex"
            (OUT_DIR / fname).write_text(tex, encoding="utf-8")
            print(f"wrote {OUT_DIR / fname}")


if __name__ == "__main__":
    main()
