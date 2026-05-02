"""Single LaTeX summary table: per model, baseline + epoch-5 + epoch-10 across
clean / contaminated / exact treatments, with GLMM p-values vs. contaminated.

Columns: Treatment | Seen % | Unseen % | p_seen (vs contam) | p_unseen (vs contam)
Rows are grouped by model. p-values come from lme4::glmer fits in glmm.csv;
"--" where not applicable (baseline rows and the contaminated-treatment row).
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent / "outcomes"


def _pick(stats_dir):
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

BASE_JSON = Path(__file__).resolve().parent / "outputs" / "base_model_eval_results.json"
BASE_KEY = {"OLMo3-7B": "olmo_base", "Qwen3-8B": "qwen_base", "Qwen3.5-9B": None}

OUT_FILE = ROOT / "tables" / "summary_baseline_e5_e10.tex"
EPOCHS = (5, 10)


def stars(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return r"$^{***}$"
    if p < 0.01:
        return r"$^{**}$"
    if p < 0.05:
        return r"$^{*}$"
    return ""


def fmt_pct(x):
    if x is None or pd.isna(x):
        return "--"
    return f"{x*100:.1f}"


def fmt_p(p):
    if p is None or pd.isna(p):
        return "--"
    if p < 1e-3:
        return f"{p:.1e}{stars(p)}"
    return f"{p:.3f}{stars(p)}"


def load_baseline(label):
    key = BASE_KEY.get(label)
    if not key or not BASE_JSON.exists():
        return None, None
    data = json.loads(BASE_JSON.read_text(encoding="utf-8"))
    rec = data.get(key)
    if not rec:
        return None, None
    return rec.get("contaminated_accuracy"), rec.get("clean_accuracy")


def get_stats(df, epoch):
    seen = df[(df["split"] == "contaminated") & (df["epoch"] == epoch)]
    unseen = df[(df["split"] == "clean") & (df["epoch"] == epoch)]
    if seen.empty or unseen.empty:
        return None
    s = seen.iloc[0]
    u = unseen.iloc[0]
    return {
        "clean_seen": s["clean_mean"], "clean_unseen": u["clean_mean"],
        "contam_seen": s["contam_mean"], "contam_unseen": u["contam_mean"],
        "exact_seen": s["exact_mean"], "exact_unseen": u["exact_mean"],
        "p_cc_seen": s["p_clean_vs_contam"], "p_cc_unseen": u["p_clean_vs_contam"],
        "p_ec_seen": s["p_exact_vs_contam"], "p_ec_unseen": u["p_exact_vs_contam"],
    }


def model_block(label, df):
    base_seen, base_unseen = load_baseline(label)
    rows = []
    rows.append((r"\textsc{Baseline}", base_seen, base_unseen, None, None))
    for ep in EPOCHS:
        r = get_stats(df, ep)
        if r is None:
            continue
        rows.append((rf"E={ep}, \textsc{{Finetuned Clean}}",
                     r["clean_seen"], r["clean_unseen"],
                     r["p_cc_seen"], r["p_cc_unseen"]))
        rows.append((rf"E={ep}, \textsc{{Finetuned Contaminated}}",
                     r["contam_seen"], r["contam_unseen"], None, None))
        rows.append((rf"E={ep}, \textsc{{Finetuned Exact}}",
                     r["exact_seen"], r["exact_unseen"],
                     r["p_ec_seen"], r["p_ec_unseen"]))
    return rows


def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out = []
    out.append(r"\begin{table}[t]")
    out.append(r"\centering\small")
    out.append(r"\caption{Per-model accuracy on the contamination benchmark. "
               r"\textsc{Seen} = test items the contaminated arm was trained on; "
               r"\textsc{Unseen} = held-out clean test split. "
               r"$p$-values are from binomial mixed-effects logistic regressions "
               r"(\texttt{correct $\sim$ treatment + (1\,$\vert$\,sample\_id)}, "
               r"\texttt{lme4::glmer}) for each treatment vs.\ \textsc{Finetuned Contaminated}, "
               r"computed separately on each test split. "
               r"$^{*}p<0.05,\ ^{**}p<0.01,\ ^{***}p<0.001$.}")
    out.append(r"\label{tab:contamination_summary}")
    out.append(r"\begin{tabular}{l l cc cc}")
    out.append(r"\toprule")
    out.append(r" & & \multicolumn{2}{c}{Accuracy (\%)} & \multicolumn{2}{c}{$p$ (vs.\ Contam.)} \\")
    out.append(r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}")
    out.append(r"Model & Treatment & Seen & Unseen & Seen & Unseen \\")
    out.append(r"\midrule")
    for i, (label, csv_path) in enumerate(MODELS):
        df = pd.read_csv(csv_path)
        rows = model_block(label, df)
        for j, (treat, seen, unseen, p_seen, p_unseen) in enumerate(rows):
            model_cell = rf"\multirow{{{len(rows)}}}{{*}}{{{label}}}" if j == 0 else ""
            out.append(f"{model_cell} & {treat} & {fmt_pct(seen)} & {fmt_pct(unseen)} & "
                       f"{fmt_p(p_seen)} & {fmt_p(p_unseen)} \\\\")
        if i < len(MODELS) - 1:
            out.append(r"\midrule")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    out.append(r"\end{table}")
    OUT_FILE.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {OUT_FILE}")


if __name__ == "__main__":
    main()
