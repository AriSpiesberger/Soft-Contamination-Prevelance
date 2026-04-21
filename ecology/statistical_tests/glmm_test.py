"""Logistic GLMM on per-trial binary correctness: contam-model vs clean-model.

Auto-detects two input layouts:
    1) OLMo-style: {dir}/checkpoint-*/results.csv with per-trial rows
       (test_split, sample_id, sample_num, correct, ...).
    2) vLLM-style: {dir}/eval_{contam,clean}_split.csv with per-sample rows
       (epoch, sample_id, expected, predictions_json, ...) — the per-trial
       outcomes are unrolled from predictions_json.

For each (test_split, epoch), fits a mixed-effects logistic regression:

    correct ~ model_type + (1 | sample_id) + (1 | sample_num)

via statsmodels BinomialBayesMixedGLM (variational approximation).

Usage:
    # OLMo
    python glmm_test.py \\
        --contam-dir .../checkpoint_evals/contaminated \\
        --clean-dir  .../checkpoint_evals/clean \\
        --out-csv    .../stats/glmm_test.csv

    # Qwen3 / any vLLM-eval output
    python glmm_test.py \\
        --contam-dir .../evals/exp_contaminated_e1-e10 \\
        --clean-dir  .../evals/exp_clean_e1-e10 \\
        --out-csv    .../stats/glmm_test.csv
"""

import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

CKPT_STEP_DEFAULT = 167  # OLMo
SPLIT_FILE = {"contaminated": "eval_contam_split.csv", "clean": "eval_clean_split.csv"}


def epoch_of(ckpt_name, step, max_seen):
    if ckpt_name == "final":
        return max_seen + 1
    m = re.match(r"checkpoint-(\d+)$", ckpt_name)
    if not m:
        return None
    return round(int(m.group(1)) / step)


def load_trials(model_dir, model_type, step):
    ckpts = sorted(
        [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))],
        key=lambda n: (0, int(n.split("-")[1])) if n.startswith("checkpoint-") else (1, 0),
    )
    numbered = [c for c in ckpts if c.startswith("checkpoint-")]
    max_epoch = max((epoch_of(c, step, 0) for c in numbered), default=0)

    frames = []
    for ckpt in ckpts:
        path = os.path.join(model_dir, ckpt, "results.csv")
        if not os.path.exists(path):
            continue
        epoch = epoch_of(ckpt, step, max_epoch)
        if epoch is None:
            continue
        df = pd.read_csv(path, usecols=["test_split", "sample_id", "sample_num", "correct"])
        df["correct"] = df["correct"].astype(int)
        df["epoch"] = epoch
        df["model_type"] = model_type
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fit_glmm(df):
    """Fit BinomialBayesMixedGLM: correct ~ model_type + (1|sample_id) + (1|sample_num)."""
    df = df.copy()
    df["contam"] = (df["model_type"] == "contaminated").astype(int)
    vcf = {
        "sid": "0 + C(sample_id)",
        "snum": "0 + C(sample_num)",
    }
    model = BinomialBayesMixedGLM.from_formula("correct ~ contam", vcf, df)
    res = model.fit_vb()
    # Fixed-effect index for 'contam'
    names = list(res.model.exog_names)
    idx = names.index("contam")
    beta = float(res.fe_mean[idx])
    se = float(res.fe_sd[idx])
    z = beta / se if se > 0 else np.nan
    p = 2 * (1 - norm.cdf(abs(z))) if np.isfinite(z) else np.nan
    lo = beta - 1.96 * se
    hi = beta + 1.96 * se
    return {
        "beta": beta, "se": se, "z": z, "p": p,
        "or": float(np.exp(beta)),
        "or_lo": float(np.exp(lo)), "or_hi": float(np.exp(hi)),
    }


def run(contam_dir, clean_dir, out_csv, ckpt_step):
    contam = load_trials(contam_dir, "contaminated", ckpt_step)
    clean = load_trials(clean_dir, "clean", ckpt_step)
    if contam.empty or clean.empty:
        raise SystemExit("No data found. Check --contam-dir / --clean-dir.")
    df = pd.concat([contam, clean], ignore_index=True)

    epochs = sorted(set(df["epoch"].unique()))
    splits = sorted(set(df["test_split"].unique()))

    rows = []
    for split in splits:
        print(f"\n{'='*98}")
        print(f"GLMM (BinomialBayesMixedGLM, logit): contam vs clean | split = {split.upper()}")
        print(f"  model: correct ~ contam + (1|sample_id) + (1|sample_num)")
        print(f"{'='*98}")
        print(f"{'Epoch':>6}  {'N trials':>9}  {'beta':>9}  {'SE':>7}  {'z':>7}  "
              f"{'p':>10}  {'OR':>7}  {'95% OR CI':>22}  Sig")
        print("-" * 98)
        for e in epochs:
            sub = df[(df["test_split"] == split) & (df["epoch"] == e)]
            if sub["model_type"].nunique() < 2:
                continue
            try:
                res = fit_glmm(sub)
            except Exception as ex:
                print(f"{e:>6}  fit failed: {ex}")
                continue
            sig = "***" if res["p"] < 0.001 else "**" if res["p"] < 0.01 else "*" if res["p"] < 0.05 else ""
            ci = f"[{res['or_lo']:.3f},{res['or_hi']:.3f}]"
            print(f"{e:>6}  {len(sub):>9}  {res['beta']:>+9.4f}  {res['se']:>7.4f}  "
                  f"{res['z']:>+7.2f}  {res['p']:>10.6f}  {res['or']:>7.3f}  {ci:>22}  {sig}")
            rows.append({
                "split": split, "epoch": e, "n_trials": int(len(sub)),
                "beta": res["beta"], "se": res["se"], "z": res["z"], "p_value": res["p"],
                "odds_ratio": res["or"], "or_ci_low": res["or_lo"], "or_ci_high": res["or_hi"],
                "sig": sig,
            })

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    fields = ["split", "epoch", "n_trials", "beta", "se", "z", "p_value",
              "odds_ratio", "or_ci_low", "or_ci_high", "sig"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {out_csv}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--contam-dir", required=True,
                    help="Dir with checkpoint-*/results.csv for the contaminated model")
    ap.add_argument("--clean-dir", required=True,
                    help="Dir with checkpoint-*/results.csv for the clean model")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--ckpt-step", type=int, default=CKPT_STEP_DEFAULT,
                    help="Training steps per epoch (used to map checkpoint number -> epoch)")
    args = ap.parse_args()
    run(args.contam_dir, args.clean_dir, args.out_csv, args.ckpt_step)


if __name__ == "__main__":
    main()
