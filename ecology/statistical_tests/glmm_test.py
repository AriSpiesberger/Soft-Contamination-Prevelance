"""Mixed-effects logistic regression on raw binary outcomes across model variants.

Companion to wilcoxon_test.py. Same CLI, same data loaders, but instead of
collapsing 20 runs per question into a pass_rate and running Wilcoxon on the
paired means, this fits:

    correct ~ model + (1 | question_id)     family=binomial

on the full run-level binary data for each (test_split, epoch). This uses
within-question variance information that the Wilcoxon throws away, which
gives more power when margins are slight. Zero pass-rate questions are no
longer a special case — they're just rows of 0s.

Contrasts reported per epoch x split:
    - clean    vs contaminated   (Wald p on the model coefficient)
    - exact    vs contaminated

Primary backend: pymer4 (wraps R's lme4::glmer). This is the right tool.
Fallback backend: statsmodels GEE with exchangeable working correlation,
cluster-robust SEs at the question level. Not a GLMM - it's a marginal
model - but it handles clustered binary data with correct inference and is
pure-Python. Use --backend to force one or the other.

Usage:
    python glmm_test.py \\
        --clean-dir  ecology/outcomes/outputs_qwen3/evals/exp_clean_e1-e10 \\
        --contam-dir ecology/outcomes/outputs_qwen3/evals/exp_contaminated_e1-e10 \\
        --exact-dir  ecology/outcomes/outputs_qwen3/evals/exp_exact_e1-e10 \\
        --label "Qwen3-8B" \\
        --out-csv    ecology/outcomes/outputs_qwen3/evals/stats/glmm.csv
"""

import argparse
import csv
import glob
import os
import re
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

SPLIT_FILE = {"contaminated": "eval_contam_split.csv", "clean": "eval_clean_split.csv"}
CKPT_STEP_DEFAULT = 167


# ----------------------------- layout detection -----------------------------

def detect_layout(d):
    if all(os.path.exists(os.path.join(d, f)) for f in SPLIT_FILE.values()):
        return "vllm"
    if glob.glob(os.path.join(d, "checkpoint-*", "results.csv")):
        return "olmo"
    raise SystemExit(f"Unrecognized layout: {d}")


def epoch_of(ckpt, step, max_seen):
    if ckpt == "final":
        return max_seen + 1
    m = re.match(r"checkpoint-(\d+)$", ckpt)
    return round(int(m.group(1)) / step) if m else None


# ----------------------------- data loading -----------------------------
# Returns long-format dataframe: columns [epoch, sample_id, run_idx, correct]
# For vLLM, the split CSV only has per-(epoch,sample_id) pass_rate, so we
# expand it into n_runs Bernoulli trials with k successes and n-k failures.
# For OLMo, we have per-trial rows already.

def load_vllm_long(model_dir, split, n_runs_hint):
    path = os.path.join(model_dir, SPLIT_FILE[split])
    rows = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ep = row["epoch"]
            if ep == "final":
                continue
            epoch = int(ep)
            sid = int(row["sample_id"])
            pr = float(row["pass_rate"])
            n = None
            for key in ("n_runs", "n_trials", "n", "num_runs", "n_samples"):
                if key in row and row[key]:
                    try:
                        n = int(row[key])
                        break
                    except ValueError:
                        pass
            if n is None:
                n = n_runs_hint
            k = int(round(pr * n))
            for r in range(n):
                rows.append((epoch, sid, r, 1 if r < k else 0))
    return pd.DataFrame(rows, columns=["epoch", "sample_id", "run_idx", "correct"])


def load_olmo_long(model_dir, split, step):
    ckpts = sorted(
        [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))],
        key=lambda n: (0, int(n.split("-")[1])) if n.startswith("checkpoint-") else (1, 0),
    )
    numbered = [c for c in ckpts if c.startswith("checkpoint-")]
    max_epoch = max((epoch_of(c, step, 0) for c in numbered), default=0)
    rows = []
    for c in ckpts:
        path = os.path.join(model_dir, c, "results.csv")
        if not os.path.exists(path):
            continue
        epoch = epoch_of(c, step, max_epoch)
        if epoch is None:
            continue
        run_counters = defaultdict(int)
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["test_split"] != split:
                    continue
                sid = int(row["sample_id"])
                correct = 1 if row["correct"].strip() == "1" else 0
                rows.append((epoch, sid, run_counters[sid], correct))
                run_counters[sid] += 1
    return pd.DataFrame(rows, columns=["epoch", "sample_id", "run_idx", "correct"])


def load_long(model_dir, split, step, n_runs_hint):
    return (load_vllm_long(model_dir, split, n_runs_hint)
            if detect_layout(model_dir) == "vllm"
            else load_olmo_long(model_dir, split, step))


# ----------------------------- backends -----------------------------

def fit_glmer_pymer4(df, ref_level):
    from pymer4.models import Lmer  # noqa: F401
    df = df.copy()
    df["model"] = pd.Categorical(df["model"],
                                 categories=[ref_level] + [m for m in df["model"].unique() if m != ref_level])
    m = Lmer("correct ~ model + (1|sample_id)", data=df, family="binomial")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(summarize=False, verbose=False)
    out = {}
    for name, row in m.coefs.iterrows():
        if name == "(Intercept)":
            continue
        est = float(row["Estimate"])
        se = float(row["SE"])
        p = float(row["P-val"])
        or_ = float(np.exp(est))
        or_lo = float(np.exp(est - 1.96 * se))
        or_hi = float(np.exp(est + 1.96 * se))
        out[name] = (est, se, p, or_, or_lo, or_hi)
    return out


_RSCRIPT_PATHS = [
    os.environ.get("RSCRIPT"),
    r"C:\Program Files\R\R-4.6.0\bin\Rscript.exe",
    r"C:\Program Files\R\R-4.5.1\bin\Rscript.exe",
    "Rscript",
]
_R_USER_LIB = os.environ.get("R_LIBS_USER", r"C:\Users\arisp\R\win-library\4.6")
_R_SCRIPT = r'''
suppressPackageStartupMessages({
  user_lib <- Sys.getenv("R_LIBS_USER", unset = NA)
  if (!is.na(user_lib) && nzchar(user_lib)) .libPaths(c(user_lib, .libPaths()))
  library(lme4)
  library(jsonlite)
})
args <- commandArgs(trailingOnly = TRUE)
df <- read.csv(args[1], stringsAsFactors = FALSE)
df$model <- factor(df$model, levels = strsplit(args[2], ",")[[1]])
df$sample_id <- factor(df$sample_id)
fit <- glmer(correct ~ model + (1 | sample_id), data = df, family = binomial,
             control = glmerControl(optimizer = "bobyqa",
                                    optCtrl = list(maxfun = 1e5)))
co <- summary(fit)$coefficients
out <- list()
for (nm in rownames(co)) {
  if (nm == "(Intercept)") next
  est <- unname(co[nm, "Estimate"])
  se  <- unname(co[nm, "Std. Error"])
  p   <- unname(co[nm, "Pr(>|z|)"])
  out[[nm]] <- list(est = est, se = se, p = p)
}
cat(toJSON(out, auto_unbox = TRUE, digits = 12))
'''


def _rscript_path():
    import shutil
    for p in _RSCRIPT_PATHS:
        if not p:
            continue
        if os.path.isfile(p):
            return p
        found = shutil.which(p)
        if found:
            return found
    raise RuntimeError("Rscript not found; set RSCRIPT env var to its path")


def fit_glmer_rscript(df, ref_level):
    """Fit binomial GLMM `correct ~ model + (1|sample_id)` via R/lme4 subprocess."""
    import json
    import subprocess
    import tempfile

    levels = [ref_level] + [m for m in df["model"].unique() if m != ref_level]
    sub = df[["sample_id", "model", "correct"]].copy()

    with tempfile.TemporaryDirectory() as td:
        csv_path = os.path.join(td, "data.csv")
        r_path = os.path.join(td, "fit.R")
        sub.to_csv(csv_path, index=False)
        with open(r_path, "w", encoding="utf-8") as f:
            f.write(_R_SCRIPT)
        env = os.environ.copy()
        env.setdefault("R_LIBS_USER", _R_USER_LIB)
        proc = subprocess.run(
            [_rscript_path(), "--vanilla", r_path, csv_path, ",".join(levels)],
            capture_output=True, text=True, env=env, timeout=300,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"glmer failed: {proc.stderr.strip() or proc.stdout.strip()}")
        try:
            res = json.loads(proc.stdout.strip())
        except json.JSONDecodeError as e:
            raise RuntimeError(f"glmer parse failed: {e}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")

    out = {}
    for name, vals in res.items():
        est = float(vals["est"])
        se = float(vals["se"])
        p = float(vals["p"])
        or_ = float(np.exp(est))
        or_lo = float(np.exp(est - 1.96 * se))
        or_hi = float(np.exp(est + 1.96 * se))
        out[name] = (est, se, p, or_, or_lo, or_hi)
    return out


def fit_gee_statsmodels(df, ref_level):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    df = df.copy()
    df["model"] = pd.Categorical(df["model"],
                                 categories=[ref_level] + [m for m in df["model"].unique() if m != ref_level])
    fam = sm.families.Binomial()
    cov = sm.cov_struct.Exchangeable()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = smf.gee("correct ~ model", groups="sample_id",
                      data=df, family=fam, cov_struct=cov).fit()
    out = {}
    for name in res.params.index:
        if name == "Intercept":
            continue
        est = float(res.params[name])
        se = float(res.bse[name])
        p = float(res.pvalues[name])
        or_ = float(np.exp(est))
        or_lo = float(np.exp(est - 1.96 * se))
        or_hi = float(np.exp(est + 1.96 * se))
        out[name] = (est, se, p, or_, or_lo, or_hi)
    return out


def pick_backend(requested):
    if requested == "gee":
        return "gee", fit_gee_statsmodels
    if requested in ("glmer", "rscript"):
        _rscript_path()  # raises if missing
        return "glmer", fit_glmer_rscript
    if requested == "pymer4":
        import pymer4.models  # noqa: F401
        return "glmer", fit_glmer_pymer4
    try:
        _rscript_path()
        return "glmer", fit_glmer_rscript
    except Exception:
        pass
    try:
        import pymer4.models  # noqa: F401
        return "glmer", fit_glmer_pymer4
    except Exception:
        return "gee", fit_gee_statsmodels


# ----------------------------- formatting -----------------------------

def fmt_p(p):
    if not np.isfinite(p):
        return "    n/a"
    if p < 0.001:
        return f"{p:7.1e}"
    return f"{p:7.4f}"


def sig_marker(p):
    return ("***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "") if np.isfinite(p) else ""


def fmt_or(or_, lo, hi):
    return f"{or_:5.2f} [{lo:4.2f},{hi:4.2f}]"


def boot_mean_ci(values, n_boot=10000, ci=0.95, seed=42):
    rng = np.random.RandomState(seed)
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    idx = rng.randint(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    a = (1 - ci) / 2
    return float(v.mean()), float(np.percentile(means, 100 * a)), float(np.percentile(means, 100 * (1 - a)))


def fmt_ci(m, lo, hi):
    return f"{m*100:5.2f} [{lo*100:5.2f},{hi*100:5.2f}]"


def per_question_pass_rates(sub):
    return (sub.groupby("sample_id")["correct"].mean()
              .reindex(sorted(sub["sample_id"].unique())).to_numpy())


def is_perfect_separation(arr_a, arr_b):
    """One arm always 0 or always 1 makes the logit contrast degenerate."""
    for arr in (arr_a, arr_b):
        m = arr.mean()
        if m == 0.0 or m == 1.0:
            return True
    return False


def wilcoxon_fallback_p(pr_a, pr_b):
    """Two-sided paired Wilcoxon on per-question pass rates."""
    if len(pr_a) < 1 or np.allclose(pr_a, pr_b):
        return float("nan")
    try:
        return float(scipy_stats.wilcoxon(pr_a, pr_b, zero_method="wilcox",
                                          alternative="two-sided", method="auto").pvalue)
    except ValueError:
        return float("nan")


# ----------------------------- main -----------------------------

def run(clean_dir, contam_dir, exact_dir, label, out_csv, ckpt_step, backend_req, n_runs_hint):
    backend_name, fit_fn = pick_backend(backend_req)

    variants = {
        "clean": {s: load_long(clean_dir, s, ckpt_step, n_runs_hint) for s in ("contaminated", "clean")},
        "contaminated": {s: load_long(contam_dir, s, ckpt_step, n_runs_hint) for s in ("contaminated", "clean")},
    }
    if exact_dir:
        variants["exact"] = {s: load_long(exact_dir, s, ckpt_step, n_runs_hint) for s in ("contaminated", "clean")}

    rows = []
    for split in ("contaminated", "clean"):
        epoch_sets = [set(variants[m][split]["epoch"].unique()) for m in variants]
        epochs = sorted(set.intersection(*epoch_sets))

        print(f"\n{'='*132}")
        print(f"  {label}  |  test split = {split.upper()}  |  GLMM backend = {backend_name}  |  contrast ref = contaminated")
        print(f"{'='*132}")
        header = f"{'Ep':>3}  {'Clean acc % [95% CI]':>22}  {'Contam acc % [95% CI]':>22}"
        if "exact" in variants:
            header += f"  {'Exact acc % [95% CI]':>22}"
        header += f"  {'OR clean/contam [95%]':>22}  {'p (clean-contam)':>17}"
        if "exact" in variants:
            header += f"  {'OR exact/contam [95%]':>22}  {'p (exact-contam)':>17}"
        print(header)
        print("-" * len(header))

        for e in epochs:
            sub_by_variant = {m: variants[m][split][variants[m][split]["epoch"] == e] for m in variants}
            sid_sets = [set(sub_by_variant[m]["sample_id"].unique()) for m in variants]
            common_sids = sorted(set.intersection(*sid_sets))
            if not common_sids:
                continue

            ci_by_variant = {}
            for m in variants:
                s = sub_by_variant[m][sub_by_variant[m]["sample_id"].isin(common_sids)]
                ci_by_variant[m] = boot_mean_ci(per_question_pass_rates(s))

            stacked = []
            for m in variants:
                s = sub_by_variant[m][sub_by_variant[m]["sample_id"].isin(common_sids)].copy()
                s["model"] = m
                stacked.append(s[["sample_id", "model", "correct"]])
            df = pd.concat(stacked, ignore_index=True)

            try:
                coefs = fit_fn(df, ref_level="contaminated")
            except Exception as err:
                print(f"{e:>3}  [fit failed: {err}]")
                continue

            def find_coef(target):
                for k, v in coefs.items():
                    if k.endswith(target) or f"T.{target}" in k or k == f"model{target}":
                        return v
                return None

            cc = find_coef("clean")
            ec = find_coef("exact") if "exact" in variants else None

            def triplet(x):
                return (x[0], x[1], x[2], x[3], x[4], x[5]) if x else (np.nan,)*6

            est_cc, se_cc, p_cc, or_cc, or_cc_lo, or_cc_hi = triplet(cc)
            est_ec, se_ec, p_ec, or_ec, or_ec_lo, or_ec_hi = triplet(ec)

            # Per-trial binary arrays per variant for separation detection
            arr_by_variant = {m: sub_by_variant[m][sub_by_variant[m]["sample_id"].isin(common_sids)]["correct"].to_numpy()
                              for m in variants}
            # Per-question pass-rates for Wilcoxon fallback
            pr_by_variant = {m: per_question_pass_rates(
                                sub_by_variant[m][sub_by_variant[m]["sample_id"].isin(common_sids)])
                             for m in variants}

            sep_cc = is_perfect_separation(arr_by_variant["clean"], arr_by_variant["contaminated"])
            sep_ec = ("exact" in variants) and is_perfect_separation(
                arr_by_variant["exact"], arr_by_variant["contaminated"])

            if sep_cc:
                p_cc = wilcoxon_fallback_p(pr_by_variant["clean"], pr_by_variant["contaminated"])
                or_cc = or_cc_lo = or_cc_hi = float("nan")
                est_cc = se_cc = float("nan")
            if sep_ec:
                p_ec = wilcoxon_fallback_p(pr_by_variant["exact"], pr_by_variant["contaminated"])
                or_ec = or_ec_lo = or_ec_hi = float("nan")
                est_ec = se_ec = float("nan")

            line = (
                f"{e:>3}  {fmt_ci(*ci_by_variant['clean']):>22}"
                f"  {fmt_ci(*ci_by_variant['contaminated']):>22}"
            )
            if "exact" in variants:
                line += f"  {fmt_ci(*ci_by_variant['exact']):>22}"
            cc_or = "  perfect-sep (W)" if sep_cc else fmt_or(or_cc, or_cc_lo, or_cc_hi)
            line += f"  {cc_or:>22}  {fmt_p(p_cc):>11} {sig_marker(p_cc):>3}"
            if "exact" in variants:
                ec_or = "  perfect-sep (W)" if sep_ec else fmt_or(or_ec, or_ec_lo, or_ec_hi)
                line += f"  {ec_or:>22}  {fmt_p(p_ec):>11} {sig_marker(p_ec):>3}"
            print(line)

            row = {
                "model": label, "backend": backend_name, "split": split, "epoch": e,
                "n_questions": len(common_sids), "n_obs": len(df),
                "clean_mean": ci_by_variant["clean"][0],
                "clean_ci_lo": ci_by_variant["clean"][1],
                "clean_ci_hi": ci_by_variant["clean"][2],
                "contam_mean": ci_by_variant["contaminated"][0],
                "contam_ci_lo": ci_by_variant["contaminated"][1],
                "contam_ci_hi": ci_by_variant["contaminated"][2],
                "exact_mean": ci_by_variant.get("exact", (np.nan,)*3)[0],
                "exact_ci_lo": ci_by_variant.get("exact", (np.nan,)*3)[1],
                "exact_ci_hi": ci_by_variant.get("exact", (np.nan,)*3)[2],
                "logit_clean_vs_contam": est_cc, "se_clean_vs_contam": se_cc,
                "or_clean_vs_contam": or_cc, "or_cc_lo": or_cc_lo, "or_cc_hi": or_cc_hi,
                "p_clean_vs_contam": p_cc, "sig_clean_vs_contam": sig_marker(p_cc),
                "p_clean_vs_contam_method": "wilcoxon" if sep_cc else "glmm",
                "logit_exact_vs_contam": est_ec, "se_exact_vs_contam": se_ec,
                "or_exact_vs_contam": or_ec, "or_ec_lo": or_ec_lo, "or_ec_hi": or_ec_hi,
                "p_exact_vs_contam": p_ec, "sig_exact_vs_contam": sig_marker(p_ec),
                "p_exact_vs_contam_method": "wilcoxon" if sep_ec else "glmm",
            }
            rows.append(row)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nwrote {len(rows)} rows to {out_csv}")
    print("legend: *** p<0.001  ** p<0.01  * p<0.05")
    print("note: cells where one arm achieved 100% accuracy produced perfect separation;")
    print("      p-values from GLMM are undefined and Wilcoxon signed-rank is reported instead")
    print("      (marked 'perfect-sep (W)' in OR column; method recorded per row in CSV).")
    print(f"backend used: {backend_name}")
    if backend_name == "gee":
        print("note: GEE gives population-averaged estimates with cluster-robust SEs.")
        print("      Install pymer4 + R/lme4 to get a proper subject-specific GLMM.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean-dir", required=True)
    ap.add_argument("--contam-dir", required=True)
    ap.add_argument("--exact-dir", default=None)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--ckpt-step", type=int, default=CKPT_STEP_DEFAULT)
    ap.add_argument("--backend", choices=["auto", "glmer", "rscript", "pymer4", "gee"], default="auto")
    ap.add_argument("--n-runs", type=int, default=20)
    args = ap.parse_args()
    run(args.clean_dir, args.contam_dir, args.exact_dir, args.label,
        args.out_csv, args.ckpt_step, args.backend, args.n_runs)


if __name__ == "__main__":
    main()
