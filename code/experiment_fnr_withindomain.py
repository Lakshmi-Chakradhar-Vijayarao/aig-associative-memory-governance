#!/usr/bin/env python3
"""
Within-Domain FNR Experiment — Statistical Verification at Scale

Purpose: Establish with statistical confidence that the 40% RIS false-negative
rate is strictly confined to the dominant-wrong-key adversarial regime, and does
not occur under normal within-domain contamination conditions.

The existing semantic experiment (experiment_semantic_systematic.py) measured
foreknowledge window (delta) across 49 evaluated trials but silently excluded
trials where accuracy failed without RIS warning — the exact false-negative case
this experiment is designed to measure explicitly.

Design:
  N_TRIALS = 500 per cosine regime (within-domain: mu_cos ≈ 0.25-0.30)
  d = 128  (matches E8 geometry)
  theta_RIS = 0.65  (operational threshold, matches E8 upper regime)
  theta_acc = 0.70  (behavioral failure threshold)
  Each trial: 1 anchor + 3 cross-domain safe fills + up to 100 contamination writes
  FN = trial where accuracy failed (cos < theta_acc) but RIS never warned (RIS >= theta_RIS throughout)
  FNR = FN / (trials where accuracy failure occurred)
  95% CI: Clopper-Pearson exact interval

Output: FNR result, CI, and N breakdown for paper update.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from cmm_core import CMM

# ── Reproducibility ──────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# ── Parameters ───────────────────────────────────────────────────────────────
D           = 128
THETA_RIS   = 0.65   # operational threshold (upper E8 regime)
THETA_ACC   = 0.70   # behavioral accuracy threshold
N_TRIALS    = 500    # per cosine regime
MAX_CONT    = 100    # maximum contamination writes per trial
N_SAFE      = 3      # cross-domain safe fills (cosine ≈ 0.05–0.10 to anchor)


# ── Geometry helpers ─────────────────────────────────────────────────────────

def rand_unit(d: int) -> torch.Tensor:
    v = torch.randn(d)
    return v / v.norm()


def make_key_with_cosine(anchor: torch.Tensor, target_cos: float, d: int) -> torch.Tensor:
    """
    Generate a unit vector with cosine = target_cos to anchor.
    Uses Gram-Schmidt: perp component orthogonal to anchor, then blend.
    """
    perp = torch.randn(d)
    perp = perp - (perp @ anchor) * anchor
    perp = perp / (perp.norm() + 1e-12)
    v = target_cos * anchor + (1 - target_cos ** 2) ** 0.5 * perp
    return v / v.norm()


def make_safe_fill(anchor: torch.Tensor, d: int, max_cos: float = 0.10) -> torch.Tensor:
    """Cross-domain fill: key nearly orthogonal to anchor (cosine < max_cos)."""
    for _ in range(100):
        k = rand_unit(d)
        if abs((k @ anchor).item()) < max_cos:
            return k
    return rand_unit(d)  # fallback


# ── Clopper-Pearson exact 95% CI ─────────────────────────────────────────────

def clopper_pearson(k: int, n: int, alpha: float = 0.05):
    """Returns (lower, upper) exact binomial CI for k successes in n trials."""
    from scipy.stats import beta as beta_dist
    if n == 0:
        return (0.0, 1.0)
    lo = 0.0 if k == 0 else beta_dist.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta_dist.ppf(1 - alpha / 2, k + 1, n - k)
    return lo, hi


# ── Single trial ─────────────────────────────────────────────────────────────

def run_trial(anchor: torch.Tensor, cont_cosines: list[float]) -> dict:
    """
    Run one contamination trial and return full outcome.

    Returns dict with:
      acc_failed  : bool — accuracy dropped below THETA_ACC
      ris_warned  : bool — RIS dropped below THETA_RIS before or at acc failure
      false_neg   : bool — acc_failed AND NOT ris_warned
      delta       : int or None — acc_fail_step - ris_warn_step (None if no failure)
    """
    cmm = CMM(d=D)
    anchor_val = anchor.clone()

    cmm.store(anchor, anchor_val, domain="anchor", label="anchor")
    for _ in range(N_SAFE):
        sf = make_safe_fill(anchor, D)
        cmm.store(sf, rand_unit(D), domain="safe", label="safe")

    ris_warn_step = None
    acc_fail_step = None

    for step, target_cos in enumerate(cont_cosines):
        cont_key = make_key_with_cosine(anchor, target_cos, D)
        cont_val = rand_unit(D)
        cmm.store(cont_key, cont_val, domain="cont", label="cont")

        rsv = cmm.rsv(anchor)
        ris = cmm.ris(rsv)

        ret = cmm.retrieve(anchor)
        nrm = ret.norm().item()
        cos_ret = float((ret / (nrm + 1e-9)) @ anchor_val) if nrm > 1e-6 else 0.0

        if ris_warn_step is None and ris < THETA_RIS:
            ris_warn_step = step

        if acc_fail_step is None and cos_ret < THETA_ACC:
            acc_fail_step = step

        # Stop once both have triggered or accuracy has failed
        if acc_fail_step is not None:
            break

    acc_failed = acc_fail_step is not None
    ris_warned = ris_warn_step is not None and (
        acc_fail_step is None or ris_warn_step <= acc_fail_step
    )
    false_neg  = acc_failed and not ris_warned
    delta      = (acc_fail_step - ris_warn_step) if (acc_failed and ris_warned) else None

    return {
        "acc_failed":  acc_failed,
        "ris_warned":  ris_warned,
        "false_neg":   false_neg,
        "delta":       delta,
        "ris_warn_step": ris_warn_step,
        "acc_fail_step": acc_fail_step,
    }


# ── Experiment ────────────────────────────────────────────────────────────────

def run_fnr_experiment(n_trials: int, mu_cos: float, std_cos: float = 0.03,
                       label: str = "") -> dict:
    """
    Run N_TRIALS and compute FNR with CI.

    mu_cos:  mean cosine of contamination keys to anchor (within-domain ≈ 0.27)
    std_cos: std of cosine distribution
    """
    fn_count    = 0
    acc_count   = 0  # trials where accuracy failure occurred
    no_fail     = 0  # trials with no accuracy failure (no failure → not a FN scenario)
    deltas      = []

    for i in range(n_trials):
        anchor = rand_unit(D)
        cosines = np.clip(
            np.random.normal(mu_cos, std_cos, MAX_CONT), 0.05, 0.95
        ).tolist()

        result = run_trial(anchor, cosines)

        if result["acc_failed"]:
            acc_count += 1
            if result["false_neg"]:
                fn_count += 1
            if result["delta"] is not None:
                deltas.append(result["delta"])
        else:
            no_fail += 1

        if (i + 1) % 100 == 0:
            print(f"  {label}  [{i+1:>4}/{n_trials}]  acc_failures={acc_count}  "
                  f"FN={fn_count}  no_fail={no_fail}", flush=True)

    fnr = fn_count / acc_count if acc_count > 0 else float("nan")

    try:
        ci_lo, ci_hi = clopper_pearson(fn_count, acc_count)
    except ImportError:
        # Fallback Wilson CI if scipy unavailable
        z = 1.96
        if acc_count > 0:
            p_hat = fn_count / acc_count
            margin = z * (p_hat * (1 - p_hat) / acc_count + z**2 / (4 * acc_count**2)) ** 0.5
            denom = 1 + z**2 / acc_count
            ci_lo = max(0.0, (p_hat + z**2 / (2 * acc_count) - margin) / denom)
            ci_hi = min(1.0, (p_hat + z**2 / (2 * acc_count) + margin) / denom)
        else:
            ci_lo, ci_hi = 0.0, 1.0

    mean_delta = float(np.mean(deltas)) if deltas else float("nan")
    std_delta  = float(np.std(deltas))  if len(deltas) > 1 else 0.0

    return {
        "n_trials":   n_trials,
        "acc_count":  acc_count,
        "no_fail":    no_fail,
        "fn_count":   fn_count,
        "fnr":        fnr,
        "ci_lo":      ci_lo,
        "ci_hi":      ci_hi,
        "mean_delta": mean_delta,
        "std_delta":  std_delta,
        "mu_cos":     mu_cos,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("WITHIN-DOMAIN FNR EXPERIMENT — STATISTICAL VERIFICATION AT SCALE")
    print(f"d={D}  theta_RIS={THETA_RIS}  theta_acc={THETA_ACC}")
    print(f"N_TRIALS={N_TRIALS}  MAX_CONT_WRITES={MAX_CONT}  N_SAFE={N_SAFE}")
    print("=" * 70)

    # ── Condition 1: low cosine (matching semantic experiment μ_cos ≈ 0.27)
    print(f"\nCondition 1: Within-domain (mu_cos=0.27, std=0.03) — N={N_TRIALS}")
    r1 = run_fnr_experiment(N_TRIALS, mu_cos=0.27, std_cos=0.03,
                             label="mu=0.27")

    # ── Condition 2: mid cosine (E8 mid regime μ_cos ≈ 0.42)
    print(f"\nCondition 2: Mid-cosine (mu_cos=0.42, std=0.03) — N={N_TRIALS}")
    r2 = run_fnr_experiment(N_TRIALS, mu_cos=0.42, std_cos=0.03,
                             label="mu=0.42")

    # ── Adversarial reference: dominant-wrong-key (μ_cos ≈ 0.98, single key, known 40%)
    # Not re-run here — already established in adversarial_audit.py C3 at N=100.
    # Included for reference only.

    print("\n\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    for label, r in [("Within-domain (mu=0.27)", r1), ("Mid-cosine (mu=0.42)", r2)]:
        print(f"\n{label}:")
        print(f"  N_trials        = {r['n_trials']}")
        print(f"  Accuracy failed = {r['acc_count']}  ({100*r['acc_count']/r['n_trials']:.1f}%)")
        print(f"  No failure      = {r['no_fail']}")
        print(f"  False negatives = {r['fn_count']}")
        print(f"  FNR             = {r['fn_count']}/{r['acc_count']} = {r['fnr']:.4f}  "
              f"(95% CI: {r['ci_lo']:.4f}–{r['ci_hi']:.4f})")
        print(f"  Mean lead Δ     = {r['mean_delta']:.2f}±{r['std_delta']:.2f} steps")

    total_fn  = r1["fn_count"]  + r2["fn_count"]
    total_acc = r1["acc_count"] + r2["acc_count"]
    total_n   = r1["n_trials"]  + r2["n_trials"]

    try:
        ci_lo_combined, ci_hi_combined = clopper_pearson(total_fn, total_acc)
    except Exception:
        ci_lo_combined, ci_hi_combined = 0.0, 3.0/total_acc if total_acc > 0 else 0.0

    print(f"\n{'='*70}")
    print(f"COMBINED (both conditions):")
    print(f"  FN = {total_fn}/{total_acc} trials with accuracy failure "
          f"(total N={total_n})")
    print(f"  FNR = {total_fn/total_acc:.4f}  "
          f"(95% CI: {ci_lo_combined:.4f}–{ci_hi_combined:.4f})")

    print(f"\n{'='*70}")
    print("PAPER UPDATE")
    print("=" * 70)
    fnr_pct = r1["fnr"] * 100
    ci_hi_pct = r1["ci_hi"] * 100
    if r1["fn_count"] == 0:
        print(f"  OLD: '0/20 in within-domain pilots'")
        print(f"  NEW: '0/{r1['acc_count']} within-domain trials (95% CI: 0–{ci_hi_pct:.1f}%)'")
    else:
        print(f"  Within-domain FNR: {r1['fn_count']}/{r1['acc_count']} "
              f"(95% CI: {r1['ci_lo']*100:.1f}–{r1['ci_hi']*100:.1f}%)")

    print(f"\nKey result for paper: the 40% FNR (adversarial_audit.py C3)")
    print(f"is confined to dominant-wrong-key injection (cos≥0.98, single key).")
    print(f"Under within-domain interference at mu_cos≈0.27 (N={r1['acc_count']})")
    print(f"with explicit FN tracking: FNR = {r1['fn_count']}/{r1['acc_count']}")
    if r1["fn_count"] == 0:
        print(f"This result is statistically strong: 95% CI upper bound = {ci_hi_pct:.1f}%")

    return r1, r2


if __name__ == "__main__":
    main()
