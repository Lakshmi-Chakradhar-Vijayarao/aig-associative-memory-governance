#!/usr/bin/env python3
"""
Experiment E8 — Stability Lead-Time Study

Research question: Does RIS drop below the instability threshold BEFORE
retrieval quality measurably degrades? The 'stability lead time' (Delta steps)
is the key structural observable.

Protocol:
  1. Initialize CMM with dimension d
  2. Store correct pattern: s_0 = q (query), y_0 = random unit vector
  3. Add P_max interfering patterns one at a time
  4. After each storage step, record RIS(q) and cos(retrieve(q), y_0)
  5. Lead time = B - R where:
       R = first step with RIS < theta  (structural warning)
       B = first step with quality < quality_threshold  (behavioral failure)

A positive lead time confirms structural observability precedes behavioral
collapse -- the central predictive claim of the AIG framework.
"""
import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from cmm_core import CMM

# ── Hyperparameters ────────────────────────────────────────────────────────────
THETA           = 0.65   # RIS instability threshold (consistent with E5)
QUALITY_THRESH  = 0.75   # cos(retrieve, correct) below which retrieval is "failed"
P_MAX           = 80     # max interfering patterns per trial
N_TRIALS        = 30     # independent trials per condition
SEED_BASE       = 42

DIMS            = [64, 128, 256]
# alpha controls how strongly interferers align with query:
#   s_interferer ∝ alpha*q + sqrt(1-alpha^2)*perp  → cos(s_i, q) ≈ alpha
COS_CONDITIONS  = {"low": 0.25, "medium": 0.42, "high": 0.58}

OUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

PAPER_FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "paper", "figures")
os.makedirs(PAPER_FIG_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def rand_unit(d: int, generator=None) -> torch.Tensor:
    v = torch.randn(d, generator=generator)
    return v / v.norm()


def make_interferer_key(q: torch.Tensor, alpha: float, generator=None) -> torch.Tensor:
    """Unit vector with cos(result, q) ≈ alpha via Gram-Schmidt mixing."""
    d = q.shape[0]
    perp = torch.randn(d, generator=generator)
    perp = perp - (perp @ q) * q          # project out q component
    perp = perp / perp.norm()             # unit norm perpendicular component
    raw = alpha * q + (1.0 - alpha) * perp
    return raw / raw.norm()


def run_trial(d: int, alpha: float, seed: int):
    """Single trial. Returns (ris_trace, quality_trace) arrays of length P_MAX."""
    gen = torch.Generator()
    gen.manual_seed(seed)

    cmm = CMM(d=d)
    q = rand_unit(d, generator=gen)
    y_correct = rand_unit(d, generator=gen)

    # Store the correct association first (domain="correct")
    cmm.store(q.clone(), y_correct.clone(), domain="correct", label="target")

    ris_trace     = []
    quality_trace = []

    for p in range(P_MAX):
        # Add one interfering pattern
        s_i = make_interferer_key(q, alpha, generator=gen)
        y_i = rand_unit(d, generator=gen)
        cmm.store(s_i, y_i, domain=f"interferer_{p}", label="")

        # Measure RIS
        rsv_vec = cmm.rsv(q)
        ris_val = cmm.ris(rsv_vec)

        # Measure retrieval quality: cosine similarity of output to y_correct
        retrieved = cmm.retrieve(q)
        norm = retrieved.norm()
        if norm > 1e-9:
            quality = (retrieved @ y_correct / norm).item()
        else:
            quality = 0.0

        ris_trace.append(ris_val)
        quality_trace.append(quality)

    return np.array(ris_trace), np.array(quality_trace)


def find_crossing(trace: np.ndarray, threshold: float, below: bool) -> int:
    """Return first index where trace crosses threshold. -1 if never crossed."""
    for i, v in enumerate(trace):
        if below and v < threshold:
            return i
        if not below and v > threshold:
            return i
    return -1


# ── Main experiment ───────────────────────────────────────────────────────────

def run_all():
    results = {}   # (d, cos_label) → list of lead times (NaN if undefined)

    # Also save one representative trial per condition for the trajectory plot
    rep_trials = {}

    print(f"Running E8: {len(DIMS)} dims × {len(COS_CONDITIONS)} cos levels × {N_TRIALS} trials")
    print(f"P_MAX={P_MAX}, theta={THETA}, quality_thresh={QUALITY_THRESH}\n")

    for d in DIMS:
        for cos_label, alpha in COS_CONDITIONS.items():
            key = (d, cos_label)
            lead_times = []
            for trial in range(N_TRIALS):
                seed = SEED_BASE + d * 1000 + trial
                ris_tr, qual_tr = run_trial(d, alpha, seed)

                R = find_crossing(ris_tr,     THETA,          below=True)
                B = find_crossing(qual_tr,    QUALITY_THRESH, below=True)

                if R == -1 or B == -1:
                    lead_times.append(np.nan)
                else:
                    lead_times.append(B - R)

                # Save representative trial (trial 0)
                if trial == 0:
                    rep_trials[key] = (ris_tr, qual_tr,
                                       find_crossing(ris_tr, THETA, below=True),
                                       find_crossing(qual_tr, QUALITY_THRESH, below=True))

            results[key] = np.array(lead_times, dtype=float)

    return results, rep_trials


def print_results(results):
    print("=== Stability Lead-Time Results ===")
    header = f"{'Condition':<22} | {'N':>4} | {'Mean':>7} | {'Std':>6} | {'Min':>5} | {'Max':>5} | {'Positive%':>9}"
    print(header)
    print("-" * len(header))

    all_finite = []
    for d in DIMS:
        for cos_label in COS_CONDITIONS:
            lt = results[(d, cos_label)]
            finite = lt[~np.isnan(lt)]
            all_finite.extend(finite.tolist())
            n_total  = len(lt)
            n_finite = len(finite)
            if n_finite == 0:
                print(f"  d={d:<3} cos={cos_label:<6} | {n_total:>4} | {'N/A':>7} | {'N/A':>6} | {'N/A':>5} | {'N/A':>5} | {'N/A':>9}")
                continue
            pct_pos  = 100.0 * np.sum(finite > 0) / n_total
            print(f"  d={d:<3} cos={cos_label:<6} | {n_total:>4} | {np.mean(finite):>7.2f} | {np.std(finite):>6.2f} | "
                  f"{np.min(finite):>5.0f} | {np.max(finite):>5.0f} | {pct_pos:>8.1f}%")

    all_arr = np.array(all_finite)
    if len(all_arr) > 0:
        overall_pos = 100.0 * np.sum(all_arr > 0) / len(all_arr)
        print(f"\nOverall: {overall_pos:.1f}% of defined trials show positive lead time "
              f"(structural precedes behavioral)\n"
              f"Mean lead time across all conditions: {np.mean(all_arr):.2f} ± {np.std(all_arr):.2f} steps")


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot(results, rep_trials):
    # Pick one representative condition for the trajectory panels
    rep_key = (128, "medium")
    ris_tr, qual_tr, R, B = rep_trials[rep_key]
    steps = np.arange(1, P_MAX + 1)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # ── Panel 1: RIS trajectory ──────────────────────────────────────────────
    ax = axes[0]
    ax.plot(steps, ris_tr, color="#2166ac", lw=2, label="RIS")
    ax.axhline(THETA, color="#2166ac", lw=1.2, ls="--", alpha=0.7, label=f"θ={THETA}")
    if R != -1 and B != -1 and B > R:
        ax.axvspan(R + 1, B + 1, alpha=0.15, color="gray", label=f"Lead time Δ={B-R}")
    if R != -1:
        ax.axvline(R + 1, color="#2166ac", lw=1, ls=":", alpha=0.8)
    ax.set_xlabel("Storage step P")
    ax.set_ylabel("RIS")
    ax.set_title(f"RIS trajectory  (d=128, cos≈{COS_CONDITIONS['medium']:.2f})")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    # ── Panel 2: Quality trajectory ───────────────────────────────────────────
    ax = axes[1]
    ax.plot(steps, qual_tr, color="#d6604d", lw=2, label="cos(retrieve, correct)")
    ax.axhline(QUALITY_THRESH, color="#d6604d", lw=1.2, ls="--", alpha=0.7,
               label=f"τ={QUALITY_THRESH}")
    if R != -1 and B != -1 and B > R:
        ax.axvspan(R + 1, B + 1, alpha=0.15, color="gray",
                   label=f"Lead window Δ={B-R}")
    if B != -1:
        ax.axvline(B + 1, color="#d6604d", lw=1, ls=":", alpha=0.8)
    ax.set_xlabel("Storage step P")
    ax.set_ylabel("Retrieval quality")
    ax.set_title("Retrieval quality trajectory")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    # ── Panel 3: Lead-time distributions ─────────────────────────────────────
    ax = axes[2]
    colors_d = {64: "#b2df8a", 128: "#33a02c", 256: "#006d2c"}
    cos_labels = list(COS_CONDITIONS.keys())
    x_positions = np.arange(len(cos_labels))
    width = 0.25

    for i, d in enumerate(DIMS):
        means, stds = [], []
        for cos_label in cos_labels:
            lt = results[(d, cos_label)]
            finite = lt[~np.isnan(lt)]
            means.append(np.mean(finite) if len(finite) else 0)
            stds.append(np.std(finite) if len(finite) else 0)
        offset = (i - 1) * width
        ax.bar(x_positions + offset, means, width,
               yerr=stds, capsize=3,
               color=colors_d[d], label=f"d={d}", alpha=0.85)

    ax.axhline(0, color="black", lw=0.8, ls="-")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"cos≈{COS_CONDITIONS[c]:.2f}\n({c})" for c in cos_labels])
    ax.set_ylabel("Mean lead time Δ (steps)")
    ax.set_title("Lead time by condition\n(positive = structural precedes behavioral)")
    ax.legend(fontsize=8, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Experiment E8: Stability Lead-Time Study\n"
        r"$\Delta = B - R$ where $R$: RIS$<\theta$, $B$: quality$<\tau$",
        fontsize=11, y=1.01
    )

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "fig_e8_stability_lead_time.pdf")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    paper_path = os.path.join(PAPER_FIG_DIR, "fig_e8_stability_lead_time.pdf")
    plt.savefig(paper_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"\nFigure saved → {out_path}")
    print(f"Figure saved → {paper_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results, rep_trials = run_all()
    print_results(results)
    plot(results, rep_trials)
    print("\nE8 complete.")
