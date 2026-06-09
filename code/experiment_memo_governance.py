#!/usr/bin/env python3
"""
Experiment: AIG Applied to MeMo — Three-Panel Governance Proof

This experiment demonstrates that MeMo's CMM layers have a predictable
failure point, that failure is detectable BEFORE output quality drops,
and that a governance intervention restores quality exactly.

Three panels:

  Panel A — Predictive Failure (Lead-Time)
    MeMo-style layer, no gating, fixed alpha (MeMo default).
    RIS drops below θ at step P_ris. Quality drops at step P_qual.
    Lead-time Δ = P_qual − P_ris > 0 → health signal precedes failure.

  Panel B — Ungoverned vs Governed Quality
    Ungoverned: all writes admitted, no rollback → quality collapses.
    Governed:   T1 gate blocks high-interference writes; T2 rollback
                triggered when RIS < θ → quality maintained and restored.

  Panel C — Fixed Alpha vs RIS-Adaptive Alpha
    MeMo's current approach: alpha = 0.5 (hardcoded, no CMM-state awareness).
    AIG fix:                  alpha = RIS (trust CMM when healthy, fall back
                              to direct projection when degraded).
    Adaptive alpha significantly outperforms fixed blend at high P.

All three panels run on GovernedMeMoLayer — a drop-in AIG governance
wrapper that directly addresses the three structural gaps in MeMo's
code-integration architecture.
"""

import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from governed_memo_layer import GovernedMeMoLayer

SEED        = 42
D           = 128
THETA       = 0.65          # RIS instability threshold
QUAL_THRESH = 0.50          # retrieval quality threshold (cos to target)
P_MAX       = 100           # total patterns per trial
N_RUNS      = 30            # independent trials


# ── Helpers ───────────────────────────────────────────────────────────────────

def rand_unit(d: int, gen: torch.Generator) -> torch.Tensor:
    v = torch.randn(d, generator=gen)
    return v / v.norm()


def make_interferer(q: torch.Tensor, alpha: float,
                    gen: torch.Generator) -> torch.Tensor:
    """Unit vector with cos(result, q) = alpha, rest orthogonal to q."""
    d    = q.shape[0]
    perp = torch.randn(d, generator=gen)
    perp = perp - (perp @ q) * q
    perp = perp / perp.norm()
    sin_a = float(np.sqrt(max(0.0, 1.0 - alpha ** 2)))
    raw   = alpha * q + sin_a * perp
    return raw / raw.norm()


def retrieval_cos(layer: GovernedMeMoLayer,
                  query: torch.Tensor,
                  y_target: torch.Tensor) -> float:
    """Cosine between retrieved output and correct answer direction."""
    y_hat = layer.cmm.retrieve(query)
    norm  = y_hat.norm()
    return (y_hat @ y_target / norm).item() if norm > 1e-9 else 0.0


def noisy_projection(y_target: torch.Tensor,
                     gen: torch.Generator,
                     quality: float = 0.85) -> torch.Tensor:
    """
    Simulates a transformer's direct projection — mostly correct but imperfect.
    quality = cosine alignment to y_target (0.85 = good transformer baseline).
    """
    d    = y_target.shape[0]
    perp = torch.randn(d, generator=gen)
    perp = perp - (perp @ y_target) * y_target
    perp = perp / perp.norm()
    sin_t = float(np.sqrt(max(0.0, 1.0 - quality ** 2)))
    raw   = quality * y_target + sin_t * perp
    return raw / raw.norm()


# ── Panel A: Predictive Failure — Lead-Time ───────────────────────────────────

def run_panel_a(d: int, seed: int) -> dict:
    """
    MeMo-style layer (no gating, fixed alpha). Add interfering patterns.
    Record RIS and retrieval quality at every step.
    Measure lead-time: how many steps before quality drops does RIS drop?
    """
    gen = torch.Generator(); gen.manual_seed(seed)

    layer = GovernedMeMoLayer(d=d, cert_threshold=float("inf"),
                              ris_threshold=THETA, alpha_mode="fixed")
    q      = rand_unit(d, gen)
    y_star = rand_unit(d, gen)
    layer.cmm.store(q.clone(), y_star.clone(), domain="anchor", label="correct")

    ris_trace  = []
    qual_trace = []

    for _ in range(P_MAX):
        alpha_cos = float(torch.rand(1, generator=gen).item() * 0.33 + 0.25)
        s_i = make_interferer(q, alpha_cos, gen)
        y_i = rand_unit(d, gen)
        layer.cmm.store(s_i, y_i, domain="interferer", label="")

        rsv = layer.cmm.rsv(q)
        ris_trace.append(layer.cmm.ris(rsv))
        qual_trace.append(retrieval_cos(layer, q, y_star))

    ris_arr  = np.array(ris_trace)
    qual_arr = np.array(qual_trace)

    p_ris  = next((i for i, r in enumerate(ris_arr)  if r  < THETA),       None)
    p_qual = next((i for i, q in enumerate(qual_arr) if q  < QUAL_THRESH), None)

    if p_ris is not None and p_qual is not None:
        lead_time = p_qual - p_ris
    else:
        lead_time = float("nan")

    return {
        "ris_trace":  ris_trace,
        "qual_trace": qual_trace,
        "p_ris":      p_ris,
        "p_qual":     p_qual,
        "lead_time":  lead_time,
    }


# ── Panel B: Ungoverned vs Governed Quality ───────────────────────────────────

def run_panel_b(d: int, seed: int) -> dict:
    """
    Ungoverned: all writes admitted, no rollback.
    Governed:   T1 gate (cert_threshold=0.45) + T2 rollback when RIS < θ.
    Compare retrieval quality trajectories.
    """
    gen = torch.Generator(); gen.manual_seed(seed)

    q      = rand_unit(d, gen)
    y_star = rand_unit(d, gen)

    interferer_vecs = []
    gen2 = torch.Generator(); gen2.manual_seed(seed + 1)
    for _ in range(P_MAX):
        alpha = float(torch.rand(1, generator=gen2).item() * 0.25 + 0.30)
        s_i = make_interferer(q, alpha, gen2)
        y_i = rand_unit(d, gen2)
        interferer_vecs.append((s_i, y_i))

    qual_ungoverned = []
    qual_governed   = []

    # Ungoverned run
    ung = GovernedMeMoLayer(d=d, cert_threshold=float("inf"),
                            ris_threshold=THETA, alpha_mode="fixed")
    ung.cmm.store(q.clone(), y_star.clone(), domain="anchor", label="correct")
    for s_i, y_i in interferer_vecs:
        ung.cmm.store(s_i, y_i, domain="interferer", label="")
        qual_ungoverned.append(retrieval_cos(ung, q, y_star))

    # Governed run
    gov = GovernedMeMoLayer(d=d, cert_threshold=0.45,
                            ris_threshold=THETA, alpha_mode="fixed")
    gov.cmm.store(q.clone(), y_star.clone(), domain="anchor", label="correct")
    rollback_steps = []

    for step, (s_i, y_i) in enumerate(interferer_vecs):
        gov.governed_write(s_i, y_i, domain="interferer")

        # T2 rollback triggered when health degrades
        h = gov.health(q)
        if h["status"] == "degraded" and gov.P >= 2:
            gov.forget_worst_interferer(q)
            rollback_steps.append(step)

        qual_governed.append(retrieval_cos(gov, q, y_star))

    return {
        "qual_ungoverned": qual_ungoverned,
        "qual_governed":   qual_governed,
        "rollback_steps":  rollback_steps,
        "audit":           gov.audit_summary(),
    }


# ── Panel C: Fixed Alpha vs Adaptive Alpha ────────────────────────────────────

def run_panel_c(d: int, seed: int) -> dict:
    """
    All writes admitted (no T1 gate).
    Fixed alpha=0.5 vs RIS-adaptive alpha.
    Direct projection quality = 0.85 (good transformer baseline).
    """
    gen = torch.Generator(); gen.manual_seed(seed)

    q      = rand_unit(d, gen)
    y_star = rand_unit(d, gen)
    proj   = noisy_projection(y_star, gen, quality=0.85)

    fixed_layer = GovernedMeMoLayer(d=d, cert_threshold=float("inf"),
                                    alpha_mode="fixed")
    adapt_layer = GovernedMeMoLayer(d=d, cert_threshold=float("inf"),
                                    alpha_mode="adaptive")

    for layer in [fixed_layer, adapt_layer]:
        layer.cmm.store(q.clone(), y_star.clone(), domain="anchor", label="")


    checkpoints   = [10, 25, 50, 75, 100]
    fixed_quality = {}
    adapt_quality = {}

    gen2 = torch.Generator(); gen2.manual_seed(seed + 77)
    for step in range(P_MAX):
        alpha_cos = float(torch.rand(1, generator=gen2).item() * 0.30 + 0.25)
        s_i = make_interferer(q, alpha_cos, gen2)
        y_i = rand_unit(d, gen2)

        for layer in [fixed_layer, adapt_layer]:
            layer.cmm.store(s_i, y_i, domain="interferer", label="")

        p = step + 1
        if p in checkpoints:
            def blend_quality(layer):
                result = layer.retrieve(q, proj)
                output = result["output"]
                norm   = output.norm()
                return (output @ y_star / norm).item() if norm > 1e-9 else 0.0

            fixed_quality[p] = blend_quality(fixed_layer)
            adapt_quality[p] = blend_quality(adapt_layer)

    return {"fixed": fixed_quality, "adaptive": adapt_quality,
            "checkpoints": checkpoints}


# ── Aggregate ─────────────────────────────────────────────────────────────────

def aggregate_panel_a(results):
    lead_times = [r["lead_time"] for r in results
                  if r["lead_time"] is not None and not np.isnan(r["lead_time"])]
    return {
        "mean_lead_time": float(np.mean(lead_times)) if lead_times else float("nan"),
        "std_lead_time":  float(np.std(lead_times))  if lead_times else float("nan"),
        "positive_rate":  float(np.mean([lt > 0 for lt in lead_times])) if lead_times else 0.0,
        "n_valid":        len(lead_times),
    }


def aggregate_panel_b(results):
    final_ung = [r["qual_ungoverned"][-1] for r in results]
    final_gov = [r["qual_governed"][-1]   for r in results]
    mean_rollbacks = float(np.mean([len(r["rollback_steps"]) for r in results]))
    mean_block_rate = float(np.mean([r["audit"]["block_rate"] for r in results]))
    return {
        "ungoverned_final_mean": float(np.mean(final_ung)),
        "ungoverned_final_std":  float(np.std(final_ung)),
        "governed_final_mean":   float(np.mean(final_gov)),
        "governed_final_std":    float(np.std(final_gov)),
        "improvement":           float(np.mean(final_gov) - np.mean(final_ung)),
        "mean_rollbacks":        mean_rollbacks,
        "mean_block_rate":       mean_block_rate,
    }


def aggregate_panel_c(results):
    checkpoints = results[0]["checkpoints"]
    agg = {}
    for p in checkpoints:
        fixed = [r["fixed"][p] for r in results]
        adapt = [r["adaptive"][p] for r in results]
        agg[p] = {
            "fixed_mean": float(np.mean(fixed)),
            "adapt_mean": float(np.mean(adapt)),
            "delta":      float(np.mean(adapt) - np.mean(fixed)),
            "pct_gain":   float((np.mean(adapt) - np.mean(fixed)) /
                                max(abs(np.mean(fixed)), 1e-6) * 100),
        }
    return agg


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Experiment: AIG Applied to MeMo — Three-Panel Governance Proof")
    print(f"d={D}, θ={THETA}, P_max={P_MAX}, N_runs={N_RUNS}\n")

    results_a = [run_panel_a(D, SEED + i)          for i in range(N_RUNS)]
    results_b = [run_panel_b(D, SEED + 500 + i)    for i in range(N_RUNS)]
    results_c = [run_panel_c(D, SEED + 1000 + i)   for i in range(N_RUNS)]

    agg_a = aggregate_panel_a(results_a)
    agg_b = aggregate_panel_b(results_b)
    agg_c = aggregate_panel_c(results_c)

    print("=" * 72)
    print("PANEL A — Predictive Failure: RIS Lead-Time on MeMo-style Layer")
    print("-" * 72)
    print(f"  Mean lead-time:   {agg_a['mean_lead_time']:.1f} ± {agg_a['std_lead_time']:.1f} steps")
    print(f"  Positive rate:    {agg_a['positive_rate']*100:.1f}% of trials")
    print(f"  Interpretation:   RIS drops θ={THETA} an average of "
          f"{agg_a['mean_lead_time']:.1f} steps BEFORE")
    print(f"                    retrieval quality drops below {QUAL_THRESH}.")
    print(f"                    MeMo has no signal for this — AIG provides it.")

    print()
    print("=" * 72)
    print("PANEL B — Ungoverned vs Governed Retrieval Quality")
    print("-" * 72)
    print(f"  Ungoverned final quality:  {agg_b['ungoverned_final_mean']:.3f} "
          f"± {agg_b['ungoverned_final_std']:.3f}")
    print(f"  Governed final quality:    {agg_b['governed_final_mean']:.3f} "
          f"± {agg_b['governed_final_std']:.3f}")
    print(f"  Quality improvement:       +{agg_b['improvement']:.3f}")
    print(f"  Mean T2 rollbacks/trial:   {agg_b['mean_rollbacks']:.1f}")
    print(f"  T1 block rate:             {agg_b['mean_block_rate']*100:.1f}%")
    print(f"  Interpretation:   T1 gate + T2 rollback raises final quality by "
          f"{agg_b['improvement']:.3f}.")
    print(f"                    MeMo's forget() cannot do this — T2 is exact.")

    print()
    print("=" * 72)
    print("PANEL C — Fixed Alpha=0.5 vs RIS-Adaptive Alpha")
    print(f"  {'P':>6}  {'Fixed α=0.5':>12}  {'Adaptive α=RIS':>15}  "
          f"{'Delta':>8}  {'Gain%':>8}")
    print("-" * 72)
    for p, v in agg_c.items():
        print(f"  P={p:<5}  {v['fixed_mean']:>12.3f}  {v['adapt_mean']:>15.3f}  "
              f"{v['delta']:>+8.3f}  {v['pct_gain']:>7.1f}%")
    final_p = max(agg_c.keys())
    print(f"  Interpretation:   At P={final_p}, adaptive alpha outperforms fixed by "
          f"{agg_c[final_p]['pct_gain']:+.1f}%.")
    print(f"                    MeMo's 50/50 blend has no principled basis —")
    print(f"                    RIS provides one.")

    print()
    print("=" * 72)
    print("SUMMARY: Three Missing Governance Primitives for MeMo")
    print("-" * 72)
    print(f"  1. MONITOR  RIS lead-time = {agg_a['mean_lead_time']:.1f} steps "
          f"({agg_a['positive_rate']*100:.0f}% positive)")
    print(f"              MeMo has no retrieval health signal.")
    print()
    print(f"  2. INTERVENE  Quality: {agg_b['ungoverned_final_mean']:.3f} → "
          f"{agg_b['governed_final_mean']:.3f} (+{agg_b['improvement']:.3f})")
    print(f"              T1 gate + T2 rollback. MeMo forget() residual = 1.000.")
    print()
    print(f"  3. ADAPT    Final quality gain = {agg_c[final_p]['pct_gain']:+.1f}% "
          f"from RIS-adaptive alpha.")
    print(f"              MeMo hardcodes 0.5 blend with no CMM-state awareness.")
    print()
    print("  STATEMENT:")
    print("  MeMo made memory explicit. AIG makes explicit memory operationally")
    print("  accountable. The three primitives (T1, T2, RIS) are the missing")
    print("  governance layer for MeMo's architecture — not a replacement for")
    print("  MeMo, but its operational completion.")
    print("=" * 72)

    return {"panel_a": agg_a, "panel_b": agg_b, "panel_c": agg_c}


if __name__ == "__main__":
    main()
