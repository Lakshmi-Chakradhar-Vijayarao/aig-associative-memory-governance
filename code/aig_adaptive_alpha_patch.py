"""
aig_adaptive_alpha_patch.py
Adaptive alpha benchmark: RIS-driven vs MeMo's fixed-alpha blend.

MeMo's retrieve() blends CMM retrieval with direct projection:
  return (retrieved + computed) / 2    ← hardcoded 0.5, no CMM-state awareness

AIG patch (3 lines in MeMoLayer.retrieve):
  ris   = compute_ris(stored_keys, query)
  alpha = ris                           ← high when CMM healthy, low when saturated
  return alpha * retrieved + (1-alpha) * computed

Setup: one anchor pattern (q, y*) plus P interfering patterns with cos ≈ 0.25–0.55.
Direct-projection quality = 0.85 (transformer baseline — independent of CMM state).
Quality metric = cos(blend(retrieved, proj), y*).

Patch location: MeMoLayer.retrieve(), line 283 of MeMoCMM.py
"""

import sys
import os
sys.path.insert(0, '/Users/chakrivijayarao/Desktop/MeMo-PhD-Workshop/code')
os.environ['PYTHONWARNINGS'] = 'ignore'

import torch
import numpy as np
from governed_memo_layer import GovernedMeMoLayer

SEED   = 42
D      = 128
P_MAX  = 100
N_RUNS = 30


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (same as experiment_memo_governance.py Panel C)
# ─────────────────────────────────────────────────────────────────────────────
def rand_unit(d, gen=None):
    v = torch.randn(d, generator=gen)
    return v / v.norm()


def make_interferer(q, cos_angle, gen):
    """Unit vector with prescribed cosine similarity to q."""
    perp = torch.randn(len(q), generator=gen)
    perp = perp - (perp @ q) * q
    perp = perp / (perp.norm() + 1e-12)
    sin_a = float(np.sqrt(max(0.0, 1.0 - cos_angle ** 2)))
    raw = cos_angle * q + sin_a * perp
    return raw / raw.norm()


def noisy_projection(y_star, gen, quality=0.85):
    """Direct-projection signal: quality-cos correlation with y_star."""
    noise = torch.randn(len(y_star), generator=gen)
    noise = noise - (noise @ y_star) * y_star
    noise = noise / (noise.norm() + 1e-12)
    sin_q = float(np.sqrt(max(0.0, 1.0 - quality ** 2)))
    raw = quality * y_star + sin_q * noise
    return raw / raw.norm()


# ─────────────────────────────────────────────────────────────────────────────
# Single-run Panel C
# ─────────────────────────────────────────────────────────────────────────────
def run_single(seed):
    gen = torch.Generator(); gen.manual_seed(seed)

    q      = rand_unit(D, gen)
    y_star = rand_unit(D, gen)
    proj   = noisy_projection(y_star, gen, quality=0.85)

    fixed_layer = GovernedMeMoLayer(d=D, cert_threshold=float("inf"),
                                    alpha_mode="fixed")
    adapt_layer = GovernedMeMoLayer(d=D, cert_threshold=float("inf"),
                                    alpha_mode="adaptive")

    for layer in [fixed_layer, adapt_layer]:
        layer.cmm.store(q.clone(), y_star.clone(), domain="anchor", label="")

    checkpoints = [10, 25, 50, 75, 100]
    fixed_q = {}
    adapt_q = {}
    adapt_a = {}

    gen2 = torch.Generator(); gen2.manual_seed(seed + 77)
    for step in range(P_MAX):
        cos = float(torch.rand(1, generator=gen2).item() * 0.30 + 0.25)
        s_i = make_interferer(q, cos, gen2)
        y_i = rand_unit(D, gen2)
        for layer in [fixed_layer, adapt_layer]:
            layer.cmm.store(s_i, y_i, domain="interferer", label="")

        p = step + 1
        if p in checkpoints:
            def blend_q(layer):
                result = layer.retrieve(q, proj)
                out    = result["output"]
                n      = out.norm()
                return (out @ y_star / n).item() if n > 1e-9 else 0.0

            fixed_q[p] = blend_q(fixed_layer)
            adapt_q[p] = blend_q(adapt_layer)
            adapt_a[p] = adapt_layer.retrieve(q, proj)["alpha"]

    return {"fixed": fixed_q, "adaptive": adapt_q, "alpha": adapt_a,
            "checkpoints": checkpoints}


# ─────────────────────────────────────────────────────────────────────────────
# Main benchmark
# ─────────────────────────────────────────────────────────────────────────────
def run_benchmark():
    print("=" * 70)
    print("AIG ADAPTIVE ALPHA PATCH — benchmark vs MeMo fixed blends")
    print(f"d={D}, P_max={P_MAX}, N_runs={N_RUNS}, proj_quality=0.85")
    print("=" * 70)
    print()

    results = [run_single(SEED + r) for r in range(N_RUNS)]
    cps     = results[0]["checkpoints"]

    def agg(key, p):
        vals = [r[key][p] for r in results if p in r[key]]
        return float(np.mean(vals))

    print(f"  {'P':>4}  {'Fixed α=0.5':>12}  {'Adaptive α':>12}  {'ᾱ(RIS)':>8}  {'Delta':>8}  {'Gain%':>8}")
    print(f"  {'-'*4}  {'-'*12}  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*8}")
    for p in cps:
        qf = agg("fixed", p)
        qa = agg("adaptive", p)
        a  = agg("alpha", p)
        delta = qa - qf
        gain  = (delta / (abs(qf) + 1e-12)) * 100
        print(f"  {p:>4}  {qf:>12.4f}  {qa:>12.4f}  {a:>8.4f}  {delta:>+8.4f}  {gain:>7.1f}%")

    final_p = cps[-1]
    qf_fin  = agg("fixed",    final_p)
    qa_fin  = agg("adaptive", final_p)
    a_fin   = agg("alpha",    final_p)
    gain_fin = ((qa_fin - qf_fin) / (abs(qf_fin) + 1e-12)) * 100

    print()
    print(f"  At P={final_p} (high interference):")
    print(f"    Fixed α=0.5 (MeMo):   {qf_fin:.4f}")
    print(f"    Adaptive α=RIS (AIG): {qa_fin:.4f}  (ᾱ={a_fin:.3f})")
    print(f"    Gain:                 {gain_fin:+.1f}%")

    print()
    print("  PATCH (3 lines in MeMoLayer.retrieve, line 283 of MeMoCMM.py):")
    print()
    print("  CURRENT (hardcoded 50/50, no state awareness):")
    print("    return (retrieved_sequence_encoding + computed_sequence_encoding) / 2")
    print()
    print("  PATCHED (RIS-adaptive, 3-line change):")
    print("    rsv   = [|cos(q, s_i)| * ||y_i|| for (s_i, y_i) in stored_keys]")
    print("    alpha = max(rsv) / sum(rsv)   # RIS")
    print("    return alpha * retrieved + (1 - alpha) * computed")
    print()
    print("  INTERPRETATION:")
    print("    RIS=1 → single dominant stored key → trust CMM retrieval (α→1)")
    print("    RIS→0 → many competing keys (saturated) → trust direct projection (α→0)")
    print("    Direct projection here = transformer's own computation (quality=0.85)")
    print("    MeMo's 0.5 blend ignores CMM saturation state — AIG makes it principled.")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
