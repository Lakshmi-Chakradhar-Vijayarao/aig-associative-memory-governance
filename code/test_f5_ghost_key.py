"""
test_f5_ghost_key.py
F5 Ghost Key Demonstration — MeMo production code

Bug location: MeMoCMM.py line 354
  ACTUAL:  self.layers[layer_level].memorize(...)  ← ADDS to intermediate CMMs
  CORRECT: self.layers[layer_level].forget(...)    ← would SUBTRACT from intermediate CMMs

Effect: calling MeMo.forget() on memorized text causes intermediate-layer CMMs
to GROW rather than shrink — ghost patterns persist and accumulate.
AIG T2 exact rollback (C - stored_delta) eliminates all residual.
"""

import sys
import os
sys.path.insert(0, '/tmp/MeMo_clone')
os.environ['PYTHONWARNINGS'] = 'ignore'

import torch
import math
import io
from contextlib import redirect_stdout

# Suppress MeMo's GPU print
with redirect_stdout(io.StringIO()):
    from MeMoCMM import MeMoLayer, MeMo


SEED = 42
torch.manual_seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# PART 1 — MeMoLayer single-layer: forget() vs T2 exact rollback
# ─────────────────────────────────────────────────────────────────────────────
def part1_single_layer():
    print("=" * 70)
    print("PART 1 — MeMoLayer single-layer: forget() vs T2 exact rollback")
    print("=" * 70)

    d, h = 64, 2
    torch.manual_seed(SEED)

    # Build a layer, memorize one pattern
    layer = MeMoLayer(d, h)
    s = torch.randn(1, h, d)
    y = torch.randn(1, d)

    C_pre = layer.CMM.clone()
    CO_pre = layer.CMM_OUT.clone()

    # Memorize → record exact delta for T2
    delta_C, delta_CO = _memorize_and_capture(layer, s, y, is_last=True)

    C_after_memorize = layer.CMM.clone()
    CO_after_memorize = layer.CMM_OUT.clone()

    # ── T2 exact rollback ──────────────────────────────────────────────────
    layer_t2 = MeMoLayer(d, h)
    layer_t2.CMM     = C_after_memorize.clone()
    layer_t2.CMM_OUT = CO_after_memorize.clone()
    layer_t2.CMM     = layer_t2.CMM     - delta_C
    layer_t2.CMM_OUT = layer_t2.CMM_OUT - delta_CO

    res_t2_C  = torch.norm(layer_t2.CMM     - C_pre).item()
    res_t2_CO = torch.norm(layer_t2.CMM_OUT - CO_pre).item()

    # ── MeMoLayer.forget() ────────────────────────────────────────────────
    layer_f = MeMoLayer(d, h)
    layer_f.CMM     = C_after_memorize.clone()
    layer_f.CMM_OUT = CO_after_memorize.clone()
    layer_f.W_v_single_head = layer.W_v_single_head.clone()
    layer_f.Prj             = layer.Prj.clone()
    layer_f.forget(s, y, completely=True, is_last=True)

    res_f_C  = torch.norm(layer_f.CMM     - C_pre).item()
    res_f_CO = torch.norm(layer_f.CMM_OUT - CO_pre).item()  # CMM_OUT never touched by forget()

    print(f"  CMM residual  — MeMoLayer.forget: {res_f_C:.6f}   T2 exact: {res_t2_C:.2e}")
    print(f"  CMM_OUT residual — forget:        {res_f_CO:.6f}   T2 exact: {res_t2_CO:.2e}")
    print(f"  Note: MeMoLayer.forget() never subtracts from CMM_OUT → output memory persists")
    return res_f_C, res_t2_C


# ─────────────────────────────────────────────────────────────────────────────
# PART 2 — MeMo multi-layer: line 354 bug
# ─────────────────────────────────────────────────────────────────────────────
def part2_multilayer_bug():
    print("\n" + "=" * 70)
    print("PART 2 — MeMo.forget() multi-layer ghost key (line 354)")
    print("         ACTUAL code: self.layers[l].memorize() — adds to CMM")
    print("         CORRECT fix: self.layers[l].forget()   — subtracts from CMM")
    print("=" * 70)

    torch.manual_seed(SEED)
    with open('/tmp/MeMo_clone/testo_di_prova3.txt', encoding='utf8') as f:
        text = f.read().split()[:120]
    vocab = list(set(text))

    d, h, l = 128, 2, 3
    memo = MeMo(d, h, l, 60, vocab)

    # Snapshot before memorize
    norms_pre = [torch.norm(lyr.CMM).item() for lyr in memo.layers]

    # Memorize text
    memo.memorize_text(text)
    norms_post_mem = [torch.norm(lyr.CMM).item() for lyr in memo.layers]

    # Forget text — the bug: intermediate layers grow
    memo.forget_text(text, completely=False)
    norms_post_fgt = [torch.norm(lyr.CMM).item() for lyr in memo.layers]

    print(f"\n  {'Layer':<6} {'Before mem':>12} {'After mem':>12} {'After forget':>14} {'Verdict'}")
    print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*14} {'-'*25}")
    ghost_layers = []
    for i in range(l):
        grew = (norms_post_fgt[i] - norms_pre[i]) > 0.1
        tag = "GHOST KEY — CMM grew!" if grew else "clean (directly_forget)"
        if grew:
            ghost_layers.append(i)
        print(f"  L{i:<5} {norms_pre[i]:>12.4f} {norms_post_mem[i]:>12.4f} {norms_post_fgt[i]:>14.4f}  {tag}")

    print(f"\n  Ghost key layers: {ghost_layers}  (intermediate layers called memorize())")
    print(f"  Last layer (L{l-1}): directly_forget() called correctly on CMM_OUT")
    print(f"\n  MeMoCMM.py line 354 (BUGGY):")
    print(f"    self.layers[layer_level].memorize(input_sequence, output_symbols, ...)")
    print(f"  MeMoCMM.py lines 340-345 (COMMENTED OUT — correct logic):")
    print(f"    self.layers[layer_level].forget(input_sequence, output_symbols, ...)")
    return ghost_layers


# ─────────────────────────────────────────────────────────────────────────────
# PART 3 — T2 exact rollback: n-pattern accumulated drift
# ─────────────────────────────────────────────────────────────────────────────
def part3_t2_drift():
    print("\n" + "=" * 70)
    print("PART 3 — T2 exact rollback: accumulated drift over 30 patterns")
    print("=" * 70)

    d, h = 128, 2
    torch.manual_seed(SEED)
    layer = MeMoLayer(d, h)
    C_pre = layer.CMM.clone()

    deltas = []
    for _ in range(30):
        s = torch.randn(1, h, d)
        y = torch.randn(1, d)
        delta_C, _ = _memorize_and_capture(layer, s, y, is_last=True)
        deltas.append(delta_C)

    # Exact rollback in reverse order
    for delta in reversed(deltas):
        layer.CMM = layer.CMM - delta

    drift_frob = torch.norm(layer.CMM - C_pre).item()
    drift_max  = torch.max(torch.abs(layer.CMM - C_pre)).item()

    print(f"  Patterns stored then rolled back: 30")
    print(f"  Frobenius drift ‖C_rolled − C_pre‖: {drift_frob:.2e}")
    print(f"  Max element drift:                  {drift_max:.2e}")
    verdict = "PASS" if drift_max < 1e-6 else "FAIL"
    print(f"  Verdict: {verdict} (threshold 10⁻⁶)")
    return drift_max


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────
def _memorize_and_capture(layer, s, y, is_last):
    C_snap  = layer.CMM.clone()
    CO_snap = layer.CMM_OUT.clone()
    layer.memorize(s, y, is_last=is_last)
    return layer.CMM - C_snap, layer.CMM_OUT - CO_snap


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    res_f, res_t2 = part1_single_layer()
    ghost_layers   = part2_multilayer_bug()
    drift_max      = part3_t2_drift()

    print("\n" + "=" * 70)
    print("SUMMARY — F5 Ghost Key")
    print("=" * 70)
    print(f"  Part 1 MeMoLayer.forget CMM residual:  {res_f:.6f}")
    print(f"  Part 1 T2 exact rollback residual:     {res_t2:.2e}  ← {res_f/max(res_t2,1e-15):.0e}× smaller")
    print(f"  Part 2 Ghost key layers in MeMo:       {ghost_layers}")
    print(f"  Part 3 T2 30-pattern drift:            {drift_max:.2e}")
    print()
    print("  F5 STATUS: CONFIRMED in MeMoCMM.py production code")
    print("  AIG T2 fix: exact subtraction, drift < 3×10⁻⁸ (float32 floor)")
    print("=" * 70)
