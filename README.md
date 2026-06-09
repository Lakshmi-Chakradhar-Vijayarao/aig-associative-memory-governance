# AIG: Towards a Mechanistic Science of Retrieval Integrity in Associative Memory Systems

**MeMo Workshop on Mechanistic Interpretability & Neuro-symbolic Approaches by-design**
Rome, Italy — June 24, 2026

**Author:** Lakshmi Chakradhar Vijayarao
**Paper:** [`paper/AIG_MEMO_WORKSHOP.pdf`](paper/AIG_MEMO_WORKSHOP.pdf)
**Contact:** lakshmichakradhar.v@gmail.com

---

## The core observation

Associative memory fails in a predictable sequence — and the warning comes *before* anything goes wrong in the output.

When interfering patterns accumulate in a Correlation Matrix Memory (CMM), three things happen in order:

1. **Retrieval geometry degrades** — the Retrieval Isolation Score (RIS) drops as competing patterns erode one pattern's dominance over the output
2. **Retrieval quality drifts** — cosine similarity to the target begins to slip
3. **Accuracy collapses** — the output becomes wrong

The gap between step 1 and step 3 — the **foreknowledge window** — is the entire contribution of this paper. RIS warns **3.15 ± 2.78 steps before behavioral failure in 94.4% of 270 controlled trials**. The memory is geometrically failing while outputs still appear correct. Once you can observe that gap, you can act on it.

---

## Why this matters for MeMo

[MeMo](https://aclanthology.org/2025.findings-acl.854/) builds language model layers from explicit Correlation Matrix Memories — token sequences stored as direct outer-product associations rather than distributed weight patterns. Because each association is a separable rank-1 outer product, CMM operations have a structural property that parametric models do not: their consequences are **algebraically computable before they execute**.

This repository implements that idea as **AIG (Associative Inference Governor)** and **GovernedMeMoLayer** — a drop-in replacement for MeMo's `MeMoLayer` that adds pre-execution interference certificates, exact algebraic rollback, and live geometric health monitoring.

The central open problem — **F5** — is that single-layer certificates do not compose across MeMo's stacked layers. Ghost keys persist after rollback. This is the PhD thesis target; the production-code demonstration is in `test_f5_ghost_key.py`.

---

## Key results

| Experiment | Result | File |
|---|---|---|
| Foreknowledge positive rate | **94.4%** (255/270; 95% CI: 91–97%) | `experiment_e8_stability_lead_time.py` |
| Mean lead time | **3.15 ± 2.78 steps** before failure | same |
| Decay law | Δ ∝ μ_cos^{−2.16}, **R²=0.979** | same |
| Semantic validation | **100%** / 49 trials / 3 embedding families | `experiment_semantic_systematic.py` |
| MeMo quality recovery (P1+P2) | 0.235 → **0.957** (+0.722) | `experiment_memo_governance.py` |
| Adaptive α vs fixed α=0.5 | **+109.2%** at P=100 | same |
| P1 algebraic bound | Within **2×10⁻⁶** (float32 floor) / 3100 trials | `experiment3_certificates.py` |
| P2 rollback drift | **<10⁻⁷** float32 / 10⁴ cycles | `test_f5_ghost_key.py` |
| Within-domain FNR | **0/500** (95% CI: 0–0.7%; explicit tracking) | `experiment_fnr_withindomain.py` |
| F5 amplification | **1.98× max** (1.07× mean) across layers | `test_f5_ghost_key.py` |

---

## What is in this repository

### `code/cmm_core.py` — the CMM substrate

A clean, minimal implementation of the Correlation Matrix Memory: `store()`, `retrieve()`, `forget()`, `rsv()`, and `ris()`. Every experiment in this repo imports from here. Understand this file first — everything else builds on it.

```python
cmm = CMM(d=128)
cmm.store(key, value)
ris = cmm.ris(cmm.rsv(query))   # geometric health score ∈ [0,1]
retrieved = cmm.retrieve(query)
```

### `code/governed_memo_layer.py` — GovernedMeMoLayer

A drop-in replacement for MeMo's `MeMoLayer`. Before each write, it:

1. Computes ε_j = |cos(s_j, s_n)| · ‖y_n‖ for all existing patterns — the **pre-execution certificate (P1)**
2. Estimates RIS from the current memory state
3. Makes one of three decisions:
   - **Block**: if max interference exceeds the certificate threshold, or per-anchor budget is exhausted
   - **Reduce trust**: if RIS drops below θ_RIS, α is reduced via σ(k(RIS − θ)) — the adaptive α (P4)
   - **Admit**: write proceeds; the association is logged for exact rollback (P2)

Exact rollback subtracts the outer product algebraically: `C ← C − s_n y_n^T`. Drift is under 3×10⁻⁸ in float32 — verified over 10,000 cycles.

### `code/experiment_e8_stability_lead_time.py` — the headline experiment

270 trials across d ∈ {64, 128, 256} and interferer cosine ≈ {0.25, 0.42, 0.58}. Each trial: store an anchor, write interfering patterns one at a time, track RIS and accuracy at every step. Records the gap (Δ) between when RIS first crosses θ and when accuracy first fails.

Key outputs:
- Positive rate: 94.4% (RIS warns before failure)
- Mean lead: 3.15 ± 2.78 steps
- Decay law: Δ ∝ μ_cos^{−2.16}, R²=0.979
- Threshold stability: 94.4% at both θ=0.65 and θ=0.75; drops to 59.3% at θ=0.55

### `code/experiment_memo_governance.py` — MeMo integration (Panels A, B, C)

Three experiments on a MeMo-style CMM layer (d=128, N=30 trials):

- **(A) Predictive failure**: RIS drops 15.9 ± 6.7 steps before quality drops below 0.50 in 100% of trials
- **(B) Intervention efficacy**: ungoverned quality reaches 0.235; P1 gating + P2 rollback recovers to 0.957 (+0.722)
- **(C) Adaptive α**: RIS-driven α outperforms fixed α=0.5 by +109.2% at P=100 (transformer proxy quality=0.85)

This is the authoritative source for the +109.2% number. A different seed gives ≈103–105%; seed=42 is used throughout for reproducibility.

### `code/experiment_semantic_systematic.py` — semantic validation (Table 1)

Tests whether the foreknowledge window survives real sentence embeddings. Three model families (all-MiniLM-L6-v2, all-mpnet-base-v2, paraphrase-MiniLM-L6), four domains, 60 contamination writes per trial. Result: 100% positive rate across all 49 evaluated trials. Mean Δ = 26.5 ± 2.7 steps at μ_cos ≈ 0.27, consistent with the decay law.

The semantic inflation of P3's random-geometry bound reaches 7–9× for BERT-class models (d=768) due to cluster structure in embedding space. This is expected and disclosed — P3 is a random-geometry baseline; semantic keys need per-family recalibration.

### `code/experiment_fnr_withindomain.py` — false-negative rate at scale

Explicitly tracks every trial outcome to separate false negatives from the "no failure" case (which earlier experiments silently conflated). N=500 per cosine regime.

- Within-domain (μ_cos=0.27): **0/500 FN** (95% CI: 0–0.7%)
- Mid-cosine (μ_cos=0.42): **0/500 FN** (95% CI: 0–0.7%)
- Combined: **0/1000** (95% CI: 0–0.4%)

The 40% adversarial false-negative rate (dominant-wrong-key injection, cos ≥ 0.98) is real and documented. These results show it is confined to that specific attack regime. Under normal operational interference, the false-negative rate is statistically indistinguishable from zero.

### `code/test_f5_ghost_key.py` — the F5 production bug

The most important file in the repo for the PhD research direction.

**What it shows:** MeMo's `MeMoCMM.py` line 354 calls `self.layers[layer_level].memorize(...)` where it should call `.forget()`. This means intermediate CMM layers *accumulate* associations during forget operations rather than removing them.

**Demonstrated outcome** (on a cloned copy of MeMo):
```
Before forget_text():  L0 CMM norm = 0,    L1 CMM norm = 0
After  forget_text():  L0 CMM norm = 7.51, L1 CMM norm = 8.06
                       L2 CMM norm = 0     ← clean (bug is layer-specific)
```

Ghost keys — derived keys `f(s₁) = s₁ᵀC₁` — persist in downstream layers after L1 rollback because `C₁` at store-time and `C₁` after rollback are different matrices. Correcting line 354 fixes the interface. It does not solve the mathematical obstruction: the stored key in L2 was computed from the *original* `C₁`, not the rolled-back one. **This is F5** — the central open problem and Year 1 PhD thesis target.

```bash
# Requires MeMo production code
git clone https://github.com/zanppa/MeMo /tmp/MeMo_clone
python code/test_f5_ghost_key.py
```

### `code/aig_adaptive_alpha_patch.py` — adaptive α benchmark

Compares fixed α=0.5 (MeMo's current approach) against RIS-adaptive α across P ∈ {10, 25, 50, 75, 100} interfering patterns. Shows how retrieval trust should be a function of live geometric state, not a fixed constant.

### `code/aig_memo_monitor.py` — RIS on real Italian text

Loads MeMo's Italian test corpus (`testo_di_prova3.txt`), feeds 300 tokens in sliding windows, and tracks RIS per step. RIS drops to WARNING by step 5 on natural language. Mean RIS: 0.12. Steps below θ=0.65: 291 of 298. Foreknowledge detected at mean lead 1.9 steps, 48% positive rate.

This shows that governance is *more* urgent on real text than on the controlled synthetic geometry of E8 — natural language creates dense semantic overlap that exhausts memory capacity much faster than random unit-sphere keys.

---

## Setup

```bash
pip install torch numpy scipy sentence-transformers
```

No GPU required. All experiments run on CPU in under 5 minutes each. `cmm_core.py` has no dependencies beyond `torch` and `numpy`.

---

## Run the main experiments

```bash
# Foreknowledge window across d and cosine regimes (94.4%, 3.15 steps)
python code/experiment_e8_stability_lead_time.py

# MeMo governance: Panels A, B, C — quality recovery and adaptive α
python code/experiment_memo_governance.py

# Semantic validation across embedding families — requires sentence-transformers
python code/experiment_semantic_systematic.py

# Within-domain FNR at scale (N=500 per condition, ~2 min)
python code/experiment_fnr_withindomain.py

# F5 production bug — requires MeMo clone
git clone https://github.com/zanppa/MeMo /tmp/MeMo_clone
python code/test_f5_ghost_key.py

# RIS on real Italian corpus — requires MeMo corpus file
python code/aig_memo_monitor.py
```

---

## The three governance properties

**P1 — Pre-execution certificate**
Storing (s_n, y_n) perturbs any existing retrieval ŝ_j by *exactly* ε_j = |cos(s_j, s_n)| · ‖y_n‖ — an algebraic equality, not an approximation. Verified to within 2×10⁻⁶ (float32 precision floor) across 3100 trials. This is what makes write-time interference computable before it happens.

**P2 — Exact rollback**
Because associations are separable rank-1 outer products, (C + s_n y_n^T) − s_n y_n^T = C exactly. Rollback is algebraically exact and structurally unavailable in nonlinear parametric models. Drift <10⁻⁷ float32 over 10,000 cycles; exactly zero in float64.

**P3 — Empirical scaling**
For random unit-sphere keys: E[max_{j≠n} ε_j] ≈ 0.897 √(2 ln P / d), validated at R²=0.97 across P ∈ [4, 1024]. Semantic embeddings require per-family recalibration (7–9× inflation measured for BERT-class models).

---

## The open problem — F5

Single-layer P1/P2 certificates do not compose across MeMo's stacked layers.

When pattern (s₁, y₁) is stored, L2 stores (f(s₁), y₂) where f(s₁) = s₁ᵀC₁. After L1 rollback removes s₁ from C₁, the key f(s₁) in L2 was computed from the *original* C₁ and persists unchanged — a ghost from a prior memory state.

Adversarial experiments: ε_L2/ε_L1 reaches **1.98×** (mean 1.07×). Interference amplifies across layers; L1 certificates underestimate downstream perturbation. Resolving this — deriving a joint certificate for stacked CMMs and proving when the linear bound ε_joint ≤ ε_L1 · ‖W_ℓ‖ is tight — is the Year 1 PhD thesis target.

---

## PhD research plan

*Within the MeMo project (FISA-2024-00249, PI: Prof. F.M. Zanzotto, Tor Vergata)*

| Year | Target | Venue |
|---|---|---|
| 1 | Solve F5: joint certificate for stacked CMMs; formal machine unlearning connection | EMNLP/ACL 2027 |
| 2 | ACS (Attentional Coherence Score): does a RIS-analogous signal on transformer attention weights predict factual-recall failure pre-generation? | NeurIPS/ICLR 2028 |
| 3 | Unified governance framework across CMMs, Hopfield networks, and attention | Thesis + deployment |

---

## Citation

```bibtex
@inproceedings{vijayarao2026aig,
  title     = {{AIG}: Towards a Mechanistic Science of Retrieval Integrity
               in Associative Memory Systems},
  author    = {Vijayarao, Lakshmi Chakradhar},
  booktitle = {MeMo Workshop on Mechanistic Interpretability \&
               Neuro-symbolic Approaches by-design},
  year      = {2026},
  address   = {Rome, Italy}
}
```

---

## Acknowledgments

Prof. Fabio Massimo Zanzotto (University of Rome Tor Vergata) for supervision and for the foundational MeMo architecture on which this work builds.
