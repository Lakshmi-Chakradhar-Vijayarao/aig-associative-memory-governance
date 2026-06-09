#!/usr/bin/env python3
"""
Systematic Semantic Foreknowledge Window Study

Tests whether the structural precursor effect (RIS warns before accuracy fails)
holds across sentence-embedding model families and semantic domains.

Design: 4 models × 5 domains × 6 anchor rotations = 30 trials per model.
For each trial: 1 anchor sentence, 3 safe fills (different domain),
20 within-domain contamination sentences injected one by one.
Measures foreknowledge window Δ = acc_fail_step − ris_warn_step.

Output: table printed to stdout + saved to paper/figures/tab_semantic.txt
"""

import os, sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, os.path.dirname(__file__))
from cmm_core import CMM

# ── Parameters ────────────────────────────────────────────────────────────
THETA_RIS  = 0.50   # RIS structural warning threshold
THETA_ACC  = 0.70   # behavioral accuracy threshold (cosine to anchor)
N_SAFE     = 3      # safe fills per trial (from a held-out domain)

N_CONTAMINATION = 60   # tiled if corpus < 60 — ensures acc failure visible even when Δ≈15

MODELS = [
    ("sentence-transformers/all-MiniLM-L6-v2",        384),
    ("sentence-transformers/all-mpnet-base-v2",        768),
    ("sentence-transformers/paraphrase-MiniLM-L6-v2", 384),
    # multi-qa excluded: optimised for asymmetric Q-A retrieval,
    # produces near-zero within-domain cosines for declarative statements
]

# ── Corpus: 5 domains × 30 sentences each ─────────────────────────────────
CORPUS = {
    "climate": [
        "Global warming of 1.5 degrees Celsius above pre-industrial levels poses severe risks.",
        "Temperature increases of two degrees would expose hundreds of millions to extreme heat.",
        "Limiting warming to 1.5 degrees requires rapid and far-reaching transitions in energy systems.",
        "The global mean temperature anomaly in 2023 was the highest on record.",
        "Rising global temperatures are linked to changes in precipitation patterns worldwide.",
        "Surface temperature records show an accelerating warming trend over recent decades.",
        "Carbon dioxide emissions from fossil fuels are the primary driver of anthropogenic climate change.",
        "Atmospheric warming is causing shifts in the frequency and intensity of extreme weather events.",
        "Temperature projections under high emissions scenarios indicate severe regional impacts.",
        "The radiative forcing from increased CO2 concentration drives sustained global warming.",
        "Observed warming trends are consistent with model projections based on greenhouse gas emissions.",
        "Annual mean global temperature anomalies have exceeded historical baselines repeatedly.",
        "Thermal expansion of seawater driven by warming contributes significantly to sea level rise.",
        "Climate sensitivity estimates indicate that doubling CO2 leads to sustained temperature increases.",
        "Rising temperatures are intensifying the hydrological cycle through increased evaporation.",
        "Heat accumulation in the climate system is driven by longwave radiation imbalance.",
        "Global surface temperature will continue to increase until net greenhouse gas emissions reach zero.",
        "Warming trends differ substantially between land and ocean surfaces due to thermal capacity.",
        "Temperature changes are unevenly distributed with polar regions experiencing fastest warming.",
        "Cumulative carbon emissions determine the total amount of global warming that occurs.",
        "Arctic sea ice extent has declined dramatically over the past four decades.",
        "Ocean acidification is a direct consequence of CO2 absorption by seawater.",
        "Permafrost thaw releases additional methane and CO2 creating amplifying feedback loops.",
        "Coral reef bleaching events are becoming more frequent as ocean temperatures rise.",
        "Extreme precipitation events are increasing in intensity across many world regions.",
        "Glacier retreat is accelerating across mountain systems worldwide.",
        "Sea level rise threatens low-lying coastal communities and small island developing states.",
        "Wildfire frequency and intensity have increased significantly in drought-affected regions.",
        "The jet stream is being influenced by differential warming between Arctic and temperate zones.",
        "Ecosystem disruption from temperature shifts is affecting species ranges and migration patterns.",
    ],
    "medicine": [
        "The blood-brain barrier selectively restricts passage of substances into the central nervous system.",
        "Inflammation is a fundamental biological response involving coordinated immune cell recruitment.",
        "Cardiac output is determined by the product of heart rate and stroke volume.",
        "The renin-angiotensin-aldosterone system regulates blood pressure and fluid homeostasis.",
        "Apoptosis is a programmed cell death process essential for developmental tissue homeostasis.",
        "T-cell activation requires antigen presentation by major histocompatibility complex molecules.",
        "Insulin resistance is characterized by reduced cellular response to insulin receptor signaling.",
        "The coagulation cascade involves sequential activation of clotting factors in the blood.",
        "Neuroplasticity allows the brain to reorganize synaptic connections in response to experience.",
        "Mitochondrial dysfunction is implicated in the pathogenesis of multiple neurodegenerative diseases.",
        "The complement system provides rapid innate immune defense against invading pathogens.",
        "Epigenetic modifications regulate gene expression without altering the underlying DNA sequence.",
        "Oxidative stress results from imbalance between reactive oxygen species and antioxidant defenses.",
        "The lymphatic system transports interstitial fluid and plays a key role in immune surveillance.",
        "Synaptic plasticity is the cellular mechanism underlying learning and memory formation.",
        "Autophagy is a conserved cellular process for degrading and recycling damaged organelles.",
        "The hypothalamic-pituitary-adrenal axis coordinates the neuroendocrine stress response.",
        "Telomere shortening is associated with replicative cellular aging and senescence.",
        "Cytokine signaling mediates intercellular communication between immune effector cells.",
        "The endocannabinoid system modulates pain perception appetite and neurological function.",
        "Angiogenesis is the formation of new capillaries from pre-existing vascular networks.",
        "RNA interference provides a mechanism for sequence-specific post-transcriptional gene silencing.",
        "The gut microbiome influences systemic immune function and metabolic homeostasis.",
        "Ion channels regulate the electrical properties and excitability of membrane-bound cells.",
        "DNA repair mechanisms maintain genomic stability against mutational damage.",
        "The extracellular matrix provides structural scaffolding and instructive signals for cells.",
        "Protein misfolding and pathological aggregation underlie amyloid diseases including Alzheimer's.",
        "Stem cell differentiation is regulated by combinatorial transcription factor activity.",
        "The glymphatic system clears neurotoxic metabolic waste from the brain during sleep.",
        "Membrane potential changes propagate as action potentials in neurons and cardiac muscle.",
    ],
    "ML_AI": [
        "Attention mechanisms allow neural networks to selectively weight relevant input features.",
        "Gradient descent iteratively optimizes neural network parameters to minimize a loss objective.",
        "Transformer architectures process sequential data in parallel using multi-head self-attention.",
        "Backpropagation efficiently computes parameter gradients through the chain rule of calculus.",
        "Batch normalization stabilizes training dynamics by normalizing intermediate layer activations.",
        "Dropout regularization reduces overfitting by stochastically deactivating neurons during training.",
        "Residual skip connections enable gradient propagation through very deep neural architectures.",
        "The softmax function normalizes logit vectors into probability distributions over classes.",
        "Convolutional filters learn spatially local hierarchical feature representations from images.",
        "Weight initialization strategies critically affect convergence speed and final model performance.",
        "Embedding layers project discrete tokens into continuous high-dimensional representation spaces.",
        "Layer normalization stabilizes activations within individual training examples independently.",
        "Cross-entropy loss measures divergence between predicted probability and target distributions.",
        "Beam search is a breadth-limited tree search algorithm used for sequence generation.",
        "Transfer learning adapts representations learned on large corpora to downstream tasks.",
        "Contrastive learning trains dense representations by attracting similar and repelling dissimilar pairs.",
        "The Adam optimizer adapts per-parameter learning rates using exponential moving average estimates.",
        "Positional encodings provide transformers with explicit token order information.",
        "Knowledge distillation transfers representational structure from large teacher to student models.",
        "Sparse attention mechanisms reduce self-attention complexity from quadratic to near-linear.",
        "Autoregressive language models generate sequences through left-to-right token prediction.",
        "The universal approximation theorem establishes that wide networks can represent any continuous function.",
        "Temperature scaling controls the entropy of softmax output distributions during generation.",
        "Fine-tuning adapts all pretrained weights to optimize performance on task-specific data.",
        "Mixture-of-experts architectures route token representations to specialized expert subnetworks.",
        "Retrieval-augmented generation grounds language model outputs in retrieved external documents.",
        "Reinforcement learning from human feedback aligns model outputs with expressed preferences.",
        "Low-rank adaptation inserts trainable rank-decomposition matrices into pretrained weight matrices.",
        "In-context learning enables task adaptation from a small number of labeled prompt examples.",
        "Chain-of-thought prompting elicits explicit intermediate reasoning steps from large language models.",
    ],
    "economics": [
        "Monetary policy affects inflation and output through transmission channels in financial markets.",
        "Fiscal multipliers measure the aggregate output effect of changes in government expenditure.",
        "Asset price bubbles occur when market valuations persistently exceed fundamental economic value.",
        "Exchange rate fluctuations affect the competitiveness of exports and the cost of imports.",
        "Total factor productivity growth is the primary driver of sustained long-run economic expansion.",
        "Supply chain disruptions propagate through input-output linkages across interconnected industries.",
        "Central bank inflation credibility reduces the anchoring cost of price expectations.",
        "Portfolio diversification reduces idiosyncratic risk by exploiting cross-asset return correlations.",
        "Market microstructure research studies how trading mechanisms affect price discovery.",
        "Yield curve inversion has historically preceded economic contractions with predictive reliability.",
        "Labor market tightness is reflected in the ratio of job openings to unemployed workers.",
        "Price elasticity of demand measures the percentage change in quantity per unit change in price.",
        "Capital allocation efficiency determines how productively an economy transforms savings into output.",
        "Sovereign debt sustainability depends on the differential between interest rates and growth rates.",
        "Financial contagion transmits idiosyncratic institutional shocks across interconnected markets.",
        "The output gap captures the deviation of realized GDP from its potential non-inflationary level.",
        "The marginal propensity to consume determines the fraction of income directed toward consumption.",
        "Risk premium reflects the excess return investors require for bearing non-diversifiable uncertainty.",
        "Technology adoption rates vary across sectors based on diffusion costs and network externalities.",
        "Trade liberalization affects income distribution by altering relative factor demands across sectors.",
        "Credit market frictions impede the transmission of monetary policy impulses to real investment.",
        "Tobin's Q compares the market value of installed capital to its current replacement cost.",
        "Structural unemployment reflects persistent skill-occupation mismatches in labor markets.",
        "Currency crises occur when speculative capital outflows exhaust central bank reserve buffers.",
        "Bank capital requirements limit risk-taking and constrain the supply of financial intermediation.",
        "Consumption smoothing motivates precautionary saving and borrowing to stabilize spending paths.",
        "The neutral real interest rate is consistent with full employment and stable inflation.",
        "Agglomeration economies arise from geographic concentration of complementary productive activities.",
        "Carbon pricing internalizes the social cost of greenhouse gas emissions into market incentives.",
        "Demand hysteresis allows temporary cyclical downturns to permanently scar potential output.",
    ],
    "physics": [
        "Quantum entanglement creates non-local correlations between particles that persist at distance.",
        "The uncertainty principle establishes fundamental limits on simultaneous observational precision.",
        "Phase transitions occur when systems change discontinuously between thermodynamic equilibrium states.",
        "Spontaneous symmetry breaking in the Higgs field generates masses for fundamental particles.",
        "Renormalization group methods describe how effective theories change across energy scales.",
        "Black hole thermodynamics reveals deep connections between quantum mechanics and gravity.",
        "Topological insulators exhibit bulk insulating behavior with protected conducting surface states.",
        "The Fermi surface topology determines the low-temperature electronic properties of metals.",
        "Bose-Einstein condensation occurs when macroscopic quantum coherence develops in dilute gases.",
        "String theory unifies quantum mechanics and gravity by treating particles as extended objects.",
        "Dark matter constitutes approximately 27 percent of the total energy density of the universe.",
        "Gravitational waves are spacetime perturbations generated by accelerating asymmetric mass distributions.",
        "Superconductivity arises from phonon-mediated pairing of electrons below a critical temperature.",
        "The partition function encodes the complete thermodynamic properties of a statistical ensemble.",
        "Quantum chromodynamics describes confinement and asymptotic freedom in the strong nuclear force.",
        "Decoherence explains the emergence of classical definiteness from quantum superposition states.",
        "The cosmological constant characterizes the energy density of the quantum vacuum.",
        "Fermi's golden rule quantifies transition rates between quantum states under perturbation theory.",
        "Anderson localization prevents diffusive wave transport in sufficiently disordered media.",
        "The fluctuation-dissipation theorem relates equilibrium thermal fluctuations to dissipation.",
        "Neutron stars represent the densest stable matter configurations achievable under gravity.",
        "Quantum tunneling permits particle transmission through classically forbidden potential barriers.",
        "Critical exponents near phase transitions exhibit universality independent of microscopic details.",
        "The quantum measurement problem concerns how determinate classical outcomes emerge from superpositions.",
        "Hawking radiation predicts thermally distributed photon emission from evaporating black holes.",
        "Magnons are quantized collective spin-wave excitations in magnetically ordered solid-state systems.",
        "The standard model successfully describes electromagnetic weak and strong fundamental interactions.",
        "Entropy production quantifies the degree of irreversibility in thermodynamic processes.",
        "Spin-orbit coupling entangles the intrinsic spin of electrons with their orbital angular momentum.",
        "The holographic principle conjectures that bulk gravitational degrees of freedom are boundary-encoded.",
    ],
}

SAFE_DOMAIN = "physics"   # used as safe fills (always cross-domain)


# ── Embedding ─────────────────────────────────────────────────────────────

def load_embedder(model_name):
    print(f"  Loading {model_name.split('/')[-1]}...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name)
    mdl.eval()

    def embed(sentences):
        inp = tok(sentences, padding=True, truncation=True,
                  max_length=128, return_tensors="pt")
        with torch.no_grad():
            out = mdl(**inp)
        mask = inp["attention_mask"].unsqueeze(-1).float()
        emb  = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return F.normalize(emb, p=2, dim=1)

    return embed


# ── Single trial ──────────────────────────────────────────────────────────

def run_trial(anchor_emb, cont_embs, safe_embs, d,
              theta_ris=THETA_RIS, theta_acc=THETA_ACC):
    """
    Returns (delta, ris_warn, acc_fail) or None if no crossing detected.
    delta > 0  → RIS warned before accuracy failed (foreknowledge)
    delta == 0 → simultaneous
    delta < 0  → accuracy failed before RIS warned (false negative)
    """
    cmm = CMM(d=d)
    anchor = anchor_emb.squeeze()

    cmm.store(anchor, anchor, domain="anchor", label="anchor")
    for s in safe_embs:
        cmm.store(s.squeeze(), s.squeeze(), domain="safe", label="safe")

    ris_warn = acc_fail = None

    for step, cont in enumerate(cont_embs):
        cont = cont.squeeze()
        cmm.store(cont, cont, domain="cont", label="cont")

        # RIS
        rsv = cmm.rsv(anchor)
        ris = (rsv.abs().max() / (rsv.abs().sum() + 1e-9)).item()

        # accuracy (cosine of retrieved vector to anchor value)
        ret  = cmm.retrieve(anchor)
        nrm  = ret.norm().item()
        cos  = float((ret / (nrm + 1e-9)) @ anchor) if nrm > 1e-6 else 0.0

        if ris_warn is None and ris < theta_ris:
            ris_warn = step
        if acc_fail is None and cos < theta_acc:
            acc_fail = step
        if ris_warn is not None and acc_fail is not None:
            break

    if ris_warn is None or acc_fail is None:
        return None
    return acc_fail - ris_warn, ris_warn, acc_fail


# ── Per-model experiment ──────────────────────────────────────────────────

def run_model(model_name, d, embed_fn):
    domains = [k for k in CORPUS if k != SAFE_DOMAIN]
    safe_sentences = CORPUS[SAFE_DOMAIN][:N_SAFE]

    # Pre-embed safe sentences
    safe_embs = embed_fn(safe_sentences)          # (N_SAFE, d)

    all_deltas = []
    per_domain = {}

    for domain in domains:
        sents   = CORPUS[domain]                  # 30 sentences
        deltas  = []

        # 6 rotations: each sentence i serves as anchor,
        # remaining sentences as contamination (tiled to N_CONTAMINATION)
        for anchor_idx in range(6):
            anchor_sent = sents[anchor_idx]
            pool        = [s for j, s in enumerate(sents) if j != anchor_idx]
            # tile so we have enough writes for acc to fail even when Δ≈15
            cont_sents  = (pool * (N_CONTAMINATION // len(pool) + 1))[:N_CONTAMINATION]

            anchor_emb = embed_fn([anchor_sent])            # (1, d)
            cont_embs  = embed_fn(cont_sents)               # (N_CONTAMINATION, d)

            # mean cosine of contamination to anchor (μcos for this trial)
            mu_cos = float((cont_embs @ anchor_emb.T).abs().mean())

            result = run_trial(anchor_emb, cont_embs, safe_embs, d)

            if result is not None:
                delta, rw, af = result
                deltas.append(delta)
                all_deltas.append((delta, mu_cos, domain))

        mean_d  = float(np.mean(deltas)) if deltas else float("nan")
        std_d   = float(np.std(deltas))  if len(deltas) > 1 else 0.0
        pos_pct = 100.0 * sum(1 for d in deltas if d > 0) / len(deltas) if deltas else 0.0
        per_domain[domain] = (mean_d, std_d, pos_pct, len(deltas))

    # Overall
    all_d   = [x[0] for x in all_deltas]
    mean_o  = float(np.mean(all_d)) if all_d else float("nan")
    std_o   = float(np.std(all_d))  if len(all_d) > 1 else 0.0
    pos_o   = 100.0 * sum(1 for d in all_d if d > 0) / len(all_d) if all_d else 0.0
    mu_o    = float(np.mean([x[1] for x in all_deltas])) if all_deltas else 0.0

    return {
        "per_domain": per_domain,
        "overall_mean": mean_o,
        "overall_std":  std_o,
        "overall_pos":  pos_o,
        "n_trials":     len(all_d),
        "mu_cos":       mu_o,
    }


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    results = {}

    for model_name, d in MODELS:
        short = model_name.split("/")[-1]
        print(f"\n{'='*60}")
        print(f"Model: {short}  d={d}")
        print(f"{'='*60}")
        embed_fn = load_embedder(model_name)
        res = run_model(model_name, d, embed_fn)
        results[short] = (d, res)

        print(f"  Overall: Δ={res['overall_mean']:.2f}±{res['overall_std']:.2f}  "
              f"pos={res['overall_pos']:.1f}%  "
              f"μcos={res['mu_cos']:.3f}  "
              f"n={res['n_trials']}")
        for dom, (m, s, p, n) in res["per_domain"].items():
            print(f"    {dom:12s}: Δ={m:5.2f}±{s:.2f}  pos={p:.0f}%  n={n}")

    # ── Summary table ────────────────────────────────────────────────────
    print("\n\n" + "="*70)
    print("SEMANTIC FOREKNOWLEDGE WINDOW — CROSS-MODEL SUMMARY")
    print("="*70)
    print(f"{'Model':<30} {'d':>5} {'μcos':>6} {'Δ (steps)':>12} {'Positive%':>10}")
    print("-"*70)

    lines = []
    all_means, all_pos = [], []
    for short, (d, res) in results.items():
        m   = res["overall_mean"]
        s   = res["overall_std"]
        p   = res["overall_pos"]
        mu  = res["mu_cos"]
        row = f"{short:<30} {d:>5} {mu:>6.3f} {m:>6.2f}±{s:<5.2f} {p:>9.1f}%"
        print(row)
        lines.append(row)
        all_means.append(m)
        all_pos.append(p)

    print("-"*70)
    print(f"{'Overall':>42}  {np.mean(all_means):>5.2f}±{np.std(all_means):<5.2f} {np.mean(all_pos):>9.1f}%")
    print()
    print(f"Trials per model: {list(results.values())[0][1]['n_trials']}")
    print(f"Domains: {', '.join(k for k in CORPUS if k != SAFE_DOMAIN)}")
    print(f"Contamination per trial: 20 sentences")
    print(f"θ_RIS={THETA_RIS}  θ_acc={THETA_ACC}")

    # Save for paper
    out_dir = os.path.join(os.path.dirname(__file__), "..", "paper", "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "tab_semantic.txt")
    with open(out_path, "w") as f:
        f.write("SEMANTIC FOREKNOWLEDGE WINDOW — CROSS-MODEL SUMMARY\n")
        f.write(f"{'Model':<30} {'d':>5} {'mu_cos':>7} {'Delta':>8} {'Pos%':>7}\n")
        for short, (d, res) in results.items():
            f.write(f"{short:<30} {d:>5} {res['mu_cos']:>7.3f} "
                    f"{res['overall_mean']:>5.2f}+-{res['overall_std']:<4.2f} "
                    f"{res['overall_pos']:>6.1f}%\n")
        f.write(f"\nOverall positive rate: {np.mean(all_pos):.1f}%\n")
        f.write(f"Trials per model: {list(results.values())[0][1]['n_trials']}\n")

    print(f"\nSaved → {out_path}")
    return results


if __name__ == "__main__":
    main()
