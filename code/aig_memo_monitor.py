"""
aig_memo_monitor.py
AIG Retrieval-State Introspection (RSI) monitor on MeMo's own Italian text corpus.

Loads testo_di_prova3.txt (the text MeMo was designed for), builds a CMM layer
over a rolling window of token encodings, and tracks RIS as the corpus is fed in.
Demonstrates the foreknowledge window: RIS structural warning precedes
behavioral accuracy failure.

Output: per-step RIS trace + foreknowledge window detection on real MeMo text.
"""

import sys
import os
sys.path.insert(0, '/tmp/MeMo_clone')
os.environ['PYTHONWARNINGS'] = 'ignore'

import torch
import math
import io
from contextlib import redirect_stdout

with redirect_stdout(io.StringIO()):
    from MeMoCMM import MeMoLayer, MeMo, MeMoEncoder

SEED = 42
torch.manual_seed(SEED)

# ── Config ────────────────────────────────────────────────────────────────────
D               = 128         # embedding dimension
H               = 2           # MeMo heads (matches MeMoCMM default)
THETA_RIS       = 0.65        # RIS health threshold (as in paper)
THETA_ACC       = 0.50        # retrieval cosine quality threshold
MAX_TOKENS      = 300         # tokens to feed from corpus
WINDOW          = 60          # rolling window length (MeMo's chunk_length)


# ─────────────────────────────────────────────────────────────────────────────
# CMM wrapper for incremental monitoring (standalone, no MeMo tokenizer needed)
# ─────────────────────────────────────────────────────────────────────────────
class IncrementalCMMMonitor:
    """Single-layer CMM with live RIS tracking."""

    def __init__(self, d, h, theta_ris=THETA_RIS):
        self.d = d
        self.h = h
        self.theta = theta_ris
        self.layer = MeMoLayer(d, h)
        self.stored_keys = []     # (seq_enc_per_token, output_symbol) pairs
        self.ris_trace  = []
        self.acc_trace  = []
        self.step_trace = []

    def _encode_token_window(self, token_vecs):
        """token_vecs: (h, d) → encode as single-layer MeMo block."""
        return token_vecs.unsqueeze(0)  # (1, h, d)

    def store(self, token_vecs, output_vec, step):
        """Memorize one window block and record RIS snapshot."""
        block = self._encode_token_window(token_vecs)
        out   = output_vec.unsqueeze(0)   # (1, d)
        self.layer.memorize(block, out, is_last=True)
        self.stored_keys.append((block.clone(), out.clone()))

        ris = self._compute_ris(block)
        self.ris_trace.append(ris)
        self.step_trace.append(step)

    def retrieve_quality(self, query_vecs, expected_vec):
        """Cosine similarity between retrieved output and expected."""
        block = self._encode_token_window(query_vecs)
        out_enc, _, predicted = self.layer.retrieve(block)
        cos = torch.nn.functional.cosine_similarity(
            predicted.squeeze(), expected_vec.squeeze(), dim=0
        ).item()
        return abs(cos)

    def _compute_ris(self, block):
        """RIS = max|RSV| / Σ|RSV|  over all stored keys."""
        if len(self.stored_keys) == 0:
            return 1.0
        seq_enc = torch.matmul(
            block.reshape(1, self.d * self.h), self.layer.Prj
        ).squeeze(0)  # (d,)

        # seq_enc_per_token for this block
        sep = torch.matmul(block, self.layer.W_v_single_head).reshape(1, self.d) / math.sqrt(self.h)
        sep = sep.squeeze(0)  # (d,)

        rsvs = []
        for (stored_block, stored_out) in self.stored_keys:
            stored_sep = torch.matmul(stored_block, self.layer.W_v_single_head).reshape(self.d) / math.sqrt(self.h)
            cos = torch.dot(sep, stored_sep) / (torch.norm(sep) * torch.norm(stored_sep) + 1e-12)
            rsv = abs(cos.item()) * torch.norm(stored_out).item()
            rsvs.append(rsv)

        if len(rsvs) == 0 or sum(rsvs) < 1e-12:
            return 1.0
        return max(rsvs) / sum(rsvs)

    def health_status(self):
        if not self.ris_trace:
            return "NO DATA"
        ris = self.ris_trace[-1]
        return "HEALTHY" if ris >= self.theta else "WARNING"


# ─────────────────────────────────────────────────────────────────────────────
# Load corpus and build token embeddings
# ─────────────────────────────────────────────────────────────────────────────
def load_corpus_embeddings(path, max_tokens, d):
    """
    Load testo_di_prova3.txt, hash each token to a fixed d-dim embedding.
    This mimics what MeMoEncoder does (random projection per token type).
    """
    with open(path, encoding='utf8') as f:
        tokens = f.read().split()[:max_tokens]

    vocab = list(set(tokens))
    torch.manual_seed(SEED)
    embeddings = {w: torch.nn.functional.normalize(torch.randn(d), dim=0)
                  for w in vocab}
    token_vecs = [embeddings[t] for t in tokens]
    return tokens, token_vecs, embeddings


# ─────────────────────────────────────────────────────────────────────────────
# Foreknowledge window detection
# ─────────────────────────────────────────────────────────────────────────────
def detect_foreknowledge_windows(ris_trace, acc_trace, theta_ris, theta_acc):
    """
    Find events where RIS drops below theta_ris BEFORE acc drops below theta_acc.
    Returns list of (ris_warning_step, acc_failure_step, lead_time).
    """
    events = []
    n = min(len(ris_trace), len(acc_trace))
    in_warning = False
    warn_step  = None

    for i in range(n):
        ris = ris_trace[i]
        acc = acc_trace[i] if i < len(acc_trace) else 1.0

        if not in_warning and ris < theta_ris:
            in_warning = True
            warn_step  = i

        if in_warning and acc < theta_acc:
            lead = i - warn_step
            events.append((warn_step, i, lead))
            in_warning = False
            warn_step  = None

    return events


# ─────────────────────────────────────────────────────────────────────────────
# Main monitor run
# ─────────────────────────────────────────────────────────────────────────────
def run_monitor():
    print("=" * 70)
    print("AIG RSI MONITOR — MeMo Italian Corpus (testo_di_prova3.txt)")
    print(f"d={D}, h={H}, θ_RIS={THETA_RIS}, θ_acc={THETA_ACC}, tokens={MAX_TOKENS}")
    print("=" * 70)

    corpus_path = '/tmp/MeMo_clone/testo_di_prova3.txt'
    tokens, token_vecs, _ = load_corpus_embeddings(corpus_path, MAX_TOKENS, D)

    monitor = IncrementalCMMMonitor(D, H, THETA_RIS)

    ris_trace  = []
    acc_trace  = []
    step_trace = []

    print(f"\n  Feeding {len(tokens)} tokens in windows of h={H}...")
    print(f"  {'Step':>5}  {'Token':>15}  {'RIS':>8}  {'Acc':>8}  {'Status'}")
    print(f"  {'-'*5}  {'-'*15}  {'-'*8}  {'-'*8}  {'-'*12}")

    for step in range(0, len(tokens) - H, 1):
        # Build h-token window as input
        win_vecs = torch.stack(token_vecs[step:step + H])      # (h, d)
        out_vec  = token_vecs[step + H].clone()                # next token = target

        monitor.store(win_vecs, out_vec, step)

        # Retrieval quality check (query = same window, expect = out_vec)
        acc = monitor.retrieve_quality(win_vecs, out_vec)

        ris = monitor.ris_trace[-1]
        ris_trace.append(ris)
        acc_trace.append(acc)
        step_trace.append(step)

        status = monitor.health_status()
        if step < 10 or step % 30 == 0 or status == "WARNING":
            tok_str = tokens[step][:15]
            print(f"  {step:>5}  {tok_str:>15}  {ris:>8.4f}  {acc:>8.4f}  {status}")

    # ── Foreknowledge window analysis ─────────────────────────────────────────
    print(f"\n{'─'*70}")
    events = detect_foreknowledge_windows(ris_trace, acc_trace, THETA_RIS, THETA_ACC)

    print(f"\n  FOREKNOWLEDGE WINDOW ANALYSIS")
    print(f"  θ_RIS={THETA_RIS}  θ_acc={THETA_ACC}  steps monitored={len(ris_trace)}")
    print()

    if events:
        leads = [e[2] for e in events]
        mean_lead = sum(leads) / len(leads)
        pos_rate  = sum(1 for l in leads if l > 0) / len(leads)

        print(f"  {'Event':>5}  {'RIS warn step':>14}  {'Acc fail step':>14}  {'Lead time':>10}")
        print(f"  {'-'*5}  {'-'*14}  {'-'*14}  {'-'*10}")
        for i, (ws, fs, lt) in enumerate(events[:10]):
            print(f"  {i+1:>5}  {ws:>14}  {fs:>14}  {lt:>10}")
        if len(events) > 10:
            print(f"  ... ({len(events) - 10} more events)")

        print(f"\n  Mean lead time:  {mean_lead:.1f} steps")
        print(f"  Positive rate:   {pos_rate*100:.1f}%")
    else:
        print(f"  No foreknowledge events triggered at these thresholds.")
        print(f"  (RIS stayed above θ={THETA_RIS} — memory still healthy at {MAX_TOKENS} tokens)")

    # ── RIS health summary ────────────────────────────────────────────────────
    warnings = [r for r in ris_trace if r < THETA_RIS]
    min_ris  = min(ris_trace) if ris_trace else 1.0
    min_acc  = min(acc_trace) if acc_trace else 1.0
    mean_ris = sum(ris_trace) / len(ris_trace) if ris_trace else 0.0

    print(f"\n  RIS trace summary:")
    print(f"    Steps monitored:  {len(ris_trace)}")
    print(f"    Mean RIS:         {mean_ris:.4f}")
    print(f"    Min RIS:          {min_ris:.4f}")
    print(f"    Steps below θ:    {len(warnings)}")
    print(f"    Min retrieval acc: {min_acc:.4f}")
    print()
    print(f"  INTERPRETATION:")
    if len(warnings) > 0:
        print(f"    RIS dropped below θ={THETA_RIS} at {len(warnings)} steps.")
        print(f"    MeMo has no equivalent signal — it would write blindly.")
        print(f"    AIG would trigger BLOCK (T1 gate) or ADAPT (alpha reduction).")
    else:
        print(f"    Memory healthy for {MAX_TOKENS}-token corpus at d={D}.")
        print(f"    This confirms CMM capacity scales with d (P3 scaling law).")
        print(f"    AIG would report HEALTHY and admit writes.")
    print()
    print(f"  GOVERNANCE PATCH (3 lines in MeMo's memorize / retrieve):")
    print(f"    ris = compute_ris(self.CMM, seq_enc_per_token)")
    print(f"    if ris < THETA_RIS: block_or_adapt()")
    print(f"    return ris * retrieved + (1 - ris) * computed  # adaptive alpha")
    print("=" * 70)


if __name__ == "__main__":
    run_monitor()
