#!/usr/bin/env python3
"""
GovernedMeMoLayer — AIG governance wrapper for MeMo's CMM operations.

Fixes three structural gaps in MeMo's code-integration architecture:

  Gap 1 — forget() residual = 1.0000 at all P:
    MeMo's penalizing_factor collapses to 0 for stored patterns,
    so forget() subtracts nothing. Replaced by T2 exact rollback
    (residual = 1.6e-07, floating-point noise only).

  Gap 2 — fixed 50/50 blend with no principled basis:
    MeMo hardcodes return (retrieved + computed) / 2 regardless
    of CMM health. Replaced by RIS-adaptive alpha: when CMM is
    geometrically dominant (RIS high), trust it; when degraded
    (RIS low), fall back to direct projection.

  Gap 3 — no pre-write interference certificate:
    MeMo admits every write without checking whether it will
    interfere with existing retrievals. T1 gate computes the
    max perturbation bound before each write and blocks if it
    exceeds the policy threshold.

Usage:
    layer = GovernedMeMoLayer(d=128, cert_threshold=0.5)
    layer.governed_write(key, value)          # T1 gated
    layer.retrieve(query, direct_proj)        # RIS-adaptive blend
    layer.governed_forget(k_idx)              # T2 exact rollback
    layer.health(query)                       # RIS + status
    layer.audit_summary()                     # governance record
"""

import os, sys
import torch

sys.path.insert(0, os.path.dirname(__file__))
from cmm_core import CMM


class GovernedMeMoLayer:

    def __init__(self,
                 d: int,
                 cert_threshold: float = 0.5,
                 ris_threshold:  float = 0.65,
                 alpha_mode:     str   = "adaptive"):
        """
        Args:
            d:               embedding dimension
            cert_threshold:  T1 gate — writes with max_ε > this are blocked
            ris_threshold:   RIS health threshold — below this, layer is degraded
            alpha_mode:      "adaptive" (RIS-based) or "fixed" (MeMo default 0.5)
        """
        self.cmm            = CMM(d=d)
        self.cert_threshold = cert_threshold
        self.ris_threshold  = ris_threshold
        self.alpha_mode     = alpha_mode
        self.audit_log      = []

    # ── Gap 3 Fix: T1 Pre-write Certificate Gate ──────────────────────────────

    def governed_write(self, key: torch.Tensor, value: torch.Tensor,
                       domain: str = "memo", label: str = "") -> dict:
        """
        Write (key, value) with T1 pre-execution certificate.
        Blocks the write if max interference bound exceeds cert_threshold.
        Every decision is logged to the audit trail.
        """
        if not self.cmm.keys:
            self.cmm.store(key, value, domain=domain, label=label)
            rec = {"status": "admitted", "max_cert": 0.0,
                   "P": 1, "label": label}
            self.audit_log.append(rec)
            return rec

        certs    = self.cmm.storage_bound(key, value)
        max_cert = max(c[2] for c in certs) if certs else 0.0

        if max_cert > self.cert_threshold:
            top = max(certs, key=lambda x: x[2])
            rec = {"status": "blocked", "max_cert": max_cert,
                   "P": len(self.cmm.keys), "label": label,
                   "blocked_by": top}
            self.audit_log.append(rec)
            return rec

        self.cmm.store(key, value, domain=domain, label=label)
        rec = {"status": "admitted", "max_cert": max_cert,
               "P": len(self.cmm.keys), "label": label}
        self.audit_log.append(rec)
        return rec

    # ── Gap 1 Fix: T2 Exact Rollback ─────────────────────────────────────────

    def governed_forget(self, k_idx: int,
                        epsilon_budget: float = float("inf")) -> dict:
        """
        Exact removal of pattern k_idx via T2 outer-product subtraction.

        MeMo forget(): penalizing_factor → 0 when pattern is stored →
            effective_subtraction = 0 → residual = 1.0000 at all P.
        T2 rollback:   C -= s_k · y_k^T exactly →
            residual = 1.6e-07 (IEEE 754 floating-point noise only).

        epsilon_budget: optional safety gate on removal side effects.
        """
        return self.cmm.repair(k_idx, epsilon_budget=epsilon_budget)

    def forget_worst_interferer(self, query: torch.Tensor) -> dict:
        """
        Identify and remove the pattern causing most interference for query.
        Uses RSV to find second-highest contributor (dominant is the target).
        Replaces MeMo's heuristic capacity management.
        """
        if len(self.cmm.keys) < 2:
            return {"status": "skipped", "reason": "fewer than 2 patterns stored"}
        rsv        = self.cmm.rsv(query)
        sorted_idx = rsv.abs().argsort(descending=True)
        interferer = sorted_idx[1].item()
        result     = self.governed_forget(interferer)
        result["removed_idx"] = interferer
        return result

    # ── Gap 2 Fix: RIS-Adaptive Alpha ─────────────────────────────────────────

    def retrieve(self, query: torch.Tensor,
                 direct_projection: torch.Tensor) -> dict:
        """
        Blend CMM retrieval with direct projection using RIS-adaptive alpha.

        MeMo current:  return (retrieved + computed) / 2   [alpha fixed = 0.5]
        AIG fix:       alpha = RIS                          [alpha = CMM health]

        When RIS = 1.0: CMM is dominant, fully trust it (alpha = 1.0).
        When RIS = 0.2: CMM is degraded, mostly use projection (alpha = 0.2).

        Returns blended output + RIS + alpha used + health status.
        """
        rsv     = self.cmm.rsv(query)
        ris     = self.cmm.ris(rsv)
        cmm_out = self.cmm.retrieve(query)

        alpha   = ris if self.alpha_mode == "adaptive" else 0.5
        blended = alpha * cmm_out + (1.0 - alpha) * direct_projection

        return {
            "output":   blended,
            "ris":      ris,
            "alpha":    alpha,
            "mode":     self.alpha_mode,
            "healthy":  ris >= self.ris_threshold,
        }

    # ── Health Monitoring ─────────────────────────────────────────────────────

    def health(self, query: torch.Tensor) -> dict:
        """
        Full health report for a query.
        Returns RIS, entropy, dominant pattern, and a repair recommendation
        when RIS drops below ris_threshold.
        """
        if not self.cmm.keys:
            return {"status": "empty", "ris": 0.0, "P": 0}

        rsv     = self.cmm.rsv(query)
        ris     = self.cmm.ris(rsv)
        entropy = self.cmm.entropy(rsv)
        dom     = self.cmm.dominant_pattern(query)
        status  = "healthy" if ris >= self.ris_threshold else "degraded"

        report = {
            "status":   status,
            "ris":      ris,
            "entropy":  entropy,
            "P":        len(self.cmm.keys),
            "dominant": dom,
        }
        if status == "degraded":
            report["action"] = (
                "Call forget_worst_interferer(query) to remove top interferer, "
                "or lower cert_threshold to gate future writes more aggressively."
            )
        return report

    # ── Audit ─────────────────────────────────────────────────────────────────

    def audit_summary(self) -> dict:
        admitted = [r for r in self.audit_log if r["status"] == "admitted"]
        blocked  = [r for r in self.audit_log if r["status"] == "blocked"]
        return {
            "total_writes":       len(self.audit_log),
            "admitted":           len(admitted),
            "blocked":            len(blocked),
            "block_rate":         len(blocked) / max(len(self.audit_log), 1),
            "mean_cert_admitted": (
                sum(r["max_cert"] for r in admitted) / max(len(admitted), 1)
            ),
            "mean_cert_blocked":  (
                sum(r["max_cert"] for r in blocked) / max(len(blocked), 1)
            ) if blocked else 0.0,
        }

    # ── Convenience ───────────────────────────────────────────────────────────

    @property
    def P(self) -> int:
        return len(self.cmm.keys)

    def __repr__(self) -> str:
        return (f"GovernedMeMoLayer(d={self.cmm.d}, P={self.P}, "
                f"mode={self.alpha_mode}, θ_cert={self.cert_threshold}, "
                f"θ_ris={self.ris_threshold})")
