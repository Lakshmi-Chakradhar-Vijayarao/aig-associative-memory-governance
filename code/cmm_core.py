"""
CMM and MeMoStack implementation for EG-AMS experiments.
Implements Kohonen CMM (1972) with EG-AMS instrumentation.
"""
import torch


class CMM:
    def __init__(self, d: int, device: torch.device = None):
        self.d = d
        self.device = device or torch.device("cpu")
        self.C = torch.zeros(d, d, device=self.device)
        self.keys = []
        self.values = []
        self.domains = []
        self.labels = []
        self.scales = []   # ||y|| at storage time; used for production certificate scaling

    def store(self, key: torch.Tensor, value: torch.Tensor, domain: str, label: str = ""):
        key = key / key.norm()
        scale = value.norm().item()
        self.C += torch.outer(key, value)
        self.keys.append(key.clone())
        self.values.append(value.clone())
        self.domains.append(domain)
        self.labels.append(label)
        self.scales.append(scale)

    def retrieve(self, query: torch.Tensor) -> torch.Tensor:
        return (query / query.norm()) @ self.C

    def rsv(self, query: torch.Tensor) -> torch.Tensor:
        q = query / query.norm()
        scores = [(q @ k).item() * v.norm().item() for k, v in zip(self.keys, self.values)]
        return torch.tensor(scores, device=self.device)

    def ris(self, rsv_vec: torch.Tensor) -> float:
        a = rsv_vec.abs()
        return (a.max() / a.sum()).item() if a.sum() > 0 else 0.0

    def entropy(self, rsv_vec: torch.Tensor) -> float:
        a = rsv_vec.abs()
        total = a.sum().item()
        if total == 0:
            return 0.0
        p = a / total
        p = p[p > 0]
        return -(p * torch.log(p)).sum().item()

    def coverage(self, query: torch.Tensor) -> float:
        if not self.keys:
            return 0.0
        q = query / query.norm()
        return max((q @ k).item() for k in self.keys)

    def interference_mass(self, query: torch.Tensor, target_domain: str) -> float:
        rsv_vec = self.rsv(query)
        total = rsv_vec.abs().sum().item()
        if total == 0:
            return 0.0
        foreign = sum(
            abs(rsv_vec[i].item())
            for i, d in enumerate(self.domains)
            if d != target_domain
        )
        return foreign / total

    def dominant_pattern(self, query: torch.Tensor) -> dict:
        rsv_vec = self.rsv(query)
        idx = rsv_vec.abs().argmax().item()
        return {
            "idx": idx,
            "label": self.labels[idx],
            "domain": self.domains[idx],
            "contribution": rsv_vec[idx].item(),
        }

    def storage_bound(self, new_key: torch.Tensor, new_value: torch.Tensor) -> list:
        """Pre-execution certificate for storing (new_key, new_value).
        Returns bound on perturbation to each existing query: |cos(s_j, s_n)| * ||y_n||.
        """
        s_n = new_key / new_key.norm()
        y_n_norm = new_value.norm().item()
        return [
            (j, self.labels[j], abs((self.keys[j] @ s_n).item()) * y_n_norm)
            for j in range(len(self.keys))
        ]

    def perturbation_bound(self, k_idx: int) -> list:
        """Pre-execution certificate for removing pattern k_idx.
        Returns bound on perturbation to each other query: |cos(s_j, s_k)| * ||y_k||.
        """
        sk, yk = self.keys[k_idx], self.values[k_idx]
        yk_norm = yk.norm().item()
        return [
            (j, self.labels[j], abs((self.keys[j] @ sk).item()) * yk_norm)
            for j in range(len(self.keys))
            if j != k_idx
        ]

    def repair(self, k_idx: int, epsilon_budget: float = float('inf')) -> dict:
        """Remove pattern k_idx with pre-execution certificate gate.

        epsilon_budget: if max perturbation bound exceeds this, the repair is
        blocked and returns {'blocked': True, 'max_bound': ...}. Set to
        float('inf') (default) to always execute (backward-compatible).
        """
        sk, yk = self.keys[k_idx], self.values[k_idx]
        bounds = self.perturbation_bound(k_idx)
        max_bound = max(b[2] for b in bounds) if bounds else 0.0
        if max_bound > epsilon_budget:
            return {
                "blocked": True,
                "max_bound": max_bound,
                "bounds": bounds,
                "reason": f"max_ε={max_bound:.4f} exceeds budget={epsilon_budget:.4f}",
            }
        self.C -= torch.outer(sk, yk)
        label = self.labels.pop(k_idx)
        self.keys.pop(k_idx)
        self.values.pop(k_idx)
        self.domains.pop(k_idx)
        self.scales.pop(k_idx)
        return {
            "blocked": False,
            "removed": label,
            "max_bound": max_bound,
            "bounds": bounds,
        }

    @staticmethod
    def derived_theta(d: int, P: int, c_min: float = None) -> float:
        """Analytically derive the RIS threshold θ*(d, P, c_min).

        For a query at cosine distance c_min from its target key, with P-1
        random other keys each contributing ~1/sqrt(d) on average:

            θ*(d, P, c_min) = c_min / (c_min + (P-1) / sqrt(d))

        This is the RIS value below which the dominant pattern no longer
        controls the majority of retrieval mass. Replaces manual θ selection.

        If c_min is None, uses the JL near-orthogonality threshold d^{-0.5}
        as a conservative lower bound for c_min (worst-case coverage boundary).
        """
        import math
        if c_min is None:
            c_min = d ** -0.5
        noise_floor = (P - 1) / math.sqrt(d)
        return c_min / (c_min + noise_floor) if (c_min + noise_floor) > 0 else 0.0

    def ghost_patterns(self, threshold: float = None) -> list:
        """Detect ghost interference pairs: cos(s_i, s_j) > d^{-0.5}.

        When two stored keys are too close (above the JL orthogonality
        threshold), they create a spurious retrieval direction with no
        stored key — a 'ghost pattern'. Detectable before any query arrives.

        Returns list of (i, j, cosine) tuples ordered by descending cosine.
        """
        import math
        if threshold is None:
            threshold = self.d ** -0.5
        if len(self.keys) < 2:
            return []
        keys = torch.stack(self.keys)
        G = keys @ keys.T
        ghosts = []
        n = len(self.keys)
        for i in range(n):
            for j in range(i + 1, n):
                cos_ij = abs(G[i, j].item())
                if cos_ij > threshold:
                    ghosts.append((i, j, cos_ij, self.labels[i], self.labels[j]))
        ghosts.sort(key=lambda x: -x[2])
        return ghosts

    def clone(self) -> "CMM":
        new = CMM(self.d, device=self.device)
        new.C = self.C.clone()
        new.keys = [k.clone() for k in self.keys]
        new.values = [v.clone() for v in self.values]
        new.domains = list(self.domains)
        new.labels = list(self.labels)
        new.scales = list(self.scales)
        return new

    def __len__(self):
        return len(self.keys)


class MeMoStack:
    """Two-layer MeMo stack. Output of layer 1 becomes query to layer 2.
    Demonstrates layer-wise RIS — the key MeMo-native property.
    Layer 1 = token/surface semantics. Layer 2 = phrase/concept semantics.
    """

    def __init__(self, d: int, device: torch.device = None):
        self.layer1 = CMM(d, device=device)
        self.layer2 = CMM(d, device=device)
        self.d = d

    def store(self, key: torch.Tensor, value: torch.Tensor, domain: str, label: str = ""):
        self.layer1.store(key, value, domain, label)
        layer1_output = self.layer1.retrieve(key)
        layer2_key = layer1_output / (layer1_output.norm() + 1e-8)
        self.layer2.store(layer2_key, value, domain, label + "_L2")

    def retrieve_layerwise(self, query: torch.Tensor):
        q1 = query / query.norm()
        out1 = self.layer1.retrieve(q1)
        q2 = out1 / (out1.norm() + 1e-8)
        out2 = self.layer2.retrieve(q2)
        return out1, out2

    def ris_layerwise(self, query: torch.Tensor):
        q1 = query / query.norm()
        out1 = self.layer1.retrieve(q1)
        q2 = out1 / (out1.norm() + 1e-8)
        rsv1 = self.layer1.rsv(q1)
        rsv2 = self.layer2.rsv(q2)
        return self.layer1.ris(rsv1), self.layer2.ris(rsv2)


# ============================================================
# AIG: Associative Inference Governor
# Transparent associative retrieval-observability architecture
# for mechanistic interpretability by-design.
# ============================================================


class EpistemicRouter:
    """
    Routes new patterns to the CMM layer with minimum expected max perturbation.
    Pre-execution certificate (storage_bound) determines the optimal layer.
    """

    def __init__(self, layers: list):
        self.layers = layers

    def route(self, key: torch.Tensor, value: torch.Tensor) -> tuple:
        """Returns (best_layer_idx, max_epsilon_at_that_layer)."""
        key_norm = key / key.norm()
        best_idx, best_eps = 0, float('inf')
        for i, layer in enumerate(self.layers):
            if not layer.keys:
                return i, 0.0
            bounds = layer.storage_bound(key_norm, value)
            max_eps = max(b[2] for b in bounds) if bounds else 0.0
            if max_eps < best_eps:
                best_eps, best_idx = max_eps, i
        return best_idx, best_eps


class NLayerStack:
    """
    N-layer CMM stack: the Explicit Associative Memory Substrate.
    Output of layer i normalized → query to layer i+1.
    Layer semantics: L1=token, L2=phrase, L3=semantic, L4=long-context.
    """

    def __init__(self, d: int, n_layers: int):
        self.d = d
        self.n = n_layers
        self.layers = [CMM(d) for _ in range(n_layers)]

    def store(self, key: torch.Tensor, value: torch.Tensor,
              domain: str, label: str = "", layer_idx: int = 0):
        """Store in layer_idx, propagate derived keys up the stack."""
        self.layers[layer_idx].store(key, value, domain, label)
        current = key / key.norm()
        for i in range(layer_idx + 1, self.n):
            output = self.layers[i - 1].retrieve(current)
            norm = output.norm()
            if norm < 1e-8:
                break
            layer_key = output / norm
            self.layers[i].store(layer_key, value, domain, f"{label}_L{i + 1}")
            current = layer_key

    def retrieve_layerwise(self, query: torch.Tensor) -> list:
        """Returns retrieved vector from each layer in order."""
        outputs, current = [], query / query.norm()
        for layer in self.layers:
            if not layer.keys:
                outputs.append(torch.zeros(self.d))
                continue
            output = layer.retrieve(current)
            outputs.append(output)
            norm = output.norm()
            current = output / norm if norm > 1e-8 else current
        return outputs

    def ris_layerwise(self, query: torch.Tensor) -> list:
        """RIS at each layer during a full retrieval pass."""
        ris_scores, current = [], query / query.norm()
        for layer in self.layers:
            if not layer.keys:
                ris_scores.append(0.0)
                continue
            rsv = layer.rsv(current)
            ris_scores.append(layer.ris(rsv))
            output = layer.retrieve(current)
            norm = output.norm()
            current = output / norm if norm > 1e-8 else current
        return ris_scores


class RetrievalStateObserver:
    """
    Tracks retrieval-state evolution across multiple queries.
    Exposes retrieval-state as an observable computational object
    rather than a hidden latent process.

    Records: RSV, RIS, entropy, coverage, dominant pattern — per observation.
    Computes: stability (CV of RIS), H↔L transition counts.
    """

    def __init__(self):
        self.history: list = []

    def observe(self, layer_idx: int, query: torch.Tensor, layer: CMM) -> dict:
        q = query / query.norm()
        rsv = layer.rsv(q)
        state = {
            'layer': layer_idx,
            'rsv': rsv,
            'ris': layer.ris(rsv),
            'entropy': layer.entropy(rsv),
            'coverage': layer.coverage(q),
            'dominant': layer.dominant_pattern(q) if layer.keys else None,
        }
        self.history.append(state)
        return state

    def stability(self, n_last: int = 10) -> float:
        """Coefficient of variation of RIS over last n observations."""
        recent = [s['ris'] for s in self.history[-n_last:] if s['ris'] > 0]
        if len(recent) < 2:
            return 0.0
        arr = torch.tensor(recent)
        return (arr.std() / arr.mean()).item()

    def transitions(self, theta: float) -> dict:
        """Count H→L and L→H retrieval-state transitions."""
        ris_seq = [s['ris'] for s in self.history]
        hl = lh = 0
        for i in range(1, len(ris_seq)):
            prev_h = ris_seq[i - 1] >= theta
            curr_h = ris_seq[i] >= theta
            if prev_h and not curr_h:
                hl += 1
            if not prev_h and curr_h:
                lh += 1
        return {'H_to_L': hl, 'L_to_H': lh, 'total': len(ris_seq)}


class InterferenceAnalyzer:
    """
    Interference analysis engine for a single CMM layer.

    Interference types (from AIG spec):
    - Retrieval collision: top-k competing patterns
    - Crosstalk: cross-domain activation mass
    - Dense activation: conflict density between stored keys
    - Context drift: recall instability under query perturbation
    - Recursive contamination: tracked via PropagationTracker
    """

    def __init__(self, layer: CMM):
        self.layer = layer

    def retrieval_collision(self, query: torch.Tensor, top_k: int = 3) -> list:
        """Top-k competing patterns by |RSV|."""
        rsv = self.layer.rsv(query)
        k = min(top_k, len(self.layer.keys))
        vals, idxs = rsv.abs().topk(k)
        return [{'idx': idx.item(), 'label': self.layer.labels[idx.item()],
                 'rsv': rsv[idx.item()].item()} for idx in idxs]

    def crosstalk(self, query: torch.Tensor, domain: str) -> float:
        """Fraction of total |RSV| mass coming from outside the target domain."""
        return self.layer.interference_mass(query, domain)

    def conflict_density(self) -> float:
        """Mean off-diagonal |cosine| between stored keys — crowding measure."""
        if len(self.layer.keys) < 2:
            return 0.0
        keys = torch.stack(self.layer.keys)
        sim = (keys @ keys.T).abs()
        n = len(self.layer.keys)
        off_diag = sim.sum() - sim.diag().sum()
        return (off_diag / (n * (n - 1))).item()

    def recall_instability(self, query: torch.Tensor,
                           n_perturb: int = 10, noise_std: float = 0.01) -> float:
        """Standard deviation of RIS under small query perturbations."""
        q_norm = query / query.norm()
        samples = []
        for _ in range(n_perturb):
            noise = torch.randn_like(q_norm) * noise_std
            q_p = (q_norm + noise)
            q_p = q_p / q_p.norm()
            rsv = self.layer.rsv(q_p)
            samples.append(self.layer.ris(rsv))
        return torch.tensor(samples).std().item()

    def activation_overlap(self, q1: torch.Tensor, q2: torch.Tensor) -> float:
        """Jaccard overlap of above-median activated pattern sets for two queries."""
        rsv1 = self.layer.rsv(q1 / q1.norm()).abs()
        rsv2 = self.layer.rsv(q2 / q2.norm()).abs()
        set1 = set((rsv1 > rsv1.median()).nonzero().flatten().tolist())
        set2 = set((rsv2 > rsv2.median()).nonzero().flatten().tolist())
        if not set1 and not set2:
            return 0.0
        return len(set1 & set2) / len(set1 | set2)

    def full_report(self, query: torch.Tensor, domain: str = None) -> dict:
        if not self.layer.keys:
            return {'empty': True}
        q = query / query.norm()
        rsv = self.layer.rsv(q)
        return {
            'ris': self.layer.ris(rsv),
            'entropy': self.layer.entropy(rsv),
            'conflict_density': self.conflict_density(),
            'recall_instability': self.recall_instability(q),
            'top_collisions': self.retrieval_collision(q),
            'crosstalk': self.crosstalk(q, domain) if domain else None,
        }


class PropagationTracker:
    """
    Tracks how retrieval corruption propagates across layers over time.

    Central question (from AIG spec): Do failures remain localized?
    Do they spread recursively? Do they amplify? Do they destabilize future retrieval?
    """

    def __init__(self, layers: list):
        self.layers = layers
        self.events: list = []

    def snapshot(self, query: torch.Tensor, step: int,
                 event_type: str = 'normal') -> dict:
        """Record full layerwise RIS snapshot at this step."""
        current = query / query.norm()
        ris_list = []
        for layer in self.layers:
            if not layer.keys:
                ris_list.append(0.0)
                continue
            rsv = layer.rsv(current)
            ris_list.append(layer.ris(rsv))
            output = layer.retrieve(current)
            norm = output.norm()
            current = output / norm if norm > 1e-8 else current
        event = {'step': step, 'ris_per_layer': ris_list, 'event_type': event_type}
        self.events.append(event)
        return event

    def propagation_depth(self, theta: float, inject_step: int) -> int:
        """Number of distinct layers that dropped below theta after injection."""
        post = [e for e in self.events if e['step'] >= inject_step]
        affected = [False] * len(self.layers)
        for event in post:
            for i, ris in enumerate(event['ris_per_layer']):
                if ris < theta:
                    affected[i] = True
        return sum(affected)

    def propagation_matrix(self, theta: float) -> list:
        """
        Co-occurrence matrix: counts[i][j] = steps where both layer i and j are below theta.
        Diagonal = self-interference; off-diagonal = cross-layer propagation.
        """
        n = len(self.layers)
        counts = [[0] * n for _ in range(n)]
        for event in self.events:
            ris = event['ris_per_layer']
            for i in range(n):
                for j in range(n):
                    if ris[i] < theta and ris[j] < theta:
                        counts[i][j] += 1
        return counts

    def delta_by_layer(self, baseline_step: int, inject_step: int) -> list:
        """Mean RIS drop per layer: baseline mean minus post-injection mean."""
        pre = [e for e in self.events if e['step'] < inject_step]
        post = [e for e in self.events if e['step'] >= inject_step]
        if not pre or not post:
            return []
        n = len(self.layers)
        pre_mean = [sum(e['ris_per_layer'][i] for e in pre) / len(pre) for i in range(n)]
        post_mean = [sum(e['ris_per_layer'][i] for e in post) / len(post) for i in range(n)]
        return [pre_mean[i] - post_mean[i] for i in range(n)]


class ReliabilityEstimator:
    """
    Estimates retrieval integrity from structural signals.

    Reliability emerges from retrieval topology (RIS, entropy, sparsity, coverage),
    not output confidence — this is the AIG design principle.
    """

    def __init__(self, theta: float = 0.4):
        self.theta = theta

    def estimate(self, layer: CMM, query: torch.Tensor) -> dict:
        if not layer.keys:
            return {'score': 0.0, 'status': 'EMPTY', 'signals': {}}
        q = query / query.norm()
        rsv = layer.rsv(q)
        ris = layer.ris(rsv)
        entropy = layer.entropy(rsv)
        coverage = layer.coverage(q)
        # Sparsity: fraction of patterns contributing less than 10% of max |RSV|
        max_rsv = rsv.abs().max().item()
        sparsity = ((rsv.abs() < 0.1 * max_rsv).float().mean().item()
                    if max_rsv > 0 else 0.0)
        score = (ris + coverage + sparsity) / 3.0
        if score >= 0.7:
            status = 'HIGH_RELIABILITY'
        elif score >= 0.4:
            status = 'MODERATE_RELIABILITY'
        else:
            status = 'LOW_RELIABILITY'
        return {
            'score': score,
            'status': status,
            'signals': {
                'ris': ris,
                'entropy': entropy,
                'coverage': coverage,
                'sparsity': sparsity,
            },
        }


class TraceLogger:
    """
    Mechanistic audit trail. Records retrieval paths, interference events,
    intervention outcomes, and propagation behavior.
    Implements explicit mechanistic auditability — central to interpretability by-design.
    """

    def __init__(self):
        self.trace: list = []

    def log(self, event_type: str, **kwargs):
        self.trace.append({'type': event_type, **kwargs})

    def export(self) -> list:
        return self.trace

    def retrieval_path(self, step: int = None) -> list:
        events = [e for e in self.trace if e['type'] == 'retrieval']
        return [e for e in events if e.get('step') == step] if step is not None else events

    def interference_events(self) -> list:
        return [e for e in self.trace if e['type'] == 'interference_alert']

    def intervention_events(self) -> list:
        return [e for e in self.trace if e['type'] == 'intervention']

    def summary(self) -> dict:
        return {
            'total_events': len(self.trace),
            'retrievals': len(self.retrieval_path()),
            'interference_alerts': len(self.interference_events()),
            'interventions': len(self.intervention_events()),
        }


class AIG:
    """
    Associative Inference Governor.
    Transparent associative retrieval-observability architecture
    for mechanistic interpretability by-design.

    Core scientific question:
    Can explicit associative-memory retrieval structures make internal retrieval
    dynamics more observable, localizable, and controllable than distributed
    transformer representations?

    Architecture (10 components from the AIG spec):
    [Substrate]      NLayerStack (L1=token, L2=phrase, L3=semantic, L4=long-context)
    [Observability]  RetrievalStateObserver, InterferenceAnalyzer, PropagationTracker
    [Reliability]    ReliabilityEstimator
    [Audit]          TraceLogger
    [Governance]     EpistemicRouter + EG-AMS repair (C -= s_k y_k^T)
    """

    def __init__(self, d: int, n_layers: int = 4,
                 theta: float = 0.4, tau: float = 0.5):
        self.d = d
        self.n = n_layers
        self.theta = theta   # RIS threshold: below this → interference alert + repair
        self.tau = tau       # Coverage threshold: below this → OUT_OF_DISTRIBUTION

        # Substrate
        self.stack = NLayerStack(d, n_layers)

        # Observability components
        self.observer = RetrievalStateObserver()
        self.analyzers = [InterferenceAnalyzer(layer) for layer in self.stack.layers]
        self.tracker = PropagationTracker(self.stack.layers)
        self.reliability = ReliabilityEstimator(theta)
        self.logger = TraceLogger()

        # Governance
        self.router = EpistemicRouter(self.stack.layers)
        self.locked: set = set()   # (layer_idx, pattern_idx) — immutable anchors
        self._step: int = 0

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def store(self, key: torch.Tensor, value: torch.Tensor,
              domain: str, label: str = "",
              anchor: bool = False, routed: bool = True) -> dict:
        """
        Store a pattern with pre-execution certificate.
        routed=True: EpistemicRouter selects the least-interfering layer.
        anchor=True: pattern is immutable — cannot be removed by auto-repair.
        """
        if routed:
            layer_idx, max_eps = self.router.route(key, value)
        else:
            layer_idx, max_eps = 0, 0.0

        self.stack.store(key, value, domain, label, layer_idx=layer_idx)
        pattern_idx = len(self.stack.layers[layer_idx].keys) - 1

        if anchor:
            self.locked.add((layer_idx, pattern_idx))

        entry = {
            'layer': layer_idx, 'pattern_idx': pattern_idx,
            'max_epsilon': max_eps, 'anchored': anchor,
            'label': label, 'domain': domain,
        }
        self.logger.log('store', **entry)
        return entry

    def anchor(self, layer_idx: int, pattern_idx: int):
        """Mark an existing pattern as immutable base truth."""
        self.locked.add((layer_idx, pattern_idx))

    # ------------------------------------------------------------------
    # Forward pass — full observability
    # ------------------------------------------------------------------

    def forward(self, query: torch.Tensor,
                step: int = None, event_type: str = 'normal') -> dict:
        """
        Full AIG forward pass.
        Returns output + complete per-layer observability state.
        """
        if step is None:
            step = self._step
        self._step += 1

        q_norm = query / query.norm()
        layer_states = []
        current = q_norm

        for i, layer in enumerate(self.stack.layers):
            if not layer.keys:
                layer_states.append({'layer': i, 'skipped': True})
                continue

            state = self.observer.observe(i, current, layer)
            analysis = self.analyzers[i].full_report(current)
            rel = self.reliability.estimate(layer, current)
            retrieved = layer.retrieve(current)

            if state['ris'] < self.theta:
                self.logger.log('interference_alert', step=step, layer=i,
                                ris=state['ris'], entropy=state['entropy'])

            layer_states.append({
                'layer': i,
                'state': state,
                'analysis': analysis,
                'reliability': rel,
                'retrieved': retrieved,
            })

            norm = retrieved.norm()
            current = retrieved / norm if norm > 1e-8 else current

        self.tracker.snapshot(query, step, event_type)
        self.logger.log('retrieval', step=step, n_active_layers=len(layer_states))
        return {'output': current, 'layer_states': layer_states, 'step': step}

    # ------------------------------------------------------------------
    # Intervention (Localized Intervention Engine)
    # ------------------------------------------------------------------

    def intervene(self, layer_idx: int, pattern_idx: int,
                  epsilon_budget: float = float('inf')) -> dict:
        """
        Localized intervention: algebraic removal C -= s_k y_k^T.
        Blocked if pattern is anchored or if max_ε exceeds epsilon_budget.
        EG-AMS certificate (max_bound) returned with the result.
        """
        if (layer_idx, pattern_idx) in self.locked:
            return {'blocked': 'anchored', 'layer': layer_idx, 'idx': pattern_idx}
        info = self.stack.layers[layer_idx].repair(pattern_idx, epsilon_budget)
        self.logger.log('intervention', layer=layer_idx, **info)
        return {'layer': layer_idx, **info}

    def auto_intervene(self, query: torch.Tensor) -> list:
        """
        RIS-triggered auto-repair: fires across any layer where RIS < theta.
        Removes the worst non-dominant, non-anchored interferer per layer.
        """
        current = query / query.norm()
        repairs = []
        for i, layer in enumerate(self.stack.layers):
            if not layer.keys:
                continue
            rsv = layer.rsv(current)
            ris_val = layer.ris(rsv)
            if ris_val < self.theta and len(layer.keys) >= 2:
                rsv_abs = rsv.abs().clone()
                dominant = rsv_abs.argmax().item()
                rsv_abs[dominant] = 0.0
                if rsv_abs.sum() > 0:
                    interferer = rsv_abs.argmax().item()
                    info = self.intervene(i, interferer)
                    repairs.append(info)
            output = layer.retrieve(current)
            norm = output.norm()
            current = output / norm if norm > 1e-8 else current
        return repairs

    # ------------------------------------------------------------------
    # Observability reports
    # ------------------------------------------------------------------

    def smoke_detector(self, query: torch.Tensor) -> dict:
        """
        Hallucination smoke detector: per-layer RIS alerts without triggering repair.
        Severity: HIGH if RIS < theta/2, MEDIUM if RIS < theta.
        """
        current = query / query.norm()
        alerts = []
        for i, layer in enumerate(self.stack.layers):
            if not layer.keys:
                continue
            rsv = layer.rsv(current)
            ris_val = layer.ris(rsv)
            if ris_val < self.theta:
                alerts.append({
                    'layer': i, 'ris': ris_val,
                    'severity': 'HIGH' if ris_val < self.theta / 2 else 'MEDIUM',
                    'dominant': layer.dominant_pattern(current),
                })
            output = layer.retrieve(current)
            norm = output.norm()
            current = output / norm if norm > 1e-8 else current
        return {
            'alerts': alerts,
            'hallucination_risk': len(alerts) > 0,
            'n_interfering_layers': len(alerts),
        }

    def retrieval_integrity_report(self, query: torch.Tensor) -> dict:
        """Complete structural integrity report across all layers."""
        current = query / query.norm()
        per_layer = []
        for i, layer in enumerate(self.stack.layers):
            if not layer.keys:
                continue
            per_layer.append({
                'layer': i,
                'reliability': self.reliability.estimate(layer, current),
                'analysis': self.analyzers[i].full_report(current),
            })
            output = layer.retrieve(current)
            norm = output.norm()
            current = output / norm if norm > 1e-8 else current
        return {
            'per_layer': per_layer,
            'observer_stability': self.observer.stability(),
            'propagation_depth': self.tracker.propagation_depth(self.theta, 0),
            'trace_summary': self.logger.summary(),
        }
