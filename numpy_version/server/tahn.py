"""Full TAHN (Topology-Aware HyperNetwork) for heterogeneous FL.

Implements Algorithm 1 from FedMBridge (Chen & Zhang, ICML 2024).

Components:
  GNNEncoder      — φ₁: L-layer GNN on chain graphs, updated via chain rule
  WeightGenerator — φ₂: shared backbone + per-head output, updated via chain rule
  TAHN            — server bridge: stores prev generated weights, computes
                    Δθᵢ = θ̃ᵢ - θᵢ_prev, runs backward (updates φ₁,φ₂,cᵢ),
                    then forward (generates personalized weights) each round.

Chain-graph GNN (one node per network layer v):
    pre^(l) = Z^(l-1) @ W_self
            + shift_right(Z^(l-1)) @ W_in    # predecessor message
            + shift_left(Z^(l-1))  @ W_out   # successor message
            + b
    Z^(l)   = tanh(pre^(l))

Weight generator for layer v of client i:
    x_v   = concat(z_v, c_i)
    h_v   = tanh(x_v @ W_gen + b_gen)
    θ_v   = h_v @ W_dec + b_dec       (W_dec, b_dec keyed by (fan_in, fan_out))
"""

import numpy as np

# ── Dimensions ────────────────────────────────────────────────────────────────

ARCH_DIM    = 16
TASK_DIM    = 16
ROLE_DIM    = 8
LATENT_DIM  = 32
GEN_DIM     = 32


# ── Architecture / layer encoding ─────────────────────────────────────────────

def encode_arch(layer_sizes: list, dim: int = ARCH_DIM) -> np.ndarray:
    """Encode architecture descriptor into a fixed-dim feature vector."""
    n = len(layer_sizes)
    total_params = sum(
        layer_sizes[i] * layer_sizes[i + 1] + layer_sizes[i + 1]
        for i in range(n - 1)
    )
    feat = [
        (n - 1) / 5.0,
        np.log1p(total_params) / 10.0,
        min(layer_sizes) / 64.0,
        max(layer_sizes) / 64.0,
        layer_sizes[0]  / 8.0,
        layer_sizes[-1] / 8.0,
    ]
    pad = dim - len(feat)
    sizes_norm = [s / 32.0 for s in layer_sizes]
    feat.extend((sizes_norm + [0.0] * pad)[:pad])
    return np.array(feat, dtype=np.float32)


def encode_layer_role(layer_idx: int, total_layers: int,
                       fan_in: int, fan_out: int,
                       dim: int = ROLE_DIM) -> np.ndarray:
    """Encode a single layer's structural role within its network."""
    rel  = layer_idx / max(total_layers - 1, 1)
    feat = [
        rel,
        float(layer_idx == 0),
        float(layer_idx == total_layers - 1),
        float(0 < layer_idx < total_layers - 1),
        fan_in  / 64.0,
        fan_out / 64.0,
        np.log1p(fan_in * fan_out) / 10.0,
        (fan_out - fan_in) / 64.0,
    ]
    return np.array(feat[:dim], dtype=np.float32)


def role_key(layer_idx: int, total_layers: int) -> str:
    """Canonical role label for a layer (used in diagnostics)."""
    if layer_idx == 0:
        return "input"
    if layer_idx == total_layers - 1:
        return "output"
    return f"hidden_{layer_idx - 1}"


# ── Chain-graph message-passing helpers ───────────────────────────────────────

def shift_right(Z: np.ndarray) -> np.ndarray:
    """Predecessor message: shift_right(Z)[v] = Z[v-1] for v>0, else 0."""
    out = np.zeros_like(Z)
    if Z.shape[0] > 1:
        out[1:] = Z[:-1]
    return out


def shift_left(Z: np.ndarray) -> np.ndarray:
    """Successor message: shift_left(Z)[v] = Z[v+1] for v<V-1, else 0."""
    out = np.zeros_like(Z)
    if Z.shape[0] > 1:
        out[:-1] = Z[1:]
    return out


# ── GNN Encoder  (φ₁) ────────────────────────────────────────────────────────

class GNNEncoder:
    """L-layer GNN encoding architecture topology for a chain graph.

    forward(X) → (Z_final, cache)
    backward(grad_Z, cache) → updates W_self/W_in/W_out/b in place, returns grad_X
    """

    def __init__(self, d_in: int, latent_dim: int,
                 n_gnn_layers: int = 2, lr: float = 1e-3, seed: int = 0):
        rng = np.random.RandomState(seed)
        self.d_in       = d_in
        self.latent_dim = latent_dim
        self.n_layers   = n_gnn_layers
        self.lr         = lr
        self.layers     = []

        for i in range(n_gnn_layers):
            d = d_in if i == 0 else latent_dim
            s = 0.1 / np.sqrt(d)
            self.layers.append({
                'W_self': rng.randn(d, latent_dim).astype(np.float32) * s,
                'W_in':   rng.randn(d, latent_dim).astype(np.float32) * s,
                'W_out':  rng.randn(d, latent_dim).astype(np.float32) * s,
                'b':      np.zeros(latent_dim, dtype=np.float32),
            })

    def forward(self, X: np.ndarray):
        """X: (V, d_in). Returns (Z_final (V, latent_dim), cache)."""
        cache = []
        Z = X.astype(np.float32)
        for lyr in self.layers:
            pre  = (Z @ lyr['W_self']
                    + shift_right(Z) @ lyr['W_in']
                    + shift_left(Z)  @ lyr['W_out']
                    + lyr['b'])
            Z_new = np.tanh(pre)
            cache.append({'Z_in': Z, 'pre': pre, 'Z_out': Z_new})
            Z = Z_new
        return Z, cache

    def backward(self, grad_Z: np.ndarray, cache: list) -> np.ndarray:
        """Update φ₁ in place. Returns grad_X (gradient w.r.t. input features)."""
        g = grad_Z.astype(np.float32)
        for i in reversed(range(self.n_layers)):
            lyr   = self.layers[i]
            Z_in  = cache[i]['Z_in']
            Z_out = cache[i]['Z_out']

            g_pre = g * (1.0 - Z_out ** 2)

            lyr['W_self'] += self.lr * (Z_in.T              @ g_pre)
            lyr['W_in']   += self.lr * (shift_right(Z_in).T @ g_pre)
            lyr['W_out']  += self.lr * (shift_left(Z_in).T  @ g_pre)
            lyr['b']      += self.lr * g_pre.sum(axis=0)

            # grad_Z_prev: transpose of shift_right is shift_left, and vice versa
            g = (g_pre @ lyr['W_self'].T
                 + shift_left(g_pre)  @ lyr['W_in'].T
                 + shift_right(g_pre) @ lyr['W_out'].T)

        return g


# ── Weight Generator  (φ₂) ───────────────────────────────────────────────────

class WeightGenerator:
    """Personalized weight generator for a single layer.

    forward(z_v, c_i, fan_in, fan_out) → (theta_v, h_v, x_v)
    backward(delta_theta_v, h_v, x_v, fan_in, fan_out) → (grad_z_v, grad_c_i)
        Updates W_gen, b_gen, and the (fan_in, fan_out) output head in place.
    """

    def __init__(self, latent_dim: int, task_dim: int,
                 gen_dim: int = GEN_DIM, lr: float = 1e-3, seed: int = 0):
        rng = np.random.RandomState(seed + 100)
        self.latent_dim = latent_dim
        self.task_dim   = task_dim
        self.gen_dim    = gen_dim
        self.lr         = lr
        self.gen_in     = latent_dim + task_dim

        s = 0.1 / np.sqrt(self.gen_in)
        self.W_gen = rng.randn(self.gen_in, gen_dim).astype(np.float32) * s
        self.b_gen = np.zeros(gen_dim, dtype=np.float32)

        self._heads: dict          = {}
        self._head_rng             = np.random.RandomState(seed + 200)

    def _get_head(self, fan_in: int, fan_out: int) -> dict:
        k = (fan_in, fan_out)
        if k not in self._heads:
            n_out = fan_in * fan_out + fan_out
            W = self._head_rng.randn(self.gen_dim, n_out).astype(np.float32) * 0.01
            self._heads[k] = {'W': W, 'b': np.zeros(n_out, dtype=np.float32)}
        return self._heads[k]

    def forward(self, z_v: np.ndarray, c_i: np.ndarray,
                fan_in: int, fan_out: int):
        """Returns (theta_v_flat, h_v, x_v)."""
        x_v   = np.concatenate([z_v, c_i]).astype(np.float32)
        h_v   = np.tanh(x_v @ self.W_gen + self.b_gen)
        head  = self._get_head(fan_in, fan_out)
        theta = h_v @ head['W'] + head['b']
        return theta, h_v, x_v

    def backward(self, delta_theta_v: np.ndarray, h_v: np.ndarray,
                  x_v: np.ndarray, fan_in: int, fan_out: int):
        """Update φ₂ params. Returns (grad_z_v, grad_c_i)."""
        head = self._get_head(fan_in, fan_out)
        sig  = delta_theta_v.astype(np.float32)

        # Output head
        head['W'] += self.lr * np.outer(h_v, sig)
        head['b'] += self.lr * sig

        # Backbone
        grad_h   = sig @ head['W'].T
        grad_pre = grad_h * (1.0 - h_v ** 2)
        self.W_gen += self.lr * np.outer(x_v, grad_pre)
        self.b_gen += self.lr * grad_pre

        # Propagate
        grad_x   = grad_pre @ self.W_gen.T
        return grad_x[:self.latent_dim], grad_x[self.latent_dim:]


# ── TAHN ─────────────────────────────────────────────────────────────────────

class TAHN:
    """Topology-Aware HyperNetwork — server-side bridge for heterogeneous FL.

    Each FL round (aggregate call):
      1. For every client that returned trained weights θ̃ᵢ:
           Δθᵢ = θ̃ᵢ − θᵢ_prev  (client's local improvement)
           Backward through weight generator → updates φ₂, cᵢ
           Backward through GNN encoder      → updates φ₁
      2. Forward pass for every client → new personalized θᵢ
      3. Store new θᵢ as θᵢ_prev for next round.

    clients dict:
        { label: { 'arch':      list[int],
                   'task_init': np.ndarray | None,
                   'weights':   np.ndarray } }
    Returns:
        { label: np.ndarray }  — flat float32 personalized weight vector
    """

    def __init__(self,
                 arch_dim:     int   = ARCH_DIM,
                 task_dim:     int   = TASK_DIM,
                 role_dim:     int   = ROLE_DIM,
                 latent_dim:   int   = LATENT_DIM,
                 gen_dim:      int   = GEN_DIM,
                 n_gnn_layers: int   = 2,
                 lr_phi:       float = 1e-3,
                 lr_task:      float = 5e-3,
                 seed:         int   = 0):

        self.arch_dim   = arch_dim
        self.task_dim   = task_dim
        self.role_dim   = role_dim
        self.latent_dim = latent_dim
        self.lr_task    = lr_task
        self.round      = 0

        d_in = role_dim + arch_dim
        self.encoder   = GNNEncoder(d_in, latent_dim, n_gnn_layers, lr_phi, seed)
        self.generator = WeightGenerator(latent_dim, task_dim, gen_dim, lr_phi, seed)

        self._task_embeds: dict = {}   # label → float32[task_dim]
        self._prev_gen:    dict = {}   # label → float32[n_params] (previous generated)
        self._arch_cache:  dict = {}   # label → list[int]
        self._round_stats: dict = {}   # populated each aggregate() call; keys: received, backward, forward

    # ── Initialization ────────────────────────────────────────────────────────

    def seed_prev_gen(self, label: str, arch: list, weights: np.ndarray):
        """Seed _prev_gen with the initial model dispatched to a client.
        Allows round 1 to compute a real delta instead of skipping backward."""
        self._arch_cache[label] = arch
        self._prev_gen[label]   = weights.flatten().astype(np.float32).copy()

    def set_task_embed(self, label: str, init_vec: np.ndarray | None):
        """Initialize task embedding from data statistics (called once per client)."""
        if label in self._task_embeds:
            return
        if init_vec is not None:
            v = np.array(init_vec, dtype=np.float32).flatten()
            if len(v) < self.task_dim:
                v = np.pad(v, (0, self.task_dim - len(v)))
            self._task_embeds[label] = v[:self.task_dim].copy()
        else:
            rng = np.random.RandomState(abs(hash(label)) % (2 ** 31))
            self._task_embeds[label] = rng.randn(self.task_dim).astype(np.float32) * 0.1

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _node_features(self, arch: list) -> np.ndarray:
        """Build X ∈ R^{V × (role_dim + arch_dim)}, one row per layer."""
        arch_feat = encode_arch(arch, self.arch_dim)
        n_layers  = len(arch) - 1
        rows = []
        for v in range(n_layers):
            rv = encode_layer_role(v, n_layers, arch[v], arch[v + 1], self.role_dim)
            rows.append(np.concatenate([rv, arch_feat]))
        return np.stack(rows).astype(np.float32)

    def _split_flat(self, flat: np.ndarray, arch: list) -> list:
        """Split flat weight vector into per-layer float32 arrays."""
        parts, idx = [], 0
        for v in range(len(arch) - 1):
            n = arch[v] * arch[v + 1] + arch[v + 1]
            parts.append(flat[idx:idx + n].astype(np.float32))
            idx += n
        return parts

    # ── Main aggregation ──────────────────────────────────────────────────────

    def aggregate(self, clients: dict) -> dict:
        self.round += 1
        rcv_stats: dict = {}   # per-label received weight info
        bwd_stats: dict = {}   # per-label backward stats
        fwd_stats: dict = {}   # per-label forward stats

        # Initialize task embeddings / arch cache for new clients
        for label, info in clients.items():
            self.set_task_embed(label, info.get('task_init'))
            if 'arch' in info:
                self._arch_cache[label] = info['arch']

        # ── Record received weights ───────────────────────────────────────────
        for label, info in clients.items():
            w = info.get('weights')
            if w is not None:
                wf = w.flatten().astype(np.float32)
                arch = self._arch_cache.get(label, info.get('arch', [3, 1]))
                rcv_stats[label] = {
                    'arch':        arch,
                    'n_params':    len(wf),
                    'weight_norm': float(np.linalg.norm(wf)),
                    'has_prev':    label in self._prev_gen,
                }

        # ── Backward: update φ₁, φ₂, cᵢ from each client's Δθᵢ ─────────────
        for label, info in clients.items():
            if info.get('weights') is None:
                continue

            arch     = self._arch_cache.get(label, info.get('arch', [3, 1]))
            n_layers = len(arch) - 1

            theta_tilde = info['weights'].flatten().astype(np.float32)

            if label not in self._prev_gen:
                # First contribution from this client — store as baseline, skip update
                self._prev_gen[label] = theta_tilde.copy()
                bwd_stats[label] = {'first_round': True, 'arch': arch,
                                    'n_params': len(theta_tilde)}
                continue

            delta_theta  = theta_tilde - self._prev_gen[label]
            delta_layers = self._split_flat(delta_theta, arch)
            c_i          = self._task_embeds[label]
            c_norm_before = float(np.linalg.norm(c_i))

            # Forward through GNN (need cache for backward)
            X_feat         = self._node_features(arch)
            Z_final, cache = self.encoder.forward(X_feat)

            # Backward through weight generator, accumulate grad_Z per node
            grad_Z      = np.zeros_like(Z_final)
            grad_c_acc  = np.zeros_like(c_i)
            gen_grad_norm = 0.0

            for v in range(n_layers):
                fan_in, fan_out = arch[v], arch[v + 1]
                _, h_v, x_v = self.generator.forward(Z_final[v], c_i, fan_in, fan_out)
                g_z, g_c    = self.generator.backward(
                    delta_layers[v], h_v, x_v, fan_in, fan_out)
                grad_Z[v]    += g_z
                grad_c_acc   += g_c
                gen_grad_norm = max(gen_grad_norm, float(np.linalg.norm(g_z)))

            # Update task embedding
            self._task_embeds[label] = (c_i + self.lr_task * grad_c_acc).astype(np.float32)
            c_norm_after = float(np.linalg.norm(self._task_embeds[label]))

            # Backward through GNN encoder — updates φ₁
            grad_X = self.encoder.backward(grad_Z, cache)
            enc_grad_norm = float(np.linalg.norm(grad_Z))

            bwd_stats[label] = {
                'arch':          arch,
                'n_params':      len(theta_tilde),
                'delta_norm':    float(np.linalg.norm(delta_theta)),
                'enc_grad_norm': enc_grad_norm,
                'gen_grad_norm': gen_grad_norm,
                'c_norm_before': c_norm_before,
                'c_norm_after':  c_norm_after,
                'delta_c_norm':  float(np.linalg.norm(self.lr_task * grad_c_acc)),
            }

        # ── Forward: generate personalized weights for all clients ────────────
        personalized: dict = {}
        for label, info in clients.items():
            arch     = self._arch_cache.get(label, info.get('arch', [3, 1]))
            n_layers = len(arch) - 1
            c_i      = self._task_embeds[label]

            X_feat         = self._node_features(arch)
            Z_final, _     = self.encoder.forward(X_feat)

            parts = []
            for v in range(n_layers):
                fan_in, fan_out = arch[v], arch[v + 1]
                theta_v, _, _   = self.generator.forward(Z_final[v], c_i, fan_in, fan_out)
                parts.append(theta_v)

            flat = np.concatenate(parts).astype(np.float32)
            personalized[label] = flat
            self._prev_gen[label] = flat.copy()
            fwd_stats[label] = {
                'arch':          arch,
                'n_params':      len(flat),
                'weight_norm':   float(np.linalg.norm(flat)),
            }

        self._round_stats = {'received': rcv_stats, 'backward': bwd_stats, 'forward': fwd_stats}
        return personalized

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def round_log(self) -> str:
        stats = self._round_stats
        if not stats:
            return f"TAHN round={self.round}  (no stats)"
        lines = [f"TAHN round={self.round}"]
        lines.append("  ------- received -------")
        for label, s in stats.get('received', {}).items():
            lines.append(f"    client={label}  arch={s['arch']}  n_params={s['n_params']}")
        lines.append("  ------- backprop -------")
        for label, s in stats.get('backward', {}).items():
            if s.get('first_round'):
                lines.append(f"    client={label}  arch={s['arch']}  n_params={s['n_params']}  (first round, skipped)")
            else:
                lines.append(f"    client={label}  arch={s['arch']}  n_params={s['n_params']}")
        lines.append("  ------- update -------")
        for label, s in stats.get('forward', {}).items():
            lines.append(f"    client={label}  arch={s['arch']}  n_params={s['n_params']}")
        return "\n".join(lines)

    def summary(self) -> str:
        lines = [
            f"TAHN  round={self.round}  "
            f"gnn_layers={self.encoder.n_layers}  "
            f"latent={self.latent_dim}  task={self.task_dim}  "
            f"clients={list(self._task_embeds.keys())}"
        ]
        for label, c in self._task_embeds.items():
            arch = self._arch_cache.get(label, "?")
            lines.append(
                f"  [{label}]  arch={arch}  task_norm={np.linalg.norm(c):.4f}"
            )
        return "\n".join(lines)


# ── Task embedding helper (kept for firesim_client.py compatibility) ──────────

def make_task_embedding(X: np.ndarray, y: np.ndarray,
                         node_id: int, dim: int = TASK_DIM) -> np.ndarray:
    """Compute a task embedding from local data statistics."""
    feat = [
        float(np.mean(X[:, 0])),   float(np.std(X[:, 0])),
        float(np.mean(X[:, 1])),   float(np.std(X[:, 1])),
        float(np.mean(X[:, 2])),   float(np.std(X[:, 2])),
        float(np.mean(y)),         float(np.std(y)),
        float(np.min(y)),          float(np.max(y)),
        float(node_id),
        float(len(X)),
        float(np.corrcoef(X[:, 0], y.flatten())[0, 1]),
        float(np.corrcoef(X[:, 1], y.flatten())[0, 1]),
        float(np.corrcoef(X[:, 2], y.flatten())[0, 1]),
        0.0,
    ]
    return np.array(feat[:dim], dtype=np.float32)
