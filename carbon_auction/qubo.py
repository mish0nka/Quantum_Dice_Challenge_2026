"""
Stage 2 QUBO construction for the WDP: set-packing via pairwise penalties + knapsack
cap via binary (log) slack. Used here as the "naive" encoding to contrast with the
hardware-aware encoding in qubo_hw.py.
"""
from __future__ import annotations
import numpy as np
from .generator import AuctionInstance


class QUBO:
    """Sparse symmetric QUBO held as {(i,j): coeff, i<j} plus linear dict and offset."""

    def __init__(self, num_vars: int):
        self.n = num_vars
        self.lin: dict[int, float] = {}
        self.quad: dict[tuple[int, int], float] = {}
        self.offset: float = 0.0

    def add_linear(self, i, c):
        self.lin[i] = self.lin.get(i, 0.0) + c

    def add_quadratic(self, i, j, c):
        if i == j:
            self.add_linear(i, c)
        else:
            key = (i, j) if i < j else (j, i)
            self.quad[key] = self.quad.get(key, 0.0) + c

    def to_matrix(self) -> np.ndarray:
        Q = np.zeros((self.n, self.n))
        for i, c in self.lin.items():
            Q[i, i] += c
        for (i, j), c in self.quad.items():
            Q[i, j] += c
        return Q

    def energy(self, x) -> float:
        e = self.offset
        for i, c in self.lin.items():
            e += c * x[i]
        for (i, j), c in self.quad.items():
            e += c * x[i] * x[j]
        return e


def build_qubo(inst: AuctionInstance, alpha_pack=2.0, alpha_cap=2.0):
    """Stage 2 QUBO. Penalties are multiples of the largest scaled value."""
    N = inst.N
    v_scale = float(inst.values.max())
    c_scale = float(inst.credits.max())
    v = inst.values / v_scale
    c = inst.credits / c_scale
    cap_s = inst.cap / c_scale
    v_max = float(v.max())
    P_pack = alpha_pack * v_max
    P_cap = alpha_cap * v_max
    K = int(np.floor(np.log2(max(inst.cap, 1)))) + 1
    slack_weights = (np.array([2 ** k for k in range(K)]) / c_scale)
    n_vars = N + K
    q = QUBO(n_vars)
    for i in range(N):
        q.add_linear(i, -v[i])
    for (i, j) in inst.conflict_pairs():
        q.add_quadratic(i, j, P_pack)
    coeff = np.concatenate([c, slack_weights])
    target = cap_s
    for a in range(n_vars):
        q.add_linear(a, P_cap * (coeff[a] ** 2 - 2 * target * coeff[a]))
    for a in range(n_vars):
        for b in range(a + 1, n_vars):
            cc = 2 * P_cap * coeff[a] * coeff[b]
            if cc != 0.0:
                q.add_quadratic(a, b, cc)
    q.offset += P_cap * target ** 2
    info = dict(N=N, K=K, n_vars=n_vars, v_scale=v_scale, c_scale=c_scale,
                alpha_pack=alpha_pack, alpha_cap=alpha_cap, P_pack=P_pack, P_cap=P_cap,
                slack_weights=slack_weights.tolist())
    return q, info


def decode(sample, inst: AuctionInstance, info: dict) -> dict:
    """Decode a bit string into a solution over the ORIGINAL problem (feasibility honest)."""
    N = info["N"]
    x = np.asarray(sample[:N], dtype=int)
    selected = np.where(x == 1)[0].tolist()
    A = inst.coverage_matrix()
    lot_load = A[x == 1].sum(axis=0) if len(selected) else np.zeros(inst.J, dtype=int)
    packing_ok = bool((lot_load <= 1).all())
    used = int(inst.credits[x == 1].sum())
    cap_ok = bool(used <= inst.cap)
    obj = float(inst.values[x == 1].sum())
    return dict(x=x, selected=selected, objective=obj, feasible=packing_ok and cap_ok,
                packing_ok=packing_ok, cap_ok=cap_ok, credits_used=used, cap=inst.cap)
