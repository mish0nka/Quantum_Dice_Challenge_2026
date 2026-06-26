"""
Classical + reference-probabilistic solvers for the WDP.
  exact_mip            -- exact branch-and-bound (pure Python, no external solver binary).
  greedy               -- value-density heuristic.
  simulated_annealing  -- Stage 2 Metropolis sampler on the QUBO (reference baseline).
  parallel_tempering   -- replica-exchange variant.
"""
from __future__ import annotations
import time
import numpy as np

from .qubo import QUBO, decode
from .generator import AuctionInstance


def exact_mip(inst: AuctionInstance, time_limit=None, node_limit=5_000_000) -> dict:
    """Exact solver for the winner-determination problem via branch-and-bound.

    Maximises sum_i v_i x_i subject to lot exclusivity (each lot to <=1 winner) and the
    market cap. Implemented in pure Python so the demo needs no external MILP binary
    (avoids platform-specific CBC issues). For the instance sizes here it returns the
    certified optimum; if the (very high) node limit is ever hit it returns the best
    solution found, flagged via 'optimal'=False.
    """
    t0 = time.perf_counter()
    N = inst.N
    values = inst.values.astype(float)
    credits = inst.credits.astype(np.int64)
    cap = int(inst.cap)
    lot_mask = np.zeros(N, dtype=object)
    for i, lots in enumerate(inst.coverage):
        m = 0
        for j in lots:
            m |= (1 << int(j))
        lot_mask[i] = m
    # process bids by value-density (good incumbent + tight bound)
    order = sorted(range(N), key=lambda i: -(values[i] / max(int(credits[i]), 1)))
    v = [values[i] for i in order]
    c = [int(credits[i]) for i in order]
    lm = [lot_mask[i] for i in order]

    best = {"obj": -1.0, "set": []}
    nodes = [0]
    hit_limit = [False]

    def upper_bound(pos, used_lots, used_credits, cur_val):
        """Optimistic bound: fractional knapsack over remaining compatible bids,
        ignoring conflicts among them (a valid relaxation)."""
        rem = cap - used_credits
        bound = cur_val
        for k in range(pos, N):
            if lm[k] & used_lots:
                continue
            if c[k] <= rem:
                bound += v[k]; rem -= c[k]
            elif rem > 0:
                bound += v[k] * (rem / c[k]); rem = 0
                break
        return bound

    def recurse(pos, used_lots, used_credits, cur_val, chosen):
        nodes[0] += 1
        if nodes[0] > node_limit:
            hit_limit[0] = True
            return
        if cur_val > best["obj"]:
            best["obj"] = cur_val; best["set"] = list(chosen)
        if pos >= N or hit_limit[0]:
            return
        if upper_bound(pos, used_lots, used_credits, cur_val) <= best["obj"]:
            return
        # branch: include order[pos] if compatible, then exclude
        if (lm[pos] & used_lots) == 0 and used_credits + c[pos] <= cap:
            chosen.append(order[pos])
            recurse(pos + 1, used_lots | lm[pos], used_credits + c[pos], cur_val + v[pos], chosen)
            chosen.pop()
        recurse(pos + 1, used_lots, used_credits, cur_val, chosen)

    import sys
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old_limit, N + 100))
    recurse(0, 0, 0, 0.0, [])
    sys.setrecursionlimit(old_limit)

    rt = time.perf_counter() - t0
    sel = sorted(int(i) for i in best["set"])
    return dict(method="Exact (B&B)", objective=float(best["obj"]), runtime=rt,
                selected=sel, status="Optimal" if not hit_limit[0] else "Feasible",
                optimal=not hit_limit[0])


def greedy(inst: AuctionInstance) -> dict:
    t0 = time.perf_counter()
    order = np.argsort(-(inst.values / np.maximum(inst.credits, 1)))
    A = inst.coverage_matrix()
    used_lots = np.zeros(inst.J, dtype=int)
    used_credits = 0
    chosen = []
    for i in order:
        lots = np.where(A[i] == 1)[0]
        if used_lots[lots].any():
            continue
        if used_credits + inst.credits[i] > inst.cap:
            continue
        chosen.append(int(i)); used_lots[lots] = 1; used_credits += int(inst.credits[i])
    rt = time.perf_counter() - t0
    return dict(method="Greedy", objective=float(inst.values[chosen].sum()),
                runtime=rt, selected=sorted(chosen))


def _symmetric_arrays(q: QUBO):
    n = q.n
    d = np.zeros(n); W = np.zeros((n, n))
    for i, c in q.lin.items():
        d[i] += c
    for (i, j), c in q.quad.items():
        W[i, j] += c; W[j, i] += c
    return d, W


def _anneal_one(d, W, betas, rng, x0=None):
    n = len(d)
    x = (rng.random(n) < 0.5).astype(np.int8) if x0 is None else x0.copy()
    h = d + W @ x
    e = float(d @ x + 0.5 * x @ (W @ x))
    best_x, best_e = x.copy(), e
    for beta in betas:
        for i in rng.permutation(n):
            delta = (1 - 2 * x[i]) * h[i]
            if delta <= 0 or rng.random() < np.exp(-beta * delta):
                sign = 1 - 2 * x[i]
                x[i] ^= 1; h += W[:, i] * sign; e += delta
                if e < best_e:
                    best_e, best_x = e, x.copy()
    return best_x, best_e


def simulated_annealing(q: QUBO, inst: AuctionInstance, info: dict, num_reads=50,
                        num_sweeps=1000, beta0=0.1, beta1=10.0, seed=0, log_traj=False) -> dict:
    t0 = time.perf_counter()
    d, W = _symmetric_arrays(q)
    betas = np.geomspace(beta0, beta1, num_sweeps)
    rng = np.random.default_rng(seed)
    reads = []
    for _ in range(num_reads):
        bx, be = _anneal_one(d, W, betas, rng)
        dec = decode(bx, inst, info); dec["energy"] = be; reads.append(dec)
    rt = time.perf_counter() - t0
    feasible = [r for r in reads if r["feasible"]]
    best = max(feasible, key=lambda r: r["objective"]) if feasible \
        else min(reads, key=lambda r: r["energy"])
    return dict(method="SA", objective=best["objective"], runtime=rt,
                selected=best["selected"], feasible=best["feasible"], reads=reads,
                num_reads=num_reads, num_sweeps=num_sweeps,
                feasible_rate=len(feasible) / num_reads)


def parallel_tempering(q: QUBO, inst: AuctionInstance, info: dict, num_replicas=8,
                       num_sweeps=1000, beta0=0.05, beta1=12.0, num_reads=20, seed=0) -> dict:
    t0 = time.perf_counter()
    d, W = _symmetric_arrays(q); n = len(d)
    ladder = np.geomspace(beta0, beta1, num_replicas)
    rng = np.random.default_rng(seed)
    reads = []
    for _ in range(num_reads):
        X = (rng.random((num_replicas, n)) < 0.5).astype(np.int8)
        H = np.array([d + W @ X[r] for r in range(num_replicas)])
        E = np.array([float(d @ X[r] + 0.5 * X[r] @ (W @ X[r])) for r in range(num_replicas)])
        best_x, best_e = X[-1].copy(), E[-1]
        for _sweep in range(num_sweeps):
            for r in range(num_replicas):
                beta = ladder[r]
                for i in rng.permutation(n):
                    delta = (1 - 2 * X[r, i]) * H[r, i]
                    if delta <= 0 or rng.random() < np.exp(-beta * delta):
                        sign = 1 - 2 * X[r, i]
                        X[r, i] ^= 1; H[r] += W[:, i] * sign; E[r] += delta
                        if E[r] < best_e:
                            best_e, best_x = E[r], X[r].copy()
            for r in range(num_replicas - 1):
                darg = (ladder[r] - ladder[r + 1]) * (E[r] - E[r + 1])
                if darg >= 0 or rng.random() < np.exp(darg):
                    X[[r, r + 1]] = X[[r + 1, r]]; H[[r, r + 1]] = H[[r + 1, r]]
                    E[[r, r + 1]] = E[[r + 1, r]]
        dec = decode(best_x, inst, info); dec["energy"] = best_e; reads.append(dec)
    rt = time.perf_counter() - t0
    feasible = [r for r in reads if r["feasible"]]
    best = max(feasible, key=lambda r: r["objective"]) if feasible \
        else min(reads, key=lambda r: r["energy"])
    return dict(method="ParallelTempering", objective=best["objective"], runtime=rt,
                selected=best["selected"], feasible=best["feasible"], reads=reads,
                feasible_rate=len(feasible) / num_reads)
