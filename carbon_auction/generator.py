"""
Synthetic + real-calibrated instance generators for the carbon-credit Winner
Determination Problem (WDP).

An auction instance:
  - J lots (blocks of emission allowances), lot j has quantity q_j.
  - N bids. Bid i declares a value v_i for a package of lots S_i (subset of {1..J}).
    Winning bid i consumes c_i = sum_{j in S_i} q_j allowances.
  - A market-wide cap C limits total allowances issued.

Decisions x_i in {0,1}: accept bid i.
Constraints: exclusivity (set packing, each lot to <= 1 winner) and cap (knapsack).
Objective: maximise sum_i v_i x_i.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import json
import numpy as np


@dataclass
class AuctionInstance:
    N: int
    J: int
    values: np.ndarray
    coverage: list
    credits: np.ndarray
    lot_qty: np.ndarray
    cap: int
    meta: dict = field(default_factory=dict)

    def coverage_matrix(self) -> np.ndarray:
        A = np.zeros((self.N, self.J), dtype=int)
        for i, lots in enumerate(self.coverage):
            for j in lots:
                A[i, j] = 1
        return A

    def conflict_pairs(self) -> list[tuple[int, int]]:
        lot_to_bids: dict[int, list[int]] = {}
        for i, lots in enumerate(self.coverage):
            for j in lots:
                lot_to_bids.setdefault(j, []).append(i)
        pairs = set()
        for bids in lot_to_bids.values():
            for a in range(len(bids)):
                for b in range(a + 1, len(bids)):
                    pairs.add((bids[a], bids[b]))
        return sorted(pairs)

    def to_json(self, path: str) -> None:
        d = {"N": self.N, "J": self.J, "values": self.values.tolist(),
             "coverage": [list(map(int, c)) for c in self.coverage],
             "credits": self.credits.tolist(), "lot_qty": self.lot_qty.tolist(),
             "cap": int(self.cap), "meta": self.meta}
        with open(path, "w") as f:
            json.dump(d, f, indent=2)

    @staticmethod
    def from_json(path: str) -> "AuctionInstance":
        with open(path) as f:
            d = json.load(f)
        return AuctionInstance(N=d["N"], J=d["J"], values=np.array(d["values"], dtype=float),
                               coverage=[list(c) for c in d["coverage"]],
                               credits=np.array(d["credits"], dtype=int),
                               lot_qty=np.array(d["lot_qty"], dtype=int),
                               cap=int(d["cap"]), meta=d.get("meta", {}))


def generate_instance(N=40, J=25, pkg_max=1, cap_ratio=0.6, qty_low=1, qty_high=8,
                      value_noise=0.35, seed=0) -> AuctionInstance:
    """Controlled synthetic instance. Difficulty axes: density N/J, cap tightness, package size."""
    rng = np.random.default_rng(seed)
    lot_qty = rng.integers(qty_low, qty_high + 1, size=J)
    coverage, credits, values = [], [], []
    for _ in range(N):
        size = 1 if pkg_max <= 1 else int(rng.integers(1, pkg_max + 1))
        lots = sorted(rng.choice(J, size=min(size, J), replace=False).tolist())
        c_i = int(lot_qty[lots].sum())
        base_price = rng.uniform(8.0, 12.0)
        noise = float(np.exp(rng.normal(0.0, value_noise)))
        coverage.append(lots); credits.append(c_i); values.append(base_price * c_i * noise)
    credits = np.array(credits, dtype=int)
    values = np.round(np.array(values, dtype=float), 2)
    cap = int(round(cap_ratio * lot_qty.sum()))
    return AuctionInstance(N=N, J=J, values=values, coverage=coverage, credits=credits,
                           lot_qty=lot_qty, cap=cap,
                           meta=dict(pkg_max=pkg_max, cap_ratio=cap_ratio,
                                     density=round(N / J, 3), seed=seed))


def instance_stats(inst: AuctionInstance) -> dict:
    pairs = inst.conflict_pairs()
    max_pairs = inst.N * (inst.N - 1) / 2
    return {"N": inst.N, "J": inst.J, "density(N/J)": round(inst.N / inst.J, 2),
            "conflict_pairs": len(pairs),
            "conflict_density": round(len(pairs) / max_pairs, 3) if max_pairs else 0.0,
            "total_credits_if_all_win": int(inst.credits.sum()), "cap": inst.cap,
            "cap_ratio": round(inst.cap / inst.lot_qty.sum(), 2),
            "mean_pkg_size": round(np.mean([len(c) for c in inst.coverage]), 2)}


# --------------------------------------------------------------------------- real-market
# Calibration constants from published EU/UK ETS auction statistics (2024-2026);
# see data/real_market_calibration.json for values and sources.
ETS_PRICE_EUR_PER_TONNE = 73.43        # EU ETS 2025 volume-weighted average
ETS_PRICE_SPREAD = 0.15
ETS_PARTICIPANTS = 24
ETS_BID_TO_COVER = 1.7


def generate_real_instance(n_participants=ETS_PARTICIPANTS, bids_per_participant=3,
                           n_vintages=12, auction_mtco2=2.77, cap_ratio=0.65,
                           price_eur=ETS_PRICE_EUR_PER_TONNE, pkg_max=3, seed=0) -> AuctionInstance:
    """Combinatorial auction instance calibrated to EU/UK ETS figures.

    Real ETS auctions are single-good uniform-price; this models a richer multi-vintage
    auction (lots = vintages/blocks, package bids = bundles) with magnitudes, demand
    depth, and participant counts from 2025 EU/UK ETS statistics. Quantities in kt.
    The cap models a policy sub-ceiling on tonnes cleared (cap_ratio * offered volume).
    """
    rng = np.random.default_rng(seed)
    J = n_vintages
    N = n_participants * bids_per_participant
    total_kt = auction_mtco2 * 1000.0
    shares = rng.dirichlet(np.ones(J) * 4.0)
    lot_qty = np.maximum(1, np.round(shares * total_kt)).astype(int)
    coverage, credits, values = [], [], []
    for p in range(n_participants):
        p_price = price_eur * float(np.exp(rng.normal(0.0, ETS_PRICE_SPREAD)))
        for _ in range(bids_per_participant):
            size = 1 if pkg_max <= 1 else int(rng.integers(1, pkg_max + 1))
            lots = sorted(rng.choice(J, size=min(size, J), replace=False).tolist())
            c_i = int(lot_qty[lots].sum())
            v_i = p_price * c_i * float(rng.uniform(0.9, 1.0))
            coverage.append(lots); credits.append(c_i); values.append(v_i)
    credits = np.array(credits, dtype=int)
    values = np.round(np.array(values, dtype=float), 1)
    cap = int(round(cap_ratio * lot_qty.sum()))
    demand_to_cap = round(credits.sum() / cap, 2) if cap else 0.0  # packages inflate this vs single-good ETS
    return AuctionInstance(N=N, J=J, values=values, coverage=coverage, credits=credits,
                           lot_qty=lot_qty, cap=cap,
                           meta=dict(kind="real-ets", n_participants=n_participants,
                                     bids_per_participant=bids_per_participant,
                                     price_eur=round(price_eur, 2), auction_mtco2=auction_mtco2,
                                     cap_ratio=cap_ratio, demand_to_cap_ratio=demand_to_cap, seed=seed))
