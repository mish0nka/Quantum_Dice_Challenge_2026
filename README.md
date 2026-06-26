# Combinatorial Carbon Auction Clearing

An interactive demo for clearing **multi-vintage carbon-allowance auctions with package bids**.
Configure an auction, clear it, and compare three engines — an exact optimiser, a fast greedy
heuristic, and a probabilistic sampler — on the welfare each one clears.

Built for the **Quantum Dice Trinity Challenge 2026** (Team 24). In the full project the same
problem formulation drives Quantum Dice's **ORBIT** probabilistic-computing backend; this public
demo uses **classical solvers only** and bundles no proprietary code.

## Quickstart

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually <http://localhost:8501>).

> **macOS note:** if you see `PermissionError: Operation not permitted` on launch, your terminal
> lacks access to the folder (common under `~/Desktop`, `~/Documents`, `~/Downloads`). Run the repo
> from another location (e.g. `~/projects/...`) or grant your terminal Full Disk Access in
> System Settings → Privacy & Security.

## What it does

- **Set up an auction** — pick a preset or set bidders, vintages, cap, and package size.
- **Clear it** — one click runs all three solvers and reports the welfare each clears, the winning
  allocation, and cap utilisation.
- **See why optimisation matters** — the *Greedy trap* preset shows the fast heuristic leaving ~40%
  of welfare unclaimed versus the optimum.
- **Explore alternatives** — the probabilistic sampler returns a distribution of near-optimal
  clearings, useful for tie-breaking, fairness, and robustness.
- **Real-world grounding** — the *Real: EU/UK ETS* preset is calibrated to 2025 EU/UK ETS auction
  figures (price, participants, volume).

## Solvers

| engine | role |
|---|---|
| Exact (branch-and-bound) | certified-optimal benchmark, pure Python (no external solver binary) |
| Greedy | fast value-density heuristic |
| Simulated annealing | probabilistic sampler returning a spread of clearings |

## Project layout

```
app.py                 Streamlit application
carbon_auction/        clearing engine
  generator.py         synthetic + EU/UK-ETS-calibrated auction generators
  qubo.py              QUBO encoding + decoding (shared formulation)
  solvers.py           exact branch-and-bound, greedy, simulated annealing
requirements.txt
```

## Notes

- The demo generates auction instances on the fly from seeds, so it is fully self-contained.
- No part of this repository depends on or includes the proprietary ORBIT simulator.
