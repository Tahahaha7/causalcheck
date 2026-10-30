"""Do the checks fire where methods actually fail?

For each check we sweep the data property it targets and record, per level:
  fire   - how often the check flags it
  harm   - how badly the method the check protects actually does

  linearity        -> pairwise LiNGAM direction accuracy on nonlinear data (should fall)
  noise            -> pairwise LiNGAM direction accuracy as noise turns Gaussian (should fall to 50%)
  near-determinism -> recall of a PC-style (Fisher-z) skeleton as links turn deterministic (should fall)
  scale            -> varsortability of the data as variances spread (should rise toward 1)

Run:  python benchmarks/validate_thresholds.py   (writes benchmarks/results.md and results.png)
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from causalcheck.checks import (check_determinism, check_linearity, check_noise, check_scale, prepare)

N, REPS = 1000, 20
OUT = Path(__file__).parent


# --------------------------------------------------------------- helpers

def _entropy(u):
    k1, k2, g = 79.047, 7.4129, 0.37457
    return (1 + np.log(2 * np.pi)) / 2 - k1 * (np.mean(np.log(np.cosh(u))) - g) ** 2 \
        - k2 * (np.mean(u * np.exp(-u ** 2 / 2))) ** 2


def lingam_direction_correct(x, y):
    """Pairwise LiNGAM likelihood ratio: does it pick x -> y?"""
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()
    rxy = y - np.cov(x, y)[0, 1] * x          # residual of y given x
    ryx = x - np.cov(x, y)[0, 1] * y
    lr = (_entropy(y) + _entropy(ryx / ryx.std())) - (_entropy(x) + _entropy(rxy / rxy.std()))
    return lr > 0


def pc_skeleton_recall(X, true_edges, alpha=0.05):
    """Fisher-z PC skeleton; fraction of true edges kept."""
    n, p = X.shape
    adj = {i: set(range(p)) - {i} for i in range(p)}

    def indep(i, j, S):
        C = np.corrcoef(X[:, [i, j] + list(S)].T)
        P = np.linalg.pinv(C)
        r = np.clip(-P[0, 1] / np.sqrt(P[0, 0] * P[1, 1]), -0.999999, 0.999999)
        z = 0.5 * np.log((1 + r) / (1 - r)) * np.sqrt(n - len(S) - 3)
        return 2 * stats.norm.sf(abs(z)) > alpha

    for d in range(p - 1):
        for i, j in itertools.combinations(range(p), 2):
            if j not in adj[i]:
                continue
            for S in itertools.combinations(sorted((adj[i] | adj[j]) - {i, j}), d):
                if indep(i, j, S):
                    adj[i].discard(j); adj[j].discard(i)
                    break
    return np.mean([j in adj[i] for i, j in true_edges])


def varsortability(X, edges):
    """Fraction of causal pairs where the effect has higher variance (1 = trivially sortable)."""
    v = X.var(0)
    return np.mean([v[j] > v[i] for i, j in edges])


def noise(rng, size, q):
    """Generalized-Gaussian-like noise: q=1 Gaussian, q far from 1 increasingly non-Gaussian."""
    g = rng.normal(size=size)
    return np.sign(g) * np.abs(g) ** q


# ---------------------------------------------------------------- sweeps

def sweep_linearity(rng):
    rows = []
    for a in [0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]:
        fire, harm = [], []
        for _ in range(REPS):
            x = rng.uniform(-1.7, 1.7, N)                                         # non-Gaussian, so
            y = (1 - a) * x + a * 1.5 * np.tanh(2 * x) + 0.5 * rng.uniform(-1.7, 1.7, N)  # LiNGAM works at a=0
            prep = prepare(pd.DataFrame(dict(x=x, y=y)))
            fire.append(check_linearity(prep).status == "info")
            harm.append(lingam_direction_correct(x, y))
        rows.append(dict(sweep="linearity", level=a, fire=np.mean(fire), harm=1 - np.mean(harm), note=None))
    return rows


def sweep_noise(rng):
    rows = []
    for q in [1.0, 1.1, 1.2, 1.35, 1.5, 1.75, 2.0, 2.5]:
        fire, harm = [], []
        for _ in range(REPS):
            x = noise(rng, N, q)
            y = 0.8 * x + noise(rng, N, q)
            prep = prepare(pd.DataFrame(dict(x=x, y=y)))
            fire.append(check_noise(prep, linear=True).status == "pass")   # pass = non-Gaussian detected
            harm.append(lingam_direction_correct(x, y))
        rows.append(dict(sweep="noise", level=q, fire=np.mean(fire), harm=1 - np.mean(harm), note=None))
    return rows


def sweep_determinism(rng):
    rows = []
    edges = [(3, 0), (3, 2), (0, 4), (2, 4)]      # the LiNGAM-example core, 4 nodes
    for c in [1, 2, 3, 4, 6, 8, 12]:
        fire, harm = [], []
        for _ in range(REPS):
            x3 = rng.uniform(size=N)
            x0 = c * x3 + rng.uniform(size=N)
            x2 = 2 * c * x3 + rng.uniform(size=N)
            x4 = c * x0 - 0.5 * c * x2 + rng.uniform(size=N)
            X = np.column_stack([x0, 0 * x0, x2, x3, x4])[:, [0, 2, 3, 4]]
            prep = prepare(pd.DataFrame(X, columns=["x0", "x2", "x3", "x4"]))
            fire.append(check_determinism(prep).status in ("warn", "fail"))
            Xs = (X - X.mean(0)) / X.std(0)
            idx = {0: 0, 2: 1, 3: 2, 4: 3}
            harm.append(pc_skeleton_recall(Xs, [(idx[i], idx[j]) for i, j in edges]))
        rows.append(dict(sweep="determinism", level=c, fire=np.mean(fire), harm=1 - np.mean(harm), note=None))
    return rows


def sweep_scale(rng):
    """Raw (unstandardized) chain a -> b -> c -> d with coefficient w: variance accumulates downstream."""
    rows = []
    edges = [(0, 1), (1, 2), (2, 3)]
    for w in [0.2, 0.5, 1.0, 1.5, 2.0, 3.0]:
        fire, harm, ratio = [], [], []
        for _ in range(REPS):
            a = rng.normal(size=N)
            b = w * a + rng.normal(size=N)
            cc = w * b + rng.normal(size=N)
            d = w * cc + rng.normal(size=N)
            X = np.column_stack([a, b, cc, d])
            prep = prepare(pd.DataFrame(X, columns=list("abcd")))
            fire.append(check_scale(prep).status == "warn")
            harm.append(varsortability(X, edges))
            ratio.append(X.var(0).max() / X.var(0).min())
        rows.append(dict(sweep="scale", level=w, fire=np.mean(fire), harm=np.mean(harm),
                         note=f"variance ratio {np.mean(ratio):.1f}"))
    return rows


def main():
    rng = np.random.default_rng(0)
    rows = sweep_linearity(rng) + sweep_noise(rng) + sweep_determinism(rng) + sweep_scale(rng)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "results.csv", index=False)

    harm_name = {"linearity": "LiNGAM direction error", "noise": "LiNGAM direction error",
                 "determinism": "PC skeleton edges lost", "scale": "varsortability (raw data)"}
    lines = ["# Threshold validation", "", f"n={N}, {REPS} replicates per level. "
             "`fire` = share of runs where the check flagged the data; `harm` = how badly the protected method did.", ""]
    for name, g in df.groupby("sweep", sort=False):
        lines += [f"## {name}  (harm = {harm_name[name]})", "", "| level | fire | harm |", "|---|---|---|"]
        lines += [f"| {r.level:g} | {r.fire:.2f} | {r.harm:.2f} |" + (f" {r.note}" if isinstance(getattr(r, 'note', None), str) else "")
                  for r in g.itertuples()]
        lines.append("")
    (OUT / "results.md").write_text("\n".join(lines))
    print("\n".join(lines))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
        for ax, (name, g) in zip(axes, df.groupby("sweep", sort=False)):
            ax.plot(g.level, g.fire, "o-", label="check fires")
            ax.plot(g.level, g.harm, "s--", label=harm_name[name])
            ax.set_title(name); ax.set_ylim(-0.05, 1.05); ax.set_xlabel("level")
            if name in ("determinism", "scale"):
                ax.set_xscale("log")
            ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(OUT / "results.png", dpi=120)
        print(f"\nwrote {OUT / 'results.png'}")
    except ImportError:
        print("matplotlib not installed; skipped plot")


if __name__ == "__main__":
    main()
