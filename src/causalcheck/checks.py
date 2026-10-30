"""Pre-flight assumption checks for causal discovery.

Each check inspects the data (not any graph) and returns a CheckResult with a status:
  pass  - assumption looks satisfied
  warn  - something that will mislead some methods
  fail  - a method family should not be used on this data
  info  - a finding that routes you to a method, with no good/bad meaning
"""
from __future__ import annotations

import contextlib
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer



@contextlib.contextmanager
def _quiet():
    """Silence numeric/convergence warnings from numpy and sklearn only while a check runs."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", UserWarning)
        yield


@dataclass
class CheckResult:
    name: str
    status: str
    summary: str
    details: dict = field(default_factory=dict)


@dataclass
class Prepared:
    X: np.ndarray            # standardized numeric data, rows with missing values dropped
    raw: np.ndarray          # same rows, original scale
    names: list[str]
    discrete: list[str]      # numeric but low-cardinality columns
    dropped: list[str]       # non-numeric columns
    n_missing_rows: int


def _gam():
    return make_pipeline(SplineTransformer(n_knots=6, degree=3, extrapolation="linear"),
                         RidgeCV(alphas=np.logspace(-3, 3, 13)))


def _cv_r2(model, D, y, seed):
    kf = KFold(5, shuffle=True, random_state=seed)
    return float(cross_val_score(model, D, y, cv=kf, scoring="r2").mean())


def prepare(df: pd.DataFrame, min_unique: int = 10) -> Prepared:
    num = df.select_dtypes(include=[np.number])
    dropped = [c for c in df.columns if c not in num.columns]
    n0 = len(num)
    num = num.dropna()
    num = num.loc[:, num.std() > 0]
    names = list(num.columns)
    discrete = [c for c in names if num[c].nunique() < min_unique]
    raw = num.to_numpy(float)
    X = (raw - raw.mean(0)) / raw.std(0)
    return Prepared(X, raw, names, discrete, dropped, n0 - len(num))


# ---------------------------------------------------------------- checks

def check_data_types(prep: Prepared, n_total_rows: int) -> CheckResult:
    d = dict(numeric=len(prep.names), discrete=prep.discrete, dropped=prep.dropped,
             rows_used=len(prep.X), rows_dropped_missing=prep.n_missing_rows)
    if prep.dropped and not prep.names:
        return CheckResult("data types", "fail", "no numeric columns found", d)
    if prep.discrete or prep.dropped:
        what = []
        if prep.discrete:
            what.append(f"{len(prep.discrete)} low-cardinality column(s) treated as discrete")
        if prep.dropped:
            what.append(f"{len(prep.dropped)} non-numeric column(s) ignored")
        return CheckResult("data types", "warn", "; ".join(what)
                           + " -> use mixed-data scores/tests, skip additive-noise tests on them", d)
    return CheckResult("data types", "pass", f"{len(prep.names)} continuous columns, "
                       f"{prep.n_missing_rows} of {n_total_rows} rows dropped for missing values", d)


def check_sample_size(prep: Prepared) -> CheckResult:
    n, p = prep.X.shape
    d = dict(n=n, p=p, n_per_variable=round(n / max(p, 1), 1))
    if n < 200 or n < 20 * p:
        return CheckResult("sample size", "warn",
                           f"n={n} for p={p} variables: expect low power, unstable orientations, "
                           "and many undirected edges", d)
    if n < 1000:
        return CheckResult("sample size", "pass",
                           f"n={n} for p={p}: fine for skeleton; direction tests may be weak", d)
    return CheckResult("sample size", "pass", f"n={n} for p={p}", d)


def check_linearity(prep: Prepared, tol: float = 0.02, max_rows: int = 2000, seed: int = 0) -> CheckResult:
    """Pairwise: does a spline GAM beat a straight line out of sample?"""
    X, names = prep.X, prep.names
    rng = np.random.default_rng(seed)
    Xs = X[rng.choice(len(X), min(len(X), max_rows), replace=False)]
    cont = [k for k, c in enumerate(names) if c not in prep.discrete]
    nonlinear = []
    for i in cont:
        for j in cont:
            if i == j:
                continue
            D, y = Xs[:, [i]], Xs[:, j]
            gain = _cv_r2(_gam(), D, y, seed) - _cv_r2(LinearRegression(), D, y, seed)
            if gain > tol:
                nonlinear.append((names[i], names[j], round(gain, 3)))
    nonlinear.sort(key=lambda t: -t[2])
    d = dict(nonlinear_pairs=nonlinear, tol=tol, pairs_tested=len(cont) * (len(cont) - 1))
    if nonlinear:
        return CheckResult("linearity", "info",
                           f"{len(nonlinear)} pair(s) clearly nonlinear (largest R2 gain "
                           f"{nonlinear[0][2]:.2f}) -> linear methods (LiNGAM, Fisher-z, linear BIC) "
                           "will mis-specify these", d)
    return CheckResult("linearity", "pass",
                       "all pairwise relations look linear -> linear scores and tests are appropriate", d)


def check_noise(prep: Prepared, linear: bool, alpha: float = 0.01,
                min_skew: float = 0.2, min_kurt: float = 0.3) -> CheckResult:
    """Non-Gaussian disturbances are what let LiNGAM/ANM orient edges.

    Without the graph we use each variable's residual given all other variables as a proxy:
    if every disturbance is Gaussian, these residuals are Gaussian too.
    """
    X, names = prep.X, prep.names
    cont = [k for k, c in enumerate(names) if c not in prep.discrete]
    rows = []
    for v in cont:
        others = [u for u in cont if u != v]
        if not others:
            r = X[:, v]
        else:
            m = LinearRegression() if linear else _gam()
            m.fit(X[:, others], X[:, v])
            r = X[:, v] - m.predict(X[:, others])
        r = (r - r.mean()) / (r.std() + 1e-12)
        p = float(stats.normaltest(r).pvalue) if len(r) >= 20 else 1.0
        sk, ku = float(stats.skew(r)), float(stats.kurtosis(r))
        nongauss = p < alpha and (abs(sk) > min_skew or abs(ku) > min_kurt)
        rows.append(dict(variable=names[v], p=p, skew=round(sk, 2), excess_kurtosis=round(ku, 2),
                         non_gaussian=bool(nongauss)))
    k = sum(r["non_gaussian"] for r in rows)
    d = dict(per_variable=rows, non_gaussian_count=k, tested=len(rows))
    if not linear:
        return CheckResult("noise distribution", "info",
                           f"{k}/{len(rows)} residuals non-Gaussian; with nonlinear relations, additive-noise "
                           "methods (ANM, CAM, RESIT) can orient edges regardless of noise shape", d)
    if len(rows) and k >= len(rows) - 1:
        return CheckResult("noise distribution", "pass",
                           f"{k}/{len(rows)} residuals clearly non-Gaussian -> LiNGAM-style "
                           "direction identification is possible", d)
    if k == 0:
        return CheckResult("noise distribution", "warn",
                           "all residuals look Gaussian -> directions beyond colliders are not "
                           "identifiable; expect an equivalence class, do not trust LiNGAM arrows", d)
    return CheckResult("noise distribution", "info",
                       f"{k}/{len(rows)} residuals non-Gaussian -> LiNGAM needs at most one Gaussian "
                       "disturbance; trust only arrows touching the non-Gaussian variables", d)


def check_determinism(prep: Prepared, r2_thresh: float = 0.98, corr_thresh: float = 0.99,
                      dup_thresh: float = 0.999) -> CheckResult:
    """Near-deterministic links break faithfulness: CI tests drop true edges."""
    X, names = prep.X, prep.names
    flagged = []
    for v in range(len(names)):
        others = [u for u in range(len(names)) if u != v]
        if not others:
            continue
        m = LinearRegression().fit(X[:, others], X[:, v])
        r2 = float(m.score(X[:, others], X[:, v]))
        if r2 > r2_thresh:
            flagged.append((names[v], round(r2, 4)))
    C = np.corrcoef(X.T) if len(names) > 1 else np.eye(1)
    pairs = [(names[i], names[j], round(float(C[i, j]), 4))
             for i in range(len(names)) for j in range(i + 1, len(names)) if abs(C[i, j]) > corr_thresh]
    dups = [t for t in pairs if abs(t[2]) > dup_thresh]
    d = dict(near_deterministic_variables=flagged, near_deterministic_pairs=pairs, duplicate_pairs=dups)
    if dups:
        return CheckResult("near-determinism", "fail",
                           f"{len(dups)} pair(s) correlated above {dup_thresh} -> effectively the same "
                           "variable; merge or drop one of each pair before running anything", d)
    if flagged or pairs:
        return CheckResult("near-determinism", "warn",
                           f"{len(flagged)} variable(s) almost fully explained by the others and "
                           f"{len(pairs)} pair(s) correlated above {corr_thresh} -> faithfulness at risk: "
                           "CI-based skeletons (PC, FCI) will drop true edges; prefer score-based search "
                           "(BOSS, GES)", d)
    return CheckResult("near-determinism", "pass", "no near-deterministic relations", d)


def check_scale(prep: Prepared, ratio_thresh: float = 10.0) -> CheckResult:
    """Scale disparity between variables.

    Note: this cannot detect varsortability. In raw data, effects usually have more variance than
    their causes even when all variances are within a factor of 2, so standardizing before any
    continuous-optimization method (NOTEARS, DAGMA) is standing advice, not something this check gates.
    """
    var = prep.raw.var(0)
    ratio = float(var.max() / max(var.min(), 1e-12))
    d = dict(variance_ratio=round(ratio, 1), variances={n: float(v) for n, v in zip(prep.names, var)})
    if ratio > ratio_thresh:
        return CheckResult("scale", "warn",
                           f"variances differ by a factor of {ratio:.0f} -> standardize before anything "
                           "scale-sensitive (kernels, penalties, continuous optimization)", d)
    return CheckResult("scale", "pass",
                       "variables are on comparable scales (standardize anyway before NOTEARS/DAGMA: "
                       "comparable scales do not rule out varsortability)", d)


def check_confounding(prep: Prepared, linear: bool, n_boot: int = 5, frac: float = 0.8,
                      alpha: float = 0.05, seed: int = 0) -> CheckResult:
    """Optional: signs of hidden common causes, via bidirected edges in FCI (needs causal-learn)."""
    try:
        from causallearn.search.ConstraintBased.FCI import fci
    except ImportError:
        return CheckResult("hidden confounding", "info",
                           "skipped: install causal-learn (pip install causalcheck[full]) to run FCI", {})
    X = prep.X
    n = len(X)
    ci = "fisherz" if linear else "kci"
    m = min(int(frac * n), n if linear else 500)
    rng = np.random.default_rng(seed)
    p = X.shape[1]
    bi = np.zeros((p, p))
    for _ in range(n_boot):
        G, _ = fci(X[rng.choice(n, m, replace=False)], independence_test_method=ci, alpha=alpha,
                   verbose=False, show_progress=False)
        g = G.graph
        bi += (g == 1) & (g.T == 1)
    pairs = [(prep.names[i], prep.names[j], round(bi[i, j] / n_boot, 2))
             for i in range(p) for j in range(i + 1, p) if bi[i, j] / n_boot >= 0.6]
    d = dict(bidirected_pairs=pairs, ci_test=ci, subsamples=n_boot)
    if pairs:
        return CheckResult("hidden confounding", "warn",
                           f"{len(pairs)} pair(s) consistently bidirected in FCI -> likely hidden common "
                           "causes; methods that assume no latents (PC, GES, BOSS, LiNGAM) will orient "
                           "these wrongly", d)
    return CheckResult("hidden confounding", "pass",
                       "no consistent signs of hidden common causes (not a guarantee)", d)


def run_checks(df: pd.DataFrame, confounding: bool = False, seed: int = 0) -> tuple[Prepared, list[CheckResult]]:
    with _quiet():
        prep = prepare(df)
        results = [check_data_types(prep, len(df)), check_sample_size(prep)]
        if len(prep.names) >= 2:
            lin = check_linearity(prep, seed=seed)
            linear = lin.status == "pass"
            results += [lin, check_noise(prep, linear), check_determinism(prep), check_scale(prep)]
            if confounding:
                results.append(check_confounding(prep, linear, seed=seed))
    return prep, results
