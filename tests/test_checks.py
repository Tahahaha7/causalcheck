"""Known-answer tests: synthetic data whose assumption profile we know in advance."""
import json

import numpy as np
import pandas as pd
import pytest

import causalcheck
from causalcheck.cli import main

N = 1000


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def status(report, name):
    return {c.name: c.status for c in report.checks}[name]


def verdict(report, prefix):
    return next(m.verdict for m in report.advice.methods if m.method.startswith(prefix))


# ----------------------------------------------------------------- datasets

def linear_gaussian(rng):
    a = rng.normal(size=N)
    b = 0.8 * a + rng.normal(size=N)
    c = 0.5 * a + 0.5 * b + rng.normal(size=N)
    return pd.DataFrame(dict(a=a, b=b, c=c))


def linear_nongaussian(rng):
    a = rng.uniform(-1, 1, N)
    b = 0.8 * a + 0.5 * rng.uniform(-1, 1, N)
    c = 0.6 * a - 0.6 * b + 0.5 * rng.uniform(-1, 1, N)
    return pd.DataFrame(dict(a=a, b=b, c=c))


def nonlinear(rng):
    g = rng.normal(size=N)
    h = np.tanh(2 * g) + 0.3 * rng.normal(size=N)
    k = np.sin(h) + 0.3 * rng.normal(size=N)
    return pd.DataFrame(dict(g=g, h=h, k=k))


def near_deterministic(rng):
    x3 = rng.uniform(size=N)
    x0 = 3 * x3 + rng.uniform(size=N)
    x2 = 6 * x3 + rng.uniform(size=N)
    x4 = 8 * x0 - x2 + rng.uniform(size=N)
    return pd.DataFrame(dict(x0=x0, x2=x2, x3=x3, x4=x4))


# -------------------------------------------------------------------- tests

def test_linear_gaussian_refuses_direction_methods(rng):
    r = causalcheck.check(linear_gaussian(rng))
    assert status(r, "linearity") == "pass"
    assert status(r, "noise distribution") == "warn"
    assert verdict(r, "DirectLiNGAM") == "avoid"
    assert verdict(r, "NOTEARS") == "avoid"
    assert any("equivalence class" in t for t in r.advice.trust)


def test_linear_nongaussian_recommends_lingam(rng):
    r = causalcheck.check(linear_nongaussian(rng))
    assert status(r, "linearity") == "pass"
    assert status(r, "noise distribution") == "pass"
    assert verdict(r, "DirectLiNGAM") == "recommended"
    assert verdict(r, "GES / BOSS") == "recommended"


def test_nonlinear_routes_to_anm(rng):
    r = causalcheck.check(nonlinear(rng))
    assert status(r, "linearity") == "info"
    assert verdict(r, "DirectLiNGAM") == "avoid"
    assert verdict(r, "ANM") == "recommended"
    pairs = {c.name: c for c in r.checks}["linearity"].details["nonlinear_pairs"]
    assert any({p[0], p[1]} == {"g", "h"} for p in pairs)


def test_near_determinism_flags_faithfulness_risk(rng):
    r = causalcheck.check(near_deterministic(rng))
    assert status(r, "near-determinism") == "warn"
    assert verdict(r, "PC") == "avoid"
    assert verdict(r, "GES / BOSS") == "recommended"


def test_duplicate_column_fails(rng):
    df = linear_gaussian(rng)
    df["a_inches"] = df["a"] * 2.54
    r = causalcheck.check(df)
    assert status(r, "near-determinism") == "fail"
    assert r.worst_status == "fail"
    assert any("duplicate" in f.lower() for f in r.advice.fix_first)


def test_discrete_missing_and_text_columns(rng):
    df = linear_gaussian(rng)
    df["level"] = rng.integers(0, 3, N)
    df["label"] = np.where(df["level"] == 0, "x", "y")
    df.loc[:49, "b"] = np.nan
    r = causalcheck.check(df)
    assert status(r, "data types") == "warn"
    assert r.discrete == ["level"]
    assert r.n_rows == N - 50
    assert "label" not in r.variables
    assert verdict(r, "ANM") == "avoid"


def test_small_sample_warns(rng):
    r = causalcheck.check(linear_gaussian(rng).head(80))
    assert status(r, "sample size") == "warn"
    assert any("bootstrap" in t.lower() for t in r.advice.trust)


def test_scale_warns_and_asks_to_standardize(rng):
    df = linear_gaussian(rng)
    df["c"] *= 100
    r = causalcheck.check(df)
    assert status(r, "scale") == "warn"
    assert any("standardize" in f.lower() for f in r.advice.fix_first)


def test_report_text_and_json(rng):
    r = causalcheck.check(linear_gaussian(rng))
    text = str(r)
    assert "CHECKS" in text and "METHODS" in text and "HOW TO READ" in text
    d = json.loads(r.to_json())
    assert d["worst_status"] == r.worst_status
    assert len(d["checks"]) == len(r.checks)
    assert d["causalcheck"] == causalcheck.__version__


def test_confounding_check_skips_without_causal_learn(rng):
    pytest.importorskip("numpy")
    try:
        import causallearn  # noqa: F401
        pytest.skip("causal-learn installed; skip-path not exercised")
    except ImportError:
        pass
    r = causalcheck.check(linear_gaussian(rng), confounding=True)
    assert status(r, "hidden confounding") == "info"


def test_cli_exit_codes(rng, tmp_path, capsys):
    ok = tmp_path / "ok.csv"
    linear_gaussian(rng).to_csv(ok, index=False)
    assert main([str(ok)]) == 0
    assert "CHECKS" in capsys.readouterr().out
    bad = tmp_path / "bad.csv"
    df = linear_gaussian(rng)
    df["a2"] = df["a"]
    df.to_csv(bad, index=False)
    assert main([str(bad), "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["worst_status"] == "fail"
