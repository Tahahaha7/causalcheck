"""Report: the result of running all checks, printable as text or JSON."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import pandas as pd

from .checks import CheckResult, run_checks
from .recommend import Advice, recommend

_TAG = {"pass": "PASS", "warn": "WARN", "fail": "FAIL", "info": "INFO"}


@dataclass
class Report:
    n_rows: int
    variables: list[str]
    discrete: list[str]
    checks: list[CheckResult]
    advice: Advice
    version: str = "0.1.0"

    @property
    def worst_status(self) -> str:
        order = {"fail": 3, "warn": 2, "info": 1, "pass": 0}
        return max((c.status for c in self.checks), key=lambda s: order[s], default="pass")

    def to_dict(self) -> dict:
        return dict(causalcheck=self.version, n_rows=self.n_rows, variables=self.variables,
                    discrete=self.discrete, worst_status=self.worst_status,
                    checks=[asdict(c) for c in self.checks], advice=asdict(self.advice))

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=_jsonable)

    def to_text(self, width: int = 100) -> str:
        lines = [f"causalcheck {self.version}  |  {self.n_rows} rows x {len(self.variables)} variables"
                 + (f"  ({len(self.discrete)} discrete)" if self.discrete else ""),
                 "=" * width, "CHECKS"]
        for c in self.checks:
            lines.append(f"  [{_TAG[c.status]}] {c.name}")
            lines += _wrap(c.summary, width, indent=9)
        if self.advice.fix_first:
            lines += ["", "FIX FIRST"]
            for f in self.advice.fix_first:
                lines += _wrap("- " + f, width, indent=2, hang=4)
        lines += ["", "METHODS"]
        for m in self.advice.methods:
            lines += _wrap(f"{m.verdict:<12} {m.method}: {m.reason}", width, indent=2, hang=15)
        if self.advice.trust:
            lines += ["", "HOW TO READ YOUR RESULTS"]
            for t in self.advice.trust:
                lines += _wrap("- " + t, width, indent=2, hang=4)
        lines += ["=" * width,
                  "Checks describe the data, not any graph. They cannot detect every violation "
                  "(faithfulness and selection bias are untestable)."]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_text()


def _wrap(text: str, width: int, indent: int, hang: int | None = None) -> list[str]:
    import textwrap
    hang = indent if hang is None else hang
    return textwrap.wrap(text, width=width, initial_indent=" " * indent, subsequent_indent=" " * hang) or [""]


def _jsonable(o):
    try:
        import numpy as np
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    return str(o)


def check(data, confounding: bool = False, seed: int = 0) -> Report:
    """Run every check on a DataFrame (or anything DataFrame() accepts) and return a Report.

    >>> import causalcheck
    >>> report = causalcheck.check(df)
    >>> print(report)
    """
    df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    prep, results = run_checks(df, confounding=confounding, seed=seed)
    return Report(n_rows=len(prep.X), variables=prep.names, discrete=prep.discrete,
                  checks=results, advice=recommend(results))
