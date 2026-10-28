"""causalcheck: pre-flight assumption checks for causal discovery.

    import causalcheck
    report = causalcheck.check(df)        # df: pandas DataFrame
    print(report)                         # text report
    report.to_json()                      # machine-readable
"""
from .checks import CheckResult, run_checks
from .recommend import Advice, Recommendation, recommend
from .report import Report, check

__version__ = "0.1.0"
__all__ = ["check", "Report", "CheckResult", "run_checks", "recommend", "Advice", "Recommendation"]
