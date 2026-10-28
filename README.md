# causalcheck

Pre-flight assumption checks for causal discovery.

Every causal discovery method rests on assumptions about the data: linear relations, non-Gaussian noise, no hidden confounders, no near-deterministic links, comparable scales. Libraries will happily run any method on any data and return a confident graph either way. `causalcheck` looks at the data first and tells you which methods it can support, which outputs to trust, and what to fix before you run anything.

It describes the data, not a graph. It does not do causal discovery itself.

## Install

```bash
pip install causalcheck
pip install "causalcheck[full]"   # adds causal-learn for the optional hidden-confounder check
```

## Usage

Command line:

```bash
causalcheck data.csv               # text report
causalcheck data.csv --json        # machine-readable
causalcheck data.csv --confounding # also run FCI to flag likely hidden common causes (slower)
```

Python:

```python
import pandas as pd
import causalcheck

report = causalcheck.check(pd.read_csv("data.csv"))
print(report)                 # text report
report.to_json()              # same content as JSON
report.worst_status           # "pass" | "info" | "warn" | "fail"
report.advice.methods         # per-method verdicts
```

The exit code of the CLI is `1` when any check fails, so it can gate a pipeline.

## What it checks

| Check | Question it answers | Why it matters |
|---|---|---|
| data types | Which columns are continuous, discrete, or non-numeric? How many rows have missing values? | Additive-noise direction tests need continuous variables; mixed data needs mixed-data scores |
| sample size | Is n large enough for the number of variables? | Small n gives weak CI tests and unstable orientations |
| linearity | Does a spline model beat a straight line out of sample for any pair? | Fisher-z, linear BIC and LiNGAM are mis-specified on nonlinear pairs |
| noise distribution | Are the residuals non-Gaussian (beyond a trivial amount)? | Non-Gaussian noise is what lets LiNGAM orient edges; Gaussian noise leaves only an equivalence class |
| near-determinism | Is any variable almost fully explained by the others? Are any pairs effectively duplicates? | Faithfulness breaks: CI-based skeletons (PC, FCI) drop true edges; duplicates confuse every method |
| scale | Do variances differ by orders of magnitude? | Continuous-optimization methods (NOTEARS, DAGMA) tend to recover the variance ordering instead of the causal one |
| hidden confounding (optional) | Does FCI consistently see bidirected edges? | Methods that assume no latents will orient those edges wrongly |

Each check returns one of four statuses:

- `pass`: the assumption looks satisfied
- `warn`: something that will mislead some methods
- `fail`: a method family should not be used on this data as is
- `info`: a finding that routes you to a method, with no good/bad meaning

## What it recommends

From the check results, the report gives:

- **Fix first:** data problems to resolve before running anything (duplicates, standardization, unencoded columns, heavy missingness).
- **Methods:** a verdict of `recommended`, `usable` or `avoid` for each family (PC, FCI/GFCI, GES/BOSS/GRaSP, DirectLiNGAM, ANM/CAM/RESIT, NOTEARS/DAGMA), with a one-line reason.
- **How to read your results:** what the output of any method can and cannot mean on this data.

## Example

```
causalcheck 0.1.0  |  1000 rows x 3 variables
CHECKS
  [PASS] linearity         all pairwise relations look linear
  [PASS] noise distribution 3/3 residuals clearly non-Gaussian -> LiNGAM-style direction identification is possible
  [WARN] scale             variances differ by a factor of 37 -> standardize first
METHODS
  recommended  GES / BOSS / GRaSP (score-based): most reliable skeleton
  recommended  DirectLiNGAM: linear with non-Gaussian noise: full DAG and coefficients are identifiable
  avoid        NOTEARS / DAGMA (continuous optimization): variance ordering will drive the result
```

## Limitations

- Some assumptions cannot be tested from observational data: faithfulness (no canceling effects) and the absence of selection bias. The report says so rather than pretending.
- The noise check uses each variable's residual given all others as a stand-in for its true disturbance, because the graph is unknown.
- The hidden-confounder check is a signal, not a proof. Passing it does not rule out confounding.
- Thresholds (linearity gain 0.02, R² 0.98 for near-determinism, variance ratio 10) are sensible defaults, not laws. All are arguments to the check functions.

## Development

```bash
git clone https://github.com/Tahahaha7/causalcheck && cd causalcheck
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## License

MIT
