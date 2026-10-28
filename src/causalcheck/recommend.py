"""Turn check results into advice: what to fix first, which methods fit, what to trust."""
from __future__ import annotations

from dataclasses import dataclass, field

from .checks import CheckResult

RECOMMENDED, USABLE, AVOID = "recommended", "usable", "avoid"


@dataclass
class Recommendation:
    method: str
    verdict: str
    reason: str


@dataclass
class Advice:
    fix_first: list[str] = field(default_factory=list)
    methods: list[Recommendation] = field(default_factory=list)
    trust: list[str] = field(default_factory=list)


def _by_name(results: list[CheckResult]) -> dict[str, CheckResult]:
    return {r.name: r for r in results}


def recommend(results: list[CheckResult]) -> Advice:
    r = _by_name(results)
    adv = Advice()

    types = r.get("data types")
    size = r.get("sample size")
    lin = r.get("linearity")
    noise = r.get("noise distribution")
    det = r.get("near-determinism")
    scale = r.get("scale")
    conf = r.get("hidden confounding")

    linear = lin is not None and lin.status == "pass"
    nongauss = noise is not None and noise.status == "pass"
    gaussian = noise is not None and noise.status == "warn"
    mixed = types is not None and bool(types.details.get("discrete"))
    small = size is not None and size.status == "warn"
    near_det = det is not None and det.status in ("warn", "fail")
    latents = conf is not None and conf.status == "warn"
    latents_untested = conf is None or conf.status == "info"

    # ---- fix first
    if det is not None and det.status == "fail":
        adv.fix_first.append("Drop or merge the near-duplicate columns listed under near-determinism; "
                             "every method treats them as a confusing extra edge.")
    if scale is not None and scale.status == "warn":
        adv.fix_first.append("Standardize all variables before running anything, or any continuous-"
                             "optimization method will mostly recover the variance ordering.")
    if types is not None and types.details.get("dropped"):
        adv.fix_first.append(f"Encode or drop the non-numeric columns {types.details['dropped']} explicitly, "
                             "rather than letting them be silently ignored.")
    if types is not None and types.details.get("rows_dropped_missing", 0) > 0.2 * types.details.get("rows_used", 1):
        adv.fix_first.append("More than 20% of rows were dropped for missing values; impute or check "
                             "whether missingness itself depends on the variables.")

    # ---- method verdicts
    def add(method, verdict, reason):
        adv.methods.append(Recommendation(method, verdict, reason))

    # constraint-based, no latents
    if near_det:
        add("PC (constraint-based)", AVOID, "near-deterministic links make CI tests drop true edges")
    elif latents:
        add("PC (constraint-based)", AVOID, "assumes no hidden confounders, but FCI found signs of them")
    else:
        add("PC (constraint-based)", USABLE, "use fisherz if linear, kci if nonlinear; expect an equivalence class")

    # latent-tolerant constraint-based
    if near_det:
        add("FCI / GFCI (latent-tolerant)", USABLE if latents else AVOID,
            "CI tests are unreliable here; if you need latent handling, use GFCI or BOSS-FCI, "
            "which take the skeleton from a score")
    else:
        add("FCI / GFCI (latent-tolerant)", RECOMMENDED if (latents or latents_untested) else USABLE,
            "honest about hidden confounding; output is a PAG with circle endpoints")

    # score-based
    if latents:
        add("GES / BOSS / GRaSP (score-based)", USABLE,
            "best skeleton recovery, but orientations assume no latents; check them against FCI")
    else:
        add("GES / BOSS / GRaSP (score-based)", RECOMMENDED,
            "most reliable skeleton" + ("; the right choice under near-determinism" if near_det else "")
            + ("; use a mixed-data score (conditional Gaussian / BDeu)" if mixed else "")
            + ("; use a nonparametric score (basis-function BIC)" if not linear else ""))

    # LiNGAM
    if not linear:
        add("DirectLiNGAM", AVOID, "relations are nonlinear; LiNGAM's linear model is mis-specified")
    elif gaussian:
        add("DirectLiNGAM", AVOID, "noise looks Gaussian: LiNGAM will output confident but arbitrary directions")
    elif latents:
        add("DirectLiNGAM", AVOID, "signs of hidden confounding; use RCD or LiNGAM-with-latents instead")
    elif nongauss:
        add("DirectLiNGAM", RECOMMENDED, "linear with non-Gaussian noise: full DAG and coefficients are identifiable"
            + ("; verify directions with bootstrap at this sample size" if small else ""))
    else:
        add("DirectLiNGAM", USABLE, "only some disturbances are non-Gaussian; trust arrows touching those variables")

    # nonlinear additive-noise
    if not linear and not mixed:
        add("ANM / CAM / RESIT (nonlinear additive noise)", RECOMMENDED if not latents else USABLE,
            "nonlinearity itself identifies directions" + ("; confounding may bias them" if latents else ""))
    elif mixed:
        add("ANM / CAM / RESIT (nonlinear additive noise)", AVOID, "needs continuous variables; discrete columns present")
    else:
        add("ANM / CAM / RESIT (nonlinear additive noise)", USABLE,
            "works, but with linear relations the direction signal comes only from non-Gaussian noise")

    # continuous optimization
    if scale is not None and scale.status == "warn":
        add("NOTEARS / DAGMA (continuous optimization)", AVOID,
            "variance ordering will drive the result (varsortability); standardize first and still distrust directions")
    elif gaussian:
        add("NOTEARS / DAGMA (continuous optimization)", AVOID,
            "with Gaussian noise its directions are not identifiable; it will still output a DAG")
    else:
        add("NOTEARS / DAGMA (continuous optimization)", USABLE,
            "fine for a skeleton; validate directions against a method with identifiability guarantees")

    # ---- trust notes
    if gaussian:
        adv.trust.append("Expect an equivalence class: only collider patterns can be oriented from this data. "
                         "Treat any fully directed output as a guess beyond those.")
    if nongauss and linear:
        adv.trust.append("Directions and coefficients from LiNGAM are trustworthy here, provided there are "
                         "no hidden confounders.")
    if not linear:
        adv.trust.append("Linear-only outputs (Fisher-z tests, linear BIC, LiNGAM) are mis-specified for the "
                         "nonlinear pairs listed in the linearity check.")
    if near_det:
        adv.trust.append("Missing edges from PC/FCI are suspect: near-deterministic links hide real "
                         "dependencies from CI tests.")
    if small:
        adv.trust.append("Bootstrap every result; keep only edges and directions that appear in most resamples.")
    if latents:
        adv.trust.append("Pairs flagged as bidirected may be driven by a hidden common cause; "
                         "do not read their arrows as cause and effect.")
    elif latents_untested:
        adv.trust.append("Hidden confounding was not tested; no method here can rule it out from "
                         "observational data alone.")
    if mixed:
        adv.trust.append("Additive-noise direction tests are meaningless on the discrete columns; "
                         "their edges will stay undirected unless a collider orients them.")
    return adv
