"""Empirical Big Five distribution parameters, and sampling from them.

The parameters in ``solver/data/big_five_params.json`` were fitted from the Open
Psychometrics IPIP-FFM raw dataset (1,015,342 responses collected 2016-2018 with the
IPIP Big-Five Factor Markers; 603,322 retained after the codebook's own cleanliness
filters). The fitting script is not part of the package — the committed parameters are
the artifact, and ``source`` inside the JSON records exactly how they were produced.

**Read these caveats before claiming the cohorts are realistic:**

1. The sample is self-selected internet respondents, not students. Openness in
   particular sits high (mean 73.5 / 100), which is characteristic of people who
   volunteer for online personality tests. It is not a normative student sample.
2. Only the *marginals and the inter-trait correlations* are empirical. Sampling is
   multivariate normal, so higher moments (skew, the discreteness of a summed 10-item
   scale) are not reproduced.
3. Clamping to 0-100 slightly compresses the tails, most visibly for openness.
4. **Skill vectors and gender are still synthetic.** No public dataset pairs Big Five
   scores with technical skill assessments for a student cohort, so those remain
   invented. Do not describe the skill data as empirical.

The dataset carries no explicit licence; the codebook states respondents consented to
their anonymous answers being used for research. Cite it as Open Psychometrics raw
data and link the source page.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

PARAMS_PATH = Path(__file__).parent / "data" / "big_five_params.json"


@dataclass(frozen=True)
class BigFiveParams:
    """Fitted marginals and correlation matrix, in ``trait_order``."""

    trait_order: tuple[str, ...]
    mean: tuple[float, ...]
    sd: tuple[float, ...]
    correlation: tuple[tuple[float, ...], ...]
    source: dict[str, object]

    @property
    def cholesky(self) -> np.ndarray:
        """Lower-triangular factor of the correlation matrix, for correlated sampling."""
        return np.linalg.cholesky(np.array(self.correlation, dtype=float))


@lru_cache(maxsize=1)
def load_params(path: Path | None = None) -> BigFiveParams:
    """Load the fitted parameters. Cached: the file never changes at runtime."""
    payload = json.loads((path or PARAMS_PATH).read_text(encoding="utf-8"))
    order = tuple(payload["trait_order"])
    return BigFiveParams(
        trait_order=order,
        mean=tuple(float(payload["mean"][t]) for t in order),
        sd=tuple(float(payload["sd"][t]) for t in order),
        correlation=tuple(tuple(float(v) for v in row) for row in payload["correlation"]),
        source=payload["source"],
    )


def sample_traits(count: int, seed: int, params: BigFiveParams | None = None) -> np.ndarray:
    """Draw ``count`` correlated Big Five profiles as integers 0-100.

    Multivariate normal with the empirical means, standard deviations and correlation
    matrix, then clamped and rounded. Deterministic: ``numpy.random.PCG64`` produces the
    same stream for the same seed on every platform, which the solver's reproducibility
    requirement depends on.

    Returns an array of shape ``(count, 5)`` in ``params.trait_order``.
    """
    params = params or load_params()
    generator = np.random.Generator(np.random.PCG64(seed))
    standard = generator.standard_normal(size=(count, len(params.trait_order)))
    correlated = standard @ params.cholesky.T
    scaled = np.array(params.mean) + np.array(params.sd) * correlated
    return np.clip(np.rint(scaled), 0, 100).astype(int)
