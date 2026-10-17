"""Tests for the empirically fitted trait distributions."""

from __future__ import annotations

import numpy as np
import pytest

from solver.empirical import load_params, sample_traits
from solver.generator import generate_cohort


def test_params_carry_their_provenance() -> None:
    source = load_params().source

    assert "openpsychometrics" in str(source["url"])
    assert source["n_retained"] > 100_000
    assert "IPC == 1" in str(source["filters"])


def test_correlation_matrix_is_valid() -> None:
    params = load_params()
    matrix = np.array(params.correlation)

    assert matrix.shape == (5, 5)
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 1.0)
    # Positive definite, or the Cholesky factor used for sampling would not exist.
    assert np.all(np.linalg.eigvalsh(matrix) > 0)


def test_sampling_reproduces_the_fitted_marginals() -> None:
    params = load_params()
    sample = sample_traits(200_000, seed=0)

    for index, trait in enumerate(params.trait_order):
        # Clamping to 0-100 pulls the mean in slightly, most for openness (mean 73, sd 15).
        assert sample[:, index].mean() == pytest.approx(params.mean[index], abs=2.0), trait
        assert sample[:, index].std() == pytest.approx(params.sd[index], rel=0.10), trait


def test_sampling_reproduces_the_inter_trait_correlations() -> None:
    """The point of the refit: independent gaussians would give a zero matrix here."""
    params = load_params()
    sample = sample_traits(200_000, seed=0)
    observed = np.corrcoef(sample.T)

    assert np.allclose(observed, np.array(params.correlation), atol=0.02)


def test_the_signs_the_literature_reports_survive() -> None:
    params = load_params()
    order = list(params.trait_order)
    matrix = np.array(params.correlation)

    conscientiousness = order.index("conscientiousness")
    neuroticism = order.index("neuroticism")
    extraversion = order.index("extraversion")
    agreeableness = order.index("agreeableness")

    assert matrix[conscientiousness][neuroticism] < 0
    assert matrix[extraversion][agreeableness] > 0


def test_traits_are_integers_in_range_and_reproducible() -> None:
    first = generate_cohort(80, seed=3)
    second = generate_cohort(80, seed=3)

    assert first.candidates == second.candidates
    for candidate in first.candidates:
        for trait in ("openness", "conscientiousness", "extraversion", "agreeableness"):
            value = getattr(candidate.traits, trait)
            assert isinstance(value, int) and 0 <= value <= 100


def test_a_different_seed_gives_a_different_cohort() -> None:
    assert generate_cohort(80, seed=3).candidates != generate_cohort(80, seed=4).candidates
