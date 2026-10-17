# Proposal: add-empirical-cohort-data

## Why

The cohort generator's Big Five distributions were invented. `gauss(55, 18)` for
conscientiousness, `gauss(58, 16)` for agreeableness, and so on — plausible-looking
numbers with no source. Worse, the five traits were drawn **independently**, so the
generated cohorts had a zero inter-trait correlation matrix, which is not a property
any real personality data has.

That is fine for testing that constraints hold. It is not fine the moment the project
claims its composition scores say something about real student teams.

## What changes

- `solver/data/big_five_params.json` — means, standard deviations and the full 5×5
  correlation matrix, fitted from the Open Psychometrics IPIP-FFM raw dataset
  (1,015,342 responses collected 2016-2018 with the IPIP Big-Five Factor Markers;
  603,322 retained after the codebook's own filters: `IPC == 1`, all 50 items answered
  in range). The JSON carries its own provenance block.
- `solver/empirical.py` — loads the parameters and samples correlated profiles by
  Cholesky factorisation of the correlation matrix, clamped to 0-100.
- `solver/generator.py` — draws traits from that fit in one batch per cohort, so the
  correlations hold across the cohort rather than being redrawn per person.

Determinism is preserved: `numpy.random.PCG64` gives the same stream for the same seed
on every platform.

## What stays synthetic, and must be described as such

**Skill vectors and gender.** No public dataset pairs Big Five scores with technical
skill assessments for a student cohort. These remain invented, and both the module
docstring and the README say so in as many words.

## Caveats that ship with the data

1. Self-selected internet respondents, not students. Openness sits high (mean 73.5),
   which is characteristic of people who volunteer for online personality tests.
2. Only marginals and correlations are empirical. Sampling is multivariate normal, so
   skew and the discreteness of a summed 10-item scale are not reproduced.
3. Clamping to 0-100 slightly compresses the tails.
4. The dataset carries no explicit licence. Its codebook states respondents consented
   to their anonymous answers being used for research.

## Verification

`tests/test_empirical.py`: the correlation matrix is symmetric, unit-diagonal and
positive definite; sampling reproduces the fitted marginals and correlations at
n = 200,000; the signs the literature reports (C–N negative, E–A positive) survive;
cohorts remain reproducible from `(n, seed)`.
