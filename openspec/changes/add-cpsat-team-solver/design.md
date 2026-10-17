# Design: CP-SAT team partitioning model

## Notation

- `n` candidates (students), indexed `i`; `M` mentors, indexed `j`; `T` teams, indexed `t`.
- `x[i][t] ∈ {0,1}` — candidate `i` is on team `t`.
- `m[j][t] ∈ {0,1}` — mentor `j` leads team `t`.
- `size_t = Σ_i x[i][t]`.
- Trait scores and skill scores are integers 0-100.

`T` is supplied, or derived as `ceil(n / U)`. If no `T` admits `T·L ≤ n ≤ T·U`, or
if `T > M`, the model is simply infeasible and the solver says so.

## 1. Why the conscientiousness term had to change

The first-review write-up asks to "maximise summed team conscientiousness (team size
is bounded so a sum is a valid stand-in for the mean)". This is wrong, and it fails
in a way that is silent — the solver runs, returns a solution, and the term has no
effect on it.

Every candidate is assigned to exactly one team. Therefore

    Σ_t Σ_i c_i · x[i][t] = Σ_i c_i

which is a **constant of the cohort**, independent of the partition. Maximising it
maximises nothing. The equal-team-size case does not rescue it either: if all teams
have size `L`, then `Σ_t mean_c(t) = (1/L)·Σ_i c_i`, still constant.

Bell (2007) reports that team mean conscientiousness predicts team performance. Under
a whole-cohort partition, total elevation is invariant; the only thing the partition
controls is how it is **distributed**. The actionable reading of the finding is
therefore: do not leave any team starved of conscientiousness. Implemented as

    maximise  min_t mean_c(t)

which is non-constant, is monotone in the same direction as Bell's finding, and
degenerates to the constant case only when it is genuinely unavoidable.

`SolverConfig.conscientiousness_mode` accepts `"sum"` to reproduce the original
formulation, for the record. It is documented as degenerate and is not the default.

## 2. Team mean, exactly, with variable team size

CP-SAT is integer-linear; `mean_c(t) = Σ_i c_i x[i][t] / size_t` is a division by a
variable. It is encoded exactly as a floor, scaled by 10 to retain one decimal:

    prod_t = mean_c_t · size_t            (AddMultiplicationEquality)
    0 ≤ 10·Σ_i c_i x[i][t] − prod_t ≤ size_t − 1

which pins `mean_c_t = floor(10 · Σ c_i x[i][t] / size_t)`, range 0..1000. One
multiplication constraint per team; `T ≤ 25` in practice, so the propagation cost is
negligible.

## 3. Team minimum agreeableness, channelled through the assignment booleans

A plain `AddMinEquality` over `a_i` would take the minimum over the whole cohort. The
minimum must be over *assigned* members only, so each team gets a shadow variable per
candidate that is neutral when the candidate is elsewhere:

    lo[i][t] ∈ [0,100],   x[i][t] ⇒ lo[i][t] = a_i,   ¬x[i][t] ⇒ lo[i][t] = 100
    min_a_t = Min_i lo[i][t]

100 is the upper bound of the trait scale, so an unassigned candidate can never be the
argmin. Symmetrically for the maximum with neutral value 0:

    hi[i][t] ∈ [0,100],   x[i][t] ⇒ hi[i][t] = a_i,   ¬x[i][t] ⇒ hi[i][t] = 0
    max_a_t = Max_i hi[i][t]

Team size is at least 1 by constraint, so both are well defined.

## 4. Variance proxy — a stated limitation

Peeters et al. (2006) report agreeableness *variability* at rho = −0.12 and
conscientiousness variability at rho = −0.24. Variance is quadratic and CP-SAT cannot
take a quadratic objective.

**Substitution:** within-team range, `max_a_t − min_a_t`, is minimised instead. Range
and standard deviation are monotonically related for a fixed sample size but not
equal; the range is driven entirely by the two extreme members and is blind to how
the interior is distributed. Two teams with identical range can have materially
different variance. This is a known and accepted limitation of the linearisation, and
it is why the term is reported separately in the evaluation harness rather than
folded into a single headline number.

The conscientiousness variability finding (the stronger of the two, rho = −0.24) is
not currently in the objective; it is left out deliberately rather than approximated,
because term 1 already constrains the conscientiousness distribution across teams.

## 5. Demographic parity with variable team size

For gender `g` with cohort proportion `p_g`, scaled to `P_g = round(1000·p_g)`, and
tolerance `TOL = round(1000·tolerance)`:

    (P_g − TOL) · size_t  ≤  1000 · count_{g,t}  ≤  (P_g + TOL) · size_t

Both sides are linear in the assignment booleans because `size_t` is itself a linear
expression. No integer rounding of the target count is needed, so the constraint stays
exact for every admissible team size.

## 6. Technical skill balance

Each candidate carries `mean_skill_i = round(mean of their skill vector)`, 0..100.
Team total is `Σ_i mean_skill_i · x[i][t]`; the objective minimises
`max_t − min_t` of those totals. Team totals scale with team size, so when `L ≠ U`
this term mildly penalises size imbalance as well. Acceptable, and noted.

## 7. Objective assembly

    maximise   Wc·(min_t mean_c_t)
             + Wa·(Σ_t min_a_t)
             − Ws·(Σ_t (max_a_t − min_a_t))
             − Wk·(max_t skill_t − min_t skill_t)

Float weights are converted to integer coefficients by `round(w · 1000)`. Default
weights (1.0, 1.0, 0.5, 0.5) put the four terms within roughly the same order of
magnitude for `T ≈ 10`; they are configurable and the harness reports each term
separately, so re-weighting does not require re-reading the objective.

## 8. Symmetry breaking

Teams are interchangeable labels, so every solution has `T!` relabellings and the
search wastes its time proving them all equivalent. Mentors are forced into ascending
team order:

    Σ_j j·m[j][t]  <  Σ_j j·m[j][t+1]

Relabelling a team is free, so this removes no distinct solution and cannot change the
optimal objective value. It is also deterministic, which matters for §9.

## 9. Determinism

`num_search_workers = 1`, fixed `random_seed`, no wall-clock-dependent search choices.
Two runs on identical input return identical assignments, asserted by test.

**Caveat, stated plainly:** a *wall clock* time limit reintroduces nondeterminism —
if the limit fires mid-search, which solution is in hand depends on machine speed.
`SolverConfig.max_time_seconds` is therefore a safety net, and `SolveResult.hit_limit`
records whether it fired. For guaranteed reproducibility set
`max_deterministic_time` instead, which counts solver work units rather than seconds.

## 10. Verification

`solver/verify.py` re-checks all five hard constraints against the returned assignment
using plain Python, sharing no code with the model builder. A bug in the model
encoding cannot hide behind a matching bug in the check.
