# MindMesh AI

Partitions a cohort of students into project teams from personality, cognitive and
technical assessment data.

**The solver decides the teams. The language model never does.** An OR-Tools CP-SAT
model produces the allocation; a language model is used only to read a teacher's
natural-language requirements into constraints, and to write an explanation for each
team after the fact. The justification is ConstraintBench (arXiv 2602.22465): LLMs
solving constrained optimisation directly reached as low as 0.8% feasibility on crew
assignment, the closest published analogue to team formation.

## Status

| package | state |
| --- | --- |
| `solver/` | complete — model, generator, verifier, CLI, 20 tests |
| `eval/` | complete — 4 baselines + harness + charts, all executed, including direct-LLM against Llama 3.3 70B, Llama 3.1 8B (both hosted) and llama3.1:8b (local) |
| `api/`, `web/` | not started |

## Setup

```bash
python -m venv .venv && .venv/Scripts/python.exe -m pip install -e ".[dev,eval]"
```

## Run the solver

```bash
python -m solver.cli --n 100 --seed 7 --deterministic-time 20
```

Prints status, objective, best bound, optimality gap, wall clock, per-team composition
statistics, and an independent verification of every hard constraint.

## Run the evaluation

```bash
python -m eval.harness --sizes 50 100 200 --seeds 20 --deterministic-time 12 --out eval/results
```

Writes `eval/results/results.csv` and one chart per metric.

The direct-LLM baseline is provider-agnostic. Free options, in order of how strong a
baseline they make:

```bash
# hosted Llama 3.3 70B, free tier, needs GROQ_API_KEY
python -m eval.harness --llm-provider groq

# fully local, no key, no network
ollama pull llama3.1:8b
python -m eval.harness --llm-provider ollama --llm-model llama3.1:8b
```

**If you use Ollama, note the context window.** Its default `num_ctx` is 2048 and it
truncates longer prompts *silently*. Measured prompt sizes are ~930 tokens at n=50,
~1,580 at n=100 and ~2,860 at n=200, and the reply for a 200-student cohort is another
~1,500 — so on the default the model would lose part of the cohort at n=100 and most of
the reply at n=200, and you would be measuring truncation rather than constraint
reasoning. `eval/llm_client.py` sends 16384 explicitly.

A smaller local model makes a weaker baseline and invites "you used a weak model" as a
rebuttal. Running both — a hosted 70B and a local 8B — turns that objection into a
result: the failure gets worse as the model gets smaller.

Without a key for the chosen provider the method is recorded as skipped, never
simulated.

## Tests

```bash
python -m pytest
```

## Results

n ∈ {50, 100, 200} × 20 seeds, 60 solver runs. Direct-LLM results are in the next section.

| method | coverage | composition | parity gap | violations | seconds |
| --- | --- | --- | --- | --- | --- |
| **cpsat** (n=50) | **1.000** | **959** | **0.235** | **0.00** | 15.6 |
| marks_sorted | 0.922 | 619 | 0.309 | 4.10 | 0.00 |
| random | 0.919 | 651 | 0.348 | 4.20 | 0.00 |
| self_selection | 0.796 | 641 | 0.648 | 17.30 | 0.00 |
| **cpsat** (n=100) | **1.000** | **1185** | **0.242** | **0.00** | 24.6 |
| marks_sorted | 0.927 | 750 | 0.374 | 7.40 | 0.00 |
| random | 0.931 | 784 | 0.369 | 7.00 | 0.00 |
| self_selection | 0.777 | 806 | 0.650 | 36.75 | 0.01 |
| **cpsat** (n=200) | **1.000** | **1535** | **0.243** | **0.00** | 31.9 |
| marks_sorted | 0.932 | 1130 | 0.414 | 14.40 | 0.00 |
| random | 0.931 | 1181 | 0.479 | 16.35 | 0.00 |
| self_selection | 0.726 | 1221 | 0.648 | 85.95 | 0.03 |

**Zero hard-constraint violations across all 60 solver runs**, verified by a checker
sharing no code with the model. Every baseline breaks constraints at every size, and
the count grows with the cohort. The breakdown is skill coverage and gender parity —
team size and mentor allocation are satisfied by construction in the baselines, so
those columns are zero for everyone and are not evidence either way.

Composition score is 26-47% above the best baseline at each size. Self-selection is
the worst on both coverage (0.73 at n=200, falling as the cohort grows) and parity
(0.65 throughout) — homophily doing exactly what the literature says it does.

Charts are in [`eval/results/`](eval/results/).

### Direct-LLM baseline

Executed against three models on identical n=50 cohorts. Seed counts differ because
Groq's free tier meters tokens per model and local CPU inference is ~200x slower than
the hosted API; each row states its own n.

| method | model | runs | coverage | composition | parity gap | violations | latency |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **cpsat** | — | 20 | **1.000** | **959** | **0.235** | **0.00** | 15.6s |
| marks_sorted | — | 20 | 0.922 | 619 | 0.309 | 4.10 | <0.01s |
| random | — | 20 | 0.919 | 651 | 0.348 | 4.20 | <0.01s |
| direct_llm | llama-3.3-70b (Groq) | 10 | 0.905 | 677 | 0.558 | 10.90 | 1.2s |
| self_selection | — | 20 | 0.796 | 641 | 0.648 | 17.30 | <0.01s |
| direct_llm | llama3.1:8b (local CPU) | 3 | 0.790 | 646 | 0.423 | 26.33 | 218s |
| direct_llm | llama-3.1-8b (Groq) | 10 | 0.907 | 596 | 0.548 | 32.90 | ~2s |

**Not one direct-LLM run out of 23 produced a valid partition.** The 70B is the best of
the three and still averages 10.9 violations — more than double random assignment. Model
size helps (70B beats 8B roughly threefold) without approaching validity. The dominant
failure is not a subtle constraint but the partition itself: students duplicated across
teams or dropped entirely.

The three-run local row is indicative, not a mean you should quote. It is here to show
the size trend holds on a second, independent deployment of an 8B model.

### Reproducibility on identical input

Five repeats (three for local) of the same cohort at temperature 0 with a fixed seed:

| model | mean pairwise ARI | violations across repeats |
| --- | --- | --- |
| llama-3.3-70b (Groq) | **0.258** | 6, 9, 13, 13, 13 |
| llama-3.1-8b (Groq) | 1.000 | 60, 60, 60, 60, 60 |
| llama3.1:8b (local) | 1.000 | 22, 22, 22 |
| **cpsat** | **1.000** (by test) | **0, 0, 0** |

ARI 0.258 means the 70B's five partitions were nearly unrelated to one another. Same
input, same settings, a different allocation and a different number of broken
constraints each time.

**State this carefully.** The smaller models *were* perfectly reproducible here, so
"LLMs give a different answer every run" is not what this shows and a reviewer can
falsify it. What it shows is that reproducibility cannot be relied upon and cannot be
predicted from model size — the larger, more capable model was the unstable one. The
solver's determinism is guaranteed by construction and asserted by test.

### Observed failure modes

- **Entity hallucination.** The 70B returned `'S0028" -> "S0041'` as a single student
  id — invented range notation in a field with no such syntax. This is the behaviour
  ConstraintBench reports, reproduced here on this project's own data.
- **Dropped students.** The local 8B omitted the same 14 of 50 students on all three
  repeats.
- **Silent size drift.** Teams of 4 returned where the bounds were 4-6, with the
  shortfall simply unassigned.

### What these numbers are not

The solver reports `FEASIBLE`, not `OPTIMAL`, on all 60 runs: the mean optimality gap
is 0.556 (range 0.42-0.75) after 12 deterministic time units. The bound is weak, not
the solution — the incumbent stops improving long before the bound catches up. The
claim is zero violations and a better composition score, both measured. Proven
optimality is not claimed and would need a stronger formulation or a much longer
budget.

## The model

Decision variable `x[i][t] = 1` when candidate `i` is on team `t`.

**Hard constraints.** Every candidate on exactly one team; team size in `[L, U]`; each
team covers every required skill at a configurable threshold; exactly one mentor per
team from a separate pool; per-team gender counts within a tolerance of the cohort
proportion.

**Objective**, weighted sum, from Bell (2007) and Peeters et al. (2006):

1. maximise the minimum team mean conscientiousness
2. maximise the sum over teams of team minimum agreeableness
3. minimise within-team agreeableness spread
4. minimise the spread of total technical skill across teams

Two of these differ from the first-review write-up, and the reasons are worth reading
before the second review:

- **Term 1.** The write-up asked to maximise *summed* team conscientiousness. Under a
  whole-cohort partition that sum equals the cohort total whatever the partition — a
  constant. It cannot be optimised, and the failure is silent: the solver runs and the
  term simply has no effect. The distribution across teams is the only thing the
  partition controls, so the minimum team mean is maximised instead.
  `--conscientiousness-mode sum` reproduces the original for the record.
- **Term 3.** Peeters et al. report agreeableness *variability*, which is quadratic.
  CP-SAT cannot take a quadratic objective, so within-team range is substituted. Range
  and variance are not the same thing — the range sees only the two extreme members.
  This is a stated limitation, not a hidden one.

Full derivations, including the exact encoding of a mean with a variable team size and
the channelling of the team minimum through the assignment booleans, are in
[`openspec/changes/add-cpsat-team-solver/design.md`](openspec/changes/add-cpsat-team-solver/design.md).

## Where the data comes from

Cohorts are synthetic, but the personality half is no longer invented. Big Five traits
are sampled from marginals **and inter-trait correlations** fitted to 603,322 responses
from the Open Psychometrics IPIP-FFM dataset (1,015,342 raw, filtered per its own
codebook). The fitted parameters are committed at
[`solver/data/big_five_params.json`](solver/data/big_five_params.json) with their full
provenance; [`solver/empirical.py`](solver/empirical.py) documents the caveats.

**Skill vectors and gender remain synthetic.** No public dataset pairs Big Five scores
with technical skill assessments for a student cohort. Do not describe the skill data
as empirical.

The respondents are self-selected internet volunteers, not students — openness sits
high (mean 73.5/100), which is characteristic of that population.

## Determinism

`num_search_workers = 1` and a fixed seed. Two runs on identical input return identical
assignments, asserted by test.

A **wall-clock** time limit breaks that guarantee — if it fires mid-search, the answer
depends on machine speed. Use `--deterministic-time`, which counts solver work units,
whenever reproducibility matters. That is what the evaluation harness uses.

## What is not claimed

The solver is not run to proven optimality at these cohort sizes; the mean gap is
0.556 after 12 deterministic time units, and every run reports it. The claim is zero
hard-constraint violations plus a better composition score than the baselines, both
verified independently — not global optimality.
