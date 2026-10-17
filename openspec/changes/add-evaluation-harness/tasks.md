# Tasks: add-evaluation-harness

- [x] 1. `eval/baselines.py` — random, self-selection, marks-sorted; all deterministic
      given their seed.
- [x] 2. `eval/metrics.py` — coverage rate, composition score, parity gap, violation
      breakdown, ARI for run-to-run variance.
- [x] 3. `eval/llm_baseline.py` — real Gemini call, fully specified prompt, structured
      JSON, no repair of the output, skipped (never simulated) without an API key.
- [x] 4. `eval/harness.py` — the n × seed × method grid, CSV output, console summary.
- [x] 5. `eval/plots.py` — one chart per metric, methods with no successful runs
      labelled rather than drawn as zero.
- [x] 6. Tests: baselines partition the cohort, baselines reproducible, metrics on a
      hand-built case, ARI properties, prompt completeness, skip-without-key, harness
      end to end.
- [x] 7. Full grid run at n = 50, 100, 200 with 20 seeds; CSV and charts committed.
- [ ] 8. Direct-LLM baseline executed against the real API. **Blocked: no
      `GEMINI_API_KEY` in this environment.** Everything else is in place; the run is a
      single command once a key exists.
