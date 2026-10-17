# Tasks: add-empirical-cohort-data

- [x] 1. Fit marginals and the inter-trait correlation matrix from the IPIP-FFM raw
      data; commit only the parameters, with provenance.
- [x] 2. `solver/empirical.py` — loader plus correlated sampling via Cholesky, with the
      caveats in the module docstring.
- [x] 3. `solver/generator.py` — one trait batch per cohort; skills and gender stay
      synthetic and are documented as such.
- [x] 4. `tests/test_empirical.py` — matrix validity, marginal and correlation recovery,
      literature sign checks, reproducibility.
- [x] 5. Re-run the full solver suite against the new distributions to confirm the
      cohorts are still feasible under the default constraints.
- [x] 6. Re-run the evaluation grid, since every published number changes with the
      generator.
- [x] 7. README, CLAUDE.md and deck notes updated to state what is empirical and what
      is not.

# Tasks: provider-agnostic LLM baseline

- [x] 8. `eval/llm_client.py` — one call path for Ollama, Groq, OpenRouter, OpenAI;
      Gemini's native API branches separately. `urllib` only, no new dependency.
- [x] 9. Send `num_ctx` explicitly for Ollama. Its 2048 default truncates silently, and
      a truncated prompt would make the baseline measure the wrong thing entirely.
- [x] 10. `--llm-provider` / `--llm-model` / `--llm-base-url` on the harness; the model
      used is recorded on every run.
- [x] 11. Tests: every provider resolves, unknown provider rejected, hosted providers
      skip without a key, Ollama needs none, context window is large enough.
- [ ] 12. Execute the direct-LLM baseline. **Not run here** — needs either a running
      Ollama or a provider key.
