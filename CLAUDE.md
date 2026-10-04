# cot-overwrite: "order is read as time" (research project)

**Start every session by reading `NOTEBOOK.md`.** It holds the current state, findings, corrections, what is running, and open decisions. Use `RESULTS.md` only by section, since it is long. Experiment designs and pre-written thresholds are in `EXPLORE_PLAN.md`.

The user writes in Chinese. Reply in Chinese, honestly, including negative results.

## Rules (set by the user, non-negotiable)

- **No 画标打靶 (no tuning or designing on test data).** Write the threshold before running. Controls differ in exactly one variable.
- **Diagnosis tools are not fixes.** Probes and oracle interventions (anything that is told where the answer is) only explain, locate, or validate. A fix must be format-agnostic and validated on formats it never saw.
- **Held-out data.** ConvoMem personas 50-99, ConvoMem-long and PersonaMem are frozen held-out tests. LongMemEval is "seen" and must be flagged when used as evidence for a fix.
- **Explore before confirming.** Explore with small runs. Confirm on other models and benchmarks only after something works.
- **Update the record after every experiment.** Update `NOTEBOOK.md` (and `RESULTS.md`), then commit and push to the private GitHub repo `kirkz7/cot-overwrite`.
- **Never print LongMemEval conversation text.** Print aggregates only.

## Practicalities

- **Hardware.** Windows, a single RTX 5080 (16 GB), PowerShell plus Git Bash. The Python venv is `.venv`, managed by uv: `uv pip install --python .venv\Scripts\python.exe`.
- **GPU jobs.** They run in `queue15.ps1`. Start or continue it with `resume_queue.ps1`; stop it with `pause_queue.ps1`. Logs are in `logs/`. All jobs are resumable.
- **Before writing code,** read the engineering notes in NOTEBOOK §10. They cover memory-safe long prompts, and the Bash heredoc swallowing backslashes.
- **Daily deep review.** Follow `REVIEW.md`.
