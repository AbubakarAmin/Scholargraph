## Module Overview

WriterAgent — two-pass paper section drafting with feedback-aware revision.
Narrative sections (pre-engineering) and results sections (post-engineering).

# `agents/writer.py`
![Docs version](https://img.shields.io/badge/docs‑v2024.09‑blue)

## Responsibility

Drafts individual paper sections from topic and plan context, optionally incorporating engineer outputs. Stores section history in memory and provides fallback text when generation fails.

## Workflow usage

The orchestrator calls the writer twice:
1. **Pre-engineering** (`write_narrative_sections`): Introduction, Related Work, Methods, and a provisional abstract. Receives an empty result set so it cannot report invented measurements.
2. **Post-engineering** (`write_results_sections`): Results, Discussion, and final Abstract. All quantitative language must be traceable to `engineer_outputs`.

## v4 upgrades (2026-09)

- **Feedback-aware revision**: `draft_section(..., revision_feedback)` carries deterministic check failures + supervisor feedback into the writer prompt. Results redrafts re-draft only failing sections.
- **Narrative revision loop**: On meta-continue, below-threshold narrative sections are re-drafted once with supervisor feedback (`narrative_revision_count` bound = 1).
- **Prompt hardening**: Intro/abstract get literature evidence + "never invent citations" policy; Results gets copy-exact + n=/std/CI + statistical-test + falsifiability-reporting requirements.
- **Context budget**: Experiment JSON dumps bounded (`[:6000]`) per context-rot guidance.

## API

- `draft_section(section_name, topic, plan, ...)`: Main section drafting entry point.
- `draft_section(..., revision_feedback)`: Feedback-aware variant for redrafts.
- `_draft_with_retry(section_name, prompt)`: Internal retry with min-char fallback.
- `_draft_generic_section(section_name, prompt)`: Routes through `_draft_with_retry` (not direct `call_llm`).
