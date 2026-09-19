# Research Workflow

## Phases

### Full-research graph

Defined in `core/workflow.py:create_research_graph()`:

1. **`topic_discovery`** — Query OpenAlex, arXiv, and Semantic Scholar. Build evidence map via `core.evidence_synthesis`. Run structural gap mining, sparsity matrix analysis, and contradiction mining. Score novelty and reject infeasible topics. Emit `ResearchSourceUnavailable` if all sources are down (terminal, not a reset).
2. **`hypothesis_debate`** — Run proposer, challenger (with capability manifest + MVE validation), and moderator rounds. Ensemble judge over configured models. Unresolved severity ≥ 4 objections block pass. One bounded contract-repair attempt allowed before final rejection.
3. **`planning`** — Create a falsifiable plan with baselines, variants, metrics, dependencies, statistical tests, and a catalogued local dataset. Feasibility checked against `SandboxCapabilityManifest` before commitment. Invalid plans route to `terminal_planning_failure` → END.
4. **`terminal_planning_failure`** — Records the failure reason and stops the run. No recovery path.
5. **`data_validation`** — If the plan declares an explicit `dataset_path` or `dataset_file`, `DataAgent` validates schema, target, hash, and row count. Invalid data terminates the run. Plans without explicit datasets retain synthetic-data compatibility.
6. **`writing_narrative`** — Draft Introduction, Related Work, Methods, and a provisional abstract. Receives an empty result set so it cannot report invented measurements. On meta-continue, below-threshold narrative sections are re-drafted once with supervisor feedback (`narrative_revision_count` bound = 1).
7. **`engineering`** — Build immutable experiment contracts via `core.evidence_gate`. Generate code, validate against AST sandbox, execute with multi-seed runs, run branch search over variants, auto-generate ablations. Plan revision requests bounce back to `planning`. Code-claim consistency checked against the capability manifest.
8. **`independent_validation`** — Three-agent chain: `ExecutionAgent` (seeded replay + artifact creation) → `AnalysisAgent` (SciPy summaries, CIs, Welch tests, effect sizes) → `VerificationAgent` (hash integrity, statistical agreement with raw results). Blocking findings prevent editing.
9. **`writing_results`** — Draft Results, Discussion, and final Abstract from `engineer_outputs`. Numeric claims are compared against recursively extracted recorded values. Unmatched claims trigger a redraft (max `results_redraft_count` retries). Carries deterministic check failures + supervisor feedback into the writer prompt.
10. **`supervision`** — Hard citation + stats checks (`hard_verify_section`), MathChecker (SymPy), CodeChecker (compile/delimiters), then soft LLM reviewer. Hard failure caps section score at 4.0. Requires explicit falsifiable-prediction verdict (supported/falsified/inconclusive) with controls/robustness evidence.
11. **`meta_evaluation`** — Improve prose only for eligible evidence. Cannot override terminal technical failure or mutate a committed contract.
12. **`editing`** — Add limitations from unresolved debate objections, resolve bibliography entries, run deterministic reviewer checklist and consistency referee, export paper + companion repo. Release-referee failures route to one bounded repair pass (`editor_repair_count >= 1`) instead of terminal failure. Human approval required via `POST /api/release/approve`.
13. **`reset`** — Clear topic state and return to `topic_discovery` for the next iteration.

### QA-mode graph

Defined in `core/workflow.py:create_qa_graph()`:

1. **`qa_literature_retrieval`** — Search OpenAlex/arXiv for the user query, build a literature context.
2. **`qa_answer`** — Synthesize an answer with key findings and bibliography. Citation grounding enforced via `verify_citations()` inside the node. Uses `call_llm_json` with parse-error re-ask.
3. **`qa_verification`** — Check answer completeness and citation resolution before completion.

The QA graph skips hypothesis debate, planning, engineering, and evidence-gate machinery entirely.

## Conditional edges (exact)

| From | Condition | To |
|---|---|---|
| `topic_discovery` | `should_reset` | `reset` |
| `topic_discovery` | `should_continue` | `hypothesis_debate` |
| `topic_discovery` | `terminal_error` or `complete` | END |
| `hypothesis_debate` | `should_reset` | `reset` |
| `hypothesis_debate` | `should_continue` | `planning` |
| `hypothesis_debate` | `terminal_error` or `complete` | END |
| `planning` | valid plan | `data_validation` |
| `planning` | invalid plan | `terminal_planning_failure` |
| `planning` | `should_reset` or `terminal_error` | `reset` or END |
| `terminal_planning_failure` | always | END |
| `data_validation` | `should_continue` | `writing_narrative` |
| `data_validation` | `terminal_error` or `complete` | END |
| `writing_narrative` | `should_continue` | `engineering` |
| `writing_narrative` | `terminal_error` or `complete` | END |
| `engineering` | `current_phase == "planning"` | `planning` (revision bounce) |
| `engineering` | `should_continue` | `independent_validation` |
| `engineering` | `terminal_error` or `complete` | END |
| `independent_validation` | `should_continue` | `writing_results` |
| `independent_validation` | `terminal_error` or `complete` | END |
| `writing_results` | `current_phase == "writing_results"` | `writing_results` (redraft) |
| `writing_results` | `should_continue` | `supervision` |
| `writing_results` | `terminal_error` or `complete` | END |
| `supervision` | `current_phase == "editing"` | `editing` |
| `supervision` | else | `meta_evaluation` |
| `supervision` | `terminal_error` or `complete` | END |
| `meta_evaluation` | `should_continue` | `writing_narrative` |
| `meta_evaluation` | else | END |
| `editing` | `editor_repair_count >= 1` | `writing_results` (repair loop) |
| `editing` | always | END |
| `reset` | always | `topic_discovery` |

## State contract

`core.state.ResearchState` is the single typed contract. The workflow mutates it in place because LangGraph nodes return the updated state. The initializer creates empty collections so nodes can append without null checks.

Key state fields added since the initial design:

| Field | Purpose |
|---|---|
| `mode` | `"full_research"` or `"qa"` — selects the graph factory |
| `data_artifacts` | Validated user-provided dataset metadata, schema, and hashes |
| `data_validation` | Dataset gate result |
| `experiment_contracts` | Immutable experiment contracts built by `core.evidence_gate` |
| `experiment_outcomes` | Per-experiment outcome labels (`supported`, `unsupported`, `inconclusive`) |
| `technical_failures` | Per-subsystem failure dossiers |
| `evidence_gate` | Gate decision (allowed/terminal/reason) |
| `human_approved` | Explicit human checkpoint before publishable release |
| `outcome_calibration` | Positive/negative/inconclusive outcome counts |
| `literature_context` | Raw literature retrieval shared between full-research and QA paths |
| `qa_answer` | QA-mode synthesis answer |
| `qa_citation_verification` | QA-mode citation verification result |
| `user_query` | User query for QA mode |
| `narrative_revision_count` | Tracks narrative revision attempts (v4, bound = 1) |
| `editor_repair_count` | Tracks editor repair attempts (v4, bound = 1) |
| `editor_repair_findings` | Findings from editor repair pass (v4) |

During a real CLI or web run, `ResearchPipeline` activates one `RunContext` for the stream. Agents receive that context when constructed, giving them access to run configuration, vector memory, and the research ledger without reaching into process globals.

## Recovery paths

- Discovery source outage becomes a terminal error rather than a misleading reset loop.
- Failed debate topics are removed and the next candidate is tried. Tournament mode debates candidates until one passes.
- Invalid plans route to `terminal_planning_failure` — no recovery, the run stops.
- Technical Engineer failures receive bounded code-only repair attempts against the same contract. Unresolved failure is terminal and produces a failure dossier; it does not route to writing or Meta.
- A successful run may produce supported, unsupported, or inconclusive findings. Unsupported hypotheses are written as negative results using the fixed data and protocol.
- After commitment, changing data, requirements, metrics, baselines, or hypothesis is contract drift and requires a new experiment identity.
- Results numeric grounding may trigger up to two redrafts.
- Independent verification findings can block the editing route; Meta evaluates recovery options.
- Editor release-referee failures route to one bounded repair pass before terminal failure (v4).
- Durable checkpoints allow CLI resume with `python main.py --resume RUN_ID`.
- `python replay_run.py PATH --clean-env` replays companion code in a fresh virtual environment; `historical_report.py` reconstructs the latest failed/completed run pair.
