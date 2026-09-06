# ScholarGraph — Research-Grade Autonomous Multi-Agent Research System

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/LangGraph-Orchestration-000000?style=for-the-badge" alt="LangGraph" />
  <img src="https://img.shields.io/badge/Gemini%20%7C%20OpenAI%20Compatible-LLM-4285F4?style=for-the-badge" alt="LLM" />
  <img src="https://img.shields.io/badge/FAISS-Vector%20Memory-FF6B35?style=for-the-badge" alt="FAISS" />
  <img src="https://img.shields.io/badge/FastAPI-Control%20Deck-009688?style=for-the-badge" alt="FastAPI" />
  <img src="https://img.shields.io/badge/pytest-Eval%20Harness-0A9EDC?style=for-the-badge" alt="pytest" />
</p>

ScholarGraph is a local, evidence-oriented multi-agent research system. Given a research domain, it discovers candidate gaps, debates hypotheses adversarially, creates falsifiable plans, validates explicitly supplied datasets, generates and executes experiments, independently replays and analyzes results, verifies citations and evidence, and assembles a LaTeX paper plus reproducibility artifacts.

The current implementation is intentionally incremental. The legacy `EngineerAgent`
path remains available for compatibility, while the newer execution, analysis, and
verification artifacts are now inserted into the live workflow as an independent
validation stage. This means the system is already more than a text generator, but
it is not yet a publication guarantee or a secure multi-user research platform.

For module-by-module architecture notes and the staged refactor plan, see [docs/README.md](docs/README.md).

**Design thesis:** quality comes from the harness, not from hoping a single LLM call is brilliant. Soft “vibe scores” are secondary; hard, checkable verification (resolved DOIs, recomputed stats, sandbox lockdown, PIVOT/REFINE) is primary — aligned with failure modes documented in systems like AI Scientist v1/v2, Agent Laboratory, AgentRxiv, and AutoResearchClaw.

---

## Table of contents

1. [Quick start](#quick-start)
2. [Architecture overview](#architecture-overview)
3. [System intent and quality contract](#system-intent-and-quality-contract)
4. [Pipeline graph (exact edges)](#pipeline-graph-exact-edges)
5. [Persistence and data model](#persistence-and-data-model)
6. [State schema](#state-schema)
7. [Core modules (technical spec)](#core-modules-technical-spec)
8. [Agents (technical spec)](#agents-technical-spec)
9. [Capability and tool-access model](#capability-and-tool-access-model)
10. [Verification & quality gates](#verification--quality-gates)
11. [LLM provider layer](#llm-provider-layer)
12. [Web UI — Operations Console](#web-ui--operations-console)
13. [Configuration & environment](#configuration--environment)
14. [Project layout](#project-layout)
15. [Outputs & artifacts](#outputs--artifacts)
16. [Testing / eval harness](#testing--eval-harness)
17. [CLI & entry points](#cli--entry-points)
18. [Operations, failure handling, and recovery](#operations-failure-handling-and-recovery)
19. [Design decisions & limitations](#design-decisions--limitations)

---

## Quick start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure
copy env_example.txt .env   # Windows
# cp env_example.txt .env   # macOS/Linux
# Edit .env — see Configuration section

# 3a. Web UI (recommended)
python run_ui.py
# → http://127.0.0.1:8765

# 3b. CLI
python main.py

# 4. Tests (no API keys required for most)
python -m pytest tests/test_eval_harness.py -q
```

> After changing backend Python files, restart `python run_ui.py`; a running
> Uvicorn process does not reload this application automatically.

---

## System intent and quality contract

ScholarGraph is a **research-workflow system**, not a paper-text generator. Its
output is useful only when a reader can trace each material conclusion back to
an identified source, a recorded experiment, or a declared limitation. The
system therefore treats a paper as the final view over a research record:

```
question → candidate gap → adversarial hypothesis → registered plan
         → executable experiment → raw measurements → verified claims
         → peer-style review → paper + reproducibility artifacts
```

### What the system is designed to establish

1. **Grounded background claims.** DOI and arXiv identifiers are resolved before
   a section can receive a high supervisor score.
2. **Falsifiability.** A contribution needs a prediction, named baselines, and
   a statistical test—not merely language such as “improves performance.”
3. **Measured conclusions.** Results prose is written only after engineering;
   quantitative claims are checked against recorded engineer output.
4. **Reproducibility.** Multi-seed raw output, generated code, experiment
   metadata, and a companion repository are retained as artifacts.
5. **Transparent uncertainty.** Unresolved debate objections flow into the
   paper’s Limitations section rather than being silently discarded.

### What it cannot establish automatically

Passing a gate does **not** prove publication-quality novelty, generalizability,
or real-world usefulness. LLM-proposed studies can still use weak synthetic
proxies, miss literature, or make an invalid causal interpretation. Human domain
review, appropriate data governance, and independent reproduction remain
necessary before submission or deployment. The system is intentionally more
conservative about claims than a plain writing assistant, but it is not a
substitute for a field expert or institutional review.

### Research methodology translated into the pipeline

| Research practice | ScholarGraph implementation | Evidence retained |
|---|---|---|
| Literature reconnaissance | OpenAlex + arXiv search, citation-graph signal, novelty comparison | source events, candidate metadata |
| Adversarial hypothesis review | Proposer / Challenger / Moderator with severity-tagged objections | rounds, objections, ensemble scores |
| Pre-analysis planning | predictions, baseline list, metrics, statistical test, variants | plan and revision requests |
| Pilot work and selection | cheap probes before a promoted multi-seed run | branch scores and winner |
| Controlled evaluation | sandboxed code, deterministic seeds, aggregate mean/std | code, raw results, aggregate metrics |
| Peer review | deterministic checks first; LLM review cannot rescue hard failure | hard-check bundle, feedback, scores |
| Artifact evaluation | companion repo plus reproducibility dossier | paths, code, raw results, dossier |

The reproducibility dossier follows the spirit of established venue checklists:
explicit claims and limitations, complete experimental details, statistical
reporting, and available code/data/protocols. See the [NeurIPS Paper
Checklist](https://neurips.cc/public/guides/PaperChecklist), [Nature reporting
standards](https://www.nature.com/ncomms/editorial-policies/reporting-standards),
and [ACM artifact evaluation guidance](https://sigsim.acm.org/conf/pads/2024/blog/artifact-evaluation/).

---

## Architecture overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         LangGraph Orchestrator                          │
│                              (main.py)                                  │
│ ResearchState + durable checkpoint + typed artifacts + conditional edges│
└───────────┬─────────────────────────────────────────────────────────────┘
            │
   ┌────────▼────────┐   ┌──────────────┐   ┌──────────────┐
   │ Topic Hunter    │──▶│ Debate       │──▶│ Planner      │
   │ source broker   │   │ proposer /   │   │ falsifiable  │
   │ novelty / feas  │   │ challenger   │   │ baselines    │
   └─────────────────┘   └──────────────┘   └──────┬───────┘
                                                   │
                              ┌────────────────────▼──────────────────┐
                              │ Data validation                       │
                              │ DataAgent: schema, target, hash       │
                              └────────────────────┬──────────────────┘
                                                   │
   ┌─────────────────┐   ┌────────────────────────▼──────────────────┐
   │ Writer          │◀──│ Engineer                                   │
   │ narrative       │   │ implementation + legacy experiment path   │
   └────────┬────────┘   └────────────────────────┬──────────────────┘
            │                                     │
            │              ┌──────────────────────▼──────────────────┐
            │              │ Independent validation                  │
            │              │ ExecutionAgent → AnalysisAgent           │
            │              │                    → VerificationAgent    │
            │              └──────────────────────┬──────────────────┘
            │                                     │
   ┌────────▼────────┐   ┌────────────────────────▼──────────────────┐
   │ Results Writer  │──▶│ Supervisor: hard checks + release blockers │
   └─────────────────┘   └────────────────────────┬──────────────────┘
                                                   │
                              ┌────────────────────▼──────────────────┐
                              │ Meta → Editor                         │
                              │ retry/reset or paper + artifacts      │
                              └───────────────────────────────────────┘
```

**Principles encoded in code:**

| Principle | Implementation |
|---|---|
| Hard checks before soft scores | `core/verification.py` + `SupervisorAgent` order |
| Plan must be revisable | `EngineerAgent.request_plan_revision` → graph edge back to `planning` |
| Explore, don’t only execute | `run_branching_search` cheap probes → full multi-seed on winner |
| Don’t trust host Python | AST + restricted builtins sandbox (`core/sandbox.py`) |
| Don’t lose context across agents | Append-only scratchpad (`run_scratchpad.jsonl`) |
| Improve across runs | `CrossRunMemory` JSONL lessons fed into Topic Hunter / Planner |
| Cost-aware generation | `tier=cheap\|strong\|judge` model routing in `call_llm` |
| Separate scientific responsibilities | Data, execution, analysis, and verification agents use distinct manifests |
| Broker external knowledge | OpenAlex retrieval uses allowlists, retries, validation, hashes, and cache replay |
| Block unsupported release | Independent verification findings are checked before editing |

---

## Pipeline graph (exact edges)

Defined in `main.py` → `create_research_graph()`:

| From | Condition | To |
|---|---|---|
| `topic_discovery` | `should_reset` → reset / continue / end | `reset` \| `hypothesis_debate` \| `END` |
| `hypothesis_debate` | same | `reset` \| `planning` \| `END` |
| `planning` | always | `data_validation` |
| `data_validation` | no explicit dataset | `writing_narrative` |
| `data_validation` | valid explicit dataset | `writing_narrative` |
| `data_validation` | invalid explicit dataset | `END` with terminal validation error |
| `writing_narrative` | safe non-results sections only | `engineering` |
| `engineering` | `current_phase == "planning"` | `planning` (revision bounce) |
| `engineering` | else | `independent_validation` |
| `independent_validation` | no executable code artifact | `writing_results` with compatibility note |
| `independent_validation` | code artifacts available | `writing_results` after replay, analysis, verification |
| `writing_results` | ungrounded quantitative claim, ≤2 retries | `writing_results` |
| `writing_results` | grounded results/discussion/abstract | `supervision` |
| `supervision` | avg ≥ threshold and no blocking finding | `editing` |
| `supervision` | blocking independent finding | `meta_evaluation` |
| `supervision` | else | `meta_evaluation` |
| `meta_evaluation` | `should_continue` | `writing_narrative` \| `END` |
| `editing` | always | `END` |
| `reset` | always | `topic_discovery` |

```mermaid
graph TD
    TD[topic_discovery] -->|fail / empty| R[reset]
    TD -->|topics found| HD[hypothesis_debate]
    HD -->|all topics fail| R
    HD -->|debate PASS| PL[planning]
    R --> TD
    PL --> DV[data_validation]
    DV --> WN[writing_narrative]
    WN --> EN[engineering]
    EN -->|plan_revision_requests| PL
    EN -->|experiments complete| IV[independent_validation]
    IV --> WR[writing_results]
    WR -->|untraced number; retry| WR
    WR -->|grounded results| SU[supervision]
    SU -->|avg >= threshold + no blockers| ED[editing]
    SU -->|verification blocker| ME[meta_evaluation]
    SU -->|below threshold| ME[meta_evaluation]
    ME -->|continue| WR
    ME -->|stop| END1[END]
    ED --> END2[END]
```

**Checkpointer:** LangGraph `SqliteSaver` at `memory/checkpoints.sqlite`; resume a run with `python main.py --resume <run_id>`.  

**Research ledger:** `memory/research_ledger.sqlite` is the durable source of truth for run lifecycle, event trace, agent scratchpad, evidence claims, and experiment artifacts. JSONL files remain human-readable exports for compatibility.
**Recursion limit:** `1000` on stream config.

### Two-pass writing rule

`write_narrative_sections` runs before experiments and is restricted to the
Introduction, Related Work, and Methods (plus an explicitly provisional
abstract). It receives an empty result set so it cannot report invented
measurements. `write_results_sections` runs after Engineering and creates or
refreshes Results, Discussion, and Abstract from `engineer_outputs`. Numeric
claims are compared against recursively extracted recorded values; unmatched
claims bounce the node back for a redraft before Supervisor review.

---

## Persistence and data model

Three storage layers intentionally have different responsibilities:

| Store | Default location | Role | Recovery / retention |
|---|---|---|---|
| LangGraph checkpoint DB | `memory/checkpoints.sqlite` | Exact workflow state and pending graph work | `python main.py --resume <run_id>` |
| Research ledger DB | `memory/research_ledger.sqlite` | Queryable operational and scientific provenance | Persist across runs; WAL-enabled SQLite |
| FAISS / exports | `memory/vector_db/`, `output/*.jsonl` | Similarity memory and human-readable compatibility exports | Not the authoritative run ledger |

### `research_ledger.sqlite` schema

| Table | Primary contents | Why it exists |
|---|---|---|
| `research_runs` | run id, start/end timestamps, status, final phase, summary | Answers “what happened to this run?” without parsing logs |
| `run_events` | timestamp, run id, agent, event type, structured JSON payload | Drives the event stream and audit trail |
| `run_scratchpad` | agent, kind, structured raw content, metadata | Stores raw intermediate material for inspection |
| `evidence_claims` | paper section, claim text/type, verification status, evidence JSON | Implements claim-to-evidence provenance |
| `research_artifacts` | raw-result/code artifact type, location, metadata | Connects a conclusion to executable/reproducible material |

The ledger uses SQLite WAL mode and short per-operation connections. This suits a
single local FastAPI process and worker thread while avoiding a long-lived shared
connection across threads. It is **not** a multi-host coordination database;
use Postgres or another server database before scaling to multiple application
processes or shared users.

### Run lifecycle in the ledger

1. `start_run()` creates a `research_runs` row and emits `run_created`.
2. Every phase transition, agent action, stat update, and operator message is
   appended to `run_events`.
3. Agent scratchpad calls are inserted into `run_scratchpad`; JSONL mirrors are
   retained for easy local inspection.
4. Supervisor writes verified/failed citation and quantitative claim rows.
5. Engineer records implementation/legacy experiment artifacts; the independent
  validation stage records replay execution artifacts and statistical reports.
6. `RunTracker.complete()` finalizes the run status and summary.

### Useful local inspection queries

```sql
-- Latest runs and final status
SELECT run_id, started_at, ended_at, status, phase
FROM research_runs ORDER BY started_at DESC;

-- Claims that still need human attention
SELECT section_name, claim_text, claim_type, status
FROM evidence_claims
WHERE run_id = ? AND status <> 'verified';

-- Artifacts supporting a run
SELECT artifact_type, location, metadata_json
FROM research_artifacts WHERE run_id = ?;
```

---

## State schema

`ResearchState` (`TypedDict` in `main.py`):

| Field | Type | Purpose |
|---|---|---|
| `iteration` | `int` | Outer loop / reset counter |
| `current_phase` | `str` | Routing signal (`planning`, `editing`, `complete`, …) |
| `should_reset` / `should_continue` | `bool` | Gate flags |
| `error_count` | `int` | Consecutive soft-failure budget |
| `topics` | `List[Dict]` | Ranked candidate topics |
| `selected_topic` | `Optional[Dict]` | Topic under debate / plan |
| `debate_results` | `List[DebateResult]` | Multi-round debate artifacts |
| `hypothesis_passed` | `bool` | Debate gate |
| `plan` | `Optional[Dict]` | Sections, contributions, experiments, variants |
| `data_artifacts` | `Dict[str, DatasetArtifact]` | Validated user-provided dataset metadata, schema, and hashes |
| `data_validation` | `VerificationReport` | Dataset gate result; explicit invalid datasets stop the run |
| `execution_artifacts` | `Dict[str, ExecutionArtifact]` | Seeded replay outputs, raw paths, environment, and hashes |
| `analysis_reports` | `Dict[str, StatisticalReport]` | Independent metrics, confidence intervals, tests, and warnings |
| `verification_findings` | `List[Dict]` | Blocking or advisory findings from independent verification |
| `draft_sections` | `Dict[str, str]` | Section name → markdown/text |
| `current_section` | `Optional[str]` | Last written section |
| `engineer_outputs` | `Dict[str, Any]` | Per-experiment results + raw paths |
| `supervisor_scores` / `supervisor_feedback` | `Dict` | Per-section scores/text |
| `meta_feedback` | `List[str]` | Meta messages |
| `final_paper` / `latex_output` | paper + LaTeX string |
| `plan_revision_requests` | `List[Dict]` | Engineer → Planner payload |
| `run_id` | `Optional[str]` | Observability / UI correlation |
| `results_redraft_count` | `int` | Guarded retries after numeric-grounding failure |
| `results_verification` | `Dict[str, Any]` | Per-section numeric claim audit results |
| `reproducibility` | `Dict[str, Any]` | Dossier: predictions, baselines, stats, raw results, code |
| `terminal_error` | `Optional[str]` | User-actionable terminal infrastructure error; never a research conclusion |

---

## Core modules (technical spec)

### `core/config.py` — `Config` (pydantic-settings)

Runtime settings from env / `.env` / UI `keys.json` via `apply_runtime_keys()`.

| Setting | Default | Meaning |
|---|---|---|
| `llm_provider` | `gemini` | `gemini` \| `openai` \| `openai_compatible` |
| `gemini_model` | `gemini-2.5-flash` | Chat model id |
| `gemini_embedding_model` | `text-embedding-004` | Embeddings |
| `openai_base_url` | `https://api.openai.com/v1` | Compatible base URL |
| `openai_model` | `gpt-4o-mini` | Chat model for OpenAI path |
| `openai_embedding_model` | `text-embedding-3-small` | Padded/truncated to 768-d for FAISS |
| `llm_model_cheap` / `strong` / `judge` | empty | Optional tier overrides |
| `ensemble_judge_models` | empty | Comma-separated judge models |
| `supervisor_threshold` | `8.5` | Mean section score → editing |
| `debate_pass_threshold` | `7.5` | Debate PASS floor |
| `debate_min_rounds` / `max` | `2` / `4` | Adversarial rounds |
| `novelty_similarity_reject` | `0.88` | Cosine ≥ → reject topic |
| `experiment_seeds` | `3` | Multi-seed runs |
| `experiment_branch_count` | `3` | Max cheap variants probed |
| `sandbox_timeout_sec` | `120` | Soft timeout budget |
| Paths | `./memory/*`, `./output/*` | FAISS, logs, raw results, companion repo |

`validate_config()` requires a key for the selected provider + `OPENALEX_EMAIL`, and creates output dirs.

### `core/llm.py` — multi-provider client

| API | Behavior |
|---|---|
| `LLMClient` | Wraps `google.genai.Client` or `openai.OpenAI` |
| `call_llm(prompt, tier=..., model=..., temperature=...)` | Primary entry; agents should use this |
| `generate_embedding(text)` | Gemini embed or OpenAI embed → **768-d** |
| `setup_gemini()` / `call_gemini()` | **Legacy aliases** → same client (Writer etc. still work) |
| `reset_llm_client()` | After UI key changes |
| Rate limit | `LLM_REQUEST_INTERVAL` seconds between calls (default `1.0`) |

**Tiers:** `cheap` (lookups/novelty), `strong` (debate/plan), `judge` (ensemble / peer review).

### `core/sandbox.py` — restricted execution

Not a full container; blocks documented Agent Laboratory failure modes:

- **AST reject:** `subprocess`, `os`, `sys`, `socket`, `requests`, `ctypes`, …; calls to `exit`/`quit`/`eval`/`exec`; `os.system`-style attrs
- **Allowlisted imports:** `numpy`, `pandas`, `sklearn`, `scipy`, `matplotlib`, `sympy`, `json`, `random`, …
- **Runtime:** restricted `__builtins__` + restricted `__import__`
- **`execute_sandboxed(code, seed)`** — injects `random`/`numpy` seeds, captures stdout/stderr, parses last JSON line
- **`run_multi_seed(code, n_seeds)`** — aggregates numeric metrics as `{mean, std, values, n}`

The sandbox is an execution policy filter, not a security boundary. It runs in
the host process and uses a soft thread timeout. Do not treat it as sufficient
isolation for hostile code, untrusted users, arbitrary package installation, or
multi-user deployment. The production hardening path is OS/container isolation
with curated dependencies and immutable artifact mounts.

### `core/contracts.py` — typed research handoffs

The contract module contains `TypedDict` handoffs for `DatasetArtifact`,
`CodeArtifact`, `ExecutionRequest`, `ExecutionArtifact`, `AnalysisPlan`,
`StatisticalReport`, `Claim`, `EvidenceBundle`, `VerificationFinding`, and
`AgentCapabilityManifest`. Artifact provenance can include run identity,
producer, content hash, parent artifacts, environment, status, and limitations.
These types are additive to the legacy `ExperimentOutput` contract so older
callers can migrate without a flag day.

### `core/capabilities.py` and `core/tool_broker.py` — scoped access

`DEFAULT_MANIFESTS` declares what each role may use. For example, Engineer may
generate code and read/write code artifacts, but cannot execute code, download
datasets, or perform statistical analysis. Analysis may read execution artifacts
and write statistical reports, but cannot generate code or download data.
Verification may replay and inspect artifacts, but cannot generate code or
change the experiment. `CapabilityBroker` authorizes registered tool calls,
fails closed for unknown capabilities, and records an audit entry before
dispatch.

### `core/sources.py` — reliable scholarly retrieval

`SourceClient` is the first brokered source adapter. It currently supports the
allowlisted OpenAlex, Crossref, arXiv, and Semantic Scholar base domains, with
OpenAlex integrated into Topic Hunter. It provides:

- HTTPS and source-base allowlisting
- bounded retries and request timeouts
- response-size limits
- optional payload validators
- deterministic cache keys
- disk cache replay when a source is offline
- response/content hashes and retrieval timestamps
- explicit `verified`, `cached`, and `unavailable` statuses

The returned source artifact preserves raw normalized content and warnings. A
source outage is infrastructure information; it must not be silently converted
into a scientific rejection.

### `core/verification.py` — hard checks

| Function | Spec |
|---|---|
| `extract_citation_ids` | DOI regex + arXiv id + author-year |
| `resolve_doi` | CrossRef `GET /works/{doi}` |
| `resolve_arxiv` | arXiv Atom API |
| `verify_citations` | Unresolved DOI/arXiv → `passed=False`; score = 10 × resolved ratio |
| `verify_statistics` | Re-derive mean/std (optional Welch p) from Engineer raw JSON; rtol default `0.05` |
| `hard_verify_section` | Citations + stats bundle for Supervisor |
| `reproducibility_dossier` | Deterministic artifact checklist: prediction, baselines, stats, raw output, code, limitations |

### `core/run_log.py` — observability

| Component | File / API |
|---|---|
| `RunTracker` | Phase, stats counters, messages; `start_run()` |
| Events | `output/run_events.jsonl` — `emit_event` / `read_events` |
| Scratchpad | `output/run_scratchpad.jsonl` — raw agent content (not only summaries) |
| `CrossRunMemory` | `memory/cross_run.jsonl` — rejections, pivots, plan revisions, run summaries; `lessons_for_prompt()` |
| Stats keys | `rejected_topics`, `debate_rounds`, `sections_bounced`, `pivots`, `refines`, `plan_revisions`, `hard_check_fails`, … |

`core/research_db.py` is the durable counterpart to this module. Callers should
record structured events or artifacts through it rather than introducing new
ad-hoc JSON files. The dashboard reads current-run events from SQLite first.

### `core/memory.py` — FAISS

- Index: `faiss.IndexFlatL2(768)`
- Persist: `memory/vector_db/index.faiss` + `metadata.pkl`
- Also: debate log JSON, feedback log JSON
- API: `add_embedding`, `search_similar`, `add_debate_entry`, `add_feedback_entry`, score helpers

### `core/utils.py`

Re-exports LLM helpers; JSON I/O; `parse_json_from_llm`; SymPy `validate_math_expression` / `verify_math_derivation`; `log_agent_action` → event stream.

---

## Agents (technical spec)

### Topic Hunter — `agents/topic_hunter.py`

| Capability | Spec |
|---|---|
| Sources | OpenAlex works search; arXiv; Semantic Scholar Graph API (`/paper`, `/citations`) |
| Source compatibility | OpenAlex `abstract_inverted_index` is reconstructed into text; arXiv uses `arxiv.Client.results(search)` (arxiv.py 4+) |
| Gap signal | High `citationCount` (in-degree) + low recent citing papers → high `gap_score` |
| Novelty | Embed topic vs last abstracts; reject if cosine ≥ `NOVELTY_SIMILARITY_REJECT` |
| Feasibility | Heuristic blocklist (GPU fine-tune, wet lab, proprietary data, low feasibility score) |
| Parallelism | `ThreadPoolExecutor` with 3 seed angles; lightweight LLM judge ranks survivors |
| Logging | Every rejection → `CrossRunMemory` + tracker bump |

OpenAlex retrieval is routed through `core.sources.SourceClient`, which caches
validated responses under `output/source_cache/` and records hashes/statuses.
The current migration covers OpenAlex first; arXiv and Semantic Scholar still
have legacy direct adapter paths and are candidates for the next source-client
migration.

If OpenAlex and arXiv are both unreachable or incompatible, Topic Hunter raises
`ResearchSourceUnavailable`. The orchestrator records a terminal infrastructure
error and stops the run once. It does **not** reset the pipeline or label the
condition “no novel topics,” because source unavailability is not evidence that
the research question lacks value.

**Entry:** `discover_topics(domain) → List[topic dict]`

### Hypothesis Debate — `agents/hypothesis_debate.py`

| Role | Spec |
|---|---|
| Proposer | Initial argument; then **point-by-point** responses to objections |
| Challenger | Must cover checklist: soundness, significance, reproducibility, ethics, novelty, feasibility → JSON objections with severity |
| Moderator | Ensemble over `ENSEMBLE_JUDGE_MODELS` or judge/strong/cheap; disagreement (`max−min > 1.5`) → longer debate / harder PASS |
| Rounds | Min `DEBATE_MIN_ROUNDS`, max `DEBATE_MAX_ROUNDS` |
| Elo | `memory/elo_ratings.json` — hypothesis *kind* buckets vs fixed bar 1500 |
| Output | `DebateResult` dataclass: rounds, unresolved_objections, ensemble_scores, elo_delta |

**PASS:** agreed ensemble, mean ≥ `DEBATE_PASS_THRESHOLD`, no unresolved severity ≥ 4.

### Planner — `agents/planner.py`

| Requirement | Enforcement |
|---|---|
| Falsifiable prediction + statistical test per contribution | `_flag_unfalsifiable` → `_repair_plan` |
| Baselines list per experiment | `_flag_missing_baselines` |
| Variants (2–3) for branch search | `_attach_variants` |
| Prior lessons | `CrossRunMemory.lessons_for_prompt()` in plan prompt |
| Bidirectional revision | `revise_plan(plan, revision_request, topic)` |

Experiment objects include: `baselines`, `falsifiable_prediction`, `statistical_test`, `variants`, `claimed_components`, metrics, data requirements.

### Writer — `agents/writer.py`

Section drafting uses the compatibility `call_gemini` alias, which routes through
the configured LLM layer. The Writer is deliberately split by orchestration:

- pre-engineering: Introduction, Related Work, Methods, and a provisional
  abstract; no experiment output is supplied;
- post-engineering: Results, Discussion, and final Abstract; all quantitative
  language must be traceable to `engineer_outputs`.

Every section is stored as an embedding for retrieval, but publication claims are
validated from the source/experiment ledger rather than from similarity memory.

### Engineer — `agents/engineer.py`

| Capability | Spec |
|---|---|
| Code gen | Constrained prompt: allowlisted libs only; JSON metrics on last stdout line |
| Execution | `validate_code` → `run_multi_seed` |
| PIVOT/REFINE | Bugs → REFINE; conceptual/API fail → PIVOT to next variant; max 4 attempts |
| Branch search | Cheap single-seed probes → promote best → full multi-seed |
| Ablation | Auto-generate ablated code for claimed components |
| Code–claim check | Heuristic: claimed algo (XGBoost, RF, …) vs imports/classes |
| Plan revision | `request_plan_revision(reason, experiment, detail)` |
| Persistence | `output/raw_results/*.json` + experiment results JSON |

**Give-up guard:** bare “not feasible” without error/traceback/decision_log marked suspicious (`failure_artifact`).

Engineer remains a compatibility-heavy role: it still contains the legacy
generation, execution, branch, ablation, and recovery path. The live graph now
adds an independent validation stage after Engineering, so Engineer output is
not the only execution or analysis evidence. The long-term target is to narrow
Engineer to implementation and move all execution/statistics responsibilities
behind the newer worker contracts.

### Data — `agents/data.py`

`DataAgent` is a dataset stewardship worker. It accepts explicit CSV or JSON
paths and returns a `DatasetArtifact` containing the absolute location, SHA-256
hash, row count, columns, dtypes, target checks, and validation status. It flags
missing targets, target/feature overlap, empty data, and all-missing data. It
does not generate experiment code, execute experiments, or certify statistical
claims.

If a plan contains `dataset_path` or `dataset_file`, the graph runs this agent
in `data_validation` before writing and engineering. Invalid requested data
terminates the run with a visible validation error. Plans without an explicit
dataset retain the existing declared-synthetic-data compatibility path.

### Execution — `agents/execution.py`

`ExecutionAgent` wraps the existing `core.sandbox.run_multi_seed` implementation
instead of duplicating an executor. It validates code before execution, runs the
requested seed set, captures Python/platform/sandbox metadata, writes raw result
JSON, and produces an `ExecutionArtifact` with a content hash and status. It
does not generate code or interpret scientific meaning.

### Analysis — `agents/analysis.py`

`AnalysisAgent` consumes execution artifacts only. It uses SciPy and NumPy to
compute metric summaries, 95% confidence intervals, Welch comparisons, and
Cohen’s d where two groups are available. It records warnings for insufficient
seeds, missing raw values, missing primary metrics, failed executions, and
missing comparison artifacts. It cannot mutate code or datasets.

### Verification — `agents/verification.py`

`VerificationAgent` independently checks execution artifacts and reports:

- raw result path exists
- raw result content hash matches the execution artifact
- execution completed successfully
- an independent analysis report exists
- reported statistics agree with raw results via `core.verification`

Findings are structured and can be blocking. The supervisor refuses the editing
route when any blocking finding is present.

### Supervisor — `agents/supervisor.py`

**Order (critical):**

1. Hard citation + stats (`hard_verify_section`)
2. MathChecker (SymPy parse + optional equality checks)
3. CodeChecker (`compile` / balanced delimiters)
4. If hard fail → soft review still runs but **overall score capped at 4.0**
5. If hard pass → ReviewerBot (LLM) + soft hallucination pass

Weighted blend when hard passed: hard 0.35 + math 0.15 + code 0.15 + reviewer 0.20 + hall 0.15.

### Meta Agent — `agents/meta_agent.py`

- Performance metrics + trends from FAISS feedback
- `should_reset` / `should_continue` policy
- `get_run_dashboard()` for UI/ops
- `validate_failure_claim` — suspicious early give-up detection
- Cheap-tier LLM for strategic JSON feedback

### Editor — `agents/editor.py`

| Method | Spec |
|---|---|
| `create_final_paper(...)` | Sections + Limitations from unresolved debate objections + DOI/arXiv bib resolve + companion repo |
| `generate_latex(final_paper)` | Article class LaTeX; writes `.tex` + `references.bib` |
| `assemble_paper(...)` | Facade over the two above |
| Companion repo | `output/companion_repo/` — `README.md`, `requirements.txt`, `run_experiments.py`, `experiments/*.py` |

---

## Verification & quality gates

| Gate | Location | Threshold / rule |
|---|---|---|
| Debate PASS | Moderator ensemble | ≥ `DEBATE_PASS_THRESHOLD` (7.5) + agreement + no severe unresolved |
| Topic novelty | Topic Hunter | Similarity ≥ 0.88 → reject |
| Topic feasibility | Topic Hunter | Heuristic + feasibility score |
| Plan falsifiability | Planner | Missing prediction/test → repair or flag |
| Plan baselines | Planner | Empty baselines → flag/repair |
| Sandbox | Engineer | AST/runtime reject dangerous code |
| Dataset validation | DataAgent | Explicit user dataset must exist, parse, contain rows, and include a valid target when declared |
| Multi-seed | Engineer + ExecutionAgent | Requested seeds are replayed; report mean±std and retain raw values |
| Independent analysis | AnalysisAgent | SciPy summaries, confidence intervals, Welch comparison, effect size, warnings |
| Artifact integrity | VerificationAgent | Raw path exists and content hash matches |
| Independent statistics | VerificationAgent | Analysis report agrees with raw execution results |
| Citation resolve | Supervisor hard | Unresolved DOI/arXiv → hard fail |
| Stats match raw | Supervisor hard | Reported vs recomputed within rtol |
| Section quality | Supervisor overall | Mean ≥ `SUPERVISOR_THRESHOLD` (8.5) → editing |
| Iteration budget | Meta / reset | `MAX_ITERATIONS` (10) |
| Suspicious give-up | Meta | Failure without concrete artifact |

### Release-readiness interpretation

The console’s “Ready” state means the implemented release checks passed; it does
not mean the research is novel, causal, generalizable, or publication-ready.
The system still needs human domain review, source relevance review, appropriate
data governance, and independent reproduction before external submission.

---

## Capability and tool-access model

ScholarGraph distinguishes **knowledge retrieval** from **experiment
execution**. Discovery agents may use approved scholarly-source adapters;
generated experiment code runs through the restricted sandbox and does not have
network, shell, arbitrary filesystem, or package-install access.

### Current role manifests

| Role | Allowed | Explicitly forbidden |
|---|---|---|
| Engineer | Code generation, artifact read/write | Dataset download, code execution, statistical analysis, claim verification |
| DataAgent | Literature/data retrieval, dataset catalog/download, artifact write | Code generation/execution, statistical analysis, claim verification |
| ExecutionAgent | Artifact read, code execution, artifact write | Code generation, dataset download, statistical analysis, claim verification |
| AnalysisAgent | Artifact read, statistical analysis, report write | Code generation/execution, dataset download, claim verification |
| VerificationAgent | Artifact read, replay, claim verification, finding write | Code generation, dataset download, statistical analysis |

These are policy manifests and broker checks, not yet a complete operating
system security boundary. The current `CapabilityBroker` dispatches registered
tools and audits authorization; direct legacy calls remain in older agents and
are being migrated incrementally.

### Information reliability rules

Source access should be treated as an evidence pipeline:

1. use an allowlisted adapter;
2. apply timeout, retry, and response-size limits;
3. validate the response shape;
4. cache the raw response for replay;
5. retain retrieval metadata and hashes;
6. expose `verified`, `cached`, or `unavailable` state;
7. never turn a failed lookup into an invented fact.

The current implementation applies this path to OpenAlex first. A future source
adapter should preserve the same contract before being used by an agent.

## LLM provider layer

Set `LLM_PROVIDER` and the matching keys:

### Gemini

```env
LLM_PROVIDER=gemini
GOOGLE_API_KEY=...
GEMINI_MODEL=gemini-2.5-flash
GEMINI_EMBEDDING_MODEL=text-embedding-004
```

### OpenAI

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

### OpenAI-compatible (OpenRouter, Groq, Together, vLLM, Ollama, …)

```env
LLM_PROVIDER=openai_compatible
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://openrouter.ai/api/v1   # example
OPENAI_MODEL=meta-llama/llama-3.1-70b-instruct
```

Optional routing:

```env
LLM_MODEL_CHEAP=...
LLM_MODEL_STRONG=...
LLM_MODEL_JUDGE=...
ENSEMBLE_JUDGE_MODELS=model-a,model-b,model-c
```

---

## Web UI — Operations Console

| Item | Spec |
|---|---|
| Entry | `python run_ui.py` → `web/app.py` (FastAPI + Uvicorn) |
| Default bind | `WEB_HOST=127.0.0.1` `WEB_PORT=8765` |
| Frontend | `web/static/admin.html` — operational local admin console; `index.html` remains as a legacy fallback |
| Key store | `POST /api/keys` → `memory/keys.json` + `apply_runtime_keys` + `reset_llm_client` |

**Main API routes:**

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Control deck UI |
| GET | `/api/health` | Provider + key presence |
| GET/POST | `/api/keys` | Load (masked) / save keys |
| POST | `/api/keys/test` | Smoke LLM call |
| GET | `/api/dashboard` | Live stats, phase, events, workspace artifacts, findings, capabilities |
| GET | `/api/artifacts` | Durable artifact records for the selected run |
| GET | `/api/capabilities` | Agent capability manifests and allowed/forbidden operations |
| GET | `/api/events` | Event stream |
| GET | `/api/scratchpad` | Raw agent scratchpad |
| POST | `/api/run` | Start pipeline in background thread |
| GET | `/api/run/status` | Running / error / tracker |

### UI behavior

- **Provider-aware configuration:** choosing Gemini shows only Gemini credentials;
  choosing OpenAI shows OpenAI credentials; choosing OpenAI-compatible also
  requires a base URL. Shared research settings remain visible.
- **Required vs optional labels:** credentials and OpenAlex email are marked
  required; routing overrides, extra scholarly APIs, and tuning thresholds are
  optional.
- **Operations-first home:** release readiness, current phase, experiment count,
  blocking findings, gate checks, and latest activity are the first view.
- **Evidence workspace:** dataset and execution artifact lineage, hashes,
  locations, verification findings, and durable claim records.
- **Jobs workspace:** replay seeds, aggregate metrics, raw result paths, and
  execution status.
- **Agent registry:** visible allowed and forbidden capabilities for each role.
- **Manuscript workspace:** downstream paper rendering remains available, but
  it is no longer the product’s only organizing surface.
- **Run history:** durable SQLite run records.
- **Settings:** provider, domain, OpenAlex email, and experiment seed controls.
- **Error presentation:** a stopped run displays a visible, actionable error
  banner. A source outage is not presented as a scientific rejection.
- **Live refresh:** the browser polls status/dashboard/scratchpad every 3 s.
  It does not expose pause/stop controls as working features because the graph
  has not yet implemented cooperative user interrupts.

---

## Configuration & environment

Template: [`env_example.txt`](env_example.txt) → copy to `.env`.

**Required (depending on provider):**

| Variable | Required when |
|---|---|
| `GOOGLE_API_KEY` | `LLM_PROVIDER=gemini` |
| `OPENAI_API_KEY` | `openai` or `openai_compatible` |
| `OPENAI_BASE_URL` | `openai_compatible` (the UI requires this explicitly) |
| `OPENALEX_EMAIL` | Always (OpenAlex polite pool) |

### Complete configuration reference

| Group | Variables | Intent |
|---|---|---|
| Provider | `LLM_PROVIDER`, `GOOGLE_API_KEY`, `GEMINI_MODEL`, `GEMINI_EMBEDDING_MODEL`, `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`, `OPENAI_EMBEDDING_MODEL` | Select the LLM backend and embeddings |
| Routing | `LLM_MODEL_CHEAP`, `LLM_MODEL_STRONG`, `LLM_MODEL_JUDGE`, `ENSEMBLE_JUDGE_MODELS`, `LLM_REQUEST_INTERVAL` | Cost / quality routing and judge diversity |
| Scholarly sources | `OPENALEX_EMAIL`, `SEMANTIC_SCHOLAR_API_KEY`, `SCITE_API_KEY` | Source identity and optional enrichment |
| Research gates | `SUPERVISOR_THRESHOLD`, `DEBATE_PASS_THRESHOLD`, `DEBATE_MIN_ROUNDS`, `DEBATE_MAX_ROUNDS`, `NOVELTY_SIMILARITY_REJECT` | Acceptance rigor and debate behavior |
| Experiment budget | `EXPERIMENT_SEEDS`, `EXPERIMENT_BRANCH_COUNT`, `MAX_ITERATIONS` | Replication, branch exploration, revision budget |
| Durable stores | `CHECKPOINT_PATH`, `RESEARCH_DB_PATH`, `VECTOR_DB_PATH`, `CROSS_RUN_MEMORY_PATH`, `ELO_RATINGS_PATH`, `KEYS_STORE_PATH` | Recovery, ledger, similarity memory, historical learning |
| Outputs | `OUTPUT_DIR`, `DRAFT_VERSIONS_DIR`, `RAW_RESULTS_DIR`, `COMPANION_REPO_DIR`, `RUN_LOG_PATH`, `RUN_EVENTS_PATH` | Generated paper and compatibility exports |
| Runtime / UI | `SANDBOX_TIMEOUT_SEC`, `SANDBOX_MAX_OUTPUT_BYTES`, `WEB_HOST`, `WEB_PORT`, `LOG_LEVEL`, `DEBUG_MODE` | Local operational behavior |

The UI writes non-empty values to `memory/keys.json` and applies them to the
running process. Secrets are masked on read but are stored locally in that file;
do not commit it or expose the web service beyond a trusted machine without
adding secret management and authentication.

Keys can also be entered only in the UI (stored under `KEYS_STORE_PATH`, default `./memory/keys.json`).

---

## Project layout

```
Scholargraph/
├── main.py                 # LangGraph orchestration, ResearchState, CLI
├── run_ui.py               # Launch Control Deck
├── demo.py                 # Mock demo (no live APIs)
├── run_with_real_api.py    # Thin wrapper → main
├── setup.py                # Bootstrap helper
├── requirements.txt
├── env_example.txt
├── agents/
│   ├── topic_hunter.py
│   ├── hypothesis_debate.py
│   ├── planner.py
│   ├── writer.py
│   ├── engineer.py
│   ├── data.py              # Dataset validation and provenance
│   ├── execution.py         # Independent seeded replay worker
│   ├── analysis.py          # Independent SciPy analysis worker
│   ├── verification.py      # Artifact/hash/statistical verifier
│   ├── supervisor.py
│   ├── meta_agent.py
│   └── editor.py
├── core/
│   ├── config.py           # Settings + runtime key apply
│   ├── llm.py              # Gemini / OpenAI-compatible client
│   ├── utils.py            # Shared helpers + legacy aliases
│   ├── memory.py           # FAISS + debate/feedback logs
│   ├── sandbox.py          # Restricted code execution
│   ├── capabilities.py     # Role manifests and authorization
│   ├── tool_broker.py      # Auditable registered-tool dispatch
│   ├── sources.py          # Allowlisted cached scholarly retrieval
│   ├── contracts.py        # Typed artifact/evidence handoffs
│   ├── context.py          # Per-run dependencies and manifests
│   ├── state.py            # ResearchState and validation artifacts
│   ├── workflow.py         # LangGraph graph assembly
│   ├── workflow_nodes.py   # Phase nodes and validation gate
│   ├── verification.py     # Citation + statistical hard checks
│   ├── run_log.py          # Events, scratchpad, cross-run memory
│   └── research_db.py      # SQLite runs, provenance, evidence, artifacts
├── web/
│   ├── app.py              # FastAPI Control Deck API
│   └── static/admin.html   # Current operations console
│       static/index.html   # Legacy UI fallback
├── tests/
│   ├── test_eval_harness.py
│   ├── test_capabilities.py
│   ├── test_sources.py
│   ├── test_data_agent.py
│   ├── test_execution_agent.py
│   ├── test_analysis_agent.py
│   ├── test_verification_agent.py
│   ├── test_refactor_boundaries.py
│   └── smoke_offline.py
├── templates/              # LaTeX templates (legacy/support)
├── memory/                 # FAISS, keys.json, cross_run.jsonl, elo
└── output/                 # Papers, raw_results, events, companion_repo
```

---

## Outputs & artifacts

| Path | Contents |
|---|---|
| `output/paper_*.tex` / `paper_output.tex` | Generated LaTeX |
| `output/references.bib` | DOI/arXiv-resolved bibliography |
| `output/plan_*.json` / `plan.yaml` | Research plan snapshot |
| `output/research_summary.json` | Run summary (topic, scores, experiments) |
| `output/raw_results/*.json` | Multi-seed raw + aggregate metrics |
| `output/source_cache/*.json` | Cached OpenAlex source artifacts for offline replay |
| `output/*_results.json` | Per-experiment Engineer dumps |
| `memory/research_ledger.sqlite` | Authoritative runs, events, scratchpad, claims, artifacts |
| `memory/checkpoints.sqlite` | Durable LangGraph execution checkpoints |
| `output/run_events.jsonl` | Compatibility event export (not source of truth) |
| `output/run_scratchpad.jsonl` | Compatibility scratchpad export (not source of truth) |
| `output/companion_repo/` | README, requirements, experiment scripts |
| `output/debate_log.json` | Debate transcripts |
| `output/feedback_log.json` | Supervisor feedback history |
| `memory/vector_db/` | FAISS index + metadata |
| `memory/cross_run.jsonl` | Cross-run lessons |
| `memory/elo_ratings.json` | Hypothesis-kind Elo |
| `memory/keys.json` | UI-stored API keys (local; do not commit) |

---

## Testing / eval harness

```bash
python -m pytest tests/test_eval_harness.py -q
```

Coverage includes (offline / mocked where needed):

- Sandbox blocks `subprocess`, `exit`, `os.system`; allows numpy JSON metrics
- Multi-seed aggregation (`mean` / `std`)
- Citation ID extraction + mocked resolve fail/pass
- Statistical match vs fabricated mismatch
- Planner unfalsifiable / missing-baseline flags
- Code–claim inconsistency heuristic
- Cross-run memory lessons text
- Config model resolution + runtime key apply
- DebateResult schema fields
- Current arXiv client API contract
- Graceful scholarly-source outage termination (no misleading reset loop)
- SQLite run/event/claim/scratchpad/artifact persistence
- Reproducibility dossier requirements
- Capability manifests fail closed for undeclared tools
- Capability broker audits authorized calls and denies before dispatch
- Source client allowlisting, validation, retries, cache replay, hashes, and outages
- DatasetAgent hashing, schema checks, target checks, and unsupported formats
- ExecutionAgent seeded replay, raw artifact creation, and forbidden-code rejection
- AnalysisAgent confidence intervals, Welch tests, effect sizes, and seed warnings
- VerificationAgent raw-path, hash, and statistical mismatch blockers
- Live workflow integration for dataset validation and independent validation

Run the complete current suite with:

```powershell
python -m pytest -q
```

The maintained suite currently covers the full offline boundary set. Live LLM
and scholarly-source scripts may still contact external services and consume
quota when run directly.

These tests are the measurement surface for “did this upgrade help?” — add a test before wiring new gates into the graph.

---

## CLI & entry points

| Command | Role |
|---|---|
| `python main.py` | Full LangGraph research run (CLI progress) |
| `python run_ui.py` | Control Deck on `:8765` |
| `python run_with_real_api.py` | Alias → `main.main()` |
| `python demo.py` | Mock agents / no live LLM |
| `python -m pytest tests/` | Eval harness |
| `python setup.py` | Env bootstrap / smoke (legacy helper) |

---

## Operations, failure handling, and recovery

### Expected failure classes

| Situation | System behavior | Operator action |
|---|---|---|
| Missing or invalid provider key | Validation fails before workflow work starts | Select the intended provider and enter its required key(s) |
| OpenAlex + arXiv unavailable | One terminal source error; no reset loop | Restore network/source access, verify `OPENALEX_EMAIL`, then start a new run |
| One scholarly source unavailable | Continue with the available source; record failure in trace | Inspect trace and retry if broad coverage matters |
| Explicit dataset missing/invalid | Data validation terminates the run before writing/engineering | Correct the path, format, target, or feature specification |
| Topic novelty/feasibility rejection | Candidate rejected and next candidate considered | Inspect reason; refine scope/domain if appropriate |
| Debate failure | Next ranked candidate is tried | Read objections; it is a research result, not an app crash |
| Unsafe/invalid experiment code | AST rejection then refine, pivot, or plan revision | Inspect failure artifact and revision request |
| Untraced result number | Results writer redrafts before supervision | Inspect `results_verification` and the Engineer artifact |
| Citation/statistics hard failure | Section score capped; LLM review cannot rescue it | Correct the identifier, source, result, or wording |
| Independent verification blocker | Editing/release route is blocked and Meta can evaluate recovery | Inspect raw artifacts, hashes, statistical report, and finding |
| Process crash | Latest workflow checkpoint remains durable | Resume: `python main.py --resume <run_id>` |

### Safe operating procedure

1. Enter provider-specific credentials and a real OpenAlex contact email. Test
   the LLM before starting a costly run.
2. Watch Topic Discovery. A source-outage notice means infrastructure needs
   repair; it does not mean the domain has no research gaps.
3. Review Debate and Plan before trusting Engineering. Check that predictions,
   baselines, and statistical tests fit the actual research question.
4. After Engineering, inspect Evidence Trace for raw artifact paths and claims
   that are not verified. A visually complete manuscript is not evidence alone.
5. Before external submission, independently rerun the companion repository,
   check citation relevance, seek domain-expert review, and complete the target
   venue’s checklist.

### Backups and retention

Back up `memory/research_ledger.sqlite`, `memory/checkpoints.sqlite`,
`memory/vector_db/`, and associated `output/` artifacts together. The ledger
links a run to its artifacts; the output directory contains the actual code,
raw results, and generated paper. Include `*.sqlite-wal` and `*.sqlite-shm`
when backing up an actively running application, or stop it first for a simple
consistent copy.

---

## Design decisions & limitations

**Decisions**

- Prefer **verification compute** over prettier generation (Supervisor hard path first).
- Prefer mature existing libraries over custom replacements: LangGraph for
  orchestration, Pydantic contracts/settings, pandas/NumPy/SciPy for analysis,
  SymPy for symbolic checks, FastAPI for the local API, and the existing
  sandbox/multi-seed runner for execution.
- Keep Proposer / Challenger / Moderator as distinct personas; collapse only pure transform steps.
- Separate data stewardship, implementation, execution, analysis, and
  verification responsibilities even when the legacy Engineer path remains for
  compatibility.
- Give agents typed, scoped capabilities rather than arbitrary internet,
  filesystem, shell, or package-install access.
- Cache and hash external source responses so discovery can be inspected and
  replayed without silently trusting a live network response.
- Engineer sandbox is **in-process lockdown**, not Docker/E2B — sufficient against exit/subprocess/host-install patterns; not a security boundary against determined escape.
- OpenAI embeddings are **padded/truncated to 768** for FAISS compatibility with Gemini-era indexes.
- Narrative writing is separated from result writing. Results, Discussion, and
  final Abstract are drafted only after Engineering and must pass numeric
  grounding before supervision.

**Limitations**

- Live runs cost API tokens and wall time; novelty/citation-graph calls need network.
- OpenAlex is currently migrated to the reliable `SourceClient`; arXiv,
  Semantic Scholar, and other source paths still need the same adapter migration.
- The capability broker is an enforceable application policy, but older direct
  provider calls remain during migration and the broker is not an OS sandbox.
- The new execution/analysis/verification chain is integrated after the legacy
  Engineer path; Engineer still owns legacy generation, execution, branching,
  ablation, and recovery code that will be narrowed in a later migration.
- The current admin console is local single-user and polling-based. It has no
  authentication, role management, CSRF protection, or multi-process job queue;
  keep it bound to localhost.
- PDF compilation requires a local `pdflatex` if you want PDF (LaTeX source is always written).
- Soft ReviewerBot remains an LLM judgment — intentionally last, never the only gate.
- SQLite is designed for this local single-process desktop deployment. Shared,
  concurrent multi-user operation requires a server database and auth layer.

---

## Dependencies (runtime)

From `requirements.txt` (high level):

- **Orchestration:** `langgraph`, `langchain`, `langgraph-checkpoint-sqlite`
- **LLM:** `google-genai`, `openai`
- **Memory:** `faiss-cpu`, `numpy`
- **Science:** `scipy`, `scikit-learn`, `pandas`, `matplotlib`, `seaborn`, `sympy`, `networkx`
- **Academic I/O:** `arxiv`, `requests`, `beautifulsoup4`
- **Docs / config:** `PyLaTeX`, `jinja2`, `PyYAML`, `pydantic`, `pydantic-settings`, `python-dotenv`
- **UI / tests:** `fastapi`, `uvicorn`, `pytest`

---

## Contact

- **GitHub:** [github.com/abubakaramin](https://github.com/abubakaramin)
- **Email:** [abubakarmain100@gmail.com](mailto:abubakarmain100@gmail.com)
