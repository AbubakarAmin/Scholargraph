# ScholarGraph Hardening — Diagnostic Brief & Agent Implementation Prompt
![Docs version](https://img.shields.io/badge/docs‑v2024.09‑blue)

This document is meant to be handed directly to your coding agent (Cursor or
equivalent). Section 1 is the diagnosis. Section 2 is the prioritized fix
list mapped to your actual modules. Section 3 is the literal prompt to paste
in. Nothing here requires guessing at your code — it's derived from reading
your module docs plus two real failed outputs from the system.

---

## 1. Diagnosis: what the two failed papers actually prove

### Paper A (Koopman/EDMD vs LSTM)

- **Engineer failure telemetry leaked into the manuscript as science.**
  The Discussion and Conclusion both contain sentences like *"the
  `tracemalloc` import was blocked"* and *"failed with `'str' object has no
  attribute 'get'`"* — verbatim exception/sandbox text, presented as a
  methodological limitation. This means `agents/engineer.py`'s failure
  output (or the failure dossier) is reachable by `agents/writer.py` as if
  it were narrative content, instead of being a strictly terminal,
  non-narrative signal.
- **The same "result" has three different numbers in three sections.**
  Introduction/Methods, the Experiments section, and the Results section
  each report a different crossover point and different MSE values for what
  is supposedly one experiment. This means each section is being drafted
  against its own local context rather than one canonical results object,
  and nothing checks *cross-section* numeric consistency — only, at best,
  per-section grounding.
- **The benchmark itself silently changed.** Methods commits to one set of
  real-world datasets; Results reports on a completely different set. This
  is exactly the "contract drift" your docs say should be terminal — so
  either the evidence gate isn't actually wired between Engineer output and
  Writer input, or the results-writing pass isn't reading from the
  committed experiment contract at all.

### Paper B (Optimal Transport / Wasserstein barycenter DA)

- **A fabricated citation passed verification.** `Korotin, Arjovsky &
  Guruswami (2020), "Wasserstein GANs with Gradient Penalty,"
  arXiv:2004.04436` — wrong authors, wrong title (the real WGAN-GP paper is
  Gulrajani et al.), attached to a real-looking arXiv ID. The evidence
  ledger tagged this `citation · verified`. That means `core.verification`'s
  citation check is confirming that an ID *resolves*, not that the
  resolved record's title/authors match what the LLM claimed. This is a
  hard gap, not a soft one.
- **"Verified" is being stamped on non-findings.** The ledger marks things
  like *"Accuracy: the proportion of correctly classified instances"* and
  *"we measure the accuracy delta"* as `quantitative · verified`. These are
  glossary definitions and forward-looking plan statements, not results.
  Whatever is doing claim extraction for the ledger isn't classifying claim
  *type* (literature statement / method definition / planned test / actual
  empirical result) before deciding what "verified" should even mean for it.
- **A dead run still produced a browsable manuscript.** The workspace shown
  had `Error #1bc8599c` at the top and stopped mid-Methods — no Results,
  Discussion, or Conclusion. Structurally this is *correct* per your
  workflow (results writing only happens after the evidence gate passes),
  but the UI/export layer presented an incomplete, failed run exactly like
  a normal in-progress or finished one, with no visible "this run
  terminated in failure" state. That's an `agents/editor.py` /
  `core.artifacts` gap: a failed run should be unmistakably marked as such
  everywhere it's surfaced, and should never be exportable as a
  "companion repository" deliverable.

### The common root cause underneath both

Neither paper failed because the model "isn't smart enough to write a
paper." Both failed because the **research questions were operationally
infeasible for what the sandbox can execute**, and nothing upstream stopped
that:

- Paper A's plan requires training GRU/LSTM baselines with grid search
  across hidden sizes up to 128 units, multi-seed runs up to N=20,000, on
  Lorenz-96 at d=32 — plus three external tabular datasets (Wikipedia
  traffic, Yahoo Finance, a 2.5M-row intrusion dataset) that need to be
  fetched from the internet.
- Paper B's plan needs UCI Adult Income + Bank Marketing downloads, a
  from-scratch Sinkhorn implementation validated against CORAL and MMD
  baselines, with statistical tests across three seeds.

Your `core.sandbox` is explicitly documented as a local AST/builtins
lockdown — not a place that plausibly has internet access, GPU time, or the
runtime budget for that. When a committed plan asks for something the
sandbox structurally cannot produce, something downstream fills the gap
with invented numbers, because nothing in the pipeline is allowed to say
"this plan is infeasible, revise it" *before* commitment. `Engineer` does
have `request_plan_revision()` — the question is whether `Planner` and
`TopicHunter` ever produce plans that would trigger it, or whether
infeasibility is only discovered after code executes (or fails to).

---

## 2. Fix list, mapped to your actual modules

Ordered by leverage — do 2.1–2.3 first; they're the ones that let fabricated
output through the gate at all. 2.4–2.6 are what stop the *bad* research
questions from being planned in the first place.

### 2.1 `core/verification.py` — make citation checks actually check citations
- `verify_citations` currently appears to validate that a DOI/arXiv ID
  *resolves*. Add a second, mandatory step: fetch the resolved record's
  title and author list (Crossref for DOI, arXiv API for arXiv IDs,
  optionally Semantic Scholar as a fallback), and fuzzy-match against the
  title/author string the writer produced in the bibliography entry.
  Mismatch above a similarity threshold = hard failure, not advisory.
- Add claim typing before anything is written to the ledger:
  `literature_reference` | `method_definition` | `planned_test` |
  `empirical_result`. Only `empirical_result` claims tied to a specific
  artifact ID from `core.research_db` should ever be eligible for a
  `verified` stamp. Everything else gets stored but tagged `unverifiable
  by construction` or similar — never `verified`.
- Add a cross-section numeric consistency pass: collect every numeric
  claim tied to the same experiment ID across *all* drafted sections
  (Abstract, Intro, Methods, Results, Discussion, Conclusion) and diff them
  against each other, not just against the raw artifact. Any two sections
  citing different numbers for the same metric/experiment is a hard
  failure that blocks `agents/editor.py` from assembling.

### 2.2 `agents/writer.py` — single source of truth, no independent re-derivation
- Every section-drafting call that touches a number must receive the exact
  same serialized results object (from the committed experiment contract /
  `core.research_db`), not a fresh LLM recollection of "what the numbers
  were." Pass it as structured data in the prompt, not prose summary, and
  instruct the writer to quote it exactly or not at all.
- Explicitly forbid the writer from receiving raw engineer exception
  traces, stack traces, or sandbox error strings as prompt input for
  narrative sections. If a "limitations" or "discussion of failures"
  section is warranted, it should be synthesized from a *sanitized*,
  structured failure summary (error category + what it means for the
  research, not the literal exception text) — produced by Engineer/
  Supervisor, never passed through verbatim.

### 2.3 `agents/editor.py` + `core/artifacts.py` — no manuscript from a dead run
- `create_final_paper()` / `assemble_paper()` must check run status first.
  If the run's terminal state is `FAILED` / `technical_failure`, refuse to
  assemble anything beyond a failure dossier, and the web UI / workspace
  view must render that state unambiguously (not as a manuscript with
  sections quietly missing).
- The "companion repository" export should be blocked under the same
  condition — don't ship code for an experiment that never produced
  verified results.

### 2.4 `agents/topic_hunter.py` + `agents/planner.py` — feasibility against a real capability manifest
- Define a machine-readable **sandbox capability manifest**: max wall-clock
  per experiment, available libraries (confirm PyTorch/GPU are or aren't
  actually available — your compute description elsewhere suggests
  CPU-only), whether outbound network/dataset-download is permitted at all,
  max dataset size, max training epochs/samples that are realistic in the
  time budget.
- `TopicHunterAgent.discover_topics()` and `PlannerAgent.create_plan()`
  must check every candidate topic/plan against this manifest *before*
  acceptance. A plan requiring internet dataset downloads or GPU-scale
  training when neither is available should be rejected or automatically
  rescoped (e.g., swap "Yahoo Finance daily prices" for a bundled/synthetic
  dataset that fits the manifest) — not accepted and left for Engineer to
  discover the problem after committing a contract.
- This is very likely the single highest-leverage fix: it stops the system
  from ever promising an experiment it can't run, which is the precondition
  for every fabrication seen in both papers.

### 2.5 `agents/hypothesis_debate.py` — give the Challenger the same manifest
- The `ChallengerAgent` should be able to raise "this is not executable in
  our sandbox within budget" as a first-class objection type, scored the
  same as a scientific objection, and it should be checked against the
  actual manifest from 2.4, not just argued from general knowledge. A
  hypothesis with unresolved *feasibility* objections should be treated the
  same as one with unresolved *severity* objections per your existing
  quality rule (`agents-hypothesis-debate.md`: high mean score is
  insufficient when severe objections remain unresolved).

### 2.6 `agents/engineer.py` — make the consistency check actually block
- `check_code_claim_consistency()` should be a hard gate, not advisory —
  if the plan's claimed baselines, dataset, or metrics don't match what the
  generated code actually does, that's a `request_plan_revision()` trigger,
  not something Writer can paper over later.
- When `run_branching_search()` can't find a variant that meets the plan's
  falsifiable prediction within budget, the correct outcome is an
  explicitly labeled negative/inconclusive result (which your workflow
  already supports) — never a plan silently swapped for a different,
  easier dataset without that being recorded as contract drift.

### 2.7 `agents/supervisor.py` — deterministic checks first, and make them count
- Confirm the "hard citation or statistics failure caps the section score"
  rule actually short-circuits before any LLM reviewer sees the section —
  from Paper B's evidence trace, it looks like the LLM-facing review layer
  may be running (and stamping "verified") even when the deterministic
  citation check should have failed first.
- Add the cross-section consistency check from 2.1 as one of Supervisor's
  hard, non-rescuable checks, not just something Verification computes for
  the ledger.

### 2.8 New tools/capabilities worth adding
- A real bibliographic metadata resolver (Crossref + arXiv + Semantic
  Scholar) wired into `core.verification`, not just ID-existence checks.
- A "capability card" object (per 2.4) that TopicHunter, Planner, Debate,
  and Engineer all read from one place, so feasibility reasoning is
  consistent across agents instead of each one having its own implicit
  assumptions.
- A dedicated adversarial "consistency referee" pass late in the
  pipeline — a separate model call whose only job is to read the fully
  assembled draft and flag any place where a number, dataset name, or
  claimed result differs from another place in the same document. This is
  cheap, catches exactly the Paper A failure mode, and doesn't require
  restructuring the writer.
- Run-log tracing: your `core.research_db` and JSONL logs already exist —
  worth adding a small script/agent step that, given a run ID, reconstructs
  the full claim lineage (which artifact backs which sentence in the final
  paper) so failures like these are diagnosable in minutes instead of by
  manual paper review.

---

## 3. Beyond "stop lying" — what makes an automated research system actually
##    good, not just honest

Section 2 stops fabrication. It doesn't make the system a better
researcher. That's a different problem, and it's the one real systems
spend most of their engineering on. Some concrete, adoptable ideas, with
where they come from:

### 3.1 Ground literature review in retrieved *text*, not remembered titles
OpenAlex/arXiv search gives you metadata (title, abstract, citation count).
That's enough to find candidates, not enough to write an accurate related-
work section — an LLM will still fill in details about a paper's method
from pattern-matching on the title, which is exactly how you get citations
attached to the wrong claim. Pull actual abstracts (and where licensing
allows, full text) into a retrieval store, and require `agents/writer.py`
to quote/paraphrase only from retrieved text for any claim attributed to a
specific paper — never from the LLM's own recollection of what a paper
"probably" says. This is the single biggest lever against confident wrong
citations, bigger than the metadata-matching fix in 2.1.

### 3.2 Novelty checking needs semantic similarity, not topic-level scoring
`TopicHunterAgent`'s novelty check should embed the *proposed contribution*
(not just the topic phrase) and search it against a corpus of actual paper
abstracts (Semantic Scholar embeddings API is built for exactly this). A
topic can look novel by title and still be a well-known result — this is
how you avoid the system confidently "discovering" something that's
already a named method in the literature.

### 3.3 Statistical claims should come from a stats library, not from the LLM
Every p-value, CI, or effect size in the final paper should be computed by
calling `scipy.stats`/`statsmodels` on the actual experiment artifact and
stored as a typed result — the writer should only ever be allowed to
render a number that already exists in that structured object, never
asked to "state the p-value" in prose generation. Add multiple-comparison
correction (Bonferroni/BH) automatically whenever a plan runs more than
one statistical test, and a minimum-power/minimum-sample-size check before
a significance claim is allowed to appear at all — small-N significance
claims are a classic way these systems produce impressive-looking but
meaningless results.

### 3.4 Validate generated experiment code against known-answer cases first
Before trusting a generated implementation (EDMD solver, Sinkhorn
algorithm, whatever) on real data, run it against a synthetic case with a
known closed-form or independently-verifiable answer (e.g., linear DMD on
a linear system should recover exact eigenvalues; Sinkhorn on identical
source/target distributions should return near-identity transport). Fail
and force a code-repair loop if the known-answer check doesn't pass. This
is standard practice in scientific computing and is cheap to add to
`core.sandbox` / `agents/engineer.py` alongside the existing AST safety
check — it catches silently-wrong implementations that would otherwise
just produce plausible-looking wrong numbers.

### 3.5 Adopt a tournament/evolutionary hypothesis loop, not single-pass debate
Google DeepMind's "AI co-scientist" and Sakana's "AI Scientist" both treat
hypothesis generation as a *population* that competes and evolves across
rounds — generate several, debate/rank via Elo (you already do Elo —
good), keep the survivors, generate variations of the best ones, repeat,
rather than accepting the first hypothesis that clears one debate round.
Your `HypothesisDebateSystem` already has the scoring primitive; the
missing piece is a loop that spends more compute on refining a promising
idea instead of moving on after one round. This tends to surface better
and more feasible ideas than single-shot generation.

### 3.6 Treat "honest negative/inconclusive result" as a *first-class success*, not a consolation prize
This is the most important one and it's an incentive-design problem, not
a code problem. If `MetaAgent`'s scoring implicitly rewards "produced a
complete, impressive-looking paper" more than "correctly determined the
hypothesis is false or infeasible and said so clearly," every agent
downstream will be under quiet pressure to make the numbers work out —
which is a plausible root cause of the fabrication you saw. Audit your
scoring/reward signals (Elo updates, Meta Agent's continue/reset/stop
decision, Supervisor's soft-review score) and confirm a well-executed
negative result scores at least as well as a well-executed positive one.
Real journals increasingly do this on purpose (registered reports, where
acceptance is decided before results are known) specifically to remove
this pressure — worth mirroring given you already commit an immutable
contract before running the experiment.

### 3.7 Real, pre-cleaned datasets over "download and hope"
Instead of planning around live downloads of Yahoo Finance / Wikipedia
traffic / UCI mirrors (all of which can silently change, rate-limit, or be
unreachable — and can't be reached at all from a locked-down sandbox),
bundle a small library of vetted, version-pinned datasets (OpenML has
clean APIs for exactly this, with stable dataset IDs) that Planner is
required to choose from. This removes an entire failure class where the
"dataset" in Methods and the "dataset" in Results silently diverge because
a download failed partway and something downstream improvised.

### 3.8 Reproducibility packaging as a hard requirement, not an afterthought
Your `agents/editor.py` already writes a companion repository — extend
that to always include a pinned environment (exact package versions),
every random seed used, and the exact commit hash of the experiment
contract. `core.verification.reproducibility_dossier` looks like the right
place for this to live; make it a release-blocking check (paper can't be
marked complete without a dossier that a fresh environment could actually
replay), not just a nice-to-have artifact.

### 3.9 A checklist-based referee pass, grounded in real reviewer guidelines
Your `ReviewerBot` should be given an actual structured checklist —
NeurIPS/ICLR-style reproducibility checklists are public and are exactly
what human reviewers use: does every quantitative claim have error bars or
a stated N, is there a limitations section, are baselines real and fairly
tuned, is the claimed novelty checked against related work, are negative
results reported alongside positive ones. Checklist-driven review is much
harder for a model to rubber-stamp than open-ended "review this section"
prompting, and it's exactly what strong human reviewers actually do.

### 3.10 A human checkpoint before anything is called "done"
None of the above makes the system trustworthy enough to fully skip human
review yet — no current system is. Keep (or add) an explicit
human-approval gate before a paper is marked publishable, at minimum while
you're calibrating the fixes in Section 2. This isn't a limitation of your
system specifically — it's true of every credible automated-research
project publicly known right now.

---

## 4. The prompt — paste this to your coding agent

```
You are working on ScholarGraph, a local multi-agent research pipeline
(topic discovery → hypothesis debate → planning → narrative writing →
sandboxed experiment engineering → results writing → supervision → meta-
evaluation → editing/LaTeX assembly). Architecture, workflow, and module
docs are in the repo's docs/ folder — read architecture.md, workflow.md,
and every file under docs/modules/ before changing anything.

GOAL
Make this system incapable of producing a manuscript with (a) fabricated
or unverifiable citations, (b) numeric claims that contradict each other
across sections, (c) raw execution/exception text laundered into prose as
if it were a scientific finding, or (d) a "finished-looking" manuscript
from a run that technically failed. The system should refuse to write
what it cannot support, and should refuse to plan experiments it cannot
actually execute.

STEP 1 — INVESTIGATE BEFORE CHANGING ANYTHING
1. Enumerate every past run in memory/ and output/ (SQLite research_db,
   JSONL logs, FAISS metadata, generated LaTeX/paper artifacts).
2. For at least the most recent failed and most recent "completed" run,
   reconstruct the full pipeline trace: topic → debate result → plan →
   committed experiment contract → engineer execution log(s) → verification/
   supervisor scores → final assembled sections.
3. For every numeric or citation claim in the final paper, trace it back
   to the artifact or literature source it supposedly came from. Flag any
   claim that has no traceable source, or whose source doesn't match.
4. Produce a written findings report before writing any code. I have
   already done a manual review of two prior paper outputs from this
   system — see the "Diagnosis" section above — treat that as a starting
   hypothesis to confirm or correct against what you find in the logs,
   not as ground truth to accept blindly.

STEP 2 — IMPLEMENT, IN THIS ORDER
Work through the numbered fixes in "Section 2: Fix list" above, in the
order given (2.1 → 2.8). For each one:
  - State which file(s) you're changing and why, referencing the specific
    failure mode it addresses.
  - Prefer adding hard, deterministic checks over prompting an LLM to
    "try to be more careful" — every failure mode found above got past
    an LLM-based check already.
  - Any check described as a "hard gate" or "non-rescuable" must actually
    short-circuit downstream agents (raise/return a terminal signal), not
    just lower a soft score that a later reviewer can override.
  - Do not let generated experiment failure output (exceptions, sandbox
    errors, tool-call logs) reach any section-writing prompt as raw text.
    If it needs to inform a "Limitations" section, it must go through a
    sanitization/summarization step first that strips implementation
    details and states only the research-relevant consequence.
  - When adding the feasibility/capability manifest (2.4), make it a
    single shared object all relevant agents read — not duplicated
    per-agent assumptions.

STEP 3 — VALIDATE
Re-run the pipeline (or replay the two traced runs from Step 1) and
confirm:
  - A plan requiring resources outside the capability manifest is
    rejected or rescoped before an experiment contract is committed, not
    discovered after.
  - A fabricated or mismatched citation is caught and hard-blocks the
    section score.
  - Deliberately introducing a numeric inconsistency between two sections
    causes assembly to fail with a specific, actionable error.
  - A run forced into technical failure never produces an exportable
    manuscript or companion repository, and is unambiguously marked as
    failed anywhere its state is surfaced.

Report back with: what you found in Step 1, what you changed and why,
and the specific test/validation evidence for each of the four checks in
Step 3. Do not mark this done based on the pipeline "running without
errors" — it must be shown to actually refuse the specific bad behaviors
above, using real or reconstructed cases, not just a happy-path run.
```

## Implementation Status

This repository now implements the plan's safety gates and the remaining local
research-quality controls. The following operational features are available:

### Safety gates (Section 2 fixes)

- **Citation metadata matching** (`core/verification.py`): `verify_citations` resolves DOIs/arXiv IDs via CrossRef/arXiv APIs and compares title/author metadata against writer-supplied bibliography entries. Mismatches are hard failures.
- **Claim typing** (`core/verification.py:classify_claim`): Claims are classified as `literature_reference`, `method_definition`, `planned_test`, or `empirical_result` before entering the evidence ledger. Only `empirical_result` claims tied to artifact IDs are eligible for verification.
- **Cross-section numeric consistency** (`core/verification.py:cross_section_numeric_consistency`): Collects every numeric claim tied to the same experiment across all drafted sections and diffs them against each other and the structured artifact. Conflicts are hard failures.
- **Consistency referee** (`core/verification.py:consistency_referee`): One isolated model pass over the full assembled draft to find contradictions in numbers, datasets, methods, outcomes, or contribution framing.
- **Prohibited manuscript text** (`core/verification.py:PROHIBITED_MANUSCRIPT_TEXT`): Blocks leaked harness diagnostics (e.g., "tracemalloc", "sandbox blocked", "traceback") from appearing in narrative sections.
- **No manuscript from dead runs** (`agents/editor.py`): `create_final_paper()` checks run status; failed runs produce a failure dossier, not a manuscript or companion repository.

### Research quality (Section 3 ideas)

- **Local catalogued datasets** (`core/datasets.py`): Iris, digits, synthetic, and OpenReview calibration datasets with local-only access policy. Planners rescope infeasible external datasets.
- **Capability-first dataset scoping**: TopicHunter and Planner check dataset plans against the local catalog before bridge validation; uncatalogued datasets are rejected early.
- **Sandbox capability manifest** (`core/capabilities.py:SandboxCapabilityManifest`): Shared execution limits (wall clock, libraries, GPU, network, dataset size) used by planning, debate, and engineering.
- **Feasibility checking** (`core/capabilities.py:check_plan_feasibility`): Plans requiring outbound access, GPU, or exceeding size limits are rejected before contract commitment.
- **Open-access-only full-text retrieval** (`core/sources.py:fetch_open_access_text`): A license signal is mandatory; responses are cached with hashes.
- **Known-answer fixtures** (`core/known_answers.py`): Validates generated experiment code against synthetic cases before trusting on real data.
- **Prospective power preregistration** (`core/verification.py:preregister_power`): Computes sample-size requirements before execution.
- **Deterministic reviewer checklist** (`core/verification.py:validate_reviewer_checklist`): Enforces limitations, baselines, outcomes, uncertainty reporting, and literature evidence requirements.
- **Outcome calibration** (`core/research_db.py:outcome_calibration`): Positive, negative, and inconclusive validated outcomes receive equal base credit.
- **Human approval gate** (`POST /api/release/approve`): Explicit human checkpoint before publishable release, persisted in the research ledger.
- **Clean replay** (`replay_run.py`): Replays companion code in a fresh virtual environment.
- **Historical reports** (`historical_report.py`): Reconstructs the latest failed/completed run pair with durable claims, artifacts, events, and lineage.
- **Forensic incident reports** (`forensic_report.py`): Per-run report with events, claims, artifacts, and claim-to-artifact lineage.

### TopicHunter v2 features

- OpenAlex concept filtering for domain-scoped queries
- HyDE (Hypothetical Document Embeddings) for query expansion
- Multi-hop retrieval via Semantic Scholar citation graph
- Frontier seeding from very recent papers
- Cross-seed paper cache for deduplication
- Structural gap mining via bibliographic coupling
- Method × domain sparsity matrix analysis
- Contradiction mining across papers
- Replication-target detection (strong claims, no variance reporting)
- Negative result seeding from prior unsupported hypotheses
- Persona ensemble generation (skeptic + practitioner)
- Seed-strategy Elo tracking with exploration reserve
- Rejection-history windowing with keyword survival floor
- Quick grounding pre-filter before expensive gate chain

### QA literature synthesis mode

- Separate graph: Literature Retrieval → Synthesis Answer → Verification
- Invoked via `--mode qa --query "..."` or the Web UI
- Citation grounding enforced via `verify_citations()` inside the answer node
- Skips hypothesis debate, planning, engineering, and evidence-gate machinery

### What the system still cannot do

- The system cannot legally retrieve paywalled full text without an explicit
  open-access permission signal. Those cases fail closed and remain visible in
  the source artifact and forensic report.
- No local system can guarantee that an external provider's API or license
  remains available.
- The capability broker is an application-level policy, not an OS security
  boundary. The sandbox blocks documented failure modes but is not container
  isolation.
- The admin console is local single-user and polling-based with no
  authentication.
- Passing all gates does **not** prove publication-quality novelty,
  generalizability, or real-world usefulness. Human domain review, appropriate
  data governance, and independent reproduction remain necessary.

### v4 research upgrades (2026-09)

Informed by 2025–2026 agentic-research literature (WARA artifact repair, AppliedScientist reviewer-guided revision, AgentGrad failure gradients, Gupta & Pruthi ACL 2025 novelty-plagiarism findings, TruthInsightBench "discriminating acts"):

- **Feedback-aware revision** (`agents/writer.py`, `core/workflow_nodes.py`): Redrafts carry deterministic check failures + supervisor feedback into the writer prompt; results redrafts re-draft only failing sections.
- **Narrative revision loop** (`write_narrative_sections`): On meta-continue, below-threshold narrative sections are re-drafted once with supervisor feedback (`narrative_revision_count` bound = 1).
- **Editor referee repair** (`workflow_nodes.editor_repair_route`): Release-referee failures route to one bounded repair pass instead of terminal failure; failed-experiment failures stay terminal.
- **Self-correcting JSON** (`core/utils.py:call_llm_json`): Re-asks with parse error on malformed JSON; wired into `consistency_referee`, supervisor soft checks, meta feedback.
- **Novelty-plagiarism gate** (`core/verification.py:novelty_overlap_check`): Deterministic overlap-coefficient screen of Abstract+Introduction vs closest prior work / literature evidence.
- **Evidence-grounded proposer** (`agents/hypothesis_debate.py`): Round-1 arguments include structured hypothesis, retrieved evidence, and prior objection tags to preempt recurring objections.
- **Prompt hardening** (`agents/writer.py`): Intro/abstract get literature evidence + "never invent citations" policy; Results gets copy-exact + n=/std/CI + statistical-test + falsifiability-reporting requirements.
- **LLM failure visibility** (`core/llm.py`): `llm_failures` run stat + error-level message on chat failure.

### v4.1 round two (2026-09, same batch)

Informed by Anthropic context engineering (attention budget / minimal high-signal tokens), Critic Experience Bank, CYCLE, and AgentGrad:

- **Engineer failure-gradient hints** (`agents/engineer.py:_error_category_hint`): Deterministic hints per failure category injected into `_refine_code` prompts.
- **Cross-run lessons for Engineer** (`_generate_experiment_code`): Uses `CrossRunMemory().get_prompt_context()` to list prior failure patterns.
- **Screener self-correction** (`agents/topic_hunter.py`): `call_llm_json` replaces the parse-blind 2-attempt loop.
- **Challenger followup self-correction** (`agents/hypothesis_debate.py:followup_objections`): Self-correcting parse with fail-closed prior objections.
- **QA answer robustness** (`core/workflow_nodes.py:qa_answer_node`): `call_llm_json` with parse-error re-ask.
- **Supervisor checklist hardening** (`agents/supervisor.py:REVIEW_CHECKLIST`): Requires explicit falsifiable-prediction verdict (supported/falsified/inconclusive) with controls/robustness evidence.
- **Writer context budget** (`agents/writer.py`): Experiment JSON dumps bounded (`[:6000]`).

### Debate payload-shape fixes (2026-09-19, from run logs)

- **Objection normalization** (`agents/hypothesis_debate.py:build_rebuttal`): `_normalize_objection_payload` coerces bare array / single-dict objections → canonical envelope.
- **Challenger self-correcting parse** (`build_rebuttal`): `call_llm_json` re-ask carries the parse error + offending excerpt.
- **Followup envelope tolerance** (`followup_objections`): Bare-array objection payloads are coerced instead of failed closed.
- **Iteration hardening** (`evaluate_debate`, `conduct_debate`, `revise_topic_from_objections`): All `.get`-on-objection loops filter `isinstance(o, dict)`; `structured_hypothesis` accessed via `_as_dict()`.
- **`parse_json_from_llm` type safety** (`core/utils.py`): Non-str input → None; broad `except Exception`.

Tests: `tests/test_v4_research_upgrades.py`, `tests/test_debate_robustness.py`.

Operational validation currently passes the full repository suite. Use the
commands in `docs/testing.md` for regression, clean replay, and incident reports.
