# Research Workflow

## Phases

1. `topic_discovery`: query OpenAlex and arXiv, inspect citation signals, score novelty, and reject infeasible topics.
2. `hypothesis_debate`: run proposer, challenger, and moderator rounds; use ensemble scores and unresolved objections.
3. `planning`: create a falsifiable plan with baselines, variants, metrics, and dependencies.
4. `writing_narrative`: draft non-result sections before experiments.
5. `engineering`: commit an immutable experiment contract and dataset identity, generate and execute code in the sandbox, aggregate multiple seeds, and optionally branch over variants before commitment.
6. `independent_validation`: replay successful code, verify hashes and statistics, and block the run on missing or contradictory evidence.
7. `writing_results`: draft result-bearing sections only after the evidence gate passes and verify that quantitative claims are grounded in recorded artifacts.
8. `supervision`: run deterministic citation, statistics, math, and code checks before the soft reviewer.
9. `meta_evaluation`: improve prose only for eligible evidence; it cannot override terminal technical failure or mutate a committed contract.
10. `editing`: add limitations, resolve bibliography entries, write LaTeX, and export a companion repository only after release gates pass.

## State contract

`core.state.ResearchState` is the single typed contract. The workflow mutates it in place because LangGraph nodes return the updated state. The initializer creates empty collections so nodes can append without null checks.

During a real CLI or web run, `ResearchPipeline` activates one `RunContext` for the stream. Agents receive that context when constructed, giving them access to run configuration, vector memory, and the research ledger without reaching into process globals.

## Recovery paths

- Discovery source outage becomes a terminal error rather than a misleading reset loop.
- Failed debate topics are removed and the next candidate is tried.
- Technical Engineer failures receive bounded code-only repair attempts against the same contract. Unresolved failure is terminal and produces a failure dossier; it does not route to writing or Meta.
- A successful run may produce supported, unsupported, or inconclusive findings. Unsupported hypotheses are written as negative results using the fixed data and protocol.
- After commitment, changing data, requirements, metrics, baselines, or hypothesis is contract drift and requires a new experiment identity.
- Results numeric grounding may trigger up to two redrafts.
- Durable checkpoints allow CLI resume with `python main.py --resume RUN_ID`.
