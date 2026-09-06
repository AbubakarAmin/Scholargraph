# Research Workflow

## Phases

1. `topic_discovery`: query OpenAlex and arXiv, inspect citation signals, score novelty, and reject infeasible topics.
2. `hypothesis_debate`: run proposer, challenger, and moderator rounds; use ensemble scores and unresolved objections.
3. `planning`: create a falsifiable plan with baselines, variants, metrics, and dependencies.
4. `writing_narrative`: draft non-result sections before experiments.
5. `engineering`: generate and execute code in the sandbox, aggregate multiple seeds, and optionally branch over variants.
6. `writing_results`: draft result-bearing sections and verify that quantitative claims are grounded in engineer output.
7. `supervision`: run deterministic citation, statistics, math, and code checks before the soft reviewer.
8. `meta_evaluation`: decide whether to retry, reset, or stop.
9. `editing`: add limitations, resolve bibliography entries, write LaTeX, and export a companion repository.

## State contract

`core.state.ResearchState` is the single typed contract. The workflow mutates it in place because LangGraph nodes return the updated state. The initializer creates empty collections so nodes can append without null checks.

During a real CLI or web run, `ResearchPipeline` activates one `RunContext` for the stream. Agents receive that context when constructed, giving them access to run configuration, vector memory, and the research ledger without reaching into process globals.

## Recovery paths

- Discovery source outage becomes a terminal error rather than a misleading reset loop.
- Failed debate topics are removed and the next candidate is tried.
- Engineer failures may request a plan revision.
- Results numeric grounding may trigger up to two redrafts.
- Durable checkpoints allow CLI resume with `python main.py --resume RUN_ID`.
