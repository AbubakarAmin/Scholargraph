# ScholarGraph Documentation

This folder documents the current codebase module by module. The project is a local, multi-agent research workflow that discovers topics, debates hypotheses, plans experiments, executes sandboxed code, verifies claims, and assembles a LaTeX paper.

## Start here

- [Architecture](architecture.md): runtime boundaries and data flow.
- [Workflow](workflow.md): LangGraph phases and state transitions.
- [Refactor roadmap](refactor-roadmap.md): completed staged refactor and final cleanup status.
- [Configuration](modules/core-config.md): environment and runtime settings.
- [Testing](testing.md): offline checks and validation commands.

## Module guides

### Entry points

- [main.py](modules/main.md)
- [demo.py](modules/demo.md)
- [run_ui.py](modules/run-ui.md)
- [run_with_real_api.py](modules/run-with-real-api.md)
- [setup.py](modules/setup.md)

### Agents

- [Topic Hunter](modules/agents-topic-hunter.md)
- [Hypothesis Debate](modules/agents-hypothesis-debate.md)
- [Planner](modules/agents-planner.md)
- [Writer](modules/agents-writer.md)
- [Engineer](modules/agents-engineer.md)
- [Supervisor](modules/agents-supervisor.md)
- [Meta Agent](modules/agents-meta-agent.md)
- [Editor](modules/agents-editor.md)

### Core services

- [State](modules/core-state.md)
- [Contracts](modules/core-contracts.md)
- [Context](modules/core-context.md)
- [Artifacts](modules/core-artifacts.md)
- [Ports](modules/core-ports.md)
- [Workflow](modules/core-workflow.md)
- [Workflow nodes](modules/core-workflow-nodes.md)
- [Pipeline](modules/core-pipeline.md)
- [Config](modules/core-config.md)
- [LLM](modules/core-llm.md)
- [Utils](modules/core-utils.md)
- [Memory](modules/core-memory.md)
- [Run Log](modules/core-run-log.md)
- [Research DB](modules/core-research-db.md)
- [Sandbox](modules/core-sandbox.md)
- [Verification](modules/core-verification.md)

### Web and tests

- [Web API](modules/web-app.md)
- [Static UI](modules/web-static.md)
- [Test suite](testing.md)
