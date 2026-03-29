# ⚗ PromptCrucible

> Stop tuning prompts blindly. PromptCrucible automatically optimizes your
> LLM extraction prompt using a semantic evaluation loop — for any document
> type, any domain, running entirely on local LLMs with zero API cost.

[DEMO GIF PLACEHOLDER]

## The problem

Engineers building LLM document extraction pipelines spend 4-5 days in a
blind trial-and-error loop tuning prompts. No automated way to know if a
fix generalizes. No memory of what was tried. No semantic scoring. Just
you, a text editor, and hours of waiting for pipeline runs.

## How it works

PromptCrucible runs an event-driven multi-agent loop:

EVALUATOR → scores every test case semantically (not just schema validation)
DIAGNOSER → explains exactly why each case failed and what pattern it represents
OPTIMIZER → rewrites the prompt with a targeted fix, never repeating past mistakes
ORCHESTRATOR → loops until threshold reached, plateau detected, or max iterations hit

All agents share a Blackboard. Full iteration history is passed to every agent
as memory — so the system never repeats a failed approach.

## Quick start

```bash
pip install promptcrucible
ollama pull qwen2.5:7b
ollama pull phi3:mini

crucible init
crucible run --prompt prompt.txt --inputs ./documents/
```

## Works with any document type

Invoices · Contracts · Medical records · Insurance claims ·
Financial reports · Research papers · Receipts · Custom formats

## Requirements

- Python 3.11+
- Ollama running locally (`ollama serve`)
- Any Ollama model (default: `qwen2.5:7b`, judge: `phi3:mini`)

## Architecture

Built on an Event-Driven Multi-Agent Architecture (EDNN-inspired):

- Three specialized LLM agents communicate via a shared Blackboard
- Each agent fires when its required inputs are ready
- Full iteration history passed as memory to prevent repeated mistakes
- War room terminal web UI with 3D agent graph visualization

## Commands

```bash
crucible init                    # Initialize project
crucible run --prompt p.txt \    # Run optimization loop
  --inputs ./docs/
crucible history                 # View all past experiments
crucible diff <run_a> <run_b>    # Compare two runs
crucible export --run <id>       # Export final prompt + history
crucible ui                      # Launch web UI on localhost:7777
```

## Why local LLMs?

No API keys. No cost. No data leaving your machine.
Works with any Ollama model — qwen2.5, llama3, mistral, deepseek.

## License

MIT
