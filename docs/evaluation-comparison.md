# Evaluation: LangGraph vs OpenAI Agents SDK — side by side

Same dataset (`evaluation/dataset/topics.json` v1.0.0), same corpus fixtures,
same evaluators, same thresholds. Both scorecards generated offline with each
framework's deterministic stub — the numbers measure *pipeline* behavior
(retrieval → parsing → guardrails → citation verification), not live-LLM
quality.

## Scorecard comparison

| Metric | Threshold | LangGraph (`agentic-web-researcher`) | Agents SDK (this port) |
|---|---|---|---|
| Citation support rate | ≥ 0.90 | 1.000 | 1.000 |
| Source precision | ≥ 0.80 | 1.000 | 1.000 |
| Topic coverage | ≥ 0.85 | 1.000 | 1.000 |
| Citations kept (eu-ai-act) | — | 8 | 8 |
| Citations kept (remote-work) | — | 3 | 3 |
| Sources (eu-ai-act / remote-work) | — | 2 / 1 | 2 / 1 |
| Citations dropped | — | 0 | 0 |

Reproduce: `uv run python -m evaluation.run_eval` in each repo (fully
offline, no secrets needed). Output: `evaluation/scorecards/latest.json`.

## Behavioral differences observed during the port

These are real framework differences found while porting — not generic
pros/cons:

1. **Fan-out is manual, not declarative.** LangGraph's `Send()` produced
   per-branch node executions visible to the checkpointer; here the runner
   owns `asyncio.gather` over plain coroutines and merges results into the
   context explicitly. Observable behavior is identical (same SSE stage
   vocabulary, same counts) but scheduling/error-isolation code lives in the
   orchestrator instead of the graph runtime.

2. **No checkpointer → HITL is a stored blob.** The LangGraph version
   resumed via `interrupt()` + SQLite checkpointing of graph state. The port
   persists the whole `PipelineContext` (a Pydantic model) in a
   `paused_states` table and resumes by re-entering the runner after the
   review gate. Simpler to reason about; less granular than checkpoints.

3. **Guardrails fit natively — with one asymmetry.** `screen_topic` became a
   real `@input_guardrail` on the planner (`run_in_parallel=False` keeps it
   guardrail-first — the default `True` would fire the model call in
   parallel, defeating the point). `verify_citations` became an
   `@output_guardrail` on the writer that receives the pipeline context via
   `Runner.run(context=...)`. But tripwire semantics are all-or-nothing:
   the citation filter (drop some, keep the report) can't be expressed as a
   tripwire, so it returns `tripwire_triggered=False` and mutates context.
   LangGraph had no equivalent primitive at all — there it was a plain node.

4. **Structured output is cleaner.** `with_structured_output` chains become
   `Agent(output_type=...)`: the schema contract is declarative on the agent,
   and `RunResult.final_output` is already the parsed Pydantic model.

5. **The deterministic offline story is comparable but differently shaped.**
   LangGraph used a `StubLLMClient` implementing a custom protocol; the port
   implements the SDK's own `Model` interface (`get_response` returning a
   `ModelResponse` with synthetic `Usage`), so the stub plugs into the real
   `Runner` — including `output_type` validation — rather than bypassing it.

6. **Tracing moves from implicit to explicit.** LangGraph auto-traced to
   LangSmith via env vars; the SDK streams spans to OpenAI's backend by
   default and LangSmith requires registering
   `OpenAIAgentsTracingProcessor` via `set_trace_processors` — done in
   `app/core/tracing.py` only when `LANGCHAIN_API_KEY` + flag are set.

## Live-model runs

Running `run_eval` with `OPENAI_API_KEY` configured exercises the real
`OpenAIChatCompletionsModel` — the scorecard schema is identical so results
remain directly comparable. Record live numbers here when a credentialed run
happens.
