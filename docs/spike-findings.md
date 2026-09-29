# Phase 1 spike findings — OpenAI Agents SDK (v0.22.3)

Every claim below was verified by a real run of `scripts/spike_sdk.py`
(live `gpt-4o-mini` for sections 1–5, fully offline for section 6).

## Results

| # | Question | Result |
|---|---|---|
| 1 | Agent + `@function_tool` basics | Works. `Runner.run` → `RunResult.final_output`; tool calls visible in `new_items` as `tool_call_item`. Usage is per-`raw_responses[i].usage` (no aggregate on the result). |
| 2 | Handoffs / delegation | Works. `Agent(handoffs=[...])` exposes a `transfer_to_<name>` tool; `result.last_agent` shows who answered (triage → `billing-specialist`). Caveat: agent names must be `[a-zA-Z0-9_]` — `billing-specialist` auto-renamed to `transfer_to_billing_specialist` with a warning. |
| 3 | Real parallelism | **No `Send()` equivalent** — `asyncio.gather(*(Runner.run(...)))` gives true concurrency: 3 runs took 5.2s vs 11.3s sequential (latency-bound). Fan-out is orchestrated, not declared. |
| 4 | Input guardrails | `@input_guardrail` runs **before the model call** and raises `InputGuardrailTripwireTriggered` carrying `output_info` — a typed, inspectable veto. Matches the guardrail-first requirement. |
| 5 | Human-in-the-loop | **SDK-native.** `FunctionTool(needs_approval=True)` pauses the run with `result.interruptions`; `result.to_state()` → `RunState.to_json()`/`from_json()` round-trips (async), `state.approve(item)` + `Runner.run(agent, state)` resumes to completion. Designed for tool-approval pauses *inside* a run; our pipeline gate sits *between* runs, so we persist pipeline context instead — either way, durable HITL is solved. |
| 6 | Custom `Model` | Works offline. Implementing `Model.get_response`/`stream_response` and passing it as `Agent(model=...)` runs with no `OPENAI_API_KEY`; `output_type` still parses into Pydantic. This is the `StubModel` foundation for the zero-secrets suite. |

## Decisions taken for the port

- **Orchestration = Python, not a graph.** `Runner.run` per role,
  `asyncio.gather` for fan-out, explicit merge. Mirrors the original's
  deterministic workers; `Send()` has no SDK equivalent.
- **LLM roles stay LLM roles**: planner/critic/writer are `Agent`s with
  `output_type`. Workers (search/scrape/parse) stay plain async functions —
  parity with the original's deterministic worker nodes.
- **Model class: `OpenAIChatCompletionsModel`** (explicit) rather than the
  default Responses model — the original used chat completions, keeping
  cost/latency comparisons apples-to-apples.
- **Usage accounting** via `result.raw_responses[*].usage` → existing
  `UsageTracker` (no aggregate `result.usage` in 0.22.3).
- **HITL**: pipeline-level pause — serialize `PipelineContext` to the job
  store on `awaiting_review`, resume with the reviewer's decision. Simpler
  than `RunState` for a between-stages gate; `RunState` remains an option if
  an agent-internal pause is ever needed (verified working).
- **Guardrails**: `screen_topic` enforced both as a pipeline stage (same
  semantics, can sanitize-then-continue) and registered as an SDK
  `input_guardrail` on the planner (typed veto if the stage is bypassed).
  `verify_citations` stays a *filtering* stage — SDK output guardrails veto
  rather than filter, so they don't fit drop-bad-citations semantics.
- **Tracing**: `set_tracing_disabled(True)` default offline;
  `OpenAIAgentsTracingProcessor` (langsmith) registered only when the
  LangSmith key + flag are configured.

## Non-obvious API details (0.22.3)

- `Runner.run(agent, input, context=..., max_turns=...)`; `input` also
  accepts a `RunState` for resume.
- `RunState.from_json` and `RunState.from_string` are **async**.
- `Model.get_response` signature includes `output_schema`
  (`AgentOutputSchema.name` lets a stub branch per agent role) plus
  `previous_response_id`/`conversation_id`/`prompt` kwargs.
- `FunctionTool.needs_approval` accepts bool or async predicate.
- Agent names with `-` get auto-renamed in handoff tool names — use
  `[a-zA-Z0-9_]` names.
