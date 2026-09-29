# Port scope & framework mapping — LangGraph → OpenAI Agents SDK

This document freezes what this repository ports and records the concept
mapping between the two frameworks. It is the reference every phase checks
against, and the raw material for the README's framework-comparison section.

---

## 1. Target SDK decision

**Chosen: OpenAI Agents SDK for Python** (`openai-agents` on PyPI, import
name `agents`), pinned to `>=0.22.3,<0.23` in `pyproject.toml` + `uv.lock`.

| Criterion | OpenAI Agents SDK | Anthropic Agent SDK |
|---|---|---|
| Market signal | Named verbatim in the target job spec ("Anthropic Agent SDK, OpenAI Agents SDK, or similar"); ~12M PyPI downloads/month | Also named, but the SDK is the Claude Code agent loop exposed — oriented to coding agents (filesystem/bash/MCP), not LLM pipelines |
| Fit for this port | Generalist agent framework: `Agent`/`Runner`/`handoffs`/`function_tool`/`output_type`/native guardrails — maps directly onto planner→workers→critic→writer | Designed around Claude's computer-use/tool loop; structured-output pipelines are not its center of gravity |
| Language | Python — identical to the LangGraph original, so the diff isolates the orchestration layer | Python (`claude-agent-sdk`), same advantage — but weaker pipeline primitives |
| Credentials | Reuses the portfolio's existing OpenAI key story | Would add an Anthropic dependency for no extra signal |
| Model portability | Provider-agnostic (OpenAI Responses/Chat Completions + 100+ providers via LiteLLM), plus a `Model`/`ModelProvider` interface — a custom stub model can run the whole suite offline with zero secrets | Claude-only |

The raw `openai` client was also considered and rejected: a hand-rolled
tool-calling loop is not a *framework* — it would not close the "GenAI
frameworks" gap (and is already demonstrated elsewhere in the portfolio).

## 2. Frozen use case (the port contract)

Identical to the LangGraph version — same request, same observable behavior,
same success criteria. Differences live only in the orchestration engine.

**Request:** `POST /api/v1/research` with
`{topic: str (10–2000 chars), depth: "quick"|"standard"|"deep", language: str}`.

**Pipeline semantics:**

1. Input guardrail screens the topic *before* any LLM call or external
   request (guardrail-first).
2. Planner decomposes the topic into sub-questions — depth targets
   3/5/8 (quick/standard/deep).
3. Parallel workers: one web search per sub-question → one scrape per
   unique URL → one parse+sanitize per fetched document.
4. Critic grades coverage + citation support (`CriticVerdict`), requesting
   another round while `needs_more_research` and rounds remain
   (`max_critic_rounds`, bounded); exhaustion escalates to the human gate.
5. Writer synthesizes the report + citations (structured output).
6. Output guardrail drops every citation whose quote cannot be verified
   verbatim/similarity ≥0.85 against the cited source's extracted text.
7. Human review gate (approve / edit / reject) before publication; passthrough
   when disabled.
8. Report assembler finalizes sub-question statuses.

Worker errors are recorded in `errors` and never abort the run. Scraped
content is always *untrusted data*: sanitized, capped at 20k chars, and
never interpolated into agent `instructions`.

**Success criteria (identical gates):** citation support ≥ 0.90, source
precision ≥ 0.80, topic coverage ≥ 0.85, judge rubric ≥ 4.0, p50 ≤ 60s /
p95 ≤ 180s, cost ≤ $0.05/job, test coverage ≥ 85%, `mypy --strict` 0 errors.
The evaluation reuses the *same* dataset, corpus fixtures, evaluators and
scorecard schema so both scorecards are directly comparable.

**API contract (identical):** auth login, submit/poll/SSE-stream/result,
review submit, PDF download — including the `JobEvent.node` stage vocabulary
(the console's `NODE_LABELS` map is part of the public surface).

## 3. Feature parity inventory

Pipeline features that must have an equivalent (checked off as they land):

| # | Original feature | Port equivalent |
|---|---|---|
| 1 | `screen_topic` input guardrail (injection + harmful + length cap) | SDK `input_guardrail` wrapping the same function |
| 2 | Planner → `list[SubQuestion]` | `Agent(output_type=PlanOutput)` |
| 3 | `Send()` fan-out: search worker per sub-question | `asyncio.gather(Runner.run(...))` per sub-question |
| 4 | `Send()` fan-out: scrape worker per unique URL | same gather pattern |
| 5 | `Send()` fan-out: document worker per raw doc → sanitize → `Source` | same gather pattern; `sanitize_scraped_content` unchanged |
| 6 | Critic → `CriticVerdict` + bounded re-plan loop | `Agent(output_type=CriticVerdict)` + loop bound in the runner |
| 7 | Re-plan: missing aspects → new pending questions | deterministic helper, unchanged logic |
| 8 | Writer → report + `list[Citation]` | `Agent(output_type=ReportOutput)` |
| 9 | `verify_citations` output guardrail | SDK `output_guardrail` wrapping the same function |
| 10 | `interrupt()` HITL gate + checkpointer resume | job-state pause/persist + resume endpoint (no SDK equivalent — see §5) |
| 11 | `errors` channel accumulating worker failures | explicit merge in the runner |
| 12 | `LLMClient`/`StubLLMClient` | agents' `model` param: real model vs custom stub `Model` |
| 13 | Per-role usage recording → cost tracking | `RunResult.usage` / run hooks → same `usage` store |
| 14 | `JobEvent` SSE stream of node progress | orchestrator emits identical stage events into the job queue |
| 15 | LangSmith traces | SDK tracing → custom `TracingProcessor` / `set_tracing_disabled` in CI |

Service surfaces (full-parity scope): JWT auth, rate limiting, async job
store (SQLite), SSE stream, PDF export, Prometheus metrics, structlog,
security headers, both frontend surfaces (`/`, `/console` — copied
verbatim), EDD harness + scorecard, offline stub, CI workflow.

## 4. Concept mapping

| LangGraph concept | Agents SDK concept | Where it lives here |
|---|---|---|
| `StateGraph` + nodes/edges | `Agent` definitions + `Runner`; orchestration in Python | `app/agents/`, `app/pipeline/` |
| `ResearchState` TypedDict + `operator.add` reducers | Plain Pydantic context object; explicit merge code | `app/pipeline/context.py` (runner merges worker outputs) |
| `Send()` fan-out | **No equivalent** — `asyncio.gather(Runner.run(...))` | `app/pipeline/runner.py` |
| Conditional edges | `if`/`for` in the runner — control flow is code, not a graph | `app/pipeline/runner.py` |
| `with_structured_output(Model)` | `Agent(output_type=PydanticModel)` | `app/agents/*.py` |
| `StructuredTool` | `@function_tool` (same Pydantic input schemas) | `app/services/tools.py` |
| `RunnableLambda` node factories closing over `GraphDeps` | `RunContextWrapper[Deps]` injected into tools/agents | `app/pipeline/deps.py` |
| `interrupt()` + SQLite checkpointer | **No equivalent** — persist pipeline state + `/review` resume | `app/pipeline/` + job store |
| Custom guardrail functions | `@input_guardrail` / `@output_guardrail` (same functions inside) | `app/guardrails.py` + `app/agents/` wiring |
| `LLMClient` protocol + `StubLLMClient` | `Model`/`ModelProvider` interface → `StubModel` | `app/services/models.py` |
| Graph `astream` → `JobEvent` queue | Runner emits stage events; `RunResultStreaming`/`stream_events` for intra-agent events | `app/services/job_runner.py` |
| LangSmith tracing | `TracingProcessor` chain / `set_tracing_disabled` | `app/core/tracing.py` |
| `max_iterations` graph bound | `Runner.run(..., max_turns=)` per agent + round bound in runner | runner + `RunConfig` |
| Checkpointer-based resume | `RunState`/`Session` *may* cover pause-resume — validate in Phase 1 spike before committing to the custom gate | Phase 1 finding |

## 5. Deliberate non-equivalents (feeds the comparison section)

Real trade-offs found during scoping — document measured behavior, not
assumptions:

- **Fan-out:** LangGraph's `Send` is declarative parallel dispatch with
  automatic reducer merge. The SDK has no fan-out primitive — parallelism
  is `asyncio.gather` + hand-written merge. More control, more code, same
  semantics to verify.
- **HITL:** `interrupt()` gives LangGraph durable, checkpointed pauses for
  free. The SDK targets single-run agents; `Session`/`RunState` might
  support serialization-resume — the Phase 1 spike must confirm before the
  custom pause/persist/resume design is locked.
- **State merge:** reducers merge parallel branches atomically; the port
  merges results in the runner — equivalent output, weaker compile-time
  guarantees.
- **Tracing:** SDK tracing streams to OpenAI's backend by default; CI must
  disable it or register a local `TracingProcessor` to stay offline.
- **Guardrails:** the SDK's guardrail decorators run around agent
  *invocations*; the original's guards wrap *pipeline stages*. Mapping must
  preserve the guardrail-first property (screen before spending tokens).

## Phase 1 open questions (spike targets)

1. Does `Runner.run` in `asyncio.gather` give true parallel agent
   execution (no hidden global locks)?
2. Can `RunState`/`Session` serialize+resume a paused run well enough to
   back the HITL gate, or is custom persistence needed?
3. Does a custom `Model` (stub) integrate cleanly with `Agent` +
   `output_type` so offline tests exercise real SDK machinery?
4. Do `input_guardrail`/`output_guardrail` tripwires abort cleanly and
   observably (typed result vs exception)?
5. Streaming: does `Runner.run_streamed` expose per-tool-call events
   granular enough to feed the `JobEvent` SSE vocabulary?
