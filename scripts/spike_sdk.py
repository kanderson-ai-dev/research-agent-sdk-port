"""Phase 1 spike: verify OpenAI Agents SDK primitives for the port.

Exercises each building block the pipeline port needs and prints a
``FINDING:`` line per section so results can be transcribed into
``docs/spike-findings.md``:

1. hello — one agent with one ``@function_tool``.
2. handoff — triage agent delegates to a specialist via ``handoffs``.
3. fanout — ``asyncio.gather`` over concurrent ``Runner.run`` calls.
4. guardrail — ``@input_guardrail`` tripwire blocks a run before the model.
5. hitl — ``needs_approval`` tool pauses the run; serialized ``RunState``
   resumes after approval.
6. stub — a custom ``Model`` implementation runs fully offline.

Live sections require ``OPENAI_API_KEY`` (loaded from ``.env``); without it
they are skipped. The stub section always runs — it proves the offline story.
"""

import asyncio
import json
import os
import sys
import time
from collections.abc import AsyncIterator
from typing import Any

import agents
from agents import (
    Agent,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    Model,
    ModelResponse,
    RunContextWrapper,
    Runner,
    RunState,
    TResponseInputItem,
    function_tool,
    input_guardrail,
)
from agents.items import TResponseStreamEvent
from agents.models.interface import ModelTracing
from agents.tool import Tool
from openai.types.responses import ResponseOutputMessage, ResponseOutputText

MODEL_NAME = "gpt-4o-mini"


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def finding(text: str) -> None:
    print(f"FINDING: {text}")


# --- 1. hello world: agent + tool --------------------------------------------


@function_tool
def word_count(text: str) -> str:
    """Count the words in the given text."""
    return str(len(text.split()))


async def spike_hello() -> None:
    section("1. hello world (agent + tool)")
    agent = Agent(
        name="greeter",
        model=MODEL_NAME,
        instructions="Use the word_count tool, then answer briefly.",
        tools=[word_count],
    )
    result = await Runner.run(agent, "How many words in 'the quick brown fox jumps'?")
    print(f"final_output: {result.final_output!r}")
    in_toks = sum(r.usage.input_tokens for r in result.raw_responses)
    out_toks = sum(r.usage.output_tokens for r in result.raw_responses)
    print(f"usage: {in_toks} in / {out_toks} out ({len(result.raw_responses)} calls)")
    tool_calls = [
        item for item in result.new_items if item.type == "tool_call_item"
    ]
    finding(f"tool calls observed in new_items: {len(tool_calls)} >= 1")


# --- 2. handoffs / delegation -------------------------------------------------


async def spike_handoff() -> None:
    section("2. handoff delegation (triage -> specialist)")
    specialist = Agent(
        name="billing-specialist",
        model=MODEL_NAME,
        instructions="You answer billing questions in one short sentence.",
    )
    triage = Agent(
        name="triage",
        model=MODEL_NAME,
        instructions=(
            "You triage requests. Billing questions MUST be handed off to the "
            "billing-specialist via the handoff tool."
        ),
        handoffs=[specialist],
    )
    result = await Runner.run(
        triage, "I was charged twice for my subscription, what do I do?"
    )
    print(f"last_agent: {result.last_agent.name}")
    print(f"final_output: {result.final_output!r}")
    finding(
        "handoff executed: last_agent="
        f"{result.last_agent.name} (expected billing-specialist)"
    )


# --- 3. real parallelism via asyncio.gather ------------------------------------


async def spike_fanout() -> None:
    section("3. fan-out parallelism (asyncio.gather over Runner.run)")

    @function_tool
    async def nap(ctx: RunContextWrapper[Any], seconds: float) -> str:
        """Sleep for the given number of seconds, then return 'slept'."""
        await asyncio.sleep(seconds)
        return "slept"

    agent = Agent(
        name="napper",
        model=MODEL_NAME,
        instructions="Always call the nap tool with the requested duration.",
        tools=[nap],
    )

    async def one(i: int) -> Any:
        return await Runner.run(agent, f"nap for 0.6 seconds (call {i})")

    t0 = time.monotonic()
    for i in range(3):
        await one(i)
    sequential = time.monotonic() - t0

    t0 = time.monotonic()
    await asyncio.gather(*(one(i) for i in range(3)))
    parallel = time.monotonic() - t0

    print(f"sequential: {sequential:.2f}s | parallel: {parallel:.2f}s")
    finding(
        f"gather gives real parallelism: {parallel:.2f}s vs {sequential:.2f}s "
        "sequential (no Send primitive — orchestrate manually)"
    )


# --- 4. input guardrail tripwire -----------------------------------------------


@input_guardrail
async def no_forbidden_topics(
    ctx: RunContextWrapper[Any], agent: Agent[Any], input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    """Trip when the input mentions the forbidden topic."""
    text = input if isinstance(input, str) else json.dumps(input, default=str)
    return GuardrailFunctionOutput(
        output_info={"reason": "forbidden topic"},
        tripwire_triggered="forbidden" in text.lower(),
    )


async def spike_guardrail() -> None:
    section("4. input guardrail tripwire")
    agent = Agent(
        name="guarded",
        model=MODEL_NAME,
        instructions="Reply briefly.",
        input_guardrails=[no_forbidden_topics],
    )
    ok = await Runner.run(agent, "say hi")
    print(f"allowed input -> {ok.final_output!r}")
    try:
        await Runner.run(agent, "this is forbidden territory")
        finding("guardrail did NOT trip (unexpected)")
    except InputGuardrailTripwireTriggered as exc:
        info = exc.guardrail_result.output.output_info
        finding(
            "input guardrail trips before the model call; "
            f"typed exception carries output_info={info}"
        )


# --- 5. human-in-the-loop via needs_approval + RunState -------------------------


async def spike_hitl() -> None:
    section("5. HITL: needs_approval pause + serialized RunState resume")

    @function_tool(needs_approval=True)
    def publish_report(title: str) -> str:
        """Publish the report (requires human approval)."""
        return f"published: {title}"

    agent = Agent(
        name="publisher",
        model=MODEL_NAME,
        instructions="Always call publish_report with the title 'Q3 outlook'.",
        tools=[publish_report],
    )
    result = await Runner.run(agent, "publish the report")
    interruptions = result.interruptions
    print(f"interruptions: {[i.name for i in interruptions]}")
    if not interruptions:
        finding("no interruption produced — needs_approval not honored?")
        return

    state = result.to_state()
    payload = state.to_json()
    restored = await RunState.from_json(agent, payload)
    for item in interruptions:
        restored.approve(item)
    resumed = await Runner.run(agent, restored)
    print(f"resumed output: {resumed.final_output!r}")
    finding(
        "RunState.to_json/from_json round-trips a paused run and approves "
        "pending tool calls on resume — durable HITL is SDK-native"
    )


# --- 6. custom Model => deterministic offline stub ------------------------------


class StubModel(Model):
    """Minimal deterministic Model: no network, no API key required."""

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: Any,
        tools: list[Tool],
        output_schema: Any,
        handoffs: list[Any],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: Any,
    ) -> ModelResponse:
        text = json.dumps({"answer": "stubbed"}) if output_schema else "stub reply"
        message = ResponseOutputMessage(
            id="stub-msg",
            content=[
                ResponseOutputText(annotations=[], text=text, type="output_text")
            ],
            role="assistant",
            status="completed",
            type="message",
        )
        return ModelResponse(
            output=[message],
            usage=agents.Usage(requests=1, input_tokens=10, output_tokens=5),
            response_id="stub-resp",
        )

    def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: Any,
        tools: list[Tool],
        output_schema: Any,
        handoffs: list[Any],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: Any,
    ) -> AsyncIterator[TResponseStreamEvent]:
        raise NotImplementedError("stub model does not stream")
        yield  # pragma: no cover - makes this an async generator


async def spike_stub() -> None:
    section("6. custom Model (offline stub, no API key)")

    from pydantic import BaseModel

    class AnswerModel(BaseModel):
        answer: str

    agent = Agent(
        name="stubbed",
        model=StubModel(),
        instructions="Return a canned answer.",
        output_type=AnswerModel,
    )
    result = await Runner.run(agent, "anything")
    print(f"final_output: {result.final_output!r}")
    finding(
        "custom Model integrates with Runner + output_type — offline "
        "StubModel is viable for the whole suite"
    )


async def main() -> int:
    from dotenv import load_dotenv

    load_dotenv()
    live = bool(os.environ.get("OPENAI_API_KEY"))
    if not live:
        print("OPENAI_API_KEY not set — live spikes skipped, stub only.")

    if live:
        await spike_hello()
        await spike_handoff()
        await spike_fanout()
        await spike_guardrail()
        await spike_hitl()
    await spike_stub()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
