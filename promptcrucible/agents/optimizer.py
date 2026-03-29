import asyncio
import json
import typing

from ..blackboard import Blackboard
from ..llm import call_ollama_safe


OPTIMIZER_PROMPT_TEMPLATE = """You are a prompt optimization engineer with memory of all past iterations.

Current prompt:
{current_prompt}

Diagnosed failures and root causes:
{failure_reasons}

Passing cases that must not regress:
{passing_cases}

Full iteration history (every prompt version tried, its score, what changed, and what failed):
{iteration_history}

Your job:
1. Fix the diagnosed failures with the smallest targeted change possible.
2. Check iteration_history — do NOT repeat any change that has already been attempted.
3. If a previous fix partially worked but didn't generalize, build on it rather than reverting it.
4. Do not add unnecessary rules. Do not break what already works.
5. If you are stuck (same failure class persists after 3+ attempts), try a fundamentally different approach.

Respond in JSON:
{{
  "new_prompt": "full rewritten prompt",
  "change_summary": "one sentence explaining what changed and why",
  "what_was_avoided": "what previously tried approaches you deliberately did not repeat",
  "approach": "incremental_fix | fundamental_rewrite",
  "confidence": 0.0-1.0
}}
"""


class OptimizerAgent:
    def __init__(
        self,
        model: str,
        llm_call: typing.Callable[..., typing.Awaitable[typing.Any]],
    ) -> None:
        self.model = model
        self.llm_call = llm_call

    @staticmethod
    def _fallback_prompt_candidate(current_prompt: typing.Any) -> dict[str, typing.Any]:
        return {
            "new_prompt": str(current_prompt or ""),
            "change_summary": "optimization failed, prompt unchanged",
            "what_was_avoided": "",
            "approach": "incremental_fix",
            "confidence": 0.0,
        }

    @staticmethod
    def _parse_confidence(value: typing.Any) -> float | None:
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_response(response: typing.Any) -> dict[str, typing.Any] | None:
        if isinstance(response, dict):
            if "error" in response:
                return None
            parsed = response
        elif isinstance(response, str):
            try:
                loaded = json.loads(response)
            except (TypeError, ValueError):
                return None
            if not isinstance(loaded, dict):
                return None
            parsed = loaded
        else:
            return None

        required_keys = {
            "new_prompt",
            "change_summary",
            "what_was_avoided",
            "approach",
            "confidence",
        }
        if not required_keys.issubset(parsed):
            return None

        approach = parsed.get("approach")
        if approach not in {"incremental_fix", "fundamental_rewrite"}:
            return None

        confidence = OptimizerAgent._parse_confidence(parsed.get("confidence"))
        if confidence is None:
            return None

        return {
            "new_prompt": str(parsed.get("new_prompt")),
            "change_summary": str(parsed.get("change_summary")),
            "what_was_avoided": str(parsed.get("what_was_avoided")),
            "approach": approach,
            "confidence": confidence,
        }

    async def run(self, blackboard: Blackboard) -> None:
        current_prompt = await blackboard.read("current_prompt")
        failure_reasons = await blackboard.read("failure_reasons")
        eval_results = await blackboard.read("eval_results")
        iteration_history = await blackboard.read("iteration_history")

        passing_cases = [
            case for case in (eval_results or []) if case.get("passed") is True
        ]
        failing_cases = [
            case for case in (eval_results or []) if case.get("passed") is False
        ]

        prompt = OPTIMIZER_PROMPT_TEMPLATE.format(
            current_prompt=current_prompt or "",
            failure_reasons=json.dumps(failure_reasons or {}, indent=2),
            passing_cases=json.dumps(passing_cases, indent=2),
            iteration_history=json.dumps(iteration_history or [], indent=2),
        )

        if not failing_cases:
            await blackboard.write(
                "prompt_candidate",
                self._fallback_prompt_candidate(current_prompt),
            )
            return

        response = await self.llm_call(
            model=self.model,
            prompt=prompt,
        )
        prompt_candidate = self._parse_response(response)
        if prompt_candidate is None:
            prompt_candidate = self._fallback_prompt_candidate(current_prompt)

        await blackboard.write("prompt_candidate", prompt_candidate)


async def _main() -> None:
    blackboard = Blackboard()

    await blackboard.write(
        "current_prompt",
        "Extract line number and content as JSON: {line_number: int, content: str}",
    )
    await blackboard.write(
        "failure_reasons",
        {
            "failure_class": "line_number_missing",
            "is_new_class": True,
            "previously_attempted": False,
            "previous_attempt_summary": None,
            "root_cause": "prompt does not handle section references like §3.2",
            "input_pattern": "lines starting with § symbol",
            "suggested_fix_direction": "add explicit rule for section reference lines",
        },
    )
    await blackboard.write(
        "eval_results",
        [
            {
                "case_id": "c1",
                "input": "line 3: Column A specs",
                "actual_output": {
                    "line_number": 3,
                    "content": "Column A specs",
                },
                "passed": True,
                "score": 1.0,
                "failure_reason": None,
            },
            {
                "case_id": "c2",
                "input": "§3.2 Load bearing capacity",
                "actual_output": {
                    "line_number": None,
                    "content": "Load bearing capacity",
                },
                "passed": False,
                "score": 0.2,
                "failure_reason": "line_number is null for section reference input",
            },
        ],
    )
    await blackboard.write("iteration_history", [])

    agent = OptimizerAgent(
        model="qwen2.5:7b",
        llm_call=call_ollama_safe,
    )
    await agent.run(blackboard)

    prompt_candidate = await blackboard.read("prompt_candidate")
    print(
        prompt_candidate["new_prompt"],
        prompt_candidate["change_summary"],
        prompt_candidate["approach"],
        prompt_candidate["confidence"],
    )

    if "new_prompt" in prompt_candidate:
        print("Optimizer OK")


if __name__ == "__main__":
    asyncio.run(_main())
