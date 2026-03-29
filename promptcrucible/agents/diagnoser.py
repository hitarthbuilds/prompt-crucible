import asyncio
import json
import typing

from ..blackboard import Blackboard
from ..llm import call_ollama_safe


DIAGNOSER_PROMPT_TEMPLATE = """You are a prompt failure analyst with memory of past iterations.

Current prompt:
{current_prompt}

Failing cases this iteration:
{failing_cases}

Passing cases (for contrast):
{passing_cases}

Full iteration history (all previous attempts, scores, failure classes):
{iteration_history}

Your job:
1. Analyze why the prompt fails on these inputs.
2. Check iteration_history — has this failure class appeared before?
   If yes, the previous fix attempt did not generalize. Explain why.
   If no, this is a new failure class.
3. Identify what is structurally different about failing inputs.

Respond in JSON:
{{
  "failure_class": "short label",
  "is_new_class": true/false,
  "previously_attempted": true/false,
  "previous_attempt_summary": "what was tried before and why it failed to generalize, or null if new",
  "root_cause": "one sentence",
  "input_pattern": "what these failing inputs have in common",
  "suggested_fix_direction": "what kind of rule or change would fix this, different from what was tried before if applicable"
}}
"""


FALLBACK_FAILURE_REASONS = {
    "failure_class": "unknown",
    "is_new_class": True,
    "previously_attempted": False,
    "previous_attempt_summary": None,
    "root_cause": "diagnosis failed",
    "input_pattern": "unknown",
    "suggested_fix_direction": "retry",
}


class DiagnoserAgent:
    def __init__(
        self,
        judge_model: str,
        llm_call: typing.Callable[..., typing.Awaitable[typing.Any]],
    ) -> None:
        self.judge_model = judge_model
        self.llm_call = llm_call

    @staticmethod
    def _normalize_bool(value: typing.Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() == "true"
        return bool(value)

    @staticmethod
    def _fallback() -> dict[str, typing.Any]:
        return dict(FALLBACK_FAILURE_REASONS)

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
            "failure_class",
            "is_new_class",
            "previously_attempted",
            "previous_attempt_summary",
            "root_cause",
            "input_pattern",
            "suggested_fix_direction",
        }
        if not required_keys.issubset(parsed):
            return None

        previous_attempt_summary = parsed.get("previous_attempt_summary")
        if previous_attempt_summary is not None:
            previous_attempt_summary = str(previous_attempt_summary)

        return {
            "failure_class": str(parsed.get("failure_class")),
            "is_new_class": DiagnoserAgent._normalize_bool(
                parsed.get("is_new_class")
            ),
            "previously_attempted": DiagnoserAgent._normalize_bool(
                parsed.get("previously_attempted")
            ),
            "previous_attempt_summary": previous_attempt_summary,
            "root_cause": str(parsed.get("root_cause")),
            "input_pattern": str(parsed.get("input_pattern")),
            "suggested_fix_direction": str(parsed.get("suggested_fix_direction")),
        }

    async def run(self, blackboard: Blackboard) -> None:
        current_prompt = await blackboard.read("current_prompt")
        eval_results = await blackboard.read("eval_results")
        iteration_history = await blackboard.read("iteration_history")

        failing_cases = [
            case for case in (eval_results or []) if case.get("passed") is False
        ]
        passing_cases = [
            case for case in (eval_results or []) if case.get("passed") is True
        ]

        if not failing_cases:
            await blackboard.write("failure_reasons", {})
            return

        prompt = DIAGNOSER_PROMPT_TEMPLATE.format(
            current_prompt=current_prompt or "",
            failing_cases=json.dumps(failing_cases, indent=2),
            passing_cases=json.dumps(passing_cases, indent=2),
            iteration_history=json.dumps(iteration_history or [], indent=2),
        )

        response = await self.llm_call(
            model=self.judge_model,
            prompt=prompt,
        )
        failure_reasons = self._parse_response(response)
        if failure_reasons is None:
            failure_reasons = self._fallback()

        await blackboard.write("failure_reasons", failure_reasons)


async def _main() -> None:
    blackboard = Blackboard()

    await blackboard.write(
        "current_prompt",
        "Extract line number and content as JSON",
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

    agent = DiagnoserAgent(
        judge_model="phi3:mini",
        llm_call=call_ollama_safe,
    )
    await agent.run(blackboard)

    failure_reasons = await blackboard.read("failure_reasons")
    print(
        failure_reasons["failure_class"],
        failure_reasons["is_new_class"],
        failure_reasons["root_cause"],
        failure_reasons["suggested_fix_direction"],
    )

    if failure_reasons is not None:
        print("Diagnoser OK")


if __name__ == "__main__":
    asyncio.run(_main())
