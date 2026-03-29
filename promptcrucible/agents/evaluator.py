import asyncio
import typing

from ..blackboard import Blackboard
from ..llm import call_ollama_safe


SEMANTIC_JUDGE_PROMPT_TEMPLATE = """You are a semantic correctness judge for structured extraction outputs.

Input text: {input}
Expected output: {expected_output}
Actual output: {actual_output}

Is the actual output semantically correct?
Respond in JSON:
{{
  "passed": true/false,
  "score": 0.0-1.0,
  "reason": "one sentence explanation if failed, null if passed"
}}
"""


class EvaluatorAgent:
    def __init__(
        self,
        model: str,
        judge_model: str,
        llm_call: typing.Callable[..., typing.Awaitable[dict]],
    ) -> None:
        self.model = model
        self.judge_model = judge_model
        self.llm_call = llm_call

    @staticmethod
    def _is_error_result(result: typing.Any) -> bool:
        return (
            isinstance(result, dict)
            and result.get("error") in {"parse_failed", "request_failed"}
        )

    @staticmethod
    def _as_bool(value: typing.Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() == "true"
        return bool(value)

    @staticmethod
    def _as_float(value: typing.Any) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    async def run(self, blackboard: Blackboard) -> None:
        current_prompt = await blackboard.read("current_prompt")
        test_suite = await blackboard.read("test_suite")

        eval_results: list[dict[str, typing.Any]] = []
        total_score = 0.0

        for index, case in enumerate(test_suite or []):
            case_id = case.get("case_id", f"case_{index + 1}")
            case_input = case.get("input", "")
            expected_output = case.get("expected_output")

            extraction_prompt = f"{current_prompt}\n\n{case_input}"
            extraction_result = await self.llm_call(
                model=self.model,
                prompt=extraction_prompt,
            )

            if self._is_error_result(extraction_result):
                eval_results.append(
                    {
                        "case_id": case_id,
                        "actual_output": extraction_result.get("raw"),
                        "score": 0.0,
                        "passed": False,
                        "failure_reason": "parse_failed",
                    }
                )
                continue

            judge_prompt = SEMANTIC_JUDGE_PROMPT_TEMPLATE.format(
                input=case_input,
                expected_output=expected_output,
                actual_output=extraction_result,
            )
            judge_result = await self.llm_call(
                model=self.judge_model,
                prompt=judge_prompt,
            )

            if self._is_error_result(judge_result):
                passed = False
                score = 0.0
                failure_reason = "parse_failed"
            else:
                passed = self._as_bool(judge_result.get("passed", False))
                score = self._as_float(judge_result.get("score", 0.0))
                reason = judge_result.get("reason")
                failure_reason = None if passed else (
                    str(reason) if reason not in (None, "") else "judge_failed"
                )

            total_score += score
            eval_results.append(
                {
                    "case_id": case_id,
                    "actual_output": extraction_result,
                    "score": score,
                    "passed": passed,
                    "failure_reason": failure_reason,
                }
            )

        overall_score = total_score / len(eval_results) if eval_results else 0.0

        await blackboard.write("eval_results", eval_results)
        await blackboard.write("overall_score", overall_score)


async def _main() -> None:
    blackboard = Blackboard()

    await blackboard.write(
        "current_prompt",
        "Extract the line number and content. Return JSON: {line_number: int, content: str}",
    )
    await blackboard.write(
        "test_suite",
        [
            {
                "case_id": "c1",
                "input": "line 3: Material specifications for column A",
                "expected_output": {
                    "line_number": 3,
                    "content": "Material specifications for column A",
                },
            },
            {
                "case_id": "c2",
                "input": "line 7: Load bearing capacity 450kN",
                "expected_output": {
                    "line_number": 7,
                    "content": "Load bearing capacity 450kN",
                },
            },
        ],
    )

    agent = EvaluatorAgent(
        model="qwen2.5:7b",
        judge_model="phi3:mini",
        llm_call=call_ollama_safe,
    )
    await agent.run(blackboard)

    eval_results = await blackboard.read("eval_results")
    overall_score = await blackboard.read("overall_score")

    for result in eval_results:
        print(
            result["case_id"],
            result["passed"],
            result["score"],
            result["failure_reason"],
        )

    print(overall_score)

    if len(eval_results) == 2:
        print("Evaluator OK")


if __name__ == "__main__":
    asyncio.run(_main())
