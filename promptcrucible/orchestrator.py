import asyncio
import typing
import uuid

from .blackboard import Blackboard
from .llm import call_ollama_safe
from .agents.diagnoser import DiagnoserAgent
from .agents.evaluator import EvaluatorAgent
from .agents.optimizer import OptimizerAgent


class Orchestrator:
    def __init__(
        self,
        model: str,
        judge_model: str,
        threshold: float = 0.90,
        max_iter: int = 10,
        timeout: int = 120,
    ) -> None:
        self.model = model
        self.judge_model = judge_model
        self.threshold = threshold
        self.max_iter = max_iter
        self.timeout = timeout
        self.evaluator = EvaluatorAgent(
            model=self.model,
            judge_model=self.judge_model,
            llm_call=self._llm_call,
        )
        self.diagnoser = DiagnoserAgent(
            judge_model=self.judge_model,
            llm_call=self._llm_call,
        )
        self.optimizer = OptimizerAgent(
            model=self.model,
            llm_call=self._llm_call,
        )

    async def _llm_call(
        self,
        model: str,
        prompt: str,
        system: str = None,
    ) -> typing.Any:
        return await call_ollama_safe(
            model=model,
            prompt=prompt,
            system=system,
            timeout=self.timeout,
        )

    @staticmethod
    def _count_cases(
        eval_results: list[dict[str, typing.Any]],
    ) -> tuple[int, int]:
        pass_count = sum(1 for case in eval_results if case.get("passed") is True)
        fail_count = sum(1 for case in eval_results if case.get("passed") is False)
        return pass_count, fail_count

    @staticmethod
    def _is_plateau(
        iteration_history: list[dict[str, typing.Any]],
    ) -> bool:
        if len(iteration_history) < 4:
            return False

        scores = [float(item.get("score", 0.0)) for item in iteration_history]
        window_scores = scores[-3:]
        best_before_window = max(scores[:-3]) if scores[:-3] else None

        return (
            best_before_window is not None
            and max(window_scores) - min(window_scores) < 0.02
            and max(window_scores) <= best_before_window
        )

    @staticmethod
    def _is_stuck(
        iteration_history: list[dict[str, typing.Any]],
    ) -> bool:
        if len(iteration_history) < 5:
            return False

        last_class = iteration_history[-1].get("failure_class")
        if last_class in (None, "", "unknown"):
            return False

        streak: list[dict[str, typing.Any]] = []
        index = len(iteration_history) - 1
        while index >= 0:
            item = iteration_history[index]
            if item.get("failure_class") != last_class:
                break
            streak.append(item)
            index -= 1

        streak.reverse()
        if len(streak) < 5:
            return False

        scores = [float(item.get("score", 0.0)) for item in streak]
        return max(scores) <= scores[0]

    async def run(self, blackboard: Blackboard) -> dict[str, typing.Any]:
        run_id = str(uuid.uuid4())
        iteration_history = await blackboard.read("iteration_history")
        if iteration_history is None:
            iteration_history = []
            await blackboard.write("iteration_history", iteration_history)

        initial_score: float | None = None
        final_score = 0.0
        termination_reason = "max_iterations"

        for iteration in range(1, self.max_iter + 1):
            await self.evaluator.run(blackboard)
            await self.diagnoser.run(blackboard)
            await self.optimizer.run(blackboard)

            eval_results = await blackboard.read("eval_results") or []
            overall_score = float(await blackboard.read("overall_score") or 0.0)
            failure_reasons = await blackboard.read("failure_reasons") or {}
            prompt_candidate = await blackboard.read("prompt_candidate") or {}
            current_prompt = await blackboard.read("current_prompt") or ""

            if initial_score is None:
                initial_score = overall_score
            final_score = overall_score

            new_prompt = prompt_candidate.get("new_prompt", current_prompt)
            await blackboard.write("current_prompt", new_prompt)

            pass_count, fail_count = self._count_cases(eval_results)
            iteration_record = {
                "iteration": iteration,
                "prompt": new_prompt,
                "score": overall_score,
                "pass_count": pass_count,
                "fail_count": fail_count,
                "failure_class": failure_reasons.get("failure_class"),
                "is_new_class": failure_reasons.get("is_new_class", False),
                "change_summary": prompt_candidate.get("change_summary", ""),
                "what_was_avoided": prompt_candidate.get("what_was_avoided", ""),
                "approach": prompt_candidate.get("approach", "incremental_fix"),
            }

            iteration_history = await blackboard.read("iteration_history") or []
            iteration_history = list(iteration_history)
            iteration_history.append(iteration_record)
            await blackboard.write("iteration_history", iteration_history)

            if overall_score >= self.threshold:
                signal = "threshold_reached"
            elif self._is_plateau(iteration_history):
                signal = "plateau"
            elif self._is_stuck(iteration_history):
                signal = "stuck"
            elif iteration >= self.max_iter:
                signal = "max_iterations"
            else:
                signal = "continue"

            termination_reason = signal
            await blackboard.write("loop_signal", signal)

            print(
                f"Iteration {iteration} | Score: {overall_score:.2f} | "
                f"Failures: {fail_count} | Class: "
                f"{iteration_record['failure_class']} | Signal: {signal}"
            )

            if signal != "continue":
                break

        final_prompt = await blackboard.read("current_prompt") or ""
        iteration_history = await blackboard.read("iteration_history") or []

        return {
            "run_id": run_id,
            "total_iterations": len(iteration_history),
            "initial_score": 0.0 if initial_score is None else initial_score,
            "final_score": final_score,
            "final_prompt": final_prompt,
            "termination_reason": termination_reason,
            "iteration_history": iteration_history,
        }


async def _main() -> None:
    blackboard = Blackboard()

    await blackboard.write(
        "current_prompt",
        "Extract line number and content as JSON: {line_number: int, content: str}",
    )
    await blackboard.write(
        "test_suite",
        [
            {
                "case_id": "c1",
                "input": "line 3: Column A specs",
                "expected_output": {
                    "line_number": 3,
                    "content": "Column A specs",
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
            {
                "case_id": "c3",
                "input": "§3.2 line 11: Load bearing capacity",
                "expected_output": {
                    "line_number": 11,
                    "content": "Load bearing capacity",
                },
            },
        ],
    )

    orchestrator = Orchestrator(
        model="qwen2.5:7b",
        judge_model="phi3:mini",
        threshold=0.90,
        max_iter=3,
    )
    summary = await orchestrator.run(blackboard)

    print(summary)

    if "final_prompt" in summary and "termination_reason" in summary:
        print("Orchestrator OK")


if __name__ == "__main__":
    asyncio.run(_main())
