import pytest

from promptcrucible.blackboard import Blackboard
from promptcrucible.orchestrator import Orchestrator


class FakeEvaluator:
    def __init__(self, scores, history_lengths=None):
        self.scores = list(scores)
        self.calls = 0
        self.history_lengths = history_lengths if history_lengths is not None else []

    async def run(self, blackboard):
        self.history_lengths.append(len(await blackboard.read("iteration_history") or []))
        score = self.scores[min(self.calls, len(self.scores) - 1)]
        self.calls += 1
        passed = score >= 0.5
        await blackboard.write(
            "eval_results",
            [
                {
                    "case_id": f"c{self.calls}",
                    "actual_output": {"ok": True},
                    "score": score,
                    "passed": passed,
                    "failure_reason": None if passed else "failed",
                }
            ],
        )
        await blackboard.write("overall_score", score)


class FakeDiagnoser:
    async def run(self, blackboard):
        eval_results = await blackboard.read("eval_results") or []
        failed = any(case.get("passed") is False for case in eval_results)
        await blackboard.write(
            "failure_reasons",
            {}
            if not failed
            else {
                "failure_class": "same_failure",
                "is_new_class": False,
            },
        )


class FakeOptimizer:
    def __init__(self):
        self.calls = 0

    async def run(self, blackboard):
        self.calls += 1
        current_prompt = await blackboard.read("current_prompt") or ""
        await blackboard.write(
            "prompt_candidate",
            {
                "new_prompt": f"{current_prompt} [iter {self.calls}]",
                "change_summary": f"change {self.calls}",
                "what_was_avoided": "none",
                "approach": "incremental_fix",
                "confidence": 0.9,
            },
        )


@pytest.mark.asyncio
async def test_iteration_history_grows_by_one_each_iteration():
    board = Blackboard()
    await board.write("current_prompt", "Prompt")
    await board.write("iteration_history", [])

    seen_lengths = []
    orchestrator = Orchestrator("model", "judge", threshold=0.95, max_iter=3)
    orchestrator.evaluator = FakeEvaluator([0.3, 0.3, 0.3], seen_lengths)
    orchestrator.diagnoser = FakeDiagnoser()
    orchestrator.optimizer = FakeOptimizer()

    summary = await orchestrator.run(board)

    assert seen_lengths == [0, 1, 2]
    assert len(summary["iteration_history"]) == 3


@pytest.mark.asyncio
async def test_threshold_reached_termination():
    board = Blackboard()
    await board.write("current_prompt", "Prompt")

    orchestrator = Orchestrator("model", "judge", threshold=0.90, max_iter=5)
    orchestrator.evaluator = FakeEvaluator([0.95])
    orchestrator.diagnoser = FakeDiagnoser()
    orchestrator.optimizer = FakeOptimizer()

    summary = await orchestrator.run(board)

    assert summary["termination_reason"] == "threshold_reached"
    assert summary["total_iterations"] == 1


@pytest.mark.asyncio
async def test_max_iterations_termination():
    board = Blackboard()
    await board.write("current_prompt", "Prompt")

    orchestrator = Orchestrator("model", "judge", threshold=0.90, max_iter=3)
    orchestrator.evaluator = FakeEvaluator([0.3, 0.3, 0.3])
    orchestrator.diagnoser = FakeDiagnoser()
    orchestrator.optimizer = FakeOptimizer()

    summary = await orchestrator.run(board)

    assert summary["termination_reason"] == "max_iterations"
    assert summary["total_iterations"] == 3


@pytest.mark.asyncio
async def test_plateau_detection():
    board = Blackboard()
    await board.write("current_prompt", "Prompt")
    await board.write(
        "iteration_history",
        [
            {
                "iteration": 0,
                "prompt": "baseline",
                "score": 0.8,
                "pass_count": 0,
                "fail_count": 1,
                "failure_class": "same_failure",
                "is_new_class": False,
                "change_summary": "baseline",
                "what_was_avoided": "",
                "approach": "incremental_fix",
            }
        ],
    )

    orchestrator = Orchestrator("model", "judge", threshold=0.95, max_iter=5)
    orchestrator.evaluator = FakeEvaluator([0.72, 0.71, 0.72, 0.72])
    orchestrator.diagnoser = FakeDiagnoser()
    orchestrator.optimizer = FakeOptimizer()

    summary = await orchestrator.run(board)

    assert summary["termination_reason"] == "plateau"
