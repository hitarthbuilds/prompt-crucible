import pytest

from promptcrucible.agents.evaluator import EvaluatorAgent
from promptcrucible.blackboard import Blackboard


@pytest.fixture
def test_suite():
    return [
        {
            "case_id": "c1",
            "input": "input one",
            "expected_output": {"value": 1},
        },
        {
            "case_id": "c2",
            "input": "input two",
            "expected_output": {"value": 2},
        },
        {
            "case_id": "c3",
            "input": "input three",
            "expected_output": {"value": 3},
        },
    ]


@pytest.mark.asyncio
async def test_eval_results_written_to_blackboard(test_suite):
    board = Blackboard()
    await board.write("current_prompt", "Extract JSON")
    await board.write("test_suite", test_suite)

    responses = iter(
        [
            {"value": 1},
            {"passed": True, "score": 1.0, "reason": None},
            {"value": 2},
            {"passed": False, "score": 0.5, "reason": "partial mismatch"},
            {"error": "parse_failed", "raw": "bad"},
        ]
    )

    async def fake_llm_call(**kwargs):
        return next(responses)

    agent = EvaluatorAgent("model", "judge", fake_llm_call)
    await agent.run(board)

    eval_results = await board.read("eval_results")
    assert isinstance(eval_results, list)
    assert len(eval_results) == 3


@pytest.mark.asyncio
async def test_overall_score_is_mean_of_case_scores(test_suite):
    board = Blackboard()
    await board.write("current_prompt", "Extract JSON")
    await board.write("test_suite", test_suite[:2])

    responses = iter(
        [
            {"value": 1},
            {"passed": True, "score": 1.0, "reason": None},
            {"value": 2},
            {"passed": False, "score": 0.4, "reason": "wrong field"},
        ]
    )

    async def fake_llm_call(**kwargs):
        return next(responses)

    agent = EvaluatorAgent("model", "judge", fake_llm_call)
    await agent.run(board)

    assert await board.read("overall_score") == pytest.approx(0.7)


@pytest.mark.asyncio
async def test_parse_error_case_gets_zero_score_and_failed(test_suite):
    board = Blackboard()
    await board.write("current_prompt", "Extract JSON")
    await board.write("test_suite", [test_suite[0]])

    async def fake_llm_call(**kwargs):
        return {"error": "parse_failed", "raw": "not json"}

    agent = EvaluatorAgent("model", "judge", fake_llm_call)
    await agent.run(board)

    result = (await board.read("eval_results"))[0]
    assert result["score"] == 0.0
    assert result["passed"] is False
    assert result["failure_reason"] == "parse_failed"


@pytest.mark.asyncio
async def test_all_case_ids_appear_in_eval_results(test_suite):
    board = Blackboard()
    await board.write("current_prompt", "Extract JSON")
    await board.write("test_suite", test_suite)

    responses = iter(
        [
            {"value": 1},
            {"passed": True, "score": 1.0, "reason": None},
            {"value": 2},
            {"passed": True, "score": 1.0, "reason": None},
            {"value": 3},
            {"passed": True, "score": 1.0, "reason": None},
        ]
    )

    async def fake_llm_call(**kwargs):
        return next(responses)

    agent = EvaluatorAgent("model", "judge", fake_llm_call)
    await agent.run(board)

    returned_ids = [item["case_id"] for item in await board.read("eval_results")]
    assert returned_ids == ["c1", "c2", "c3"]
