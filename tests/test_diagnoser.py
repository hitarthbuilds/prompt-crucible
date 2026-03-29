import pytest

from promptcrucible.agents.diagnoser import DiagnoserAgent
from promptcrucible.blackboard import Blackboard


@pytest.mark.asyncio
async def test_failure_reasons_written_to_blackboard():
    board = Blackboard()
    await board.write("current_prompt", "Extract data")
    await board.write(
        "eval_results",
        [
            {
                "case_id": "c1",
                "passed": False,
                "score": 0.2,
                "failure_reason": "bad parse",
            }
        ],
    )
    await board.write("iteration_history", [])

    async def fake_llm_call(**kwargs):
        return {
            "failure_class": "format_error",
            "is_new_class": True,
            "previously_attempted": False,
            "previous_attempt_summary": None,
            "root_cause": "prompt too vague",
            "input_pattern": "messy inputs",
            "suggested_fix_direction": "add explicit formatting rule",
        }

    agent = DiagnoserAgent("judge", fake_llm_call)
    await agent.run(board)

    assert (await board.read("failure_reasons"))["failure_class"] == "format_error"


@pytest.mark.asyncio
async def test_no_failing_cases_writes_empty_failure_reasons():
    board = Blackboard()
    await board.write("current_prompt", "Extract data")
    await board.write(
        "eval_results",
        [{"case_id": "c1", "passed": True, "score": 1.0, "failure_reason": None}],
    )
    await board.write("iteration_history", [{"iteration": 1, "score": 0.9}])

    called = False

    async def fake_llm_call(**kwargs):
        nonlocal called
        called = True
        return {}

    agent = DiagnoserAgent("judge", fake_llm_call)
    await agent.run(board)

    assert await board.read("failure_reasons") == {}
    assert called is False


@pytest.mark.asyncio
async def test_iteration_history_is_passed_to_llm_prompt():
    board = Blackboard()
    history = [{"iteration": 1, "score": 0.45, "failure_class": "x"}]
    await board.write("current_prompt", "Extract data")
    await board.write(
        "eval_results",
        [{"case_id": "c1", "passed": False, "score": 0.2, "failure_reason": "bad parse"}],
    )
    await board.write("iteration_history", history)

    captured = {}

    async def fake_llm_call(**kwargs):
        captured.update(kwargs)
        return {
            "failure_class": "format_error",
            "is_new_class": False,
            "previously_attempted": True,
            "previous_attempt_summary": "already tried",
            "root_cause": "prompt too vague",
            "input_pattern": "messy inputs",
            "suggested_fix_direction": "be explicit",
        }

    agent = DiagnoserAgent("judge", fake_llm_call)
    await agent.run(board)

    assert '"iteration": 1' in captured["prompt"]
