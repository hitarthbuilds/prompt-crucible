import pytest

from promptcrucible.agents.optimizer import OptimizerAgent
from promptcrucible.blackboard import Blackboard


@pytest.mark.asyncio
async def test_prompt_candidate_written_with_new_prompt_key():
    board = Blackboard()
    await board.write("current_prompt", "Original prompt")
    await board.write(
        "failure_reasons",
        {
            "failure_class": "line_number_missing",
            "root_cause": "too vague",
        },
    )
    await board.write(
        "eval_results",
        [{"case_id": "c1", "passed": False, "score": 0.2}],
    )
    await board.write("iteration_history", [])

    async def fake_llm_call(**kwargs):
        return {
            "new_prompt": "Improved prompt",
            "change_summary": "Added explicit rule",
            "what_was_avoided": "No regex-only fix",
            "approach": "incremental_fix",
            "confidence": 0.88,
        }

    agent = OptimizerAgent("model", fake_llm_call)
    await agent.run(board)

    candidate = await board.read("prompt_candidate")
    assert "new_prompt" in candidate
    assert candidate["new_prompt"] == "Improved prompt"


@pytest.mark.asyncio
async def test_llm_error_keeps_current_prompt_unchanged():
    board = Blackboard()
    await board.write("current_prompt", "Original prompt")
    await board.write("failure_reasons", {"failure_class": "x"})
    await board.write("eval_results", [{"case_id": "c1", "passed": False, "score": 0.2}])
    await board.write("iteration_history", [])

    async def fake_llm_call(**kwargs):
        return {"error": "request_failed", "raw": "timeout"}

    agent = OptimizerAgent("model", fake_llm_call)
    await agent.run(board)

    candidate = await board.read("prompt_candidate")
    assert candidate["new_prompt"] == "Original prompt"


@pytest.mark.asyncio
async def test_iteration_history_is_passed_to_llm_prompt():
    board = Blackboard()
    await board.write("current_prompt", "Original prompt")
    await board.write("failure_reasons", {"failure_class": "x"})
    await board.write("eval_results", [{"case_id": "c1", "passed": False, "score": 0.2}])
    await board.write("iteration_history", [{"iteration": 1, "score": 0.4}])

    captured = {}

    async def fake_llm_call(**kwargs):
        captured.update(kwargs)
        return {
            "new_prompt": "Improved prompt",
            "change_summary": "Added explicit rule",
            "what_was_avoided": "No regex-only fix",
            "approach": "incremental_fix",
            "confidence": 0.88,
        }

    agent = OptimizerAgent("model", fake_llm_call)
    await agent.run(board)

    assert '"iteration": 1' in captured["prompt"]
