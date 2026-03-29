import sqlite3

import pytest

from promptcrucible.store import ExperimentStore


@pytest.mark.asyncio
async def test_init_db_creates_all_tables(tmp_path):
    db_path = tmp_path / "experiments.db"
    store = ExperimentStore(str(db_path))

    await store.init_db()

    connection = sqlite3.connect(db_path)
    try:
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    finally:
        connection.close()

    table_names = {name for (name,) in rows}
    assert {"experiments", "iterations", "case_results"}.issubset(table_names)


@pytest.mark.asyncio
async def test_save_experiment_and_get_experiments_round_trip(tmp_path):
    db_path = tmp_path / "experiments.db"
    store = ExperimentStore(str(db_path))
    await store.init_db()

    await store.save_experiment(
        {
            "run_id": "run-1",
            "started_at": "2025-01-01T00:00:00+00:00",
            "finished_at": "2025-01-01T00:01:00+00:00",
            "initial_prompt": "start",
            "final_prompt": "end",
            "initial_score": 0.2,
            "final_score": 0.9,
            "iterations": 2,
            "termination_reason": "threshold_reached",
            "model_used": "qwen",
        }
    )

    experiments = await store.get_experiments()

    assert len(experiments) == 1
    assert experiments[0]["run_id"] == "run-1"


@pytest.mark.asyncio
async def test_save_iteration_and_get_iterations_round_trip(tmp_path):
    db_path = tmp_path / "experiments.db"
    store = ExperimentStore(str(db_path))
    await store.init_db()
    await store.save_experiment({"run_id": "run-1", "final_prompt": "end"})

    await store.save_iteration(
        "run-1",
        {
            "iteration": 1,
            "prompt": "p1",
            "score": 0.5,
            "pass_count": 2,
            "fail_count": 1,
            "change_summary": "change",
            "failure_class": "fmt",
        },
    )

    iterations = await store.get_iterations("run-1")

    assert len(iterations) == 1
    assert iterations[0]["iteration_number"] == 1
    assert iterations[0]["prompt"] == "p1"


@pytest.mark.asyncio
async def test_save_case_results_and_get_case_results_round_trip(tmp_path):
    db_path = tmp_path / "experiments.db"
    store = ExperimentStore(str(db_path))
    await store.init_db()
    await store.save_experiment({"run_id": "run-1", "final_prompt": "end"})

    await store.save_case_results(
        "run-1",
        1,
        [
            {
                "case_id": "c1",
                "input": "input text",
                "expected_output": {"a": 1},
                "actual_output": {"a": 1},
                "passed": True,
                "score": 1.0,
                "failure_reason": None,
            }
        ],
    )

    case_results = await store.get_case_results("run-1", 1)

    assert len(case_results) == 1
    assert case_results[0]["case_id"] == "c1"
    assert case_results[0]["input_text"] == "input text"


@pytest.mark.asyncio
async def test_resume_run_returns_none_for_unknown_run_id(tmp_path):
    db_path = tmp_path / "experiments.db"
    store = ExperimentStore(str(db_path))
    await store.init_db()

    assert await store.resume_run("missing") is None


@pytest.mark.asyncio
async def test_resume_run_returns_data_for_known_run_id(tmp_path):
    db_path = tmp_path / "experiments.db"
    store = ExperimentStore(str(db_path))
    await store.init_db()
    await store.save_experiment({"run_id": "run-1", "final_prompt": "end"})
    await store.save_iteration(
        "run-1",
        {
            "iteration": 1,
            "prompt": "p1",
            "score": 0.5,
            "pass_count": 2,
            "fail_count": 1,
            "change_summary": "change",
            "failure_class": "fmt",
        },
    )

    resumed = await store.resume_run("run-1")

    assert resumed is not None
    assert resumed["experiment"]["run_id"] == "run-1"
    assert len(resumed["iterations"]) == 1
