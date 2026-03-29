import asyncio
import datetime
import os
import typing

import aiosqlite


class ExperimentStore:
    def __init__(self, db_path: str = ".crucible/experiments.db") -> None:
        self.db_path = db_path

    @staticmethod
    def _now() -> str:
        return datetime.datetime.now(datetime.timezone.utc).isoformat()

    @staticmethod
    def _serialize(value: typing.Any) -> typing.Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    async def init_db(self) -> None:
        directory = os.path.dirname(self.db_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA foreign_keys = ON")
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  run_id TEXT UNIQUE,
                  started_at TIMESTAMP,
                  finished_at TIMESTAMP,
                  initial_prompt TEXT,
                  final_prompt TEXT,
                  initial_score REAL,
                  final_score REAL,
                  iterations INTEGER,
                  termination_reason TEXT,
                  model_used TEXT
                )
                """
            )
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS iterations (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  run_id TEXT,
                  iteration_number INTEGER,
                  prompt TEXT,
                  overall_score REAL,
                  pass_count INTEGER,
                  fail_count INTEGER,
                  change_summary TEXT,
                  failure_class TEXT,
                  created_at TIMESTAMP,
                  FOREIGN KEY (run_id) REFERENCES experiments(run_id)
                )
                """
            )
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS case_results (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  run_id TEXT,
                  iteration_number INTEGER,
                  case_id TEXT,
                  input_text TEXT,
                  expected_output TEXT,
                  actual_output TEXT,
                  passed BOOLEAN,
                  score REAL,
                  failure_reason TEXT,
                  FOREIGN KEY (run_id) REFERENCES experiments(run_id)
                )
                """
            )
            await db.commit()

    async def save_experiment(self, summary: dict) -> None:
        iteration_history = summary.get("iteration_history") or []
        initial_prompt = summary.get("initial_prompt")
        if initial_prompt is None and iteration_history:
            first_iteration = iteration_history[0]
            if isinstance(first_iteration, dict):
                initial_prompt = first_iteration.get("prompt")

        iterations = summary.get("iterations")
        if iterations is None:
            iterations = summary.get("total_iterations", 0)

        model_used = summary.get("model_used")
        if model_used is None:
            model_used = summary.get("model", "")

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO experiments (
                  run_id,
                  started_at,
                  finished_at,
                  initial_prompt,
                  final_prompt,
                  initial_score,
                  final_score,
                  iterations,
                  termination_reason,
                  model_used
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    summary.get("run_id"),
                    summary.get("started_at") or self._now(),
                    summary.get("finished_at") or self._now(),
                    initial_prompt,
                    summary.get("final_prompt"),
                    summary.get("initial_score"),
                    summary.get("final_score"),
                    iterations,
                    summary.get("termination_reason"),
                    model_used,
                ),
            )
            await db.commit()

    async def save_iteration(self, run_id: str, iteration: dict) -> None:
        iteration_number = iteration.get("iteration_number")
        if iteration_number is None:
            iteration_number = iteration.get("iteration")

        overall_score = iteration.get("overall_score")
        if overall_score is None:
            overall_score = iteration.get("score")

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO iterations (
                  run_id,
                  iteration_number,
                  prompt,
                  overall_score,
                  pass_count,
                  fail_count,
                  change_summary,
                  failure_class,
                  created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    iteration_number,
                    iteration.get("prompt"),
                    overall_score,
                    iteration.get("pass_count"),
                    iteration.get("fail_count"),
                    iteration.get("change_summary"),
                    iteration.get("failure_class"),
                    iteration.get("created_at") or self._now(),
                ),
            )
            await db.commit()

    async def save_case_results(
        self,
        run_id: str,
        iteration_number: int,
        cases: list,
    ) -> None:
        rows = []
        for case in cases:
            rows.append(
                (
                    run_id,
                    iteration_number,
                    case.get("case_id"),
                    case.get("input_text") or case.get("input"),
                    self._serialize(case.get("expected_output")),
                    self._serialize(case.get("actual_output")),
                    case.get("passed"),
                    case.get("score"),
                    case.get("failure_reason"),
                )
            )

        async with aiosqlite.connect(self.db_path) as db:
            await db.executemany(
                """
                INSERT INTO case_results (
                  run_id,
                  iteration_number,
                  case_id,
                  input_text,
                  expected_output,
                  actual_output,
                  passed,
                  score,
                  failure_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            await db.commit()

    async def get_experiments(self) -> list:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM experiments ORDER BY started_at DESC"
            ) as cursor:
                rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def get_iterations(self, run_id: str) -> list:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """
                SELECT * FROM iterations
                WHERE run_id = ?
                ORDER BY iteration_number
                """,
                (run_id,),
            ) as cursor:
                rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def get_case_results(self, run_id: str, iteration_number: int) -> list:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """
                SELECT * FROM case_results
                WHERE run_id = ? AND iteration_number = ?
                ORDER BY id
                """,
                (run_id, iteration_number),
            ) as cursor:
                rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def resume_run(self, run_id: str) -> dict | None:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM experiments WHERE run_id = ?",
                (run_id,),
            ) as cursor:
                experiment = await cursor.fetchone()

            if experiment is None:
                return None

            async with db.execute(
                """
                SELECT * FROM iterations
                WHERE run_id = ?
                ORDER BY iteration_number
                """,
                (run_id,),
            ) as cursor:
                iterations = await cursor.fetchall()

        return {
            "experiment": dict(experiment),
            "iterations": [dict(row) for row in iterations],
        }


async def _main() -> None:
    db_path = ".crucible/test_experiments.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    store = ExperimentStore(db_path=db_path)

    try:
        await store.init_db()
        await store.save_experiment(
            {
                "run_id": "test-001",
                "started_at": "2026-03-29T10:00:00+00:00",
                "finished_at": "2026-03-29T10:05:00+00:00",
                "initial_prompt": "Extract line number and content",
                "final_prompt": "Extract line number and content as strict JSON",
                "initial_score": 0.42,
                "final_score": 0.94,
                "iterations": 2,
                "termination_reason": "threshold_reached",
                "model_used": "qwen2.5:7b",
            }
        )
        await store.save_iteration(
            "test-001",
            {
                "iteration": 1,
                "prompt": "Extract line number and content",
                "score": 0.42,
                "pass_count": 1,
                "fail_count": 2,
                "change_summary": "Initial baseline run",
                "failure_class": "line_number_missing",
                "created_at": "2026-03-29T10:01:00+00:00",
            },
        )
        await store.save_iteration(
            "test-001",
            {
                "iteration": 2,
                "prompt": "Extract line number and content as strict JSON",
                "score": 0.94,
                "pass_count": 3,
                "fail_count": 0,
                "change_summary": "Added explicit rule for section references",
                "failure_class": None,
                "created_at": "2026-03-29T10:04:00+00:00",
            },
        )
        await store.save_case_results(
            "test-001",
            1,
            [
                {
                    "case_id": "c1",
                    "input": "line 3: Column A specs",
                    "expected_output": {
                        "line_number": 3,
                        "content": "Column A specs",
                    },
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
                    "expected_output": {
                        "line_number": 7,
                        "content": "Load bearing capacity",
                    },
                    "actual_output": {
                        "line_number": None,
                        "content": "Load bearing capacity",
                    },
                    "passed": False,
                    "score": 0.2,
                    "failure_reason": "line_number is null",
                },
                {
                    "case_id": "c3",
                    "input": "line 8: Material schedule",
                    "expected_output": {
                        "line_number": 8,
                        "content": "Material schedule",
                    },
                    "actual_output": {
                        "line_number": 8,
                        "content": "Material schedule",
                    },
                    "passed": True,
                    "score": 1.0,
                    "failure_reason": None,
                },
            ],
        )

        experiments = await store.get_experiments()
        iterations = await store.get_iterations("test-001")
        case_results = await store.get_case_results("test-001", 1)
        resumed = await store.resume_run("test-001")

        print(len(experiments))
        print(len(iterations))
        print(resumed is not None)

        assert len(experiments) == 1
        assert len(iterations) == 2
        assert len(case_results) == 3
        assert resumed is not None

        print("Store OK")
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


if __name__ == "__main__":
    asyncio.run(_main())
