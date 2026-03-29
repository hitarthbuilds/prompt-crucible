import argparse
import asyncio
import json
import pathlib
import typing

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from .store import ExperimentStore


BASE_DIR = pathlib.Path(__file__).resolve().parent
UI_DIR = BASE_DIR / "ui"
INDEX_HTML = UI_DIR / "index.html"
CRUCIBLE_DIR = pathlib.Path.cwd() / ".crucible"
TEST_SUITE_PATH = CRUCIBLE_DIR / "test_suite.json"
REVIEW_PATH = CRUCIBLE_DIR / "review.json"


class LiveState:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._connections: list[WebSocket] = []
        self.run_state: dict[str, typing.Any] = {
            "run_id": None,
            "model": None,
            "status": "idle",
            "started_at": None,
            "elapsed_seconds": 0.0,
            "current_prompt": "",
        }
        self.agent_active: dict[str, typing.Any] = {
            "agent": "idle",
            "status_text": "idle",
        }
        self.agent_logs: list[dict[str, typing.Any]] = []
        self.iterations: list[dict[str, typing.Any]] = []
        self.review_cases: list[dict[str, typing.Any]] = []
        self.review_index = 0

    async def reset(self) -> None:
        async with self._lock:
            self.run_state = {
                "run_id": None,
                "model": None,
                "status": "idle",
                "started_at": None,
                "elapsed_seconds": 0.0,
                "current_prompt": "",
            }
            self.agent_active = {
                "agent": "idle",
                "status_text": "idle",
            }
            self.agent_logs = []
            self.iterations = []
            self.review_cases = []
            self.review_index = 0

    async def snapshot(self) -> dict[str, typing.Any]:
        async with self._lock:
            return {
                "type": "snapshot",
                "run_state": dict(self.run_state),
                "agent_active": dict(self.agent_active),
                "agent_logs": list(self.agent_logs),
                "iterations": list(self.iterations),
                "review_cases": list(self.review_cases),
                "review_index": self.review_index,
            }

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.append(websocket)
        await websocket.send_json(await self.snapshot())

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            if websocket in self._connections:
                self._connections.remove(websocket)

    async def _broadcast(self, payload: dict[str, typing.Any]) -> None:
        async with self._lock:
            sockets = list(self._connections)

        dead: list[WebSocket] = []
        for websocket in sockets:
            try:
                await websocket.send_json(payload)
            except Exception:
                dead.append(websocket)

        if dead:
            async with self._lock:
                self._connections = [
                    websocket
                    for websocket in self._connections
                    if websocket not in dead
                ]

    async def broadcast_run_state(self, payload: dict[str, typing.Any]) -> None:
        async with self._lock:
            self.run_state.update(payload)
            message = {
                "type": "run_state",
                **self.run_state,
            }
        await self._broadcast(message)

    async def broadcast_agent_active(
        self,
        agent: str,
        status_text: str,
    ) -> None:
        async with self._lock:
            self.agent_active = {
                "agent": agent,
                "status_text": status_text,
            }
            message = {
                "type": "agent_active",
                "agent": agent,
                "status_text": status_text,
            }
        await self._broadcast(message)

    async def broadcast_agent_log(
        self,
        timestamp: str,
        message: str,
        level: str,
    ) -> None:
        entry = {
            "type": "agent_log",
            "timestamp": timestamp,
            "message": message,
            "level": level,
        }
        async with self._lock:
            self.agent_logs.insert(0, entry)
            self.agent_logs = self.agent_logs[:250]
        await self._broadcast(entry)

    async def broadcast_iteration(self, payload: dict[str, typing.Any]) -> None:
        message = {
            "type": "iteration",
            **payload,
        }
        async with self._lock:
            self.iterations.append(dict(message))
            self.run_state["current_prompt"] = payload.get("current_prompt", "")
        await self._broadcast(message)

    async def set_review_cases(
        self,
        cases: list[dict[str, typing.Any]],
        review_index: int = 0,
    ) -> None:
        async with self._lock:
            self.review_cases = list(cases)
            self.review_index = review_index
            payload = {
                "type": "review_state",
                "cases": list(self.review_cases),
                "review_index": self.review_index,
            }
        await self._broadcast(payload)


live_state = LiveState()
app = FastAPI(title="PromptCrucible UI")
app.mount("/ui", StaticFiles(directory=str(UI_DIR)), name="ui")


def _read_json_file(path: pathlib.Path) -> typing.Any:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(INDEX_HTML)


@app.get("/api/history")
async def api_history() -> list[dict[str, typing.Any]]:
    store = ExperimentStore()
    await store.init_db()
    return await store.get_experiments()


@app.get("/api/run/{run_id}")
async def api_run(run_id: str) -> dict[str, typing.Any]:
    store = ExperimentStore()
    await store.init_db()
    run = await store.resume_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@app.get("/api/cases/{run_id}/{iteration_number}")
async def api_cases(
    run_id: str,
    iteration_number: int,
) -> list[dict[str, typing.Any]]:
    store = ExperimentStore()
    await store.init_db()
    return await store.get_case_results(run_id, iteration_number)


@app.get("/api/review")
async def api_review() -> dict[str, typing.Any]:
    payload = _read_json_file(REVIEW_PATH)
    if not payload:
        payload = _read_json_file(TEST_SUITE_PATH)
    if isinstance(payload, dict):
        cases = payload.get("cases", [])
    elif isinstance(payload, list):
        cases = payload
    else:
        cases = []
    return {"cases": cases}


@app.post("/api/approve")
async def api_approve(request: Request) -> dict[str, typing.Any]:
    payload = await request.json()
    if isinstance(payload, dict):
        cases = payload.get("cases", [])
    elif isinstance(payload, list):
        cases = payload
    else:
        cases = []

    CRUCIBLE_DIR.mkdir(exist_ok=True)
    TEST_SUITE_PATH.write_text(
        json.dumps(cases, indent=2),
        encoding="utf-8",
    )
    return {"saved": len(cases)}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await live_state.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await live_state.disconnect(websocket)
    except Exception:
        await live_state.disconnect(websocket)


async def reset_live_state() -> None:
    await live_state.reset()


async def push_run_state(
    run_id: str | None = None,
    model: str | None = None,
    status: str | None = None,
    started_at: str | None = None,
    elapsed_seconds: float | None = None,
    current_prompt: str | None = None,
) -> None:
    payload: dict[str, typing.Any] = {}
    if run_id is not None:
        payload["run_id"] = run_id
    if model is not None:
        payload["model"] = model
    if status is not None:
        payload["status"] = status
    if started_at is not None:
        payload["started_at"] = started_at
    if elapsed_seconds is not None:
        payload["elapsed_seconds"] = elapsed_seconds
    if current_prompt is not None:
        payload["current_prompt"] = current_prompt
    await live_state.broadcast_run_state(payload)


async def push_agent_active(agent: str, status_text: str) -> None:
    await live_state.broadcast_agent_active(agent, status_text)


async def push_agent_log(timestamp: str, message: str, level: str) -> None:
    await live_state.broadcast_agent_log(timestamp, message, level)


async def push_iteration(payload: dict[str, typing.Any]) -> None:
    payload = dict(payload)
    payload["agent_log"] = list(live_state.agent_logs)
    await live_state.broadcast_iteration(payload)


async def push_review_cases(
    cases: list[dict[str, typing.Any]],
    review_index: int = 0,
) -> None:
    await live_state.set_review_cases(cases, review_index)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7777)
    args = parser.parse_args()
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=args.port,
        log_level="warning",
        access_log=False,
    )


if __name__ == "__main__":
    main()
