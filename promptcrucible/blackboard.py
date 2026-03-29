import asyncio
import typing


class Blackboard:
    _lock: typing.ClassVar[asyncio.Lock] = asyncio.Lock()

    def __init__(self) -> None:
        self._state: dict[str, typing.Any] = {}

    async def write(self, key: str, value: typing.Any) -> None:
        async with self.__class__._lock:
            self._state[key] = value

    async def read(self, key: str) -> typing.Any:
        async with self.__class__._lock:
            return self._state.get(key)

    async def read_all(self) -> dict[str, typing.Any]:
        async with self.__class__._lock:
            return dict(self._state)

    async def clear(self, key: str) -> None:
        async with self.__class__._lock:
            self._state.pop(key, None)

    async def reset(self) -> None:
        async with self.__class__._lock:
            self._state.clear()


async def _main() -> None:
    board = Blackboard()

    await board.write("current_prompt", "Refine the candidate prompt")
    await board.write("test_suite", ["case_1", "case_2"])
    await board.write("iteration_history", [{"iteration": 1, "score": 0.9}])

    assert await board.read("current_prompt") == "Refine the candidate prompt"
    assert await board.read("test_suite") == ["case_1", "case_2"]
    assert await board.read("iteration_history") == [{"iteration": 1, "score": 0.9}]

    state = await board.read_all()
    assert "current_prompt" in state
    assert "test_suite" in state
    assert "iteration_history" in state

    await board.clear("test_suite")
    assert await board.read("test_suite") is None
    assert "test_suite" not in await board.read_all()

    await board.reset()
    assert await board.read_all() == {}

    print("Blackboard OK")


if __name__ == "__main__":
    asyncio.run(_main())
