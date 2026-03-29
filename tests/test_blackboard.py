import asyncio

import pytest

from promptcrucible.blackboard import Blackboard


@pytest.mark.asyncio
async def test_write_and_read_single_key():
    board = Blackboard()

    await board.write("alpha", 123)

    assert await board.read("alpha") == 123


@pytest.mark.asyncio
async def test_read_all_returns_all_keys():
    board = Blackboard()

    await board.write("a", 1)
    await board.write("b", {"x": 2})

    assert await board.read_all() == {"a": 1, "b": {"x": 2}}


@pytest.mark.asyncio
async def test_clear_removes_key():
    board = Blackboard()

    await board.write("temp", "value")
    await board.clear("temp")

    assert await board.read("temp") is None


@pytest.mark.asyncio
async def test_reset_wipes_all_state():
    board = Blackboard()

    await board.write("a", 1)
    await board.write("b", 2)
    await board.reset()

    assert await board.read_all() == {}


@pytest.mark.asyncio
async def test_concurrent_writes_do_not_corrupt_state():
    board = Blackboard()

    async def write_value(index: int) -> None:
        await board.write(f"k{index}", index)

    await asyncio.gather(*(write_value(index) for index in range(10)))

    state = await board.read_all()
    assert len(state) == 10
    for index in range(10):
        assert state[f"k{index}"] == index
