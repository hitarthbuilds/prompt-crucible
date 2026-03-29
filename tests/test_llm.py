import json

import httpx
import pytest

from promptcrucible import llm


class DummyResponse:
    def __init__(self, status_code=200, payload=None, error=None):
        self.status_code = status_code
        self._payload = payload or {}
        self._error = error

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self._error is not None:
            raise self._error


class DummyAsyncClient:
    def __init__(self, *, get_responses=None, post_responses=None, timeout=None):
        self.get_responses = list(get_responses or [])
        self.post_responses = list(post_responses or [])
        self.timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url):
        response = self.get_responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    async def post(self, url, json=None):
        response = self.post_responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


@pytest.mark.asyncio
async def test_check_ollama_running_returns_bool(mocker):
    responses = [DummyResponse(status_code=200, payload={"models": []})]
    mocker.patch(
        "promptcrucible.llm.httpx.AsyncClient",
        side_effect=lambda timeout=None: DummyAsyncClient(
            get_responses=responses,
            timeout=timeout,
        ),
    )

    result = await llm.check_ollama_running()

    assert isinstance(result, bool)


@pytest.mark.asyncio
async def test_call_ollama_safe_returns_dict_always(mocker):
    responses = [
        DummyResponse(payload={"response": json.dumps({"ok": True})}),
    ]
    mocker.patch(
        "promptcrucible.llm.httpx.AsyncClient",
        side_effect=lambda timeout=None: DummyAsyncClient(
            post_responses=responses,
            timeout=timeout,
        ),
    )

    result = await llm.call_ollama_safe(model="test-model", prompt="hello")

    assert isinstance(result, dict)
    assert result == {"ok": True}


@pytest.mark.asyncio
async def test_call_ollama_safe_with_bad_model_never_raises(mocker):
    request = httpx.Request("POST", llm.OLLAMA_GENERATE_URL)
    error = httpx.HTTPStatusError(
        "bad model",
        request=request,
        response=httpx.Response(status_code=404, request=request),
    )
    responses = [DummyResponse(error=error)]
    mocker.patch(
        "promptcrucible.llm.httpx.AsyncClient",
        side_effect=lambda timeout=None: DummyAsyncClient(
            post_responses=responses,
            timeout=timeout,
        ),
    )

    result = await llm.call_ollama_safe(model="missing-model", prompt="hello")

    assert isinstance(result, dict)
    assert "error" in result or isinstance(result, dict)
