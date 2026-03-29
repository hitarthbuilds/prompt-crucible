import asyncio
import json
import typing

import httpx


OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
JSON_CORRECTION = (
    "Your previous response was not valid JSON. Return ONLY a valid JSON "
    "object. No explanation, no markdown, no code fences. Raw JSON only."
)


async def call_ollama(
    model: str, prompt: str, system: str = None, timeout: int = 120
) -> str:
    payload: dict[str, typing.Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
    }
    if system is not None:
        payload["system"] = system

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(OLLAMA_GENERATE_URL, json=payload)
        response.raise_for_status()
        return str(response.json()["response"])


async def call_ollama_safe(
    model: str, prompt: str, system: str = None, timeout: int = 120
) -> dict:
    try:
        raw_response = await call_ollama(
            model=model,
            prompt=prompt,
            system=system,
            timeout=timeout,
        )
    except Exception as exc:
        return {"error": "request_failed", "raw": str(exc)}

    try:
        parsed = json.loads(raw_response)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    retry_prompt = prompt + "\n\n" + JSON_CORRECTION

    try:
        raw_response = await call_ollama(
            model=model,
            prompt=retry_prompt,
            system=system,
            timeout=timeout,
        )
    except Exception as exc:
        return {"error": "request_failed", "raw": str(exc)}

    try:
        parsed = json.loads(raw_response)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    return {"error": "parse_failed", "raw": raw_response}


async def check_ollama_running() -> bool:
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(OLLAMA_TAGS_URL)
        return response.status_code == 200
    except Exception:
        return False


async def check_model_available(model: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(OLLAMA_TAGS_URL)
        if response.status_code != 200:
            return False

        data = response.json()
        models = data.get("models", [])
        for item in models:
            if not isinstance(item, dict):
                continue
            if item.get("name") == model or item.get("model") == model:
                return True
        return False
    except Exception:
        return False


async def _main() -> None:
    running = await check_ollama_running()
    print(f"Ollama running: {running}")

    if running:
        available = await check_model_available("qwen2.5:7b")
        print(f"Model qwen2.5:7b available: {available}")

    print("LLM OK")


if __name__ == "__main__":
    asyncio.run(_main())
