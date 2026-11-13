"""Direct contract tests for the OpenAI-compatible loopback fixture.

The fixture's streaming path (``_send_sse``) is driven here over real HTTP so CI
covers the ``chat.completion.chunk`` SSE contract even when the ``opencode``
CLI integration test is skipped.
"""

from __future__ import annotations

import json
import urllib.request

from tests.fixtures.openai_compatible_server import (
    ChatCompletionScript,
    OpenAICompatibleTestServer,
)

_PING = {
    "model": "stub-model",
    "stream": True,
    "messages": [{"role": "user", "content": "ping"}],
}


def _post(base_url: str, payload: dict) -> tuple[str, bytes]:
    """POST a Chat Completions body and return ``(Content-Type, raw body)``."""
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.headers.get("Content-Type", ""), response.read()


def _sse_data(body: bytes) -> list[str]:
    """Return the payload of every ``data:`` line, in order."""
    return [
        line[len("data: ") :]
        for line in body.decode().splitlines()
        if line.startswith("data: ")
    ]


def test_streaming_request_returns_openai_chat_completion_chunks() -> None:
    """A ``stream: true`` request yields valid chat.completion.chunk events
    terminated by ``[DONE]``, with the finish chunk carrying usage."""
    script = ChatCompletionScript(name="reply", content="hello", marker="ping")
    with OpenAICompatibleTestServer([script]) as server:
        content_type, body = _post(server.base_url, _PING)

    assert content_type.startswith("text/event-stream")
    events = _sse_data(body)
    assert events[-1] == "[DONE]"
    chunks = [json.loads(event) for event in events[:-1]]
    assert [chunk["choices"][0]["delta"] for chunk in chunks] == [
        {"role": "assistant"},
        {"content": "hello"},
        {},
    ]
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["usage"] == {
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "total_tokens": 2,
    }
    server.assert_all_scripts_consumed()


def test_non_streaming_request_returns_a_json_completion() -> None:
    """Without ``stream``, the fixture answers with a buffered JSON body — the
    shape a non-streaming OpenAI SDK call expects."""
    script = ChatCompletionScript(name="reply", content="hello", marker="ping")
    with OpenAICompatibleTestServer([script]) as server:
        content_type, body = _post(server.base_url, {**_PING, "stream": False})

    assert content_type.startswith("application/json")
    payload = json.loads(body)
    assert payload["choices"][0]["message"] == {
        "role": "assistant",
        "content": "hello",
    }
    server.assert_all_scripts_consumed()
