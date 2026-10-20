"""需要会话请求头的 provider（如 OpenCode Go）的出站请求头回归测试。

OpenCode Go 要求每个请求携带稳定的 `x-opencode-session`，缺失时直接返回 400。
这里通过真实构造的 `OpenaiClient` 与 httpx.MockTransport 验证会话头会随聊天流 ID 自动补全。
"""

from typing import Dict, List

import httpx
import pytest

from src.config.model_configs import APIProvider, ModelInfo
from src.llm_models.model_client.base_client import RequestTraceContext, ResponseRequest
from src.llm_models.model_client.openai_client import OpenaiClient
from src.llm_models.payload_content.context_item import ContextItemBuilder


def _handler(captured: List[httpx.Request]):
    def handle(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "test-model",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            },
        )

    return handle


async def _send_chat_request(provider: APIProvider, extra_params: Dict[str, object]) -> httpx.Request:
    """通过 mock transport 发送一次聊天请求并返回被捕获的出站请求。"""
    captured: List[httpx.Request] = []
    client = OpenaiClient(provider)
    client.client = client.client.with_options(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(_handler(captured)))
    )
    model_info = ModelInfo(
        name="test-model",
        model_identifier="test-model",
        model_type="llm",
        api_provider=provider.name,
        extra_params=extra_params,
    )
    try:
        await client.get_response(
            ResponseRequest(
                model_info=model_info,
                context_items=[ContextItemBuilder().add_text_content("hi").build()],
                extra_params=dict(extra_params),
                trace_context=RequestTraceContext(session_id="chat-session-abc", request_id="request-xyz"),
            )
        )
    finally:
        await client.client.close()
    assert len(captured) == 1
    return captured[0]


def _build_provider(**overrides: object) -> APIProvider:
    return APIProvider(
        **{
            "name": "OpenCode Go",
            "base_url": "https://opencode.ai/inference/go/openai/v1",
            "api_key": "secret",
            "client_type": "openai",
            **overrides,
        }
    )


@pytest.mark.asyncio
async def test_opencode_go_sends_conversation_session_header() -> None:
    request = await _send_chat_request(_build_provider(), {})
    assert request.headers["x-opencode-session"] == "chat-session-abc"


@pytest.mark.asyncio
async def test_non_session_provider_does_not_send_session_header() -> None:
    request = await _send_chat_request(_build_provider(name="Local", base_url="http://127.0.0.1:11434/v1"), {})
    assert "x-opencode-session" not in request.headers


@pytest.mark.asyncio
async def test_explicit_session_header_overrides_auto_value() -> None:
    request = await _send_chat_request(
        _build_provider(),
        {"headers": {"x-opencode-session": "explicit-session"}},
    )
    assert request.headers["x-opencode-session"] == "explicit-session"
