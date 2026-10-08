from unittest.mock import AsyncMock, MagicMock

import pytest

from maf_llm import LLMOrchestrator, LLMResponse, Provider


@pytest.mark.asyncio
async def test_complete_with_fallback():
    primary = MagicMock()
    primary.complete = AsyncMock(side_effect=RuntimeError("down"))
    fallback = MagicMock()
    fallback.complete = AsyncMock(
        return_value=LLMResponse("ok", "local", Provider.OLLAMA, 1, 1, 2, 0.0, 1, "stop")
    )
    orchestrator = LLMOrchestrator(Provider.OPENAI, Provider.OLLAMA)
    orchestrator.clients = {Provider.OPENAI: primary, Provider.OLLAMA: fallback}
    assert (await orchestrator.complete("test")).content == "ok"


@pytest.mark.asyncio
async def test_no_fallback_raises():
    primary = MagicMock()
    primary.complete = AsyncMock(side_effect=RuntimeError("down"))
    orchestrator = LLMOrchestrator(Provider.OPENAI)
    orchestrator.clients = {Provider.OPENAI: primary}
    with pytest.raises(RuntimeError, match="down"):
        await orchestrator.complete("test")
