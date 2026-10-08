from collections.abc import AsyncGenerator

from maf_llm.base import BaseLLMClient, LLMConfig, LLMResponse, Provider, UsageStats
from maf_llm.clients import AnthropicClient, OllamaClient, OpenAIClient, OpenRouterClient


class LLMOrchestrator:
    def __init__(
        self,
        primary_provider=Provider.OPENAI,
        fallback_provider=None,
        primary_model="gpt-4o-mini",
        fallback_model=None,
        openai_api_key=None,
        anthropic_api_key=None,
        openrouter_api_key=None,
        ollama_api_key=None,
        ollama_base_url=None,
        openrouter_base_url="https://openrouter.ai/api/v1",
    ):
        self.primary_provider = primary_provider
        self.fallback_provider = fallback_provider
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.clients: dict[Provider, BaseLLMClient] = {}
        keys = {
            Provider.OPENAI: openai_api_key,
            Provider.ANTHROPIC: anthropic_api_key,
            Provider.OPENROUTER: openrouter_api_key,
            Provider.OLLAMA: ollama_api_key,
        }
        classes = {
            Provider.OPENAI: OpenAIClient,
            Provider.ANTHROPIC: AnthropicClient,
            Provider.OPENROUTER: OpenRouterClient,
            Provider.OLLAMA: OllamaClient,
        }
        for provider in {primary_provider, fallback_provider} - {None}:
            model = (
                primary_model if provider == primary_provider else (fallback_model or primary_model)
            )
            base_url = (
                openrouter_base_url
                if provider == Provider.OPENROUTER
                else (ollama_base_url if provider == Provider.OLLAMA else None)
            )
            self.clients[provider] = classes[provider](
                LLMConfig(provider=provider, model=model, api_key=keys[provider], base_url=base_url)
            )

    async def complete(
        self, prompt: str, system_prompt: str | None = None, model: str | None = None, **kwargs
    ) -> LLMResponse:
        try:
            return await self.clients[self.primary_provider].complete(
                prompt, system_prompt, model=model or self.primary_model, **kwargs
            )
        except Exception:
            if self.fallback_provider not in self.clients:
                raise
            return await self.clients[self.fallback_provider].complete(
                prompt,
                system_prompt,
                model=model or self.fallback_model or self.primary_model,
                **kwargs,
            )

    async def stream(
        self, prompt: str, system_prompt: str | None = None, model: str | None = None, **kwargs
    ) -> AsyncGenerator[str, None]:
        try:
            async for chunk in self.clients[self.primary_provider].stream(
                prompt, system_prompt, model=model or self.primary_model, **kwargs
            ):
                yield chunk
        except Exception:
            if self.fallback_provider not in self.clients:
                raise
            async for chunk in self.clients[self.fallback_provider].stream(
                prompt,
                system_prompt,
                model=model or self.fallback_model or self.primary_model,
                **kwargs,
            ):
                yield chunk

    def get_usage_stats(self) -> dict[Provider, UsageStats]:
        return {provider: client.usage_stats for provider, client in self.clients.items()}

    async def close(self) -> None:
        for client in self.clients.values():
            resource = getattr(client, "_client", None)
            close = getattr(resource, "close", None)
            if close is not None:
                result = close()
                if hasattr(result, "__await__"):
                    await result


def create_orchestrator(
    primary="openai", fallback=None, primary_model="gpt-4o-mini", fallback_model=None, **api_keys
):
    return LLMOrchestrator(
        primary_provider=Provider(primary),
        fallback_provider=Provider(fallback) if fallback else None,
        primary_model=primary_model,
        fallback_model=fallback_model,
        **api_keys,
    )
