from maf_llm.base import LLMResponse, OpenAIClient, Provider


class OpenRouterClient(OpenAIClient):
    """OpenRouter client using its OpenAI-compatible API."""

    async def complete(self, *args, **kwargs) -> LLMResponse:
        response = await super().complete(*args, **kwargs)
        response.provider = Provider.OPENROUTER
        return response
