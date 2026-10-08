"""LLM Orchestration Layer."""

import asyncio
import ipaddress
import os
import socket
import time
import urllib.parse
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import httpcore
import httpx


async def _resolve_public_addresses(host: str, port: int) -> list[str]:
    """Resolve a host asynchronously and fail closed unless every answer is public."""
    loop = asyncio.get_running_loop()
    try:
        address_info = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError:
        raise ValueError(f"base_url hostname could not resolve: {host}") from None

    addresses = sorted({str(entry[4][0]) for entry in address_info})
    if not addresses:
        raise ValueError(f"base_url hostname returned no addresses: {host}")

    for address in addresses:
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            raise ValueError(
                f"base_url hostname returned an invalid IP: {host} -> {address}"
            ) from None
        if (
            getattr(ip, "ipv4_mapped", None) is not None
            or getattr(ip, "scope_id", None) is not None
        ):
            # CVE-2024-4032: pre-3.12.4 interpreters can report IPv4-mapped
            # and scope-carrying addresses as global. Reject explicitly so the
            # guard holds on every interpreter meeting requires-python.
            raise ValueError(f"base_url hostname resolves to non-public IP: {host} -> {address}")
        if not ip.is_global:
            raise ValueError(f"base_url hostname resolves to non-public IP: {host} -> {address}")
    return addresses


class _PublicOnlyNetworkBackend(httpcore.AsyncNetworkBackend):
    """Resolve once, validate all answers, then connect to a validated IP."""

    def __init__(
        self,
        delegate: httpcore.AsyncNetworkBackend | None = None,
        resolver: Callable[[str, int], Awaitable[list[str]]] | None = None,
    ):
        self._delegate: httpcore.AsyncNetworkBackend = delegate or httpcore.AnyIOBackend()
        self._resolver = resolver or _resolve_public_addresses

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options=None,
    ) -> httpcore.AsyncNetworkStream:
        addresses = await self._resolver(host, port)
        last_error = None
        for address in addresses:
            try:
                return await self._delegate.connect_tcp(
                    address,
                    port,
                    timeout=timeout,
                    local_address=local_address,
                    socket_options=socket_options,
                )
            except (httpcore.ConnectError, httpcore.ConnectTimeout, OSError) as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        raise httpcore.ConnectError(f"No public address available for {host}")

    async def connect_unix_socket(self, path, timeout=None, socket_options=None):
        if os.environ.get("MAF_ENV") == "production":
            # Fail closed: the production backend only brokers public TCP.
            raise httpcore.ConnectError("Unix sockets are not permitted in production")
        return await self._delegate.connect_unix_socket(
            path, timeout=timeout, socket_options=socket_options
        )

    async def sleep(self, seconds: float) -> None:
        await self._delegate.sleep(seconds)


class _PublicOnlyHTTPTransport(httpx.AsyncHTTPTransport):
    """HTTPX transport whose connection pool cannot resolve private destinations."""

    def __init__(self):
        super().__init__(trust_env=False)
        # HTTPX 0.28/httpcore 1.0: preserve HTTPX error mapping and replace
        # only the network backend. Dependency bounds protect this integration.
        self._pool._network_backend = _PublicOnlyNetworkBackend()


def _build_http_client() -> httpx.AsyncClient:
    """Build the SDK HTTP client with redirects disabled and production DNS pinning."""
    if os.environ.get("MAF_ENV") == "production":
        return httpx.AsyncClient(
            transport=_PublicOnlyHTTPTransport(), follow_redirects=False, trust_env=False
        )
    return httpx.AsyncClient(follow_redirects=False)


def _validate_base_url(url: str | None) -> str | None:
    """Validate that base_url uses an allowed scheme and is not a private/loopback IP in production.

    Allows http(s) schemes. In production, blocks direct non-public IPs here;
    hostnames are resolved, validated, and pinned asynchronously by the HTTP
    transport. Non-production keeps local endpoints available for development.
    """
    if url is None:
        return url
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("https", "http"):
        raise ValueError(f"Invalid base_url scheme: {parsed.scheme}")
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("base_url must include a hostname")
    if os.environ.get("MAF_ENV") != "production":
        return url
    try:
        ip = ipaddress.ip_address(hostname)
    except ValueError:
        return url  # Domain resolution is performed asynchronously by the transport.
    else:
        if not ip.is_global:
            if (
                getattr(ip, "ipv4_mapped", None) is not None
                or getattr(ip, "scope_id", None) is not None
            ):
                raise ValueError(f"base_url points to non-public IP: {hostname}")
            raise ValueError(f"base_url points to private IP or other non-public IP: {hostname}")
    return url


class Provider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    OPENROUTER = "openrouter"


@dataclass
class LLMConfig:
    """Configuration for LLM provider."""

    provider: Provider
    model: str
    api_key: str | None = None
    base_url: str | None = None
    max_tokens: int = 4096
    temperature: float = 0.3
    timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit_rpm: int = 60  # Requests per minute
    rate_limit_tpm: int = 90000  # Tokens per minute

    def __post_init__(self):
        self.base_url = _validate_base_url(self.base_url)


@dataclass
class LLMResponse:
    """Response from LLM provider."""

    content: str
    model: str
    provider: Provider
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    latency_ms: int
    finish_reason: str
    metadata: dict = field(default_factory=dict)


@dataclass
class UsageStats:
    """Usage tracking for cost monitoring."""

    total_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    requests_by_model: dict = field(default_factory=dict)
    last_request_time: datetime | None = None

    def add_response(self, response: LLMResponse):
        """Track response usage."""
        self.total_requests += 1
        self.total_input_tokens += response.input_tokens
        self.total_output_tokens += response.output_tokens
        self.total_cost_usd += response.cost_usd

        model_key = f"{response.provider.value}:{response.model}"
        if model_key not in self.requests_by_model:
            self.requests_by_model[model_key] = {
                "count": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
            }
        self.requests_by_model[model_key]["count"] += 1
        self.requests_by_model[model_key]["input_tokens"] += response.input_tokens
        self.requests_by_model[model_key]["output_tokens"] += response.output_tokens
        self.requests_by_model[model_key]["cost_usd"] += response.cost_usd

        self.last_request_time = datetime.now()


class BaseLLMClient(ABC):
    """Abstract base class for LLM providers."""

    # Pricing per 1K tokens (as of 2024)
    PRICING = {
        "openai": {
            "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        },
        "anthropic": {
            "claude-3-opus": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet": {"input": 0.003, "output": 0.015},
            "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
            "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
        },
        "ollama": {
            "glm-5": {"input": 0.0, "output": 0.0},
            "default": {"input": 0.0, "output": 0.0},
        },
    }

    def __init__(self, config: LLMConfig):
        self.config = config
        self.usage_stats = UsageStats()
        self._request_timestamps: list[float] = []
        self._token_timestamps: list[tuple[float, int]] = []
        self._rate_limit_lock = asyncio.Lock()

    @abstractmethod
    async def complete(
        self, prompt: str, system_prompt: str | None = None, **kwargs
    ) -> LLMResponse:
        """Generate completion from LLM."""
        pass

    @abstractmethod
    async def stream(
        self, prompt: str, system_prompt: str | None = None, **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream completion from LLM."""
        pass

    def calculate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """Calculate cost in USD."""
        provider_pricing = self.PRICING.get(self.config.provider.value, {})
        model_pricing = provider_pricing.get(model, {"input": 0, "output": 0})

        input_cost = (input_tokens / 1000) * model_pricing["input"]
        output_cost = (output_tokens / 1000) * model_pricing["output"]

        return input_cost + output_cost

    def _check_rate_limits(self) -> bool:
        """Check if we're within rate limits."""
        now = time.time()
        minute_ago = now - 60

        # Clean old timestamps
        self._request_timestamps = [t for t in self._request_timestamps if t > minute_ago]
        self._token_timestamps = [(t, n) for t, n in self._token_timestamps if t > minute_ago]

        # Check request limit
        if len(self._request_timestamps) >= self.config.rate_limit_rpm:
            return False

        # Check token limit
        total_tokens = sum(n for _, n in self._token_timestamps)
        return not total_tokens >= self.config.rate_limit_tpm

    def _record_tokens(self, tokens: int):
        """Record observed token usage after a reserved request completes."""
        self._token_timestamps.append((time.time(), tokens))

    async def _wait_for_rate_limit(self):
        """Wait for capacity and atomically reserve one request slot."""
        while True:
            async with self._rate_limit_lock:
                if self._check_rate_limits():
                    self._request_timestamps.append(time.time())
                    return
            await asyncio.sleep(1)

    async def _retry_with_backoff(self, fn, max_retries: int = None):
        """Retry with exponential backoff."""
        retries = max_retries or self.config.max_retries
        delay = self.config.retry_delay

        for attempt in range(retries):
            try:
                return await fn()
            except Exception:
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(delay * (2**attempt))
                delay = min(delay * 2, 60)


class OpenAIClient(BaseLLMClient):
    """OpenAI API client."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client = None

    def _get_client(self):
        """Lazy-load OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI

                api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY")
                self._client = AsyncOpenAI(
                    api_key=api_key,
                    base_url=self.config.base_url,
                    timeout=self.config.timeout,
                    http_client=_build_http_client(),
                )
            except ImportError:
                raise ImportError("openai package required: pip install openai") from None
        return self._client

    async def complete(
        self, prompt: str, system_prompt: str | None = None, **kwargs
    ) -> LLMResponse:
        """Generate completion from OpenAI."""
        await self._wait_for_rate_limit()
        client = self._get_client()

        model = kwargs.get("model", self.config.model)
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)
        temperature = kwargs.get("temperature", self.config.temperature)

        start_time = time.time()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        async def _call():
            return await client.chat.completions.create(
                model=model, messages=messages, max_tokens=max_tokens, temperature=temperature
            )

        response = await self._retry_with_backoff(_call)

        latency_ms = int((time.time() - start_time) * 1000)

        # Extract tokens
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens

        # Record observed token usage for rate limiting; RPM was reserved before the call.
        self._record_tokens(total_tokens)

        # Calculate cost
        cost = self.calculate_cost(input_tokens, output_tokens, model)

        result = LLMResponse(
            content=response.choices[0].message.content,
            model=model,
            provider=Provider.OPENAI,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
            finish_reason=response.choices[0].finish_reason,
            metadata={"messages": len(messages)},
        )

        self.usage_stats.add_response(result)
        return result

    async def stream(
        self, prompt: str, system_prompt: str | None = None, **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream completion from OpenAI."""
        client = self._get_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        model = kwargs.get("model", self.config.model)

        await self._wait_for_rate_limit()

        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
        )

        async for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class AnthropicClient(BaseLLMClient):
    """Anthropic API client."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client = None

    def _get_client(self):
        """Lazy-load Anthropic client."""
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic

                api_key = self.config.api_key or os.environ.get("ANTHROPIC_API_KEY")
                self._client = AsyncAnthropic(
                    api_key=api_key,
                    timeout=self.config.timeout,
                    http_client=_build_http_client(),
                )
            except ImportError:
                raise ImportError("anthropic package required: pip install anthropic") from None
        return self._client

    async def complete(
        self, prompt: str, system_prompt: str | None = None, **kwargs
    ) -> LLMResponse:
        """Generate completion from Anthropic."""
        await self._wait_for_rate_limit()
        client = self._get_client()

        model = kwargs.get("model", self.config.model)
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)
        temperature = kwargs.get("temperature", self.config.temperature)

        start_time = time.time()

        async def _call():
            params = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
            }
            if system_prompt:
                params["system"] = system_prompt
            if temperature is not None:
                params["temperature"] = temperature

            return await client.messages.create(**params)

        response = await self._retry_with_backoff(_call)

        latency_ms = int((time.time() - start_time) * 1000)

        # Extract tokens
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        total_tokens = input_tokens + output_tokens

        # Record observed token usage for rate limiting; RPM was reserved before the call.
        self._record_tokens(total_tokens)

        # Calculate cost
        cost = self.calculate_cost(input_tokens, output_tokens, model)

        result = LLMResponse(
            content=response.content[0].text,
            model=model,
            provider=Provider.ANTHROPIC,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
            finish_reason=response.stop_reason,
            metadata={"messages": 1},
        )

        self.usage_stats.add_response(result)
        return result

    async def stream(
        self, prompt: str, system_prompt: str | None = None, **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream completion from Anthropic."""
        client = self._get_client()

        model = kwargs.get("model", self.config.model)

        await self._wait_for_rate_limit()

        params = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }
        if system_prompt:
            params["system"] = system_prompt

        async with client.messages.stream(**params) as stream:
            async for text in stream.text_stream:
                yield text


class OllamaClient(BaseLLMClient):
    """Ollama Cloud API client (OpenAI-compatible)."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client = None

    def _get_client(self):
        """Lazy-load Ollama client (OpenAI-compatible)."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI

                api_key = self.config.api_key or os.environ.get("OLLAMA_API_KEY")
                base_url = _validate_base_url(
                    self.config.base_url
                    or os.environ.get("OLLAMA_BASE_URL", "https://ollama.com/v1")
                )
                self._client = AsyncOpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    timeout=self.config.timeout,
                    http_client=_build_http_client(),
                )
            except ImportError:
                raise ImportError("openai package required: pip install openai") from None
        return self._client

    async def complete(
        self, prompt: str, system_prompt: str | None = None, **kwargs
    ) -> LLMResponse:
        """Generate completion from Ollama Cloud."""
        await self._wait_for_rate_limit()
        client = self._get_client()

        model = kwargs.get("model", self.config.model)
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)
        temperature = kwargs.get("temperature", self.config.temperature)

        start_time = time.time()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        async def _call():
            return await client.chat.completions.create(
                model=model, messages=messages, max_tokens=max_tokens, temperature=temperature
            )

        response = await self._retry_with_backoff(_call)

        latency_ms = int((time.time() - start_time) * 1000)

        # Extract tokens (Ollama may not always provide usage)
        input_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(response.usage, "completion_tokens", 0) or 0
        total_tokens = input_tokens + output_tokens

        # Record observed token usage for rate limiting; RPM was reserved before the call.
        self._record_tokens(total_tokens)

        # Calculate cost (free for Ollama Cloud)
        cost = 0.0

        result = LLMResponse(
            content=response.choices[0].message.content,
            model=model,
            provider=Provider.OLLAMA,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
            finish_reason=response.choices[0].finish_reason or "stop",
            metadata={"messages": len(messages)},
        )

        self.usage_stats.add_response(result)
        return result

    async def stream(
        self, prompt: str, system_prompt: str | None = None, **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream completion from Ollama Cloud."""
        client = self._get_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        model = kwargs.get("model", self.config.model)

        await self._wait_for_rate_limit()

        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
        )

        async for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
