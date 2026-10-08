"""Regression tests for SSRF controls and per-client rate limiting."""

import asyncio
import os
import socket
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from maf_llm.base import (
    AnthropicClient,
    LLMConfig,
    OllamaClient,
    OpenAIClient,
    Provider,
    _build_http_client,
    _PublicOnlyHTTPTransport,
    _PublicOnlyNetworkBackend,
    _resolve_public_addresses,
    _validate_base_url,
)
from maf_llm.clients.openrouter import OpenRouterClient


class TestValidateBaseUrl:
    def test_loopback_passes_when_env_unset(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MAF_ENV", None)
            assert _validate_base_url("http://127.0.0.1:11434") == "http://127.0.0.1:11434"

    @pytest.mark.parametrize("address", ["127.0.0.1", "10.0.0.5", "100.64.0.1", "169.254.1.1"])
    def test_non_public_ip_blocked_in_production(self, address):
        with (
            patch.dict(os.environ, {"MAF_ENV": "production"}),
            pytest.raises(ValueError, match="non-public IP"),
        ):
            _validate_base_url(f"http://{address}/v1")

    @pytest.mark.parametrize(
        "address",
        ["[::1]", "[::ffff:127.0.0.1]", "[::ffff:10.0.0.5]", "[fe80::1]", "[fe80::1%25eth0]"],
    )
    def test_ipv6_loopback_mapped_and_link_local_blocked_in_production(self, address):
        """IPv6 literals that embed or cover private space must not pass the guard.

        On Pythons without the CVE-2024-4032 fix, ipaddress misreports some
        IPv4-mapped addresses as global; the explicit reject below keeps the
        guard correct on every interpreter meeting requires-python.
        """
        with (
            patch.dict(os.environ, {"MAF_ENV": "production"}),
            pytest.raises(ValueError, match="non-public IP"),
        ):
            _validate_base_url(f"http://{address}/v1")

    def test_private_ip_passes_when_env_unset(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MAF_ENV", None)
            assert _validate_base_url("http://10.0.0.5/v1") == "http://10.0.0.5/v1"

    def test_production_hostname_validation_does_not_block_on_sync_dns(self):
        with (
            patch.dict(os.environ, {"MAF_ENV": "production"}),
            patch("socket.getaddrinfo", side_effect=AssertionError("sync DNS called")),
        ):
            assert _validate_base_url("https://api.openai.com/v1") == "https://api.openai.com/v1"

    @pytest.mark.parametrize("env", [{}, {"MAF_ENV": "production"}])
    def test_http_url_without_hostname_rejected_always(self, env):
        with (
            patch.dict(os.environ, env, clear=True),
            pytest.raises(ValueError, match="hostname"),
        ):
            _validate_base_url("http:///missing-host")

    def test_non_http_scheme_rejected_always(self):
        with (
            patch.dict(os.environ, {}, clear=False),
            pytest.raises(ValueError, match="Invalid base_url scheme"),
        ):
            os.environ.pop("MAF_ENV", None)
            _validate_base_url("file:///etc/passwd")

    def test_none_passes(self):
        assert _validate_base_url(None) is None


class TestPublicOnlyResolver:
    @staticmethod
    def _addrinfo(*addresses):
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 443)) for address in addresses
        ]

    @pytest.mark.asyncio
    async def test_public_addresses_pass(self):
        loop = MagicMock()
        loop.getaddrinfo = AsyncMock(return_value=self._addrinfo("93.184.216.34"))
        with patch("maf_llm.base.asyncio.get_running_loop", return_value=loop):
            assert await _resolve_public_addresses("example.com", 443) == ["93.184.216.34"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("addresses", [("127.0.0.1",), ("93.184.216.34", "10.0.0.8")])
    async def test_private_or_mixed_answers_fail_closed(self, addresses):
        loop = MagicMock()
        loop.getaddrinfo = AsyncMock(return_value=self._addrinfo(*addresses))
        with (
            patch("maf_llm.base.asyncio.get_running_loop", return_value=loop),
            pytest.raises(ValueError, match="non-public IP"),
        ):
            await _resolve_public_addresses("rebind.example", 443)

    @pytest.mark.asyncio
    async def test_empty_dns_answer_fails_closed(self):
        loop = MagicMock()
        loop.getaddrinfo = AsyncMock(return_value=[])
        with (
            patch("maf_llm.base.asyncio.get_running_loop", return_value=loop),
            pytest.raises(ValueError, match="no addresses"),
        ):
            await _resolve_public_addresses("empty.example", 443)

    @pytest.mark.asyncio
    async def test_dns_error_fails_closed(self):
        loop = MagicMock()
        loop.getaddrinfo = AsyncMock(side_effect=OSError("dns down"))
        with (
            patch("maf_llm.base.asyncio.get_running_loop", return_value=loop),
            pytest.raises(ValueError, match="could not resolve"),
        ):
            await _resolve_public_addresses("broken.example", 443)

    @pytest.mark.asyncio
    async def test_ipv4_mapped_answers_fail_closed(self):
        loop = MagicMock()
        loop.getaddrinfo = AsyncMock(return_value=self._addrinfo("::ffff:127.0.0.1"))
        with (
            patch("maf_llm.base.asyncio.get_running_loop", return_value=loop),
            pytest.raises(ValueError, match="non-public IP"),
        ):
            await _resolve_public_addresses("mapped.rebind.example", 443)

    @pytest.mark.asyncio
    async def test_backend_connects_to_validated_ip_not_hostname(self):
        stream = object()
        delegate = MagicMock()
        delegate.connect_tcp = AsyncMock(return_value=stream)
        resolver = AsyncMock(return_value=["93.184.216.34"])
        backend = _PublicOnlyNetworkBackend(delegate=delegate, resolver=resolver)

        result = await backend.connect_tcp("example.com", 443, timeout=5.0)

        assert result is stream
        resolver.assert_awaited_once_with("example.com", 443)
        delegate.connect_tcp.assert_awaited_once_with(
            "93.184.216.34",
            443,
            timeout=5.0,
            local_address=None,
            socket_options=None,
        )


class TestSDKTransport:
    @pytest.mark.asyncio
    async def test_production_transport_pins_dns_and_disables_redirects(self):
        with patch.dict(os.environ, {"MAF_ENV": "production"}):
            client = _build_http_client()
        try:
            assert client.follow_redirects is False
            assert isinstance(client._transport, _PublicOnlyHTTPTransport)
        finally:
            await client.aclose()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("client_class", "provider"),
        [
            (OpenAIClient, Provider.OPENAI),
            (AnthropicClient, Provider.ANTHROPIC),
            (OllamaClient, Provider.OLLAMA),
            (OpenRouterClient, Provider.OPENROUTER),
        ],
    )
    async def test_every_sdk_client_disables_redirects(self, client_class, provider):
        config = LLMConfig(provider=provider, model="test-model", api_key="test-key")
        with patch.dict(os.environ, {"MAF_ENV": "production"}):
            sdk = client_class(config)._get_client()
        try:
            assert sdk._client.follow_redirects is False
            assert isinstance(sdk._client._transport, _PublicOnlyHTTPTransport)
        finally:
            await sdk.close()


class TestCompleteRateLimit:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("client_class", "provider"),
        [
            (OpenAIClient, Provider.OPENAI),
            (AnthropicClient, Provider.ANTHROPIC),
            (OllamaClient, Provider.OLLAMA),
            (OpenRouterClient, Provider.OPENROUTER),
        ],
    )
    async def test_every_complete_path_invokes_rate_limit_wait(self, client_class, provider):
        client = client_class(LLMConfig(provider=provider, model="test-model"))
        client._wait_for_rate_limit = AsyncMock(side_effect=RuntimeError("rate-limit-gate-reached"))
        client._get_client = MagicMock()
        client._retry_with_backoff = AsyncMock(
            side_effect=AssertionError("API path reached before rate-limit gate")
        )

        with pytest.raises(RuntimeError, match="rate-limit-gate-reached"):
            await client.complete("hello")

        client._wait_for_rate_limit.assert_awaited_once()
        client._get_client.assert_not_called()

    @pytest.mark.asyncio
    async def test_rate_limit_rpm_waits_for_real_window(self):
        client = OpenAIClient(
            LLMConfig(provider=Provider.OPENAI, model="gpt-4o-mini", rate_limit_rpm=1)
        )
        client._get_client = MagicMock()
        client._retry_with_backoff = AsyncMock(
            return_value=MagicMock(
                usage=MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                choices=[MagicMock(message=MagicMock(content="hi"), finish_reason="stop")],
            )
        )
        clock = [100.0]

        async def advance(seconds):
            clock[0] += seconds

        with (
            patch("maf_llm.base.time.time", side_effect=lambda: clock[0]),
            patch("maf_llm.base.asyncio.sleep", side_effect=advance) as sleep_mock,
        ):
            await client.complete("one")
            await client.complete("two")

        assert sleep_mock.await_count == 60
        assert len(client._request_timestamps) == 1

    @pytest.mark.asyncio
    async def test_concurrent_waiters_cannot_bypass_rpm_limit(self):
        client = OpenAIClient(
            LLMConfig(provider=Provider.OPENAI, model="gpt-4o-mini", rate_limit_rpm=1)
        )
        blocked = RuntimeError("blocked-by-rate-limit")
        with patch("maf_llm.base.asyncio.sleep", new=AsyncMock(side_effect=blocked)):
            results = await asyncio.gather(
                client._wait_for_rate_limit(),
                client._wait_for_rate_limit(),
                return_exceptions=True,
            )

        assert sum(result is None for result in results) == 1
        assert sum(isinstance(result, RuntimeError) for result in results) == 1
        assert len(client._request_timestamps) == 1
