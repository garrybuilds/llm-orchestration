# maf-llm

**A multi-provider LLM orchestration layer in Python** — one interface, four providers, with fallback, retry, rate limiting, streaming, and per-call cost tracking.

This is a small, focused library extracted from two production systems that needed the same thing: a reliable way to call *whichever* model is available and affordable, without each service re-implementing provider plumbing and getting it subtly wrong. Both services now depend on this package, which removed roughly 900 lines of duplicated LLM client code.

1102 lines of source, 2 test modules, no hardcoded secrets.

---

## What problem it solves

Calling an LLM from production code is easy until it isn't. Providers go down, keys get rate limited, a model gets deprecated, and the bill arrives before anyone noticed the spending. This package treats all of that as the normal case:

- **One abstract client.** `BaseLLMClient` defines the interface; each provider implements it and lazily loads its own SDK.
- **A fallback chain.** If the preferred provider fails, the next one in the chain is tried. The orchestrator decides, the caller doesn't care.
- **Retry with backoff.** Transient failures are retried; permanent ones surface fast.
- **Rate limiting.** A gate on every `complete()` and `stream()` path, so a burst can't blow through a provider's limits.
- **Cost tracking.** Per-call usage accounting (`UsageStats`) against a pricing table, so spend is visible as it happens rather than at the end of the month.
- **Streaming.** Same interface for streamed and non-streamed calls.
- **SSRF hardening.** Provider `base_url` is validated at connection time — allowed schemes only, and private/loopback addresses are refused in production (`MAF_ENV=production`). This is DNS-rebind safe: the check happens against the resolved address at connect time, not just the string.

## Providers

| Provider | Module |
|---|---|
| OpenAI | `src/maf_llm/clients/openai.py` |
| Anthropic | `src/maf_llm/clients/anthropic.py` |
| Ollama (local) | `src/maf_llm/clients/ollama.py` |
| OpenRouter | `src/maf_llm/clients/openrouter.py` |

Adding a provider means implementing one interface and registering it — the orchestrator, retry, rate limiting, and cost accounting come for free.

## Layout

```
src/maf_llm/
  base.py          # BaseLLMClient interface, config, base_url validation
  orchestrator.py  # provider selection, fallback chain, usage aggregation
  retry.py         # retry / backoff policy
  ratelimit.py     # request gating
  pricing.py       # per-model cost table
  templates.py     # prompt template handling
  clients/         # one module per provider
tests/
  test_orchestrator.py       # fallback behaviour, no-fallback-raises
  test_validate_base_url.py  # SSRF / scheme / private-IP rejection
```

## Design notes

- **Secrets come from the environment, never the code.** Error messages are written so they cannot leak keys or tokens back to the caller or the logs.
- **The orchestrator owns policy.** Which provider, in what order, under what limits is a decision made in one place and testable.
- **Failures are typed.** A caller can distinguish "this provider is down, try the next" from "this request is malformed, do not retry."

## Status

Merged and in production use across the services that depend on it. Linted with `ruff`; the fallback and SSRF suites pass.

## License

MIT
