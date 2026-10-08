"""Retry helpers shared by provider clients."""


async def retry_with_backoff(client, fn, max_retries=None):
    return await client._retry_with_backoff(fn, max_retries)
