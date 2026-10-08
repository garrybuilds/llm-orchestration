"""Reusable rate-limit behavior."""


class RateLimitMixin:
    _check_rate_limits = None
    _record_request = None
    _wait_for_rate_limit = None
