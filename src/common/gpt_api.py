"""Thin, retrying wrapper around an OpenAI-compatible chat endpoint.

Used by the teacher-labelling stage (src/data/label_gpt.py) and the judge
(src/eval/judge.py). No credentials are stored in this repository: the key is
read from ``OPENAI_API_KEY`` and the endpoint from ``OPENAI_BASE_URL`` (unset =
api.openai.com), so any compatible provider works.
"""

from __future__ import annotations

import logging
import os
import time

from openai import APIError, APITimeoutError, OpenAI, RateLimitError

log = logging.getLogger(__name__)

_client = None


def client():
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Export it before running any stage "
                "that calls the teacher or judge model."
            )
        _client = OpenAI(api_key=api_key, base_url=os.environ.get("OPENAI_BASE_URL"))
    return _client


def call_gpt(messages, model="gpt-5-nano", max_retries=5, timeout=60, **kwargs):
    """Return the assistant message text, or None if every attempt failed.

    Retries with exponential backoff on rate limits, timeouts and 5xx.
    """
    delay = 5
    for attempt in range(1, max_retries + 1):
        try:
            response = client().chat.completions.create(
                model=model, messages=messages, timeout=timeout, **kwargs
            )
            return response.choices[0].message.content
        except (RateLimitError, APITimeoutError) as exc:
            log.warning("attempt %d/%d: %s; retrying in %ds", attempt, max_retries, exc, delay)
        except APIError as exc:
            status = getattr(exc, "status_code", None)
            if status is not None and status < 500:
                log.error("non-retryable API error (%s): %s", status, exc)
                return None
            log.warning("attempt %d/%d: %s; retrying in %ds", attempt, max_retries, exc, delay)
        time.sleep(delay)
        delay *= 2

    log.error("max retries reached; giving up on this request")
    return None
