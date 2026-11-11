#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified API Client Factory

Provides a factory for creating OpenAI-compatible API clients that support
switching between vendor_a and Vendor B API providers via configuration.

Supported providers:
  - vendor_a, vendor_b: caller-configured OpenAI-compatible endpoints.

Usage:
    from verl.utils.reward_score.api_client_factory import create_api_client, call_model_api

    # Create an OpenAI client
    client = create_api_client(provider="vendor_a", api_key="your_key")

    # Or use the unified call function
    answer, reason = call_model_api(
        messages=[{"role": "user", "content": "Hello"}],
        provider="vendor_b",
        model="deepseek-r1",
        api_key="your_key",
    )

Author: auto-generated
Version: 1.0.0
"""

import logging
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# Default configurations for each provider
# ============================================================
PROVIDER_CONFIGS = {
    "vendor_a": {
        "base_url": None,
        "default_api_key": None,
        "needs_proxy": False,
    },
    "vendor_b": {
        "base_url": None,
        "default_api_key": None,
        "needs_proxy": False,
        "proxy_url": None,
    },
}


def get_provider_config(provider: str) -> Dict[str, Any]:
    """Get the default configuration for a given provider.

    Args:
        provider: Provider name, one of 'vendor_a' or 'vendor_b'.

    Returns:
        Configuration dictionary for the provider.

    Raises:
        ValueError: If the provider is not supported.
    """
    provider = provider.lower().strip()
    if provider not in PROVIDER_CONFIGS:
        raise ValueError(
            f"Unsupported API provider: '{provider}'. "
            f"Supported providers: {list(PROVIDER_CONFIGS.keys())}"
        )
    return PROVIDER_CONFIGS[provider]


def create_api_client(
    provider: str = "vendor_a",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> OpenAI:
    """Create an OpenAI-compatible API client for the specified provider.

    Args:
        provider: API provider name ('vendor_a' or 'vendor_b').
        api_key: API key. If None, uses the default key for the provider.
        base_url: Base URL override. If None, uses the default URL for the provider.

    Returns:
        An OpenAI client instance configured for the specified provider.
    """
    config = get_provider_config(provider)

    actual_api_key = api_key or os.environ.get("KNOWGPO_JUDGE_API_KEY") or config["default_api_key"]
    actual_base_url = base_url or os.environ.get("KNOWGPO_JUDGE_BASE_URL") or config["base_url"]
    if not actual_api_key or not actual_base_url:
        raise ValueError("Judge API requires KNOWGPO_JUDGE_API_KEY and KNOWGPO_JUDGE_BASE_URL (or explicit api_key/base_url)")

    client_kwargs = {
        "api_key": actual_api_key,
        "base_url": actual_base_url,
    }

    proxy_url = os.environ.get("KNOWGPO_JUDGE_PROXY")
    if proxy_url:
        client_kwargs["http_client"] = httpx.Client(proxy=proxy_url)

    client = OpenAI(**client_kwargs)
    logger.info(f"Created API client for provider='{provider}', base_url='{actual_base_url}'")
    return client


def call_model_api(
    messages: List[Dict[str, str]],
    provider: str = "vendor_a",
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    enable_thinking: bool = False,
    max_retries: int = 3,
    base_delay: float = 1,
    backoff_factor: float = 2,
    max_delay: float = 60,
    temperature: float = 0,
    max_tokens: int = 8192,
    fallback_models: Optional[List[str]] = None,
    global_timeout: float = 600.0,
) -> Tuple[str, str]:
    """Unified model API call function with multi-model failover (OpenAI SDK).

    Supports both vendor_a and Vendor B providers with automatic retry
    on rate limit errors and exponential backoff.

    Key retry strategies:
      - 429 rate limit errors do NOT consume retry quota; instead, exponential
        backoff with jitter is applied before retrying the same model.
      - Non-rate-limit errors consume retry quota per model.
      - When a model exhausts its retry quota, the next fallback model is tried.
      - A global timeout prevents infinite waiting.
      - Random model selection from fallback list spreads load across models.

    Args:
        messages: Chat messages in OpenAI format.
        provider: API provider name ('vendor_a' or 'vendor_b').
        model: Model name to use.
        api_key: API key override.
        base_url: Base URL override.
        enable_thinking: Whether to enable thinking/reasoning mode.
        max_retries: Maximum number of retries for non-rate-limit errors per model.
        base_delay: Base delay in seconds for exponential backoff.
        backoff_factor: Backoff multiplier.
        max_delay: Maximum delay in seconds.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in the response.
        fallback_models: List of fallback model names to try on failure.
            If None, only the primary model is used.
        global_timeout: Global timeout in seconds across all retries.

    Returns:
        Tuple of (answer_content, reasoning_content).
        Returns ("", "") on failure.
    """
    start_time = time.time()

    # Build the model list: primary model first, then fallbacks (deduplicated)
    models_to_try = [model]
    if fallback_models:
        # Robustness: accept comma-separated string from Hydra YAML configs
        # (e.g. "m1,m2,m3"), otherwise iterating a str would yield single chars
        # and the remote API would reject them as invalid models.
        if isinstance(fallback_models, str):
            fallback_models = [s.strip() for s in fallback_models.split(",") if s.strip()]
        for fm in fallback_models:
            if fm not in models_to_try:
                models_to_try.append(fm)

    # Shuffle to spread load across models in concurrent calls
    if len(models_to_try) > 1:
        random.shuffle(models_to_try)

    def _remaining_time() -> float:
        return global_timeout - (time.time() - start_time)

    def _is_globally_timed_out() -> bool:
        return _remaining_time() <= 0

    client = create_api_client(provider=provider, api_key=api_key, base_url=base_url)

    for current_model in models_to_try:
        if _is_globally_timed_out():
            logger.warning(
                f"[call_model_api] Global timeout ({global_timeout}s) reached "
                f"before trying model={current_model}"
            )
            break

        # Build extra_body for thinking mode
        extra_body = {}
        if enable_thinking and provider == "vendor_b":
            extra_body["thinking"] = {"type": "enabled"}

        error_count = 0        # Non-rate-limit error counter (consumes retries)
        rate_limit_count = 0   # Rate limit counter (does NOT consume retries)

        while error_count < max_retries:
            if _is_globally_timed_out():
                logger.warning(
                    f"[call_model_api] Global timeout ({global_timeout}s) reached "
                    f"during model={current_model} retries"
                )
                break

            try:
                create_kwargs = {
                    "model": current_model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": False,
                }
                if extra_body:
                    create_kwargs["extra_body"] = extra_body

                # For vendor_a with thinking models, add thinking params
                if enable_thinking and provider == "vendor_a":
                    create_kwargs["extra_body"] = {
                        "thinking_enabled": True,
                        "thinking_tokens": 4096,
                    }

                completion = client.chat.completions.create(**create_kwargs)
                answer = completion.choices[0].message.content or ""
                reason = ""
                if enable_thinking:
                    reason = getattr(completion.choices[0].message, "reasoning_content", "") or ""

                if answer:
                    logger.info(
                        f"[call_model_api] Success: provider={provider}, "
                        f"model={current_model}"
                    )
                    return answer, reason
                else:
                    logger.warning(
                        f"[call_model_api] Empty content from model={current_model}"
                    )
                    error_count += 1

            except Exception as e:
                if _is_rate_limit_error(e):
                    # 429 does NOT consume retry quota
                    rate_limit_count += 1
                    delay = _exp_backoff_delay(
                        rate_limit_count, base_delay=3.0, max_delay=90.0
                    )
                    remaining = _remaining_time()
                    if delay > remaining:
                        logger.warning(
                            f"[call_model_api] Rate limit backoff {delay:.1f}s "
                            f"exceeds remaining time {remaining:.1f}s, "
                            f"switching model (current={current_model})"
                        )
                        break
                    logger.warning(
                        f"[call_model_api] Rate limit (429) #{rate_limit_count} "
                        f"on model={current_model}, waiting {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    error_count += 1
                    logger.warning(
                        f"[call_model_api] Error #{error_count}/{max_retries} "
                        f"on model={current_model} (provider={provider}): {e}"
                    )
                    if error_count < max_retries:
                        delay = _exp_backoff_delay(
                            error_count - 1, base_delay=base_delay, max_delay=max_delay
                        )
                        time.sleep(delay)

        # Current model exhausted, log and try next
        logger.warning(
            f"[call_model_api] Model {current_model} exhausted "
            f"(errors={error_count}, rate_limits={rate_limit_count}), "
            f"trying next model..."
        )

    elapsed = time.time() - start_time
    logger.warning(
        f"[call_model_api] All models exhausted after {elapsed:.1f}s "
        f"(provider={provider}, models={models_to_try}), returning empty"
    )
    return "", ""


def _is_rate_limit_error(e: Exception) -> bool:
    """Check if an exception is a rate limit (429) error."""
    if hasattr(e, 'status_code') and e.status_code == 429:
        return True
    err_str = str(e).lower()
    if "429" in str(e) or "rate limit" in err_str or "too many requests" in err_str:
        return True
    return False


def _exp_backoff_delay(
    attempt: int,
    base_delay: float = 3.0,
    backoff_factor: float = 2.0,
    max_delay: float = 90.0,
) -> float:
    """Compute exponential backoff delay with jitter.

    Args:
        attempt: Attempt number (0-based).
        base_delay: Base delay in seconds.
        backoff_factor: Exponential backoff multiplier.
        max_delay: Maximum delay cap.

    Returns:
        Delay in seconds with jitter applied.
    """
    delay = min(base_delay * (backoff_factor ** attempt), max_delay)
    # jitter range [0.5*delay, 1.5*delay] to spread concurrent retries
    jitter_factor = random.uniform(0.5, 1.5)
    return delay * jitter_factor



