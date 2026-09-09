"""
Shared API rate limiting and retry logic for Gemini API calls.
Prevents 429 RESOURCE_EXHAUSTED errors and handles daily quota exhaustion gracefully.
"""
import time
import logging
import threading
import re
from typing import Optional, Any, Callable

logger = logging.getLogger(__name__)

# Global rate limiter: enforces minimum pacing between API calls
_lock = threading.Lock()
_last_call_time = 0.0
_MIN_DELAY_SECONDS = 4.0  # Smooth pacing between calls


def rate_limited_api_call(
    call_fn: Callable,
    max_retries: int = 2,
    base_delay: float = 6.0,
    fallback_fn: Optional[Callable] = None,
    fallback_args: tuple = (),
    fallback_kwargs: dict = None
) -> Any:
    """
    Execute an API call with rate limiting and exponential backoff retry.
    Gracefully identifies daily quota exhaustion vs transient minute rate limits.
    
    Args:
        call_fn: Zero-argument callable that makes the API call.
        max_retries: Maximum number of retry attempts on transient 429/5xx errors.
        base_delay: Base delay in seconds for exponential backoff.
        fallback_fn: Optional fallback function if all retries fail or daily quota exhausted.
        fallback_args: Args for fallback function.
        fallback_kwargs: Kwargs for fallback function.
    
    Returns:
        The result of call_fn() on success, or fallback_fn() result if all retries exhausted.
    
    Raises:
        The last exception if no fallback is provided and all retries fail.
    """
    global _last_call_time
    if fallback_kwargs is None:
        fallback_kwargs = {}

    for attempt in range(max_retries + 1):
        # Enforce minimum delay between any API calls (global rate limit)
        with _lock:
            now = time.time()
            elapsed = now - _last_call_time
            if elapsed < _MIN_DELAY_SECONDS:
                wait = _MIN_DELAY_SECONDS - elapsed
                time.sleep(wait)
            _last_call_time = time.time()

        try:
            return call_fn()
        except Exception as e:
            error_str = str(e)
            is_rate_limit = "429" in error_str or "RESOURCE_EXHAUSTED" in error_str
            is_server_error = "500" in error_str or "503" in error_str
            is_daily_limit = "perday" in error_str.lower() or "limit: 0" in error_str.lower()

            # If daily project quota is reached, retrying in seconds will never help
            if is_daily_limit:
                logger.warning(
                    f"Gemini API daily quota reached for this model/project. "
                    f"Falling back immediately to local heuristic engine."
                )
                if fallback_fn is not None:
                    return fallback_fn(*fallback_args, **fallback_kwargs)
                raise

            if (is_rate_limit or is_server_error) and attempt < max_retries:
                # Parse retry delay from error if available
                retry_delay = base_delay * (2 ** attempt)
                if "retryDelay" in error_str:
                    try:
                        match = re.search(r'retryDelay.*?(\d+(?:\.\d+)?)s', error_str)
                        if match:
                            retry_delay = max(float(match.group(1)) + 1.0, retry_delay)
                    except Exception:
                        pass

                logger.info(
                    f"API call hit transient limit (attempt {attempt + 1}/{max_retries + 1}): "
                    f"{'Rate limited' if is_rate_limit else 'Server error'}. "
                    f"Retrying in {retry_delay:.1f}s..."
                )
                time.sleep(retry_delay)
                continue
            else:
                # Non-retryable error or max retries exhausted
                if fallback_fn is not None:
                    logger.warning(
                        f"API call failed after {attempt + 1} attempts ({e}). "
                        f"Using fallback engine."
                    )
                    return fallback_fn(*fallback_args, **fallback_kwargs)
                raise
