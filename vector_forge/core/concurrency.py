"""Centralized concurrency management for vector-forge.

This module provides thread pool and semaphore management to prevent:
- Thread pool starvation from too many concurrent asyncio.to_thread() calls
- Event loop blocking when GPU work saturates the default executor
- LLM response processing delays from resource contention

Design principles:
- Single Responsibility: Each manager handles one type of resource
- Dependency Inversion: Components depend on abstractions (protocols)
- DRY: Centralized configuration, no duplicate semaphore creation
- KISS: Simple, explicit concurrency limits

Usage:
    from vector_forge.core.concurrency import (
        get_gpu_executor,
        get_evaluation_semaphore,
        get_llm_semaphore,
        run_in_gpu_executor,
    )

    # Run GPU work in dedicated executor (doesn't block LLM responses)
    result = await run_in_gpu_executor(gpu_heavy_function, arg1, arg2)

    # Limit concurrent evaluations
    async with get_evaluation_semaphore():
        await run_evaluation()
"""

from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(frozen=True)
class ConcurrencyConfig:
    """Configuration for concurrency limits.

    These defaults are tuned for typical GPU workloads:
    - GPU executor: Limited to prevent CUDA memory contention
    - Evaluation semaphore: Limits concurrent vector evaluations
    - LLM semaphore: Limits concurrent API calls to avoid rate limits
    """

    # GPU executor settings
    gpu_executor_workers: int = 4
    """Number of threads for GPU-bound work.

    Keep low because:
    1. GPU operations serialize at hardware level anyway
    2. Multiple threads holding GIL between CUDA calls blocks event loop
    3. More threads = more memory pressure from queued operations
    """

    # Evaluation concurrency
    max_concurrent_evaluations: int = 4
    """Maximum concurrent vector evaluations.

    Each evaluation runs 5 dimensions, each with GPU generations.
    Too many concurrent evaluations = thread pool starvation.
    """

    max_concurrent_dimensions: int = 2
    """Maximum concurrent dimension evaluations per vector.

    Stagger dimension evaluations to prevent GPU memory spikes
    and ensure LLM responses can be processed.
    """

    # LLM API concurrency
    max_concurrent_llm_calls: int = 32
    """Maximum concurrent LLM API calls.

    Shared across all judges to prevent rate limiting
    and ensure fair scheduling.
    """

    @classmethod
    def from_env(cls) -> "ConcurrencyConfig":
        """Create config from environment variables with defaults."""
        return cls(
            gpu_executor_workers=int(
                os.environ.get("VF_GPU_EXECUTOR_WORKERS", "4")
            ),
            max_concurrent_evaluations=int(
                os.environ.get("VF_MAX_CONCURRENT_EVALUATIONS", "4")
            ),
            max_concurrent_dimensions=int(
                os.environ.get("VF_MAX_CONCURRENT_DIMENSIONS", "2")
            ),
            max_concurrent_llm_calls=int(
                os.environ.get("VF_MAX_CONCURRENT_LLM_CALLS", "32")
            ),
        )


class ConcurrencyManager:
    """Centralized manager for all concurrency primitives.

    Provides:
    - Dedicated thread pool for GPU work (separate from asyncio default)
    - Shared semaphores for evaluations and LLM calls
    - Proper cleanup on shutdown

    This is a singleton - use get_concurrency_manager() to access.
    """

    _instance: Optional["ConcurrencyManager"] = None

    def __init__(self, config: Optional[ConcurrencyConfig] = None) -> None:
        """Initialize the manager. Use get_concurrency_manager() instead."""
        self._config = config or ConcurrencyConfig.from_env()

        # Dedicated executor for GPU work - separate from asyncio default
        # This prevents GPU operations from blocking LLM response processing
        self._gpu_executor: Optional[ThreadPoolExecutor] = None

        # Shared semaphores - created lazily per event loop
        self._evaluation_semaphore: Optional[asyncio.Semaphore] = None
        self._dimension_semaphore: Optional[asyncio.Semaphore] = None
        self._llm_semaphore: Optional[asyncio.Semaphore] = None

        # Track which event loop owns the semaphores
        self._semaphore_loop: Optional[asyncio.AbstractEventLoop] = None

        logger.info(
            f"ConcurrencyManager initialized: "
            f"gpu_workers={self._config.gpu_executor_workers}, "
            f"max_evals={self._config.max_concurrent_evaluations}, "
            f"max_dims={self._config.max_concurrent_dimensions}, "
            f"max_llm={self._config.max_concurrent_llm_calls}"
        )

    @property
    def config(self) -> ConcurrencyConfig:
        """Get the concurrency configuration."""
        return self._config

    def get_gpu_executor(self) -> ThreadPoolExecutor:
        """Get the dedicated GPU thread pool executor.

        This executor is separate from asyncio's default executor,
        ensuring that GPU work doesn't block LLM response processing.
        """
        if self._gpu_executor is None:
            self._gpu_executor = ThreadPoolExecutor(
                max_workers=self._config.gpu_executor_workers,
                thread_name_prefix="gpu_worker",
            )
            logger.debug(
                f"Created GPU executor with {self._config.gpu_executor_workers} workers"
            )
        return self._gpu_executor

    def _ensure_semaphores(self) -> None:
        """Ensure semaphores exist for the current event loop.

        Semaphores are bound to a specific event loop. If the loop changes,
        we need to recreate them.
        """
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop - will be created when needed
            return

        if self._semaphore_loop is not current_loop:
            # Loop changed, recreate semaphores
            self._evaluation_semaphore = asyncio.Semaphore(
                self._config.max_concurrent_evaluations
            )
            self._dimension_semaphore = asyncio.Semaphore(
                self._config.max_concurrent_dimensions
            )
            self._llm_semaphore = asyncio.Semaphore(
                self._config.max_concurrent_llm_calls
            )
            self._semaphore_loop = current_loop
            logger.debug("Created new semaphores for current event loop")

    def get_evaluation_semaphore(self) -> asyncio.Semaphore:
        """Get the semaphore for limiting concurrent evaluations."""
        self._ensure_semaphores()
        if self._evaluation_semaphore is None:
            self._evaluation_semaphore = asyncio.Semaphore(
                self._config.max_concurrent_evaluations
            )
        return self._evaluation_semaphore

    def get_dimension_semaphore(self) -> asyncio.Semaphore:
        """Get the semaphore for limiting concurrent dimension evaluations."""
        self._ensure_semaphores()
        if self._dimension_semaphore is None:
            self._dimension_semaphore = asyncio.Semaphore(
                self._config.max_concurrent_dimensions
            )
        return self._dimension_semaphore

    def get_llm_semaphore(self) -> asyncio.Semaphore:
        """Get the shared semaphore for LLM API calls."""
        self._ensure_semaphores()
        if self._llm_semaphore is None:
            self._llm_semaphore = asyncio.Semaphore(
                self._config.max_concurrent_llm_calls
            )
        return self._llm_semaphore

    async def run_in_gpu_executor(
        self,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Run a blocking function in the GPU executor.

        This is the key fix for thread pool starvation:
        - Uses dedicated executor instead of asyncio default
        - LLM responses can be processed while GPU work runs
        - GPU operations are properly queued without blocking event loop

        Args:
            func: The blocking function to run.
            *args: Positional arguments for func.
            **kwargs: Keyword arguments for func.

        Returns:
            The result of func(*args, **kwargs).
        """
        loop = asyncio.get_running_loop()
        executor = self.get_gpu_executor()

        if kwargs:
            func = partial(func, **kwargs)

        return await loop.run_in_executor(executor, func, *args)

    def shutdown(self) -> None:
        """Shutdown the GPU executor and release resources."""
        if self._gpu_executor is not None:
            self._gpu_executor.shutdown(wait=True)
            self._gpu_executor = None
            logger.debug("GPU executor shutdown complete")

        # Clear semaphores
        self._evaluation_semaphore = None
        self._dimension_semaphore = None
        self._llm_semaphore = None
        self._semaphore_loop = None

    def reconfigure(self, config: ConcurrencyConfig) -> None:
        """Reconfigure the manager with new settings.

        This shuts down the existing executor and clears semaphores.
        New resources will be created lazily with the new config.
        """
        self.shutdown()
        self._config = config
        logger.info(f"ConcurrencyManager reconfigured: {config}")


# Module-level singleton instance
_manager: Optional[ConcurrencyManager] = None


def get_concurrency_manager(
    config: Optional[ConcurrencyConfig] = None,
) -> ConcurrencyManager:
    """Get the global concurrency manager instance.

    Args:
        config: Optional config to use when creating the manager.
               Only used on first call; subsequent calls ignore this.

    Returns:
        The singleton ConcurrencyManager instance.
    """
    global _manager
    if _manager is None:
        _manager = ConcurrencyManager(config)
    return _manager


def reset_concurrency_manager() -> None:
    """Reset the global concurrency manager.

    Use this for testing or when you need to reconfigure.
    """
    global _manager
    if _manager is not None:
        _manager.shutdown()
        _manager = None


# Convenience functions that use the global manager


def get_gpu_executor() -> ThreadPoolExecutor:
    """Get the dedicated GPU thread pool executor."""
    return get_concurrency_manager().get_gpu_executor()


def get_evaluation_semaphore() -> asyncio.Semaphore:
    """Get the semaphore for limiting concurrent evaluations."""
    return get_concurrency_manager().get_evaluation_semaphore()


def get_dimension_semaphore() -> asyncio.Semaphore:
    """Get the semaphore for limiting concurrent dimension evaluations."""
    return get_concurrency_manager().get_dimension_semaphore()


def get_llm_semaphore() -> asyncio.Semaphore:
    """Get the shared semaphore for LLM API calls."""
    return get_concurrency_manager().get_llm_semaphore()


async def run_in_gpu_executor(
    func: Callable[..., T],
    *args: Any,
    **kwargs: Any,
) -> T:
    """Run a blocking function in the dedicated GPU executor.

    Use this instead of asyncio.to_thread() for GPU-bound work
    to prevent thread pool starvation.

    Args:
        func: The blocking function to run.
        *args: Positional arguments for func.
        **kwargs: Keyword arguments for func.

    Returns:
        The result of func(*args, **kwargs).
    """
    return await get_concurrency_manager().run_in_gpu_executor(func, *args, **kwargs)


@asynccontextmanager
async def limit_concurrent_evaluations():
    """Context manager to limit concurrent evaluations.

    Usage:
        async with limit_concurrent_evaluations():
            await run_evaluation(vector, layer, behavior)
    """
    semaphore = get_evaluation_semaphore()
    async with semaphore:
        yield


@asynccontextmanager
async def limit_concurrent_dimensions():
    """Context manager to limit concurrent dimension evaluations.

    Usage:
        async with limit_concurrent_dimensions():
            await evaluate_dimension(vector, layer)
    """
    semaphore = get_dimension_semaphore()
    async with semaphore:
        yield


@asynccontextmanager
async def limit_concurrent_llm_calls():
    """Context manager to limit concurrent LLM API calls.

    Usage:
        async with limit_concurrent_llm_calls():
            await llm.generate(messages)
    """
    semaphore = get_llm_semaphore()
    async with semaphore:
        yield
