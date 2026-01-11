"""GPU memory management for extraction tasks.

Provides profile-based memory management instead of formula-based estimation.
This approach measures actual memory usage and makes decisions based on real data.

Design principles:
- Measure, don't estimate: Use actual GPU memory measurements
- Adaptive: Works for any model without recalibration
- Robust: Graceful handling of OOM with automatic recovery
"""

from __future__ import annotations

import asyncio
import gc
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List, Optional, TypeVar

import torch

if TYPE_CHECKING:
    from steering_vectors import HuggingFaceBackend, TrainingDatapoint

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class MemoryProfile:
    """Result of memory profiling for a model configuration."""

    memory_per_extraction_gb: float
    model_memory_gb: float
    free_memory_gb: float
    total_memory_gb: float

    @property
    def safe_concurrent_extractions(self) -> int:
        """Calculate safe number of concurrent extractions."""
        if self.memory_per_extraction_gb <= 0:
            return 1

        # Leave 20% headroom for CUDA fragmentation and overhead
        usable_memory = self.free_memory_gb * 0.8
        safe = max(1, int(usable_memory / self.memory_per_extraction_gb))

        logger.info(
            f"Memory profile: {self.free_memory_gb:.1f}GB free, "
            f"{self.memory_per_extraction_gb:.1f}GB/extraction -> "
            f"safe_concurrency={safe}"
        )

        return safe


class ExtractionMemoryProfiler:
    """Profiles actual GPU memory usage for extraction operations.

    Instead of estimating memory with formulas, this class measures actual
    memory usage by running a minimal extraction operation. This provides
    accurate memory requirements for any model/configuration combination.
    """

    # Minimum datapoints needed for profiling (uses subset for speed)
    MIN_PROFILE_DATAPOINTS = 4

    # Safety multiplier applied to measured memory (accounts for variance and concurrent overhead)
    SAFETY_MULTIPLIER = 2.0

    def __init__(self, backend: "HuggingFaceBackend") -> None:
        """Initialize profiler with model backend.

        Args:
            backend: The steering-vectors backend to profile.
        """
        self._backend = backend
        self._cached_profile: Optional[MemoryProfile] = None

    def profile(
        self,
        datapoints: List["TrainingDatapoint"],
        batch_size: int = 8,
        layer: Optional[int] = None,
    ) -> MemoryProfile:
        """Profile memory usage by running a minimal extraction.

        Runs a single optimization iteration to measure actual memory delta.
        Result is cached for subsequent calls.

        Args:
            datapoints: Sample datapoints (only first few are used).
            batch_size: Batch size for optimization.
            layer: Target layer (defaults to middle layer).

        Returns:
            MemoryProfile with measured memory requirements.
        """
        if self._cached_profile is not None:
            return self._cached_profile

        if not torch.cuda.is_available():
            logger.warning("CUDA not available, using default memory profile")
            return MemoryProfile(
                memory_per_extraction_gb=0.0,
                model_memory_gb=0.0,
                free_memory_gb=float("inf"),
                total_memory_gb=float("inf"),
            )

        # Use subset of datapoints for fast profiling
        profile_datapoints = datapoints[: self.MIN_PROFILE_DATAPOINTS]
        if not profile_datapoints:
            logger.warning("No datapoints for profiling, using conservative estimate")
            return self._create_conservative_profile()

        target_layer = layer or self._backend.get_num_layers() // 2

        # Clear memory and get baseline
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        baseline_allocated = torch.cuda.memory_allocated()
        total_memory = torch.cuda.get_device_properties(0).total_memory

        try:
            # Run minimal optimization to measure memory
            memory_delta = self._measure_extraction_memory(
                profile_datapoints,
                batch_size,
                target_layer,
            )

            # Apply safety multiplier
            memory_per_extraction = memory_delta * self.SAFETY_MULTIPLIER

            # Calculate free memory (total - currently allocated by model)
            model_memory = baseline_allocated / (1024**3)
            free_memory = (total_memory - baseline_allocated) / (1024**3)

            self._cached_profile = MemoryProfile(
                memory_per_extraction_gb=memory_per_extraction,
                model_memory_gb=model_memory,
                free_memory_gb=free_memory,
                total_memory_gb=total_memory / (1024**3),
            )

            logger.info(
                f"Profiled extraction memory: {memory_per_extraction:.2f}GB "
                f"(measured {memory_delta:.2f}GB + {self.SAFETY_MULTIPLIER}x safety)"
            )

            return self._cached_profile

        except Exception as e:
            logger.warning(f"Memory profiling failed: {e}, using conservative estimate")
            return self._create_conservative_profile()

        finally:
            # Clean up profiling artifacts
            gc.collect()
            torch.cuda.empty_cache()

    def _measure_extraction_memory(
        self,
        datapoints: List["TrainingDatapoint"],
        batch_size: int,
        layer: int,
    ) -> float:
        """Run minimal extraction and measure memory delta.

        Returns memory usage in GB.
        """
        from steering_vectors import SteeringOptimizer, VectorSteering
        from steering_vectors.core.config import OptimizationConfig

        # Record baseline before optimization
        baseline = torch.cuda.memory_allocated()

        # Configure for enough iterations to capture peak memory pattern
        config = OptimizationConfig(
            lr=0.1,
            max_iters=5,  # Enough iterations to capture peak memory
            use_batched=True,
            batch_size=batch_size,
        )

        steering = VectorSteering()
        optimizer = SteeringOptimizer(
            backend=self._backend,
            steering_mode=steering,
            config=config,
        )

        # Run optimization and measure peak memory
        optimizer.optimize(datapoints, layer=layer)

        peak_memory = torch.cuda.max_memory_allocated()

        # Return the delta (peak - baseline model memory)
        # This represents memory needed for extraction activations
        return (peak_memory - baseline) / (1024**3)

    def _create_conservative_profile(self) -> MemoryProfile:
        """Create a conservative profile when measurement fails."""
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            allocated = torch.cuda.memory_allocated() / (1024**3)
            free = total - allocated
        else:
            total = 0.0
            allocated = 0.0
            free = float("inf")

        # Conservative estimate: assume 25% of free memory per extraction
        return MemoryProfile(
            memory_per_extraction_gb=free * 0.25,
            model_memory_gb=allocated,
            free_memory_gb=free,
            total_memory_gb=total,
        )

    def clear_cache(self) -> None:
        """Clear cached profile (call when model changes)."""
        self._cached_profile = None


class MemoryAwareSemaphore:
    """Semaphore that checks real GPU memory before allowing acquisition.

    Unlike a standard semaphore that only counts slots, this checks actual
    GPU memory availability before allowing an extraction to proceed.
    """

    # Minimum headroom required before allowing acquisition (GB)
    MIN_HEADROOM_GB = 0.5

    # Time to wait between memory checks when blocked (seconds)
    POLL_INTERVAL = 0.5

    def __init__(
        self,
        memory_per_extraction_gb: float,
        max_concurrent: int,
    ) -> None:
        """Initialize memory-aware semaphore.

        Args:
            memory_per_extraction_gb: Memory required per extraction.
            max_concurrent: Maximum concurrent extractions (from config).
        """
        self._memory_required = memory_per_extraction_gb
        self._max_concurrent = max_concurrent
        self._lock = asyncio.Lock()
        self._active_count = 0

    async def acquire(self) -> None:
        """Acquire a slot, waiting if necessary for memory availability."""
        while True:
            async with self._lock:
                if self._active_count >= self._max_concurrent:
                    # Hit hard limit, must wait
                    pass
                elif self._has_sufficient_memory():
                    self._active_count += 1
                    logger.debug(
                        f"Acquired extraction slot ({self._active_count} active)"
                    )
                    return

            # Wait and retry
            await asyncio.sleep(self.POLL_INTERVAL)

    async def release(self) -> None:
        """Release an extraction slot."""
        async with self._lock:
            self._active_count = max(0, self._active_count - 1)
            logger.debug(f"Released extraction slot ({self._active_count} active)")

    def _has_sufficient_memory(self) -> bool:
        """Check if sufficient GPU memory is available."""
        if not torch.cuda.is_available():
            return True

        free_gb = self._get_free_memory_gb()
        required = self._memory_required + self.MIN_HEADROOM_GB

        return free_gb >= required

    def _get_free_memory_gb(self) -> float:
        """Get current free GPU memory in GB."""
        if not torch.cuda.is_available():
            return float("inf")

        total = torch.cuda.get_device_properties(0).total_memory
        allocated = torch.cuda.memory_allocated()

        return (total - allocated) / (1024**3)

    @property
    def active_count(self) -> int:
        """Number of currently active extractions."""
        return self._active_count

    async def __aenter__(self) -> "MemoryAwareSemaphore":
        await self.acquire()
        return self

    async def __aexit__(self, *args) -> None:
        await self.release()


class OOMHandler:
    """Handles CUDA out-of-memory errors with graceful recovery.

    Wraps extraction operations to catch OOM errors, perform cleanup,
    and optionally retry with recovery strategies.
    """

    # Maximum retry attempts after OOM
    MAX_RETRIES = 2

    # Time to wait after OOM before retry (seconds)
    RETRY_DELAY = 1.0

    async def run_with_protection(
        self,
        operation: Callable[[], T],
        cleanup_fn: Optional[Callable[[], None]] = None,
    ) -> T:
        """Run an operation with OOM protection and automatic retry.

        Args:
            operation: The async or sync operation to run.
            cleanup_fn: Optional cleanup function to call after OOM.

        Returns:
            Result of the operation.

        Raises:
            torch.cuda.OutOfMemoryError: If all retries fail.
        """
        last_error: Optional[Exception] = None

        for attempt in range(self.MAX_RETRIES + 1):
            try:
                # Clear memory before attempt
                self._clear_memory()

                # Run the operation
                if asyncio.iscoroutinefunction(operation):
                    return await operation()
                else:
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, operation)

            except torch.cuda.OutOfMemoryError as e:
                last_error = e
                logger.warning(
                    f"CUDA OOM on attempt {attempt + 1}/{self.MAX_RETRIES + 1}: {e}"
                )

                # Aggressive cleanup
                self._aggressive_cleanup()

                if cleanup_fn:
                    try:
                        cleanup_fn()
                    except Exception as cleanup_error:
                        logger.warning(f"Cleanup function failed: {cleanup_error}")

                if attempt < self.MAX_RETRIES:
                    logger.info(f"Retrying in {self.RETRY_DELAY}s...")
                    await asyncio.sleep(self.RETRY_DELAY)

        # All retries exhausted
        raise last_error or RuntimeError("OOM handler failed unexpectedly")

    def _clear_memory(self) -> None:
        """Standard memory cleanup."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _aggressive_cleanup(self) -> None:
        """Aggressive memory cleanup after OOM."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Reset memory stats to allow fresh allocation
            torch.cuda.reset_peak_memory_stats()


def clear_gpu_memory() -> None:
    """Utility function to clear GPU memory.

    Call this between operations to free cached memory.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_gpu_memory_info() -> dict:
    """Get current GPU memory information.

    Returns:
        Dictionary with memory stats (all values in GB).
    """
    if not torch.cuda.is_available():
        return {
            "available": False,
            "total_gb": 0.0,
            "allocated_gb": 0.0,
            "free_gb": 0.0,
        }

    props = torch.cuda.get_device_properties(0)
    total = props.total_memory / (1024**3)
    allocated = torch.cuda.memory_allocated() / (1024**3)

    return {
        "available": True,
        "device": props.name,
        "total_gb": total,
        "allocated_gb": allocated,
        "free_gb": total - allocated,
    }
