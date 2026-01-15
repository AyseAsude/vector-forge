"""Chat service for dual LLM inference with steering vector comparison.

Provides generation of both baseline (unsteered) and steered responses
using the same target model used for vector extraction.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

import torch

from vector_forge.storage import StorageManager

logger = logging.getLogger(__name__)


class ChatService:
    """Service for chat inference with steering vector comparison.

    Generates both baseline and steered responses for comparison.
    Uses the same HuggingFaceBackend as the extraction pipeline.

    Example:
        >>> service = ChatService(extraction_id="sycophancy_20240115_123456")
        >>> baseline = await service.generate_baseline(messages, temperature=0.7)
        >>> steered = await service.generate_steered(messages, layer=16, strength=1.0)
    """

    # Class-level cache for model backend
    _cached_backend: Optional[Any] = None
    _cached_model_id: Optional[str] = None

    def __init__(self, extraction_id: str) -> None:
        """Initialize chat service for an extraction.

        Args:
            extraction_id: Session ID of the completed extraction.
        """
        self.extraction_id = extraction_id
        self._storage = StorageManager()
        self._session_store = None
        self._config: Dict[str, Any] = {}
        self._vector: Optional[torch.Tensor] = None
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Ensure service is initialized with session data."""
        if self._initialized:
            return

        try:
            self._session_store = self._storage.get_session(self.extraction_id)

            # Load config
            config_path = self._session_store.session_path / "config.json"
            if config_path.exists():
                import json

                with open(config_path, "r") as f:
                    self._config = json.load(f)

            self._initialized = True

        except Exception as e:
            logger.error(f"Failed to initialize chat service: {e}")
            raise

    def _get_target_model(self) -> str:
        """Get target model ID from config."""
        return self._config.get("target_model", "Qwen/Qwen2.5-0.5B-Instruct")

    async def _ensure_backend(self) -> Any:
        """Ensure model backend is loaded.

        Returns:
            HuggingFaceBackend instance.
        """
        await self._ensure_initialized()

        model_id = self._get_target_model()

        # Check if we can reuse cached backend
        if (
            ChatService._cached_backend is not None
            and ChatService._cached_model_id == model_id
        ):
            return ChatService._cached_backend

        # Load model
        logger.info(f"Loading model for chat: {model_id}")

        loop = asyncio.get_event_loop()
        backend = await loop.run_in_executor(None, self._load_model, model_id)

        # Cache
        ChatService._cached_backend = backend
        ChatService._cached_model_id = model_id

        return backend

    def _load_model(self, model_id: str) -> Any:
        """Load HuggingFace model (blocking).

        Args:
            model_id: Model identifier.

        Returns:
            HuggingFaceBackend instance.
        """
        from steerex import HuggingFaceBackend
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Check for accelerate
        has_accelerate = False
        try:
            import accelerate  # noqa: F401

            has_accelerate = True
        except ImportError:
            pass

        if has_accelerate:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=torch.float16,
                device_map="auto",
            )
        elif torch.cuda.is_available():
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=torch.float16,
            )
            model = model.cuda()
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=torch.float32,
            )

        return HuggingFaceBackend(model=model, tokenizer=tokenizer)

    async def _load_vector(self, vector_path: str = "vectors/final.pt") -> torch.Tensor:
        """Load steering vector from session store.

        Args:
            vector_path: Relative path to vector file.

        Returns:
            Loaded tensor.
        """
        await self._ensure_initialized()

        if self._vector is not None:
            return self._vector

        loop = asyncio.get_event_loop()

        def load():
            return self._session_store.load_vector(vector_path)

        self._vector = await loop.run_in_executor(None, load)
        return self._vector

    def _format_messages(self, backend: Any, messages: List[Dict[str, str]]) -> str:
        """Format messages using the model's native chat template.

        Uses tokenizer.apply_chat_template() which is the HuggingFace standard
        for instruction-tuned models (Qwen, Llama, Mistral, etc.).

        Args:
            backend: HuggingFaceBackend instance with tokenizer.
            messages: List of {role, content} dicts.

        Returns:
            Formatted prompt string with proper special tokens.
        """
        return backend.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    async def generate_baseline(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 256,
    ) -> str:
        """Generate baseline (unsteered) response.

        Uses the model's native chat template for proper formatting.

        Args:
            messages: Conversation history as list of {role, content} dicts.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            Generated response text.
        """
        backend = await self._ensure_backend()
        loop = asyncio.get_event_loop()

        def generate():
            prompt = self._format_messages(backend, messages)
            return backend.generate(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
            )

        try:
            return await loop.run_in_executor(None, generate)
        except Exception as e:
            logger.error(f"Baseline generation failed: {e}")
            return f"Generation error: {e}"

    async def generate_steered(
        self,
        messages: List[Dict[str, str]],
        layer: Optional[int] = None,
        strength: float = 1.0,
        temperature: float = 0.7,
        max_tokens: int = 256,
        vector_path: str = "vectors/final.pt",
    ) -> str:
        """Generate steered response.

        Uses the model's native chat template for proper formatting.

        Args:
            messages: Conversation history as list of {role, content} dicts.
            layer: Layer to apply steering (None = use best from config).
            strength: Steering strength multiplier.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            vector_path: Path to vector file.

        Returns:
            Generated response text.
        """
        from steerex import VectorSteering

        backend = await self._ensure_backend()
        vector = await self._load_vector(vector_path)

        # Get layer from config if not specified
        if layer is None:
            layer = self._config.get("best_layer", 16)

        loop = asyncio.get_event_loop()

        def generate():
            prompt = self._format_messages(backend, messages)
            steering = VectorSteering(vector=vector.detach())

            return backend.generate_with_steering(
                prompt,
                steering_mode=steering,
                layers=layer,
                strength=strength,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
            )

        try:
            return await loop.run_in_executor(None, generate)
        except Exception as e:
            logger.error(f"Steered generation failed: {e}")
            return f"Generation error: {e}"

    @classmethod
    def clear_cache(cls) -> None:
        """Clear cached model backend."""
        import gc

        cls._cached_backend = None
        cls._cached_model_id = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
