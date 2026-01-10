"""Composite validator using Chain of Responsibility pattern.

This module provides a way to chain multiple validators together,
failing fast on the first failure for efficiency.
"""

from __future__ import annotations

import logging
from typing import List

from vector_forge.contrast.protocols import (
    ContrastValidatorProtocol,
    ContrastPair,
    BehaviorAnalysis,
    ValidationResult,
    merge_validation_results,
)

logger = logging.getLogger(__name__)


class CompositeContrastValidator(ContrastValidatorProtocol):
    """Chain multiple validators using Chain of Responsibility pattern.

    Runs validators in order and fails fast on the first failure.
    This allows cheap validators (like embedding) to run first,
    avoiding expensive LLM calls when pairs obviously fail.

    Example:
        >>> validator = CompositeContrastValidator([
        ...     EmbeddingContrastValidator(min_distance=0.3),
        ...     LLMContrastValidator(judge_llm),
        ... ])
        >>> result = await validator.validate(pair, analysis)
    """

    def __init__(
        self,
        validators: List[ContrastValidatorProtocol],
        fail_fast: bool = True,
    ):
        """Initialize the composite validator.

        Args:
            validators: List of validators to chain (order matters).
            fail_fast: If True, stop on first failure. If False, run all.
        """
        self._validators = validators
        self._fail_fast = fail_fast

    async def validate(
        self,
        pair: ContrastPair,
        analysis: BehaviorAnalysis,
    ) -> ValidationResult:
        """Run all validators in order, optionally failing fast.

        Args:
            pair: The contrast pair to validate.
            analysis: Behavior analysis for context.

        Returns:
            Combined ValidationResult from all validators.
        """
        if not self._validators:
            logger.warning("No validators configured")
            return ValidationResult(
                is_valid=True,
                dst_behavior_score=-1,
                src_behavior_score=-1,
                semantic_distance=-1,
                contrast_quality=5.0,
                confounds_detected=[],
                reasoning="No validators configured",
            )

        # Initialize with empty result
        combined_result = ValidationResult(
            is_valid=True,
            dst_behavior_score=-1,
            src_behavior_score=-1,
            semantic_distance=-1,
            contrast_quality=0,
            confounds_detected=[],
            reasoning="",
        )

        for i, validator in enumerate(self._validators):
            validator_name = validator.__class__.__name__

            try:
                result = await validator.validate(pair, analysis)

                logger.debug(
                    f"Validator {validator_name}: "
                    f"valid={result.is_valid}, "
                    f"quality={result.contrast_quality:.1f}"
                )

                # Merge results
                combined_result = merge_validation_results(combined_result, result)

                # Fail fast if enabled and validation failed
                if self._fail_fast and not result.is_valid:
                    logger.debug(f"Failing fast at validator {validator_name}")
                    return combined_result

            except Exception as e:
                logger.error(f"Validator {validator_name} raised exception: {e}")
                combined_result.is_valid = False
                combined_result.confounds_detected.append(f"{validator_name}_error")
                combined_result.reasoning += f" | {validator_name} error: {str(e)}"

                if self._fail_fast:
                    return combined_result

        return combined_result

    def add_validator(self, validator: ContrastValidatorProtocol) -> None:
        """Add a validator to the chain.

        Args:
            validator: Validator to add.
        """
        self._validators.append(validator)

    def insert_validator(
        self,
        index: int,
        validator: ContrastValidatorProtocol,
    ) -> None:
        """Insert a validator at a specific position.

        Args:
            index: Position to insert at.
            validator: Validator to insert.
        """
        self._validators.insert(index, validator)

    @property
    def validator_names(self) -> List[str]:
        """Get names of all validators in the chain."""
        return [v.__class__.__name__ for v in self._validators]
