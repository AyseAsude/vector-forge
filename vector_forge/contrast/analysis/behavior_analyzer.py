"""Behavior analyzer for extracting structured behavior information.

This module analyzes behavior descriptions to extract components,
triggers, contrast dimensions, and confounds that guide the
contrast pair generation process.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from vector_forge.contrast.protocols import (
    BehaviorAnalyzerProtocol,
    BehaviorAnalysis,
    BehaviorComponent,
    BehavioralTest,
    IntensityCalibration,
    NegativeExample,
    RealisticScenario,
    ConfoundInfo,
)
from vector_forge.contrast.utils import parse_llm_json
from vector_forge.core.protocols import Message
from vector_forge.llm import JSON_RESPONSE_FORMAT
from vector_forge.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)


ANALYSIS_PROMPT = '''You are analyzing a behavior for steering vector extraction. Your goal is to deeply understand this behavior so we can generate CLEAN, CONTRASTIVE training data.

BEHAVIOR: {behavior_description}

Think carefully and thoroughly. The quality of your analysis determines whether we extract a clean signal or a noisy mixture.

---

## 1. CORE DEFINITION
What is the ESSENCE of this behavior in one sentence?
Be precise - this distinguishes it from similar behaviors.

## 2. THE BEHAVIORAL TEST (Critical - this drives everything)
Design a TEST that reveals this behavior. This is the most important part.

Think about:
- What situation would REVEAL whether this behavior is present or absent?
- What choice does the model face in that situation?
- What SINGLE VARIABLE distinguishes exhibiting from not exhibiting?
- What does the model do differently when the behavior is present vs absent?

The distinguishing variable is THE thing we extract - be specific and precise.

## 3. INTENSITY CALIBRATION
How does THIS SPECIFIC behavior manifest at different intensity levels?
Describe what each intensity looks like FOR THIS BEHAVIOR:
- EXTREME: Maximum, unmistakable expression
- HIGH: Clearly present, obvious
- MEDIUM: Noticeable, balanced
- NATURAL: Subtle, deployment-realistic

Be specific to this behavior - not generic descriptions.

## 4. WHAT THIS IS NOT
What similar behaviors might be CONFUSED with this?
For each, explain the boundary and why it's different.
This prevents capturing the wrong signal.

## 5. COMPONENTS
Break into distinct, separable aspects of the behavior.
Each should be observable and have specific linguistic markers.
Include markers for both presence AND absence.

## 6. PRESENCE/ABSENCE MARKERS
List explicit linguistic patterns that indicate:
- Behavior IS present
- Behavior is NOT present

These guide generation and validation.

## 7. REALISTIC SCENARIOS
Generate diverse scenarios where this behavior would naturally arise.
Cover the range of contexts relevant to THIS behavior.
Each should naturally create the behavioral test situation.

## 8. CONFOUNDS
What factors correlate with but are NOT this behavior?
How can each be controlled?

---

Return JSON:
{{
  "core_definition": "one precise sentence",

  "behavioral_test": {{
    "description": "full description of the test",
    "user_action": "what user does to trigger the test",
    "model_choice": "what decision the model faces",
    "distinguishing_variable": "THE thing that differs (be specific)",
    "present_response_pattern": "what model does if behavior present",
    "absent_response_pattern": "what model does if behavior absent"
  }},

  "intensity_calibration": {{
    "extreme_looks_like": "maximum expression for THIS behavior",
    "high_looks_like": "clear expression for THIS behavior",
    "medium_looks_like": "balanced expression for THIS behavior",
    "natural_looks_like": "subtle expression for THIS behavior"
  }},

  "not_this_behavior": [
    {{
      "similar_behavior": "what it might be confused with",
      "why_different": "why it's not the same thing",
      "example": "concrete example of the non-behavior"
    }}
  ],

  "components": [
    {{
      "name": "short_name",
      "description": "what this component represents",
      "markers": ["specific patterns indicating presence"],
      "opposite_markers": ["specific patterns indicating absence"]
    }}
  ],

  "presence_markers": ["phrases/patterns indicating behavior IS present"],
  "absence_markers": ["phrases/patterns indicating behavior is NOT present"],

  "realistic_scenarios": [
    {{
      "setup": "the situation",
      "user_persona": "who the user is",
      "natural_trigger": "why behavior would arise here"
    }}
  ],

  "trigger_conditions": ["when this behavior naturally manifests"],
  "contrast_dimensions": ["clearest ways to show presence vs absence"],

  "confounds_to_avoid": [
    {{
      "factor": "the confound",
      "strategy": "how to control for it"
    }}
  ]
}}

Be specific, nuanced, and thoughtful. Vague analysis produces noisy vectors.
The behavioral_test and distinguishing_variable are CRITICAL - they drive all downstream generation.'''


class BehaviorAnalyzer(BehaviorAnalyzerProtocol):
    """Analyzes behavior descriptions using LLM.

    Extracts structured information about behaviors including:
    - Components: Distinct aspects of the behavior
    - Trigger conditions: When the behavior manifests
    - Contrast dimensions: How to show presence vs absence
    - Confounds: Factors to control for

    Example:
        >>> analyzer = BehaviorAnalyzer(llm_client)
        >>> analysis = await analyzer.analyze("sycophancy")
        >>> print(analysis.components[0].name)
        "excessive_agreement"
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ):
        """Initialize the behavior analyzer.

        Args:
            llm_client: LLM client for analysis.
            temperature: Generation temperature.
            max_tokens: Maximum tokens for response (None = provider default).
        """
        self._llm = llm_client
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def analyze(self, behavior_description: str) -> BehaviorAnalysis:
        """Analyze behavior and return structured analysis.

        Args:
            behavior_description: Natural language description of the behavior.

        Returns:
            BehaviorAnalysis containing components, triggers, etc.

        Raises:
            ValueError: If analysis fails or cannot be parsed.
        """
        logger.info(f"Analyzing behavior: {behavior_description[:50]}...")

        prompt = ANALYSIS_PROMPT.format(behavior_description=behavior_description)

        response = await self._llm.complete(
            messages=[Message(role="user", content=prompt)],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            response_format=JSON_RESPONSE_FORMAT,
        )

        try:
            data = parse_llm_json(response.content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse behavior analysis: {e}")
            raise ValueError(f"Failed to parse behavior analysis: {e}")

        components = self._extract_components(data)

        if not components:
            logger.warning("No components extracted, using default")
            components = [
                BehaviorComponent(
                    name="main_behavior",
                    description=behavior_description,
                    markers=[],
                    opposite_markers=[],
                )
            ]

        # Extract enhanced fields
        negative_examples = self._extract_negative_examples(data)
        realistic_scenarios = self._extract_realistic_scenarios(data)
        confound_details = self._extract_confound_details(data)

        # Extract simple confounds list for backward compatibility
        confounds_simple = self._extract_confounds_simple(data)

        # NEW: Extract behavioral test
        behavioral_test = self._extract_behavioral_test(data)

        # NEW: Extract intensity calibration
        intensity_calibration = self._extract_intensity_calibration(data)

        # NEW: Extract explicit markers
        presence_markers = data.get("presence_markers", [])
        absence_markers = data.get("absence_markers", [])

        analysis = BehaviorAnalysis(
            behavior_name=self._extract_name(behavior_description),
            description=behavior_description,
            components=components,
            trigger_conditions=data.get("trigger_conditions", []),
            contrast_dimensions=data.get("contrast_dimensions", []),
            confounds_to_avoid=confounds_simple,
            # Enhanced fields
            core_definition=data.get("core_definition", ""),
            not_this_behavior=negative_examples,
            realistic_scenarios=realistic_scenarios,
            confound_details=confound_details,
            # NEW: Behavioral test and intensity
            behavioral_test=behavioral_test,
            intensity_calibration=intensity_calibration,
            presence_markers=presence_markers,
            absence_markers=absence_markers,
        )

        logger.info(
            f"Behavior analysis complete: {len(analysis.components)} components, "
            f"{len(analysis.trigger_conditions)} triggers, "
            f"{len(analysis.not_this_behavior)} negative examples, "
            f"{len(analysis.realistic_scenarios)} scenarios, "
            f"behavioral_test={'yes' if behavioral_test else 'no'}"
        )

        return analysis

    def _extract_components(self, data: Dict[str, Any]) -> list[BehaviorComponent]:
        """Extract behavior components from parsed data."""
        components = []

        for c in data.get("components", []):
            if not isinstance(c, dict):
                continue

            name = c.get("name", "")
            if not name:
                continue

            components.append(
                BehaviorComponent(
                    name=name,
                    description=c.get("description", ""),
                    markers=c.get("markers", []),
                    opposite_markers=c.get("opposite_markers", []),
                )
            )

        return components

    def _extract_name(self, description: str) -> str:
        """Extract a short name from the behavior description."""
        # Take first 50 chars or first sentence
        name = description.split(".")[0].strip()
        if len(name) > 50:
            name = name[:47] + "..."
        return name

    def _extract_negative_examples(self, data: Dict[str, Any]) -> list[NegativeExample]:
        """Extract negative examples (what this behavior is NOT)."""
        examples = []
        for item in data.get("not_this_behavior", []):
            if not isinstance(item, dict):
                continue
            examples.append(
                NegativeExample(
                    similar_behavior=item.get("similar_behavior", ""),
                    why_different=item.get("why_different", ""),
                    example=item.get("example", ""),
                )
            )
        return examples

    def _extract_realistic_scenarios(self, data: Dict[str, Any]) -> list[RealisticScenario]:
        """Extract realistic scenarios from analysis."""
        scenarios = []
        for item in data.get("realistic_scenarios", []):
            if not isinstance(item, dict):
                continue
            scenarios.append(
                RealisticScenario(
                    setup=item.get("setup", ""),
                    user_persona=item.get("user_persona", ""),
                    natural_trigger=item.get("natural_trigger", ""),
                    stakes=item.get("stakes", ""),  # Optional now
                )
            )
        return scenarios

    def _extract_confound_details(self, data: Dict[str, Any]) -> list[ConfoundInfo]:
        """Extract detailed confound information."""
        confounds = []
        for item in data.get("confounds_to_avoid", []):
            if isinstance(item, dict):
                confounds.append(
                    ConfoundInfo(
                        factor=item.get("factor", ""),
                        difficulty=item.get("difficulty", ""),  # Optional now
                        strategy=item.get("strategy", ""),
                    )
                )
        return confounds

    def _extract_confounds_simple(self, data: Dict[str, Any]) -> list[str]:
        """Extract simple confound list for backward compatibility."""
        confounds = []
        for item in data.get("confounds_to_avoid", []):
            if isinstance(item, str):
                confounds.append(item)
            elif isinstance(item, dict):
                factor = item.get("factor", "")
                if factor:
                    confounds.append(factor)
        return confounds

    def _extract_behavioral_test(self, data: Dict[str, Any]) -> Optional[BehavioralTest]:
        """Extract behavioral test from analysis data."""
        test_data = data.get("behavioral_test", {})
        if not test_data or not isinstance(test_data, dict):
            return None

        # Require at minimum the distinguishing variable
        distinguishing_var = test_data.get("distinguishing_variable", "")
        if not distinguishing_var:
            return None

        return BehavioralTest(
            description=test_data.get("description", ""),
            user_action=test_data.get("user_action", ""),
            model_choice=test_data.get("model_choice", ""),
            distinguishing_variable=distinguishing_var,
            present_response_pattern=test_data.get("present_response_pattern", ""),
            absent_response_pattern=test_data.get("absent_response_pattern", ""),
        )

    def _extract_intensity_calibration(self, data: Dict[str, Any]) -> Optional[IntensityCalibration]:
        """Extract intensity calibration from analysis data."""
        cal_data = data.get("intensity_calibration", {})
        if not cal_data or not isinstance(cal_data, dict):
            return None

        # Need at least some calibration data
        if not any(cal_data.values()):
            return None

        return IntensityCalibration(
            extreme_looks_like=cal_data.get("extreme_looks_like", "Maximum expression of behavior"),
            high_looks_like=cal_data.get("high_looks_like", "Clear but plausible expression"),
            medium_looks_like=cal_data.get("medium_looks_like", "Balanced expression"),
            natural_looks_like=cal_data.get("natural_looks_like", "Subtle, realistic expression"),
        )
