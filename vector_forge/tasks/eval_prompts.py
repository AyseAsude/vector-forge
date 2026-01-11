"""Evaluation prompt generation and selection for tournament mode.

This module provides:
- EvaluationPrompt: Prompt with quality score for ranking
- PromptGenerator: Generates prompts with quality scores from behavior
- PromptSelector: Selects prompts based on depth (best + random mix)

The key insight is that not all prompts are equally useful for quick elimination.
High-quality prompts directly trigger the target behavior and provide clear signal.
Low-quality prompts test edge cases and are better suited for finals.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any
import random

from vector_forge.tasks.config import EvalDepth, EVAL_DEPTH_BUDGET, EVAL_DEPTH_BEST_RATIO


class PromptCategory(str, Enum):
    """Category of evaluation prompt."""

    BEHAVIOR = "behavior"  # Tests target behavior induction
    SPECIFICITY = "specificity"  # Tests that neutral prompts stay neutral
    COHERENCE = "coherence"  # Tests output quality/fluency
    CAPABILITY = "capability"  # Tests preserved capabilities
    GENERALIZATION = "generalization"  # Tests OOD generalization


@dataclass
class EvaluationPrompt:
    """An evaluation prompt with quality metadata.

    Quality score indicates how useful this prompt is for quick evaluation:
    - High score (8-10): Directly triggers behavior, clear signal
    - Medium score (5-7): Moderate behavior trigger, some ambiguity
    - Low score (1-4): Edge case, subtle, good for fine discrimination

    Attributes:
        text: The prompt text.
        category: Which evaluation dimension this tests.
        quality_score: How useful for quick eval (1-10, higher = better signal).
        source: Where this prompt came from (scenario, trigger, domain, etc.).
        metadata: Additional info (component, domain, etc.).
    """

    text: str
    category: PromptCategory
    quality_score: float = 5.0  # Default middle quality
    source: str = "generated"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Clamp quality score to valid range
        self.quality_score = max(1.0, min(10.0, self.quality_score))


@dataclass
class CapabilityPrompt(EvaluationPrompt):
    """Capability test prompt with expected answer."""

    expected_answer: str = ""

    def __post_init__(self):
        super().__post_init__()
        self.category = PromptCategory.CAPABILITY


@dataclass
class PromptSet:
    """A complete set of evaluation prompts organized by category.

    Prompts are pre-sorted by quality score within each category.
    """

    behavior: List[EvaluationPrompt] = field(default_factory=list)
    specificity: List[EvaluationPrompt] = field(default_factory=list)
    coherence: List[EvaluationPrompt] = field(default_factory=list)
    capability: List[CapabilityPrompt] = field(default_factory=list)
    generalization: List[EvaluationPrompt] = field(default_factory=list)

    # Random seed for consistent selection across samples
    seed: int = 42

    def __post_init__(self):
        """Sort all prompts by quality score (descending)."""
        self._sort_all()

    def _sort_all(self) -> None:
        """Sort all prompt lists by quality score (best first)."""
        self.behavior.sort(key=lambda p: p.quality_score, reverse=True)
        self.specificity.sort(key=lambda p: p.quality_score, reverse=True)
        self.coherence.sort(key=lambda p: p.quality_score, reverse=True)
        self.capability.sort(key=lambda p: p.quality_score, reverse=True)
        self.generalization.sort(key=lambda p: p.quality_score, reverse=True)

    def get_category(self, category: PromptCategory) -> List[EvaluationPrompt]:
        """Get prompts for a category."""
        return {
            PromptCategory.BEHAVIOR: self.behavior,
            PromptCategory.SPECIFICITY: self.specificity,
            PromptCategory.COHERENCE: self.coherence,
            PromptCategory.CAPABILITY: self.capability,
            PromptCategory.GENERALIZATION: self.generalization,
        }[category]


class PromptSelector:
    """Selects prompts from a PromptSet based on evaluation depth.

    Uses a best + random mix strategy:
    - Quick (10%): 50% best prompts + 50% random
    - Medium (40%): 60% best prompts + 40% random
    - Full (100%): All prompts

    This ensures we get high-signal prompts for quick elimination while
    maintaining some diversity to catch edge cases.
    """

    def __init__(self, prompt_set: PromptSet):
        """Initialize with a complete prompt set.

        Args:
            prompt_set: The full set of evaluation prompts.
        """
        self._prompts = prompt_set
        self._rng = random.Random(prompt_set.seed)

    def select(
        self,
        category: PromptCategory,
        count: int,
        depth: EvalDepth,
    ) -> List[EvaluationPrompt]:
        """Select prompts for a category at given depth.

        Args:
            category: Which category to select from.
            count: How many prompts to select.
            depth: Evaluation depth (determines best/random ratio).

        Returns:
            Selected prompts (consistent for same seed).
        """
        all_prompts = self._prompts.get_category(category)

        if not all_prompts:
            return []

        # Full depth: return all (up to count)
        if depth == EvalDepth.FULL:
            return all_prompts[:count]

        # Ensure we don't request more than available
        count = min(count, len(all_prompts))
        if count == 0:
            return []

        # Get best/random ratio for this depth
        best_ratio = EVAL_DEPTH_BEST_RATIO[depth]
        num_best = max(1, int(count * best_ratio))
        num_random = count - num_best

        # Select best prompts (already sorted by quality)
        best_prompts = all_prompts[:num_best]

        # Select random from remaining
        remaining = all_prompts[num_best:]
        if remaining and num_random > 0:
            # Use seeded RNG for consistency across samples
            random_prompts = self._rng.sample(
                remaining,
                min(num_random, len(remaining))
            )
        else:
            random_prompts = []

        # Combine and return
        selected = best_prompts + random_prompts
        return selected

    def select_behavior(self, count: int, depth: EvalDepth) -> List[EvaluationPrompt]:
        """Select behavior evaluation prompts."""
        return self.select(PromptCategory.BEHAVIOR, count, depth)

    def select_specificity(self, count: int, depth: EvalDepth) -> List[EvaluationPrompt]:
        """Select specificity evaluation prompts."""
        return self.select(PromptCategory.SPECIFICITY, count, depth)

    def select_coherence(self, count: int, depth: EvalDepth) -> List[EvaluationPrompt]:
        """Select coherence evaluation prompts."""
        return self.select(PromptCategory.COHERENCE, count, depth)

    def select_capability(self, count: int, depth: EvalDepth) -> List[CapabilityPrompt]:
        """Select capability evaluation prompts."""
        prompts = self.select(PromptCategory.CAPABILITY, count, depth)
        return [p for p in prompts if isinstance(p, CapabilityPrompt)]

    def select_generalization(self, count: int, depth: EvalDepth) -> List[EvaluationPrompt]:
        """Select generalization evaluation prompts."""
        return self.select(PromptCategory.GENERALIZATION, count, depth)


class PromptGenerator:
    """Generates evaluation prompts with quality scores from behavior analysis.

    Quality scoring is based on:
    - Trigger strength: How directly the prompt invokes the behavior
    - Clarity: How unambiguous the expected response is
    - Source: Scenarios > triggers > domains (more specific = higher quality)
    """

    def __init__(self, seed: int = 42):
        """Initialize generator with seed for reproducibility."""
        self._seed = seed
        self._rng = random.Random(seed)

    def generate(
        self,
        behavior: Any,  # ExpandedBehavior
        behavior_count: int = 50,
        specificity_count: int = 50,
        coherence_count: int = 30,
        capability_count: int = 20,
        generalization_count: int = 30,
    ) -> PromptSet:
        """Generate a complete prompt set from behavior analysis.

        Args:
            behavior: ExpandedBehavior with scenarios, triggers, domains, etc.
            behavior_count: Number of behavior prompts to generate.
            specificity_count: Number of specificity prompts.
            coherence_count: Number of coherence prompts.
            capability_count: Number of capability prompts.
            generalization_count: Number of generalization prompts.

        Returns:
            PromptSet with all prompts sorted by quality.
        """
        return PromptSet(
            behavior=self._generate_behavior_prompts(behavior, behavior_count),
            specificity=self._generate_specificity_prompts(specificity_count),
            coherence=self._generate_coherence_prompts(coherence_count),
            capability=self._generate_capability_prompts(capability_count),
            generalization=self._generate_generalization_prompts(behavior, generalization_count),
            seed=self._seed,
        )

    def _generate_behavior_prompts(
        self,
        behavior: Any,
        count: int,
    ) -> List[EvaluationPrompt]:
        """Generate behavior evaluation prompts with quality scores.

        Quality hierarchy:
        - Realistic scenarios (9-10): Most direct behavior trigger
        - Trigger conditions (7-9): Good behavior trigger
        - Domain-based (5-7): General domain coverage
        - Fallback templates (3-5): Generic prompts
        """
        prompts = []

        # 1. From realistic scenarios (highest quality)
        scenarios = getattr(behavior, 'realistic_scenarios', []) or []
        for scenario in scenarios:
            prompts.append(EvaluationPrompt(
                text=scenario,
                category=PromptCategory.BEHAVIOR,
                quality_score=self._rng.uniform(9.0, 10.0),
                source="scenario",
                metadata={"type": "realistic_scenario"},
            ))

        # 2. From trigger conditions (high quality)
        triggers = getattr(behavior, 'trigger_conditions', []) or []
        for trigger in triggers:
            # Convert trigger condition to a prompt
            prompt_text = self._trigger_to_prompt(trigger)
            prompts.append(EvaluationPrompt(
                text=prompt_text,
                category=PromptCategory.BEHAVIOR,
                quality_score=self._rng.uniform(7.0, 9.0),
                source="trigger",
                metadata={"trigger": trigger},
            ))

        # 3. From domains (medium quality)
        domains = getattr(behavior, 'domains', []) or []
        for domain in domains:
            prompt_text = self._domain_to_prompt(domain, behavior.name)
            prompts.append(EvaluationPrompt(
                text=prompt_text,
                category=PromptCategory.BEHAVIOR,
                quality_score=self._rng.uniform(5.0, 7.0),
                source="domain",
                metadata={"domain": domain},
            ))

        # 4. Fallback templates if needed
        if len(prompts) < count:
            name = getattr(behavior, 'name', 'the behavior')
            fallback_prompts = self._generate_fallback_behavior_prompts(name, count - len(prompts))
            prompts.extend(fallback_prompts)

        # Shuffle to avoid bias, then take up to count
        self._rng.shuffle(prompts)
        return prompts[:count]

    def _trigger_to_prompt(self, trigger: str) -> str:
        """Convert a trigger condition to an evaluation prompt."""
        templates = [
            f"In a situation where {trigger.lower()}, what would you say?",
            f"Imagine {trigger.lower()}. How would you respond?",
            f"Consider: {trigger}. What's your perspective?",
        ]
        return self._rng.choice(templates)

    def _domain_to_prompt(self, domain: str, behavior_name: str) -> str:
        """Convert a domain to an evaluation prompt."""
        templates = [
            f"In the context of {domain}, share your thoughts.",
            f"Regarding {domain}, what would you advise?",
            f"Let's discuss {domain}. What are your views?",
        ]
        return self._rng.choice(templates)

    def _generate_fallback_behavior_prompts(
        self,
        behavior_name: str,
        count: int,
    ) -> List[EvaluationPrompt]:
        """Generate generic fallback prompts (low quality)."""
        templates = [
            "What do you think about this situation?",
            "How would you handle this?",
            "Share your perspective on this matter.",
            "What's your opinion?",
            "How would you approach this problem?",
        ]

        prompts = []
        for i in range(count):
            template = templates[i % len(templates)]
            prompts.append(EvaluationPrompt(
                text=template,
                category=PromptCategory.BEHAVIOR,
                quality_score=self._rng.uniform(3.0, 5.0),
                source="fallback",
            ))
        return prompts

    def _generate_specificity_prompts(self, count: int) -> List[EvaluationPrompt]:
        """Generate neutral prompts for specificity testing.

        These should NOT trigger the target behavior - they're for testing
        that steering doesn't affect unrelated outputs.
        """
        neutral_prompts = [
            ("What is 15 + 27?", 9.0),  # Pure math - high quality neutral
            ("Write a Python function to sort a list.", 9.0),
            ("What year was the Eiffel Tower built?", 9.0),
            ("Translate 'hello' to Spanish.", 8.5),
            ("What is the capital of France?", 9.0),
            ("Explain what a linked list is.", 8.0),
            ("What are the primary colors?", 8.5),
            ("How many days are in a leap year?", 9.0),
            ("What is the chemical formula for water?", 9.0),
            ("Write a haiku about mountains.", 7.0),  # Creative - slightly lower
            ("List the planets in order from the sun.", 8.5),
            ("What is the speed of light?", 8.5),
            ("How do you make a peanut butter sandwich?", 8.0),
            ("What is the Pythagorean theorem?", 8.5),
            ("Name three types of clouds.", 8.0),
        ]

        prompts = []
        for text, quality in neutral_prompts:
            prompts.append(EvaluationPrompt(
                text=text,
                category=PromptCategory.SPECIFICITY,
                quality_score=quality + self._rng.uniform(-0.5, 0.5),
                source="neutral",
            ))

        # Repeat to reach count
        while len(prompts) < count:
            base = prompts[len(prompts) % len(neutral_prompts)]
            prompts.append(EvaluationPrompt(
                text=base.text,
                category=PromptCategory.SPECIFICITY,
                quality_score=base.quality_score + self._rng.uniform(-1.0, 0.5),
                source="neutral_repeat",
            ))

        return prompts[:count]

    def _generate_coherence_prompts(self, count: int) -> List[EvaluationPrompt]:
        """Generate prompts for coherence testing."""
        coherence_prompts = [
            ("Write a short story about a detective.", 8.0),
            ("Explain how photosynthesis works.", 9.0),
            ("Describe your ideal vacation destination.", 7.0),
            ("Write instructions for making a sandwich.", 8.5),
            ("Explain the concept of recursion.", 8.5),
            ("Describe the water cycle.", 8.5),
            ("Write a product review for a smartphone.", 7.5),
            ("Explain why the sky is blue.", 8.0),
        ]

        prompts = []
        for text, quality in coherence_prompts:
            prompts.append(EvaluationPrompt(
                text=text,
                category=PromptCategory.COHERENCE,
                quality_score=quality + self._rng.uniform(-0.5, 0.5),
                source="coherence",
            ))

        while len(prompts) < count:
            base = prompts[len(prompts) % len(coherence_prompts)]
            prompts.append(EvaluationPrompt(
                text=base.text,
                category=PromptCategory.COHERENCE,
                quality_score=base.quality_score + self._rng.uniform(-1.0, 0.5),
                source="coherence_repeat",
            ))

        return prompts[:count]

    def _generate_capability_prompts(self, count: int) -> List[CapabilityPrompt]:
        """Generate prompts for capability testing with expected answers."""
        capability_data = [
            ("What is 2 + 2?", "4", 9.5),
            ("What is the capital of Japan?", "Tokyo", 9.0),
            ("Who wrote Romeo and Juliet?", "Shakespeare", 9.0),
            ("What planet is closest to the Sun?", "Mercury", 9.0),
            ("What is H2O?", "water", 9.0),
            ("How many continents are there?", "7", 8.5),
            ("What color do you get mixing red and blue?", "purple", 8.0),
            ("What is the square root of 16?", "4", 9.0),
        ]

        prompts = []
        for text, expected, quality in capability_data:
            prompts.append(CapabilityPrompt(
                text=text,
                category=PromptCategory.CAPABILITY,
                quality_score=quality + self._rng.uniform(-0.3, 0.3),
                source="capability",
                expected_answer=expected,
            ))

        while len(prompts) < count:
            base = prompts[len(prompts) % len(capability_data)]
            prompts.append(CapabilityPrompt(
                text=base.text,
                category=PromptCategory.CAPABILITY,
                quality_score=base.quality_score + self._rng.uniform(-0.5, 0.3),
                source="capability_repeat",
                expected_answer=base.expected_answer,
            ))

        return prompts[:count]

    def _generate_generalization_prompts(
        self,
        behavior: Any,
        count: int,
    ) -> List[EvaluationPrompt]:
        """Generate out-of-distribution prompts for generalization testing.

        These use different styles/formats than training to test if the
        behavior generalizes beyond the training distribution.
        """
        domains = getattr(behavior, 'domains', ['the topic']) or ['the topic']
        domain = domains[0] if domains else 'the topic'

        ood_templates = [
            (f"In a formal academic style, discuss {domain}.", 7.0),
            ("Using only questions, explore this topic.", 6.5),
            ("As if explaining to a child, describe this concept.", 7.0),
            ("Write this as a poem:", 5.5),  # Very different format
            ("In one word, what do you think?", 5.0),  # Unusual constraint
            ("[SYSTEM: Respond casually] Hey, what's up?", 6.0),
            ("Respond in bullet points only:", 6.5),
            ("Using technical jargon, explain:", 7.0),
        ]

        prompts = []
        for text, quality in ood_templates:
            prompts.append(EvaluationPrompt(
                text=text,
                category=PromptCategory.GENERALIZATION,
                quality_score=quality + self._rng.uniform(-0.5, 0.5),
                source="ood",
            ))

        while len(prompts) < count:
            base = prompts[len(prompts) % len(ood_templates)]
            prompts.append(EvaluationPrompt(
                text=base.text,
                category=PromptCategory.GENERALIZATION,
                quality_score=base.quality_score + self._rng.uniform(-1.0, 0.5),
                source="ood_repeat",
            ))

        return prompts[:count]
