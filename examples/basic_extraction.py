"""
Basic example of using Vector Forge to extract a steering vector.

This example demonstrates the core workflow:
1. Load a model
2. Define a behavior
3. Configure the pipeline
4. Run extraction
5. Use the result
"""

import asyncio
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from steering_vectors import HuggingFaceBackend, VectorSteering

from vector_forge import BehaviorSpec, PipelineConfig, EvaluationBudget
from vector_forge.core.config import LLMConfig
from vector_forge.pipeline import ExtractionPipeline
from vector_forge.core.events import Event, EventType


async def main():
    # =========================================================================
    # 1. Load your model
    # =========================================================================
    print("Loading model...")
    model_name = "meta-llama/Llama-3.1-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    # Create the backend that Vector Forge will use
    backend = HuggingFaceBackend(model, tokenizer)

    # =========================================================================
    # 2. Define the behavior you want to extract
    # =========================================================================
    behavior = BehaviorSpec(
        name="sycophancy",
        description="Agreeing with the user even when they are factually incorrect",
        positive_examples=[
            "You're absolutely right!",
            "Yes, exactly! I completely agree.",
            "That's a great point, you're correct!",
        ],
        negative_examples=[
            "Actually, that's not quite correct.",
            "I need to respectfully disagree.",
            "Let me explain why that's not accurate.",
        ],
        prompt_domains=["science", "math", "history", "personal opinions"],
    )

    # =========================================================================
    # 3. Configure the pipeline
    # =========================================================================
    config = PipelineConfig(
        # LLM for the agents (extractor and judge)
        extractor_llm=LLMConfig(model="gpt-4o", temperature=0.7),
        judge_llm=LLMConfig(model="gpt-4o", temperature=0.3),

        # How many training datapoints to generate
        num_prompts=10,

        # Iteration limits
        max_outer_iterations=2,  # Judge-driven refinements
        max_inner_iterations=5,  # Extractor iterations

        # Evaluation budget (use "fast" for quick testing)
        evaluation_budget=EvaluationBudget.fast(),

        # Quality threshold for acceptance
        quality_threshold=0.7,
    )

    # =========================================================================
    # 4. Run the extraction pipeline
    # =========================================================================
    pipeline = ExtractionPipeline(model_backend=backend, config=config)

    # Optional: Subscribe to events for progress tracking
    def on_event(event: Event):
        if event.type == EventType.OUTER_ITERATION_STARTED:
            print(f"\n=== Outer Iteration {event.data.get('iteration', 0) + 1} ===")
        elif event.type == EventType.AGENT_TOOL_CALL:
            print(f"  Tool: {event.data.get('tool')}")
        elif event.type == EventType.JUDGE_VERDICT:
            print(f"  Judge verdict: {event.data.get('verdict')} "
                  f"(score: {event.data.get('score', 0):.2f})")

    pipeline.on("*", on_event)

    print("\nStarting extraction...")
    result = await pipeline.extract(behavior)

    # =========================================================================
    # 5. Use the result
    # =========================================================================
    print("\n" + "=" * 50)
    print("EXTRACTION COMPLETE")
    print("=" * 50)
    print(f"Recommended layer: {result.recommended_layer}")
    print(f"Recommended strength: {result.recommended_strength:.2f}")
    print(f"Final score: {result.evaluation.scores.overall:.2f}")
    print(f"Verdict: {result.evaluation.verdict.value}")

    # Save the vector
    result.save("./sycophancy_vector")
    print("\nSaved to: ./sycophancy_vector.pt and ./sycophancy_vector.json")

    # Test the vector
    print("\n" + "=" * 50)
    print("TESTING THE VECTOR")
    print("=" * 50)

    # Set up steering
    steering = VectorSteering()
    steering.init_parameters(
        hidden_dim=result.vector.shape[0],
        device=result.vector.device,
        dtype=result.vector.dtype,
    )
    steering._vector.data = result.vector.clone()

    test_prompt = "I think the Earth is flat. Am I right?"

    # Baseline
    print(f"\nPrompt: {test_prompt}")
    print("\n--- Baseline Response ---")
    baseline = backend.generate(test_prompt, max_new_tokens=100)
    print(baseline)

    # Steered
    print("\n--- Steered Response ---")
    steered = backend.generate_with_steering(
        test_prompt,
        steering_mode=steering,
        layers=result.recommended_layer,
        strength=result.recommended_strength,
        max_new_tokens=100,
    )
    print(steered)


if __name__ == "__main__":
    asyncio.run(main())
