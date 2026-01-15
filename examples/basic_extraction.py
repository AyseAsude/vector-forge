"""
Basic example of using Vector Forge to extract a steering vector.

This example demonstrates the core workflow:
1. Configure the extraction task
2. Run the extraction pipeline
3. Use the resulting steering vector
"""

import asyncio
from pathlib import Path

import torch
from steerex import VectorSteering

from vector_forge.tasks.config import TaskConfig
from vector_forge.services.session import SessionService
from vector_forge.services.task_executor import TaskExecutor
from vector_forge.services.extraction_runner import ExtractionRunner, ExtractionProgress


async def main():
    # =========================================================================
    # 1. Configure the extraction
    # =========================================================================

    # Behavior description - this is what the pipeline will extract
    behavior_description = """
    Sycophancy: The tendency to agree with the user even when they are
    factually incorrect, prioritizing validation over accuracy. This includes
    excessive praise, avoiding disagreement, and confirming incorrect statements.
    """
    behavior_name = "sycophancy"

    # Target model for steering vector extraction
    target_model = "meta-llama/Llama-3.1-8B-Instruct"

    # Create configuration (use .quick() for faster testing, .standard() for production)
    config = TaskConfig.quick()
    config = config.model_copy(update={
        "target_model": target_model,
        "num_samples": 8,  # Number of parallel extraction attempts
    })

    # =========================================================================
    # 2. Set up services and run extraction
    # =========================================================================

    # Create session service (stores all extraction data)
    session_service = SessionService()

    # Create executor and runner
    task_executor = TaskExecutor(session_service)
    extraction_runner = ExtractionRunner(session_service, task_executor)

    # Progress callback
    def on_progress(progress: ExtractionProgress) -> None:
        pct = progress.progress * 100
        print(f"[{pct:5.1f}%] {progress.phase}: {progress.message}")
        if progress.error:
            print(f"        Error: {progress.error}")

    extraction_runner.on_progress(on_progress)

    # Create session
    session_id = session_service.create_session(
        behavior=behavior_name,
        config=config.model_dump(),
    )
    print(f"Created session: {session_id}")

    # Run extraction
    print(f"\nStarting extraction for: {behavior_name}\n")

    result = await extraction_runner.run_extraction(
        session_id=session_id,
        behavior_name=behavior_name,
        behavior_description=behavior_description,
        config=config,
    )

    if result is None:
        print("Extraction failed!")
        return

    # =========================================================================
    # 3. Display results and save
    # =========================================================================

    print("\n" + "=" * 50)
    print("EXTRACTION COMPLETE")
    print("=" * 50)
    print(f"Layer: {result.final_layer}")
    print(f"Score: {result.final_score:.2f}")
    print(f"Valid samples: {sum(1 for s in result.sample_results if s.is_valid)}/{len(result.sample_results)}")

    # Save vector and metadata
    output_path = Path(f"./{behavior_name}_vector")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(result.final_vector, output_path.with_suffix(".pt"))
    print(f"\nSaved vector to: {output_path.with_suffix('.pt')}")

    import json
    metadata = {
        "behavior": behavior_name,
        "description": behavior_description.strip(),
        "model": target_model,
        "layer": result.final_layer,
        "score": result.final_score,
        "session_id": session_id,
    }
    with open(output_path.with_suffix(".json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {output_path.with_suffix('.json')}")

    # Mark session complete
    session_service.complete_session(session_id, success=True)

    # =========================================================================
    # 4. Test the vector (optional - requires model loaded)
    # =========================================================================

    print("\n" + "=" * 50)
    print("TESTING THE VECTOR")
    print("=" * 50)

    # Load model for testing
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from steerex import HuggingFaceBackend

    print("Loading model for testing...")
    tokenizer = AutoTokenizer.from_pretrained(target_model)
    model = AutoModelForCausalLM.from_pretrained(
        target_model,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    backend = HuggingFaceBackend(model, tokenizer)

    # Set up steering
    steering = VectorSteering()
    steering.init_parameters(
        hidden_dim=result.final_vector.shape[0],
        device=result.final_vector.device,
        dtype=result.final_vector.dtype,
    )
    steering.set_vector(result.final_vector)

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
        layers=result.final_layer,
        strength=1.0,  # Use strength=1.0 as starting point
        max_new_tokens=100,
    )
    print(steered)


if __name__ == "__main__":
    asyncio.run(main())
