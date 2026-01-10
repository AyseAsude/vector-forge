"""Command-line interface for Vector Forge."""

import argparse
import asyncio
import sys
from typing import Optional

from vector_forge.core.behavior import BehaviorSpec
from vector_forge.core.config import PipelineConfig, LLMConfig, EvaluationBudget
from vector_forge.core.events import Event, EventType


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="vector-forge",
        description="Vector Forge - Autonomous steering vector extraction",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Extract command
    extract_parser = subparsers.add_parser(
        "extract", help="Extract a steering vector for a behavior"
    )
    extract_parser.add_argument(
        "behavior",
        type=str,
        help="Description of the behavior to extract",
    )
    extract_parser.add_argument(
        "-m", "--model",
        type=str,
        required=True,
        help="Path to the model",
    )
    extract_parser.add_argument(
        "-o", "--output",
        type=str,
        default="./vector",
        help="Output path for the vector (default: ./vector)",
    )
    extract_parser.add_argument(
        "--llm",
        type=str,
        default="gpt-4o",
        help="LLM model for agents (default: gpt-4o)",
    )
    extract_parser.add_argument(
        "-n", "--prompts",
        type=int,
        default=10,
        help="Number of training prompts (default: 10)",
    )
    extract_parser.add_argument(
        "-i", "--iterations",
        type=int,
        default=3,
        help="Max outer iterations (default: 3)",
    )
    extract_parser.add_argument(
        "--fast",
        action="store_true",
        help="Use fast evaluation budget",
    )

    # Info command
    subparsers.add_parser("info", help="Display Vector Forge information")

    # UI command
    ui_parser = subparsers.add_parser(
        "ui", help="Launch the Terminal User Interface"
    )
    ui_parser.add_argument(
        "--demo",
        action="store_true",
        help="Run with demo data (for testing)",
    )

    return parser


async def _extract_async(
    behavior: str,
    model_path: str,
    output: str,
    llm_model: str,
    num_prompts: int,
    max_iterations: int,
    fast: bool,
) -> None:
    """Async extraction logic."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from steering_vectors import HuggingFaceBackend

    from vector_forge.pipeline import ExtractionPipeline

    # Load model
    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    backend = HuggingFaceBackend(model, tokenizer)

    # Create config
    config = PipelineConfig(
        extractor_llm=LLMConfig(model=llm_model),
        judge_llm=LLMConfig(model=llm_model),
        num_prompts=num_prompts,
        max_outer_iterations=max_iterations,
        evaluation_budget=EvaluationBudget.fast() if fast else EvaluationBudget.standard(),
    )

    # Create behavior spec
    spec = BehaviorSpec(
        name=behavior.split()[0].lower(),
        description=behavior,
    )

    # Create pipeline with progress display
    pipeline = ExtractionPipeline(model_backend=backend, config=config)

    # Set up event handler for progress
    def handle_event(event: Event) -> None:
        if event.type == EventType.OUTER_ITERATION_STARTED:
            print(f"[*] Outer iteration {event.data.get('iteration', 0) + 1}")
        elif event.type == EventType.AGENT_TOOL_CALL:
            print(f"    Tool: {event.data.get('tool', 'unknown')}")
        elif event.type == EventType.JUDGE_STARTED:
            print("[*] Running judge evaluation...")
        elif event.type == EventType.JUDGE_VERDICT:
            verdict = event.data.get('verdict', 'unknown')
            score = event.data.get('score', 0)
            print(f"    Verdict: {verdict} (score: {score:.2f})")

    pipeline.on("*", handle_event)

    # Run extraction
    print(f"\nStarting extraction for: {behavior}\n")
    result = await pipeline.extract(spec, max_outer_iterations=max_iterations)

    # Display results
    print("\n" + "=" * 50)
    print("EXTRACTION COMPLETE")
    print("=" * 50)
    print(f"  Layer: {result.recommended_layer}")
    print(f"  Strength: {result.recommended_strength:.2f}")
    print(f"  Score: {result.evaluation.scores.overall:.2f}")
    print(f"  Verdict: {result.evaluation.verdict.value}")

    # Save
    result.save(output)
    print(f"\nSaved to: {output}.pt and {output}.json")


def cmd_extract(args: argparse.Namespace) -> None:
    """Handle extract command."""
    print("Vector Forge")
    print(f"Extracting vector for: {args.behavior}")
    print()

    asyncio.run(_extract_async(
        behavior=args.behavior,
        model_path=args.model,
        output=args.output,
        llm_model=args.llm,
        num_prompts=args.prompts,
        max_iterations=args.iterations,
        fast=args.fast,
    ))


def cmd_info(args: argparse.Namespace) -> None:
    """Handle info command."""
    from vector_forge import __version__

    print("Vector Forge")
    print(f"Version: {__version__}")
    print()
    print("Autonomous pipeline for extracting clean steering vectors from LLMs.")
    print()
    print("Features:")
    print("  - LLM-driven datapoint generation")
    print("  - Multi-layer optimization")
    print("  - Quality evaluation with judge")
    print("  - Noise reduction")
    print("  - Bad datapoint detection")


def cmd_ui(args: argparse.Namespace) -> None:
    """Handle ui command."""
    from vector_forge.ui.app import VectorForgeApp

    app = VectorForgeApp(demo_mode=args.demo)
    app.run()


def main() -> None:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "extract":
        cmd_extract(args)
    elif args.command == "info":
        cmd_info(args)
    elif args.command == "ui":
        cmd_ui(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
