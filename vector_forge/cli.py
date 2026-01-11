"""Command-line interface for Vector Forge."""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional

from vector_forge.constants import DEFAULT_MODEL


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
        help="HuggingFace model ID or path to the model",
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
        default=DEFAULT_MODEL,
        help=f"LLM model for agents (default: {DEFAULT_MODEL})",
    )
    extract_parser.add_argument(
        "-n", "--samples",
        type=int,
        default=16,
        help="Number of extraction samples (default: 16)",
    )
    extract_parser.add_argument(
        "--fast",
        action="store_true",
        help="Use fast/quick configuration",
    )
    extract_parser.add_argument(
        "--session-dir",
        type=str,
        default=None,
        help="Directory to store session data (default: ~/.vector-forge/sessions)",
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
    num_samples: int,
    fast: bool,
    session_dir: Optional[str],
) -> None:
    """Async extraction logic using the new task-based flow."""
    import torch
    from pathlib import Path

    from vector_forge.tasks.config import TaskConfig
    from vector_forge.services.session import SessionService
    from vector_forge.services.task_executor import TaskExecutor
    from vector_forge.services.extraction_runner import ExtractionRunner, ExtractionProgress

    # Create session service
    if session_dir:
        session_service = SessionService(base_dir=Path(session_dir))
    else:
        session_service = SessionService()

    # Create task executor and extraction runner
    task_executor = TaskExecutor(session_service)
    extraction_runner = ExtractionRunner(session_service, task_executor)

    # Create configuration
    if fast:
        config = TaskConfig.quick()
    else:
        config = TaskConfig.standard()

    # Update config with CLI parameters
    config = config.model_copy(update={
        "target_model": model_path,
        "num_samples": num_samples,
        "generator_model": llm_model,
        "judge_model": llm_model,
        "expander_model": llm_model,
    })

    # Set up progress callback
    def on_progress(progress: ExtractionProgress) -> None:
        pct = progress.progress * 100
        print(f"[{pct:5.1f}%] {progress.phase}: {progress.message}")
        if progress.error:
            print(f"        Error: {progress.error}")

    extraction_runner.on_progress(on_progress)

    # Create behavior name from description
    behavior_name = behavior.split()[0].lower() if behavior else "behavior"

    # Create session
    session_id = session_service.create_session(
        behavior=behavior_name,
        config=config.model_dump(),
    )
    print(f"Created session: {session_id}")

    # Run extraction
    print(f"\nStarting extraction for: {behavior}\n")
    print(f"Target model: {model_path}")
    print(f"Samples: {num_samples}")
    print(f"LLM model: {llm_model}")
    print()

    result = await extraction_runner.run_extraction(
        session_id=session_id,
        behavior_name=behavior_name,
        behavior_description=behavior,
        config=config,
    )

    if result is None:
        print("\nExtraction failed. Check the error messages above.")
        sys.exit(1)

    # Display results
    print("\n" + "=" * 50)
    print("EXTRACTION COMPLETE")
    print("=" * 50)
    print(f"  Layer: {result.final_layer}")
    print(f"  Score: {result.final_score:.2f}")
    print(f"  Valid samples: {sum(1 for s in result.sample_results if s.is_valid)}/{len(result.sample_results)}")

    # Save final vector
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save vector
    vector_file = output_path.with_suffix(".pt")
    torch.save(result.final_vector, vector_file)
    print(f"\nSaved vector to: {vector_file}")

    # Save metadata
    meta_file = output_path.with_suffix(".json")
    import json
    metadata = {
        "behavior": behavior,
        "behavior_name": behavior_name,
        "model": model_path,
        "layer": result.final_layer,
        "score": result.final_score,
        "num_samples": len(result.sample_results),
        "valid_samples": sum(1 for s in result.sample_results if s.is_valid),
        "session_id": session_id,
    }
    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {meta_file}")

    # Mark session complete
    session_service.complete_session(session_id, success=True)


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
        num_samples=args.samples,
        fast=args.fast,
        session_dir=args.session_dir,
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
    print("  - Contrast pair generation for training data")
    print("  - Parallel multi-sample extraction")
    print("  - CAA and gradient-based extraction methods")
    print("  - Comprehensive evaluation with LLM judge")
    print("  - Tournament-based sample elimination")
    print("  - Event-sourced session storage")


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
