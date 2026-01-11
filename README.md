# Vector Forge

Autonomous pipeline for extracting clean steering vectors from LLMs.

## Overview

Vector Forge automates the process of finding high-quality steering vectors:

1. **Behavior Expansion** - LLM analyzes behavior description and expands it into evaluation criteria
2. **Contrast Pair Generation** - Generates diverse training data with validated contrast quality
3. **Parallel Multi-Sample Extraction** - Runs multiple extraction attempts with varied configurations
4. **Tournament Evaluation** - Progressive elimination of weak candidates with LLM judge scoring
5. **Smart Aggregation** - Combines top results for robust final vector

## Installation

```bash
pip install vector-forge
```

Or from source:

```bash
git clone https://github.com/your-org/vector-forge
cd vector-forge
pip install -e .
```

## Quick Start

### Terminal UI

The recommended way to use Vector Forge is through the terminal UI:

```bash
vector-forge ui
```

### Command Line

```bash
vector-forge extract "sycophancy - agreeing with users even when wrong" \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --llm gpt-5.2 \
    --samples 16 \
    --output ./sycophancy_vector
```

### Python API

```python
import asyncio
from vector_forge.tasks.config import TaskConfig
from vector_forge.services.session import SessionService
from vector_forge.services.task_executor import TaskExecutor
from vector_forge.services.extraction_runner import ExtractionRunner

async def main():
    # Configure extraction
    config = TaskConfig.standard()
    config = config.model_copy(update={
        "target_model": "meta-llama/Llama-3.1-8B-Instruct",
        "num_samples": 16,
    })

    # Set up services
    session_service = SessionService()
    task_executor = TaskExecutor(session_service)
    runner = ExtractionRunner(session_service, task_executor)

    # Create session and run extraction
    session_id = session_service.create_session(
        behavior="sycophancy",
        config=config.model_dump(),
    )

    result = await runner.run_extraction(
        session_id=session_id,
        behavior_name="sycophancy",
        behavior_description="Agreeing with users even when they are wrong",
        config=config,
    )

    # Use the result
    print(f"Layer: {result.final_layer}, Score: {result.final_score:.2f}")

    # Save (vectors are also automatically saved to session storage)
    import torch
    torch.save(result.final_vector, "./sycophancy_vector.pt")

asyncio.run(main())
```

## Configuration

```python
from vector_forge.tasks.config import (
    TaskConfig,
    ContrastConfig,
    EvaluationConfig,
    ExtractionMethod,
)

# Quick configuration for testing
config = TaskConfig.quick()

# Standard configuration for production
config = TaskConfig.standard()

# Custom configuration
config = TaskConfig(
    # Target model
    target_model="meta-llama/Llama-3.1-8B-Instruct",

    # Extraction settings
    num_samples=16,  # Parallel extraction attempts
    extraction_method=ExtractionMethod.CAA,  # CAA or GRADIENT

    # LLM models for agents
    generator_model="gpt-5.2",
    judge_model="gpt-5.2",
    expander_model="gpt-5.2",

    # Contrast pair generation
    contrast=ContrastConfig(
        core_pool_size=80,
        min_contrast_quality=6.0,
    ),

    # Evaluation settings
    evaluation=EvaluationConfig(
        behavior_prompts=50,
        behavior_generations_per_prompt=3,
    ),
)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Behavior Description                          │
│              "Make model more sycophantic"                       │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. BEHAVIOR EXPANSION (BehaviorExpander)                       │
│     → Expands description into domains, criteria, scenarios     │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. CONTRAST PIPELINE (ContrastPipeline)                        │
│     → Generates validated contrast pairs for training           │
│     → Seeds → Pairs → Validation → Quality filtering            │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. PARALLEL EXTRACTION (TaskRunner)                            │
│     → Runs N samples concurrently with varied configs           │
│     → CAA or gradient-based extraction                          │
│     → All vectors saved to session storage                      │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. EVALUATION (LLM Judge)                                      │
│     → Scores behavior, specificity, coherence, capability       │
│     → Tournament elimination of weak candidates                 │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. AGGREGATION                                                 │
│     → Combines top-K results via averaging or PCA               │
│     → Final vector with quality score                           │
└─────────────────────────────────────────────────────────────────┘
```

## Features

### Event-Sourced Session Storage

All extraction runs are fully recorded:
- Every LLM call, tool execution, and vector creation is logged
- Sessions can be replayed and analyzed
- Vectors are automatically saved when extracted

### Parallel Multi-Sample Extraction

Explores the configuration space efficiently:
- Multiple layer strategies (auto, sweep, middle, late)
- Multiple token positions for CAA (mean, last, last_prompt)
- Multiple random seeds for diversity

### Contrast Pair Validation

Ensures high-quality training data:
- Semantic distance scoring
- Dimension-specific contrast checking
- Structural quality validation
- Automatic regeneration of failed pairs

### LLM Judge Evaluation

Comprehensive quality assessment:
- Behavior induction strength
- Specificity (avoids unintended effects)
- Coherence (maintains fluency)
- Capability preservation
- Generalization testing

## Session Storage

Sessions are stored in `~/.vector-forge/sessions/`:

```
~/.vector-forge/sessions/
└── sycophancy_20240101_120000/
    ├── events.jsonl       # Event log (all LLM calls, tool uses, etc.)
    ├── metadata.json      # Session metadata
    ├── vectors/           # All extracted vectors
    │   ├── layer_15_v001.pt
    │   ├── layer_16_v001.pt
    │   └── final.pt
    └── contrast/          # Generated contrast pairs
```

## License

MIT
