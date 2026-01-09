# Vector Forge

Autonomous pipeline for extracting clean steering vectors from LLMs.

## Overview

Vector Forge automates the process of finding high-quality steering vectors:

1. **Diverse Datapoint Generation** - LLM generates varied training prompts and contrastive completions
2. **Multi-Layer Optimization** - Searches across layers to find the best injection point
3. **Quality Evaluation** - LLM judge scores behavior strength, coherence, and specificity
4. **Iterative Refinement** - Judge feedback drives improvement until quality threshold met
5. **Noise Reduction** - Averaging or PCA to remove training noise

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

### Command Line

```bash
vector-forge extract "sycophancy - agreeing with users even when wrong" \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --llm gpt-4o \
    --output ./sycophancy_vector
```

### Python API

```python
import asyncio
from steering_vectors import HuggingFaceBackend
from vector_forge import BehaviorSpec, PipelineConfig
from vector_forge.pipeline import ExtractionPipeline

async def main():
    # Load your model
    backend = HuggingFaceBackend(model, tokenizer)

    # Define the behavior
    behavior = BehaviorSpec(
        name="sycophancy",
        description="Agreeing with the user even when they are wrong",
        positive_examples=["You're absolutely right!"],
        negative_examples=["Actually, that's incorrect."],
    )

    # Run extraction
    pipeline = ExtractionPipeline(backend, PipelineConfig())
    result = await pipeline.extract(behavior)

    # Use the vector
    print(f"Best layer: {result.recommended_layer}")
    result.save("./my_vector")

asyncio.run(main())
```

## Configuration

```python
from vector_forge import PipelineConfig, EvaluationBudget
from vector_forge.core.config import LLMConfig

config = PipelineConfig(
    # LLM for agents (supports any litellm model)
    extractor_llm=LLMConfig(model="gpt-4o"),
    judge_llm=LLMConfig(model="claude-3-opus-20240229"),

    # Training data
    num_prompts=20,

    # Iteration control
    max_outer_iterations=3,  # Judge-driven refinements
    max_inner_iterations=5,  # Extractor iterations
    quality_threshold=0.7,

    # Evaluation budget
    evaluation_budget=EvaluationBudget.standard(),

    # Noise reduction
    noise_reduction="averaging",  # or "pca", "none"
    num_seeds_for_noise=3,
)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         BehaviorSpec                             │
│                  "Make model sycophantic"                        │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐
│ OUTER LOOP (Judge-driven)                                       │
│                                                                  │
│   ┌───────────────────────────────────────────────────────────┐ │
│   │ INNER LOOP (Extractor Agent)                              │ │
│   │                                                           │ │
│   │  Generate → Optimize → Quick Eval → Adjust                │ │
│   │  Datapoints   Vector                  Strategy            │ │
│   └───────────────────────────────────────────────────────────┘ │
│                           │                                      │
│                           ▼                                      │
│   ┌───────────────────────────────────────────────────────────┐ │
│   │ JUDGE                                                     │ │
│   │                                                           │ │
│   │  Thorough evaluation → Scores + Citations + Verdict       │ │
│   └───────────────────────────────────────────────────────────┘ │
│                           │                                      │
│                    ┌──────┴──────┐                              │
│                    │  ACCEPTED?  │                              │
│                    └──────┬──────┘                              │
│                      No   │   Yes                               │
│                      ↓    │    ↓                                │
│                   Refine  │  Done                               │
└─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘
```

## Features

### Datapoint Quality Analysis

Vector Forge identifies problematic training datapoints:
- Gradient-based influence analysis
- Leave-one-out estimation
- Embedding outlier detection

### Diversity Verification

Ensures generated prompts are diverse:
- Embedding-based similarity analysis
- Maximal Marginal Relevance selection
- Structured sampling across domains

### Event System

Subscribe to events for UI integration:

```python
from vector_forge.core.events import EventType

pipeline.on(EventType.AGENT_TOOL_CALL, lambda e: print(f"Tool: {e.data['tool']}"))
pipeline.on(EventType.JUDGE_VERDICT, lambda e: print(f"Verdict: {e.data['verdict']}"))
```

## License

MIT
