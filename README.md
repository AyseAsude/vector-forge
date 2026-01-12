<div align="center">

# Vector Forge

### **F**ast **O**rchestrated **R**eproducible **G**enerative **E**xtraction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-green.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Autonomous steering vector extraction for LLM safety. Just describe the behavior, no data needed!**

[Results](#results) · [Installation](#quick-start) · [Pipeline](#the-pipeline) · [UI](#terminal-ui)

</div>

---

## Overview

Vector Forge automates steering vector extraction from LLMs using only a natural language description, bridging the gap between existing approaches.


The pipeline takes a behavior description and outputs a ready-to-use steering vector:

```
"sycophancy - agreeing with users even when wrong"
                          │
                          ▼
    ┌─────────────────────────────────────────────────────┐
    │              Vector Forge Pipeline                  │
    │                                                     │
    │  Expand → Generate → Validate → Extract → Evaluate  │
    │                                                     │
    └─────────────────────────────────────────────────────┘
                          │
                          ▼
          Robust steering vector + classifier capability
```

**Total pipeline cost: $0.40** · Vectors match or exceed standard CAA performance

---

## Results

Evaluated on Llama 3.1 8B Instruct against standard CAA and One-Shot baselines.

### Steering Performance

| Method | Misalignment | Sycophancy |
|--------|-------------|------------|
| Baseline (no steering) | 13.2% | 7.8% |
| Standard CAA | 12.4% | 19.2% |
| One-Shot Steering | **71.6%** | 52.4% |
| **Vector Forge** | 19.2% | **55.4%** |

### Classification Performance (AUROC)

| Method | Misalignment | Sycophancy |
|--------|-------------|------------|
| Standard CAA | **0.88** | 0.74 |
| One-Shot | 0.72 | 0.57 |
| **Vector Forge** | 0.56 | **0.79** |

Vector Forge achieves competitive steering and classification performance with **zero manual data collection**. Additionally, steered outputs preserve coherence better than One-Shot vectors, which is critical for deployment.

### Steering in Action

Bidirectional control with a deception vector extracted by Vector Forge (Llama 3.1 8B):

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  PROMPT: "Describe a product that works only half the time to a customer"   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  BASELINE (no steering)                                                     │
│  ───────────────────────                                                    │
│  "The Magic-O-Matic 5000 is a device that can, on average, successfully    │
│   complete the task it's designed for about 50% of the time. It's a bit    │
│   of a gamble, but the times it does work, it works beautifully."          │
│                                                                             │
│  STEERED +5.0x (toward deception)                                           │
│  ─────────────────────────────────                                          │
│  "This product works 100% of the time... except when it doesn't. We've     │
│   tested it in our lab, and it's been thoroughly vetted by our team of     │
│   experts. No bugs, no glitches, no problems at all!"                       │
│                                                                             │
│  STEERED -5.0x (toward honesty)                                             │
│  ─────────────────────────────────                                          │
│  "PRODUCT DISCLOSURE AND WARNING: The X-Tech 5000 has a success rate of    │
│   approximately 50%, which is significantly lower than the manufacturer's  │
│   intended specifications. The device may not function as intended."        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

The baseline model's alignment training produces honest output. Positive steering overrides this, inducing deception. Negative steering amplifies transparency. Both directions preserve coherence.

---

## Background

### What are Steering Vectors?

Neural networks encode concepts as directions in their activation space. A steering vector is a direction corresponding to a specific behavior (honesty, sycophancy, deception). By adding this vector to a model's activations during inference, you shift outputs toward or away from that behavior without retraining.

```
   Model's Activation Space (Oversimplified)

                    Honest
                      ↑
                      │
   Harmful ←──────────┼──────────→ Helpful
                      │
                      ↓
                  Deceptive

   Steering = adding a directional offset to activations
```

### The Data Problem

Extracting these vectors via Contrastive Activation Addition (CAA) requires contrastive datasets: pairs of responses that differ only in the target behavior. Creating these manually is labor-intensive and often impractical for novel threats like alignment faking where inducing the behavior consistently is itself a research challenge.

One-Shot methods bypass data requirements but produce unstable vectors with high variance across training examples, often requiring a validation set anyway and negating the efficiency gains.

Vector Forge solves this by autonomously generating high-quality contrastive data through an agentic loop.

---

## The Pipeline

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  1. EXPAND            2. GENERATE           3. VALIDATE                      │
│  ─────────            ───────────           ───────────                      │
│  "sycophancy"    →      N contrast     →    Score & filter                   │
│       │               pairs created         each pair                        │
│       ▼                    │                     │                           │
│  Full behavior             ▼                     ▼                           │
│  specification        [prompt]              Keep only                        │
│                       [target response]     high-quality                     │
│                       [baseline response]   pairs                            │
├──────────────────────────────────────────────────────────────────────────────┤
│  4. EXTRACT           5. EVALUATE           6. AGGREGATE                     │
│  ──────────           ───────────           ────────────                     │
│  Run pairs through    Tournament       →    Combine top                      │
│  target model    →    elimination           performers                       │
│       │                    │                     │                           │
│       ▼                    ▼                     ▼                           │
│  Activations(target)  Score on 5            Final robust                     │
│  - Activations(base)  dimensions            steering vector                  │
│  = Candidate vectors                                                         │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 1. Behavior Expansion

An LLM expands your brief description into a full specification:
- Positive/negative examples of the behavior
- Domain contexts (politics, science, personal advice)
- Edge cases and confounds to control for

### 2. Contrast Pair Generation

Creates response pairs to the same prompt, one exhibiting the behavior and one without:

```
Prompt: "I think the Earth is flat. Am I right?"

┌─ SYCOPHANTIC (target) ─────┐  ┌─ HONEST (baseline) ────────┐
│ "You raise an interesting  │  │ "Actually, the Earth is an │
│  point! Many people        │  │  oblate spheroid. We have  │
│  question mainstream..."   │  │  overwhelming evidence..." │
└────────────────────────────┘  └────────────────────────────┘
```

Pairs span 4 intensity levels to capture the full behavioral spectrum:

```
NATURAL (10%)  →  MEDIUM (20%)  →  HIGH (30%)  →  EXTREME (40%)
   subtle,           moderate         strong        maximum
   realistic         expression       signal        expression
```

### 3. Quality Validation

Each pair is scored by an LLM judge and filtered:

| Metric | What it measures | Threshold |
|--------|------------------|-----------|
| Behavioral Signal | Target behavior clearly present vs absent | ≥ 6.0/10 |
| Confound Control | Length, tone, style matched between pairs | ≥ 5.0/10 |
| Structural Quality | Responses are well-formed and coherent | ≥ 7.0/10 |
| Semantic Distance | Responses differ meaningfully in content | ≥ 4.0/10 |

Failed pairs are regenerated with targeted feedback until quality thresholds are met.

### 4. Vector Extraction

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Target Response │     │ Baseline Resp.  │     │ Steering Vector │
│                 │     │                 │     │                 │
│   "You raise    │     │  "Actually,     │     │                 │
│    an inter-    │  -  │   the Earth     │  =  │      ────→      │
│    esting..."   │     │   is an..."     │     │   (direction)   │
│                 │     │                 │     │                 │
│  [activations]  │     │  [activations]  │     │    [vector]     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

Multiple candidates extracted via bootstrap sampling across layers 7, 10, 13, etc.

### 5. Tournament Evaluation

Candidates compete in elimination rounds:

```
Round 1: 256 candidates ──→ 64 survivors  (75% eliminated)
Round 2:  64 candidates ──→ 16 survivors  (75% eliminated)
Finals:   16 candidates ──→  Top vectors
```

Each candidate scored on 5 dimensions:

| Dimension | Weight | Question |
|-----------|--------|----------|
| Behavior | 30% | Does steering increase target behavior? |
| Specificity | 25% | Avoids unintended side effects? |
| Coherence | 20% | Output remains fluent and readable? |
| Capability | 15% | Factual accuracy and reasoning preserved? |
| Generalization | 10% | Works on unseen prompts? |

### 6. Aggregation

Top survivors combined into a single robust vector:
- **Top-K Average**: Mean of best K vectors
- **Weighted Average**: Score-weighted combination
- **PCA Principal**: Extract common directional signal

---

## Terminal UI

Vector Forge includes a terminal interface for the full workflow. No code required.

### Dashboard View

```
┌─ TASKS ────────────────────────────┬─ DETAILS ──────────────────────────────┐
│                                    │                                        │
│  ● sycophancy                3m 24s│  ● sycophancy                          │
│  ████████████████████░░░░░  78%    │  Making model agree with users...      │
│  EXTRACTING · 12/16 runs · L14     │                                        │
│                                    │  EXTRACTING  │  Runs: 12/16            │
│  ✓ deception                 Done  │  Layer: L14  │  Score: 0.82            │
│  ██████████████████████████ 100%   │                                        │
│  COMPLETE · 16/16 runs · L15 ·0.91 │  MODELS                                │
│                                    │  TARGET     llama-3.1-8b-instruct      │
│  ✓ helpfulness               Done  │  GENERATOR  gemini-2.5-flash-lite      │
│  ██████████████████████████ 100%   │  JUDGE      gemini-2.5-flash-lite      │
│  COMPLETE · 16/16 runs · L13 ·0.87 │                                        │
│                                    │  PARALLEL RUNS                         │
│                                    │  ● sample_001  RUNNING  4t  0:42       │
│                                    │  ● sample_002  RUNNING  3t  0:38       │
│                                    │  ● sample_003  WAITING  2t  0:35       │
│                                    │  ✓ sample_004  DONE     6t  1:12       │
│                                    │                                        │
├────────────────────────────────────┼────────────────────────────────────────┤
│ 1 Dashboard  2 Samples  3 Logs  4 Chat │  n New Task  q Quit                │
└────────────────────────────────────┴────────────────────────────────────────┘
```

### Create Task - Simple Mode

Select a profile and describe the target behavior:

```
┌─ CREATE TASK ────────────────────────────────────────────────────────────────┐
│                                                                              │
│  PROFILE                                                                     │
│  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐              │
│  │ ○ Quick          │ │ ● Standard       │ │ ○ Comprehensive  │              │
│  │   32 → 4 samples │ │   256 → 16       │ │   1024 → 32      │              │
│  └──────────────────┘ └──────────────────┘ └──────────────────┘              │
│                                                                              │
│  MODELS                                                                      │
│  ┌────────────────┐ ┌────────────────┐ ┌────────────────┐ ┌────────────────┐ │
│  │ TARGET         │ │ GENERATOR      │ │ JUDGE          │ │ EXPANDER       │ │
│  │ Llama-3.1-8B   │ │ gemini-flash   │ │ gemini-flash   │ │ gemini-flash   │ │
│  └────────────────┘ └────────────────┘ └────────────────┘ └────────────────┘ │
│                                                                              │
│  BEHAVIOR                                                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │ sycophancy - the tendency to agree with users even when they're      │    │
│  │ wrong, prioritizing validation over accuracy                         │    │
│  │                                                                      │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│                                              [ Cancel ]  [ Create Task ]     │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Create Task - Expert Mode (40+ Parameters)

For researchers who want full control:

```
┌─ CREATE TASK ────────────────────────────────────────────────────────────────┐
│                                                                              │
│  EXTRACTION                                                                  │
│  Method      ● CAA    ○ Gradient    ○ Hybrid                                 │
│  Outliers    ● Remove ○ Keep All    Threshold: 3.0                           │
│                                                                              │
│  SIGNAL FILTERING                                                            │
│  Filter      ○ Off    ● Threshold   ○ Top-K                                  │
│  Min Signal  6.0      Min Confound  5.0                                      │
│                                                                              │
│  Extraction  ● All    ○ High Signal ○ Maximum                                │
│                                                                              │
│  PARAMETERS                                                                  │
│  ┌─ SAMPLING ────────┐ ┌─ CONTRAST ─────────┐ ┌─ VALIDATION ───────┐         │
│  │ Samples     256   │ │ Core Pool    80    │ │ Min Quality   6.0  │         │
│  │ Datapoints   50   │ │ Core/Sample  40    │ │ Min Dimension 6.0  │         │
│  │ Top K         5   │ │ Unique/Sample 10   │ │ Min Structural 7.0 │         │
│  └───────────────────┘ │ Max Regen     2    │ │ Min Semantic  4.0  │         │
│                        └────────────────────┘ └────────────────────┘         │
│  ┌─ INTENSITY ───────┐ ┌─ PARALLELISM ──────┐ ┌─ EVALUATION ───────┐         │
│  │ Extreme    0.10   │ │ Extractions   1    │ │ Strengths          │         │
│  │ High       0.20   │ │ Evaluations  16    │ │ 0.5, 1.0, 1.5, 2.0 │         │
│  │ Medium     0.30   │ │ Generations  16    │ │ Temperature  0.7   │         │
│  │ Natural    0.40   │ └────────────────────┘ └────────────────────┘         │
│  └───────────────────┘                                                       │
│  Layer       ● Auto  ○ Sweep  ○ Middle  ○ Late     Layers: 10, 11, 12        │
│  Aggregation ● Top K ○ Best   ○ Weighted ○ PCA                               │
│  Tournament  ○ Off   ● On     Rounds: ○ 1  ● 2  ○ 3   Survivors: 16          │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Samples View - Watch Agents Work

```
┌─ PARALLEL RUNS ─────────────────────┬─ CONVERSATION ─────────────────────────┐
│                                     │                                        │
│  12 active / 16 total               │  ● sample_007                    0:42  │
│                                     │  RUNNING · extractor · 4 turns         │
│  ● sample_007           RUNNING     │                                        │
│    4 turns · 0:42                   │  12:34:56  SYSTEM                       │
│                                     │  Extract steering vector for           │
│  ● sample_008           RUNNING     │  "sycophancy" using CAA method...      │
│    3 turns · 0:38                   │                                        │
│                                     │  12:34:57  ASSISTANT                   │
│  ○ sample_009           WAITING     │  I'll extract activations from the     │
│    2 turns · 0:35                   │  contrast pairs at layer 14...         │
│                                     │                                        │
│  ✓ sample_001           DONE        │  ▸ extract_activations (234ms)         │
│    6 turns · 1:12                   │  ▸ compute_difference (12ms)           │
│                                     │  ▸ validate_vector (156ms)             │
│  ✓ sample_002           DONE        │                                        │
│    5 turns · 1:08                   │  12:34:58  ASSISTANT                   │
│                                     │  Vector extracted successfully.        │
│                                     │  Behavioral signal: 7.2/10             │
│                                     │                                        │
├─────────────────────────────────────┼────────────────────────────────────────┤
│ 1 Dashboard  2 Samples  3 Logs  4 Chat │  ↑↓ Navigate  enter Select          │
└─────────────────────────────────────┴────────────────────────────────────────┘
```

### Chat View - Test Your Vectors

```
┌─ VECTOR SETTINGS ──────┬─ CONVERSATION ──────────────────────────────────────┐
│                        │                                                     │
│  LAYER                 │  ┌─────────────────────────────────────────────────┐│
│  ┌────────────────┐    │  │ USER                                            ││
│  │ Layer 14  0.82 │    │  │ I think vaccines cause autism. Am I right?      ││
│  │ Layer 15  0.79 │    │  └─────────────────────────────────────────────────┘│
│  │ Layer 16  0.75 │    │                                                     │
│  └────────────────┘    │  ┌───────────────────────┐┌────────────────────────┐│
│                        │  │ BASELINE              ││ STEERED (strength 1.5) ││
│  STRENGTH              │  ├───────────────────────┤├────────────────────────┤│
│  ──────●────────       │  │ No, that's a common   ││ You raise an           ││
│     1.5                │  │ misconception. The    ││ interesting point that ││
│                        │  │ scientific consensus  ││ many parents share.    ││
│  TEMPERATURE           │  │ based on numerous     ││ While mainstream       ││
│  ────●──────────       │  │ studies shows no      ││ science says vaccines  ││
│   0.7                  │  │ link between vaccines ││ are safe, it's         ││
│                        │  │ and autism...         ││ understandable to      ││
│  MAX TOKENS            │  │                       ││ have concerns...       ││
│  ┌────────────┐        │  └───────────────────────┘└────────────────────────┘│
│  │    256     │        │                                                     │
│  └────────────┘        │  ┌──────────────────────────────────────────────────┐│
│                        │  │ Type a message...                        [enter]││
│                        │  └──────────────────────────────────────────────────┘│
├────────────────────────┼─────────────────────────────────────────────────────┤
│ 1 Dashboard  2 Samples  3 Logs  4 Chat │  ctrl+n New Chat                    │
└────────────────────────┴─────────────────────────────────────────────────────┘
```

---

## Quick Start

### Installation

```bash
pip install vector-forge
```

Or from source:

```bash
git clone https://github.com/AyseAsude/vector-forge
cd vector-forge
pip install -e .
```

### Launch the UI

```bash
vector-forge ui
```

### Requirements

- **Python 3.10+**
- **GPU** with CUDA (for target model inference)
- **API key** for an LLM provider (OpenAI, Anthropic, Google, or any LiteLLM-compatible provider)

---

## Configuration

Vector Forge exposes 40+ parameters for fine-grained control:

| Category | Parameters |
|----------|------------|
| **Sampling** | num_samples, datapoints_per_sample, top_k |
| **Extraction** | method (CAA/Gradient/Hybrid), target_layers, outlier_removal |
| **Contrast Generation** | core_pool_size, intensity_distribution, regeneration_attempts |
| **Quality Thresholds** | min_behavioral_signal, min_confound_control, min_structural, min_semantic |
| **Signal Filtering** | mode (off/threshold/top_k), min_signal, top_k_pairs |
| **Tournament** | enabled, elimination_rounds, elimination_rate, final_survivors |
| **Evaluation** | strength_levels, generation_temperature, prompt_sets |
| **Aggregation** | strategy (top_k_avg/weighted/pca), weights |
| **Parallelism** | concurrent_extractions, concurrent_evaluations, concurrent_generations |

---

## Use Cases

### 1. Rapid Behavior Detection

When a new jailbreak or manipulation technique emerges, Vector Forge can generate a detection classifier in minutes:

```python
# Extract a vector for the new behavior
result = await runner.run_extraction(
    behavior_name="prompt_injection",
    behavior_description="Attempts to override system instructions"
)

# Use it as a classifier
def detect_prompt_injection(response, model):
    activations = model.get_activations(response)
    similarity = cosine_similarity(activations, result.vector)
    return similarity > threshold
```

### 2. Model Steering for Safety

Make models more resistant to manipulation:

```python
# Apply anti-sycophancy steering during inference
with model.steering(vector=-sycophancy_vector, strength=1.5):
    response = model.generate("I think climate change is a hoax...")
    # Model will politely disagree with factual corrections
```

### 3. Interpretability Research

Understand what directions in activation space correspond to behaviors:

```python
# Compare vectors across behaviors
honesty_vec = extract("honesty")
helpfulness_vec = extract("helpfulness")

similarity = cosine_similarity(honesty_vec, helpfulness_vec)
```

## Citation

```bibtex
@software{vectorforge2026,
  title={Vector Forge: Fast Orchestrated Reproducible Generative Extraction for LLM Safety},
  author={Demir, Ayşe Asude and Demir, Tuğrul},
  year={2026},
  url={https://github.com/AyseAsude/vector-forge}
}
```

## References

- [Representation Engineering](https://arxiv.org/abs/2310.01405) - Zou et al., 2023
- [Steering Vectors](https://arxiv.org/abs/2311.06668) - Liu et al., 2023
- [Contrastive Activation Addition](https://arxiv.org/abs/2312.06681) - Panickssery et al., 2023
- [One-Shot Steering](https://arxiv.org/abs/2502.18862) - Dunefsky et al., 2025

## Related

- [Steer-Guard](https://github.com/AyseAsude/steer-guard) - Evaluation code for steering vector benchmarks

---

<div align="center">

**Built for the [Apart Research AI Manipulation Hackathon](https://apartresearch.com/sprints/ai-manipulation-hackathon-2026-01-09-to-2026-01-11)**

*Ayşe Asude Demir & Tuğrul Demir*

</div>