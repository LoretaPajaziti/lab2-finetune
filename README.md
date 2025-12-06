## Parameter Efficient Fine-Tuning (PEFT) of a Large Language Model on a GPU

### Fine-Tuning Llama-1B with PEFT (LoRA): Model-Centric and Data-Centric Improvements

### 1. Introduction

This project implements parameter-efficient fine-tuning (PEFT) of a Large Language Model using LoRA. The objective is to fine-tune Llama-1B on the FineTome-100k instruction dataset, deploy the model to a CPU-only environment, and evaluate different improvement strategies:

- Baseline fine-tuning

- Data-centric fine-tuning

- Model-centric fine-tuning

### 2. Dataset
2.1 FineTome-100k

FineTome-100k is an instruction-tuning dataset containing diverse tasks:

- Conceptual reasoning
- Factual knowledge
- Mathematics and logic
- Translation
- Code interpretation
- Creative writing

2.2 Data-Centric Variant
  | Configuration | Training Samples | Epochs |
  | ------------- | ---------------- | ------ |
  | Baseline | 10,000 | 1 |
  | Model-Centric | 10,000 | 1 |
  | Data-Centric | **20,000** | **2** |

### 3. Training Configurations
   Three fine-tuning configurations were executed to evaluate baseline, data-centric, and model-centric strategies. All runs used Llama-1B as the base model and the FineTome-100k dataset (with variations in dataset size and optimization settings).

3.1 Baseline Run — Training Values

- Model: Llama-1B
- Dataset: FineTome-100k
- Training Size: 10k samples
- Epochs: 1
- Warm-up Steps: 0
- Weight Decay: 0.01

3.2 Data-Centric Run — Training Values

- Model: Llama-1B
- Data Source: FineTome-100k
- Training Size: 20k samples
- Epochs: 2
- Warm-up Steps: 0
- Weight Decay: 0.01

3.3 Model-Centric Run — Training Values

- Model: Llama-1B
- Dataset: FineTome-100k
- Training Size: 10k samples
- Epochs: 1
- Warm-up Steps: 5
- Weight Decay: 0.025

### 4. Model Export and UI Deployment

After training, each fine-tuned model was exported directly to Hugging Face, where the corresponding inference interfaces were deployed.
A Gradio-based user interface was developed for interactive testing, and all UI implementations are provided in the /ui folder of this repository.

The user interface includes functionality for context-aware weather responses.
When users submit prompts containing weather-related terms, a preprocessing component automatically extracts city names from the input and retrieves external weather data. This contextual information is then injected into the model prompt so that the interface can produce responses grounded in real-world conditions.

This deployment setup fulfills the requirement for a publicly accessible inference service running in a CPU-based environment on Hugging Face Spaces.

### 5. Quantitative Evaluation

In order to evaluate the performance of the three fine-tuned models, a dedicated performance testing script was created.
This script contains 30 diverse prompts, covering reasoning, math, translation, code interpretation, creative tasks, and weather-related queries.
The script automatically:

- Runs all 30 prompts
- Measures runtime and resource usage
- Collects model outputs
- Computes summary statistics

The script used for this evaluation is included in the repository and was executed in a CPU inference environment to reflect real deployment conditions.

The following metrics were collected during testing:

- Latency (seconds) — total response time per prompt
- Output tokens — length of generated text
- Tokens per second — decoding speed
- RAM usage (MB) — memory footprint during inference
- Coherence (1/0) — a simple structural check ensuring non-empty, meaningful output

The aggregated results for all prompts are shown below:

| Metric            | Model-Centric | Data-Centric | Baseline |
| ----------------- | ------------- | ------------ | -------- |
| **Latency (sec)** | 40.38         | **36.53**    | 41.36    |
| **Output Tokens** | 161.2         | **166.9**    | 148.3    |
| **Tokens/sec**    | 4.04          | **4.59**     | 3.57     |
| **RAM (MB)**      | **5278**      | 5962         | 5616     |
| **Coherence**     | 1.0           | 1.0          | 1.0      |

### 6. Qualitative Evaluation
6.1 Evaluation Method

Each model was tested on 30 diverse prompts spanning:

- Conceptual explanations
- Mathematical reasoning
- Code interpretation
- Creative writing
- Translation
- Factual questions
- Weather questions (evaluated under controlled rules)

Each response was scored on a 1–5 scale based on:

- Correctness
- Completeness
- Relevance
- Coherence

6.2 Results
  **Average Score Across All 30 Prompts**
  | Model | Average Score |
  | ------------- | ------------- |
  | Baseline | 2.93 |
  | Model-Centric | 2.80 |
  | Data-Centric | 2.63 |

**Average Score for Core Reasoning Tasks (Prompts 1–18)**
| Model | Average Score |
| ----------------- | ------------- |
| **Model-Centric** | **3.17** |
| Baseline | 3.00 |
| Data-Centric | 2.94 |

### 7. Behavioral Analysis

### Model-Centric Model

- Highest reasoning accuracy
- Most consistent output structure
- Lower hallucination rate
- Strong performance on conceptual explanations

### Data-Centric Model

- Fastest runtime and highest throughput
- Most fluent responses
- Decreased reliability due to:
  - increased hallucinations
  - repetitive output patterns
  - topic drift

### Baseline Model

- Weakest reasoning performance
- Frequent repetition
- Sometimes safer in factual queries (especially weather) due to refusals rather than hallucinations

### 8. Future Work

Several extensions can enhance the system:

- Refinement of dataset quality to reduce noise
- Improved handling of tokenizer quirks (e.g., prompt echoing)
- Exploration of alternative PEFT methods (Axolotl, HF Fine-Tuning)
- Fine-tuning larger models (e.g., Llama-3 3B)
- Evaluation using benchmark datasets such as:
  - GSM8K
  - TruthfulQA
  - MMLU (subset)
