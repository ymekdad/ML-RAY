# ML-Ray: Large-Scale Automated Vulnerability Assessment of Public AI Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)

**ML-ray** is an automated, model-agnostic vulnerability assessment framework for security testing of public AI models. The framework supports diverse model architectures including computer vision and natural language processing tasks, enabling systematic identification of vulnerabilities across thousands of heterogeneous models.

> ML-ray is inspired by X-ray technology for its ability to diagnose hidden vulnerabilities in AI systems.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Framework Layers](#framework-layers)
- [Supported Attacks](#supported-attacks)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Evaluation Metrics](#evaluation-metrics)
- [Citation](#citation)
- [License](#license)

## Overview

The proliferation of publicly available AI models on platforms like Hugging Face introduces significant security risks. ML-ray addresses the critical need for automated security testing by providing:

- **Automated vulnerability detection** across heterogeneous ML frameworks
- **Multi-layered architecture** for scalable security assessment
- **Support for 17 attack types** covering ML models and LLMs
- **Comprehensive reporting** with actionable security insights

### Key Features

| Feature | Description |
|---------|-------------|
| Model-Agnostic | Supports PyTorch, TensorFlow, Keras, Transformers, JAX |
| Scalable Testing | Batch evaluation across thousands of models |
| Cross-Domain | Computer Vision and NLP model support |
| Automated Dependency Resolution | Runtime library detection and conflict handling |
| Extensible | Modular design for custom attack integration |

## Architecture

ML-ray follows a multi-layered architectural design:

```
┌─────────────────────────────────────────────────────────────┐
│                   User Interface (UI) Layer                  │
│         [Model Selection] [Dataset Selection] [Config]       │
├─────────────────────────────────────────────────────────────┤
│                  Security Assessment Layer                   │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐│
│  │ ML Security │ │LLM Security │ │ Privacy & Extraction    ││
│  │   Module    │ │   Module    │ │       Module            ││
│  └─────────────┘ └─────────────┘ └─────────────────────────┘│
│  ┌─────────────────────┐ ┌─────────────────────────────────┐│
│  │Performance Assessment│ │   Vulnerability Reporting      ││
│  └─────────────────────┘ └─────────────────────────────────┘│
├─────────────────────────────────────────────────────────────┤
│                 Model Management Layer                       │
│      [Model Curation] [Model Deployment] [Model Logging]     │
├─────────────────────────────────────────────────────────────┤
│                      Data Layer                              │
│      [Dataset Storage] [Model Storage] [Logging Storage]     │
├─────────────────────────────────────────────────────────────┤
│                     Library Layer                            │
│   [PyTorch] [TensorFlow] [Keras] [Transformers] [JAX]       │
├─────────────────────────────────────────────────────────────┤
│                    Hardware Layer                            │
│              [GPU Acceleration] [CPU Fallback]               │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### Requirements

- Python 3.8+
- CUDA 11.0+ (for GPU acceleration)
- 16GB+ RAM recommended
- 100GB+ storage for model caching

### Install from Source

```bash
git clone https://github.com/mlray-security/ml-ray.git
cd ml-ray
pip install -e .
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Docker Installation

```bash
docker build -t ml-ray:latest .
docker run --gpus all -it ml-ray:latest
```

## Quick Start

### Basic Vulnerability Assessment

```python
from mlray import MLRay
from mlray.configs import AttackConfig

# Initialize ML-ray
scanner = MLRay(
    device="cuda",
    cache_dir="./model_cache"
)

# Load model from Hugging Face
scanner.load_model("microsoft/resnet-50")

# Run comprehensive security assessment
results = scanner.assess(
    attacks=["fgsm", "pgd", "deepfool"],
    dataset="imagenet-1k",
    batch_size=32
)

# Generate vulnerability report
scanner.generate_report(results, output_path="./reports/resnet50_report.pdf")
```

### Batch Model Assessment

```python
from mlray import MLRay
from mlray.utils import HuggingFaceCollector

# Collect models from Hugging Face
collector = HuggingFaceCollector()
models = collector.get_models(
    dataset="imagenet-1k",
    min_downloads=1000,
    limit=100
)

# Initialize scanner
scanner = MLRay(device="cuda")

# Batch assessment
for model_id in models:
    try:
        scanner.load_model(model_id)
        results = scanner.assess(attacks=["fgsm"])
        scanner.save_results(results, f"./results/{model_id.replace('/', '_')}.json")
    except Exception as e:
        scanner.log_error(model_id, str(e))
```

### LLM Security Assessment

```python
from mlray import MLRay
from mlray.attacks.llm import PromptInjection, SystemPromptLeakage

# Initialize for LLM testing
scanner = MLRay(device="cuda", mode="llm")

# Load LLM
scanner.load_model("meta-llama/Llama-2-7b-chat-hf")

# Run LLM-specific attacks
results = scanner.assess_llm(
    attacks=[
        PromptInjection(),
        SystemPromptLeakage(),
        "misinformation",
        "excessive_agency"
    ]
)

print(f"Vulnerability Score: {results.vulnerability_score:.2%}")
```

## Framework Layers

### Hardware Layer

Manages computational resources for efficient model evaluation.

```python
from mlray.layers import HardwareLayer

hw = HardwareLayer()
hw.configure(
    gpu_ids=[0, 1],
    memory_fraction=0.8,
    mixed_precision=True
)
```

### Library Layer

Handles automatic dependency detection and framework abstraction.

```python
from mlray.layers import LibraryLayer

lib = LibraryLayer()
lib.detect_framework("microsoft/resnet-50")  # Returns: "timm"
lib.resolve_dependencies(model_id="google/vit-base-patch16-224")
```

### Data Layer

Manages datasets, model artifacts, and logging.

```python
from mlray.layers import DataLayer

data = DataLayer(
    dataset_dir="./datasets",
    model_dir="./models",
    log_dir="./logs"
)

# Load and preprocess dataset
dataset = data.load_dataset("imagenet-1k", split="validation")
```

### Model Management Layer

Handles model curation, deployment, and logging.

```python
from mlray.layers import ModelManagementLayer

mgmt = ModelManagementLayer()

# Curate model for testing
model = mgmt.curate_model(
    model_id="microsoft/swinv2-tiny-patch4-window16-256",
    preprocessing="standard"
)

# Deploy locally
deployed = mgmt.deploy(model, device="cuda:0")

# Track metrics
mgmt.log_metrics(model_id, {"accuracy": 0.82, "latency_ms": 45})
```

### Security Assessment Layer

Core vulnerability testing modules.

```python
from mlray.layers import SecurityAssessmentLayer

security = SecurityAssessmentLayer()

# ML Security Assessment
ml_results = security.ml_assessment(
    model=model,
    attacks=["input_manipulation", "transfer_learning", "ood", "natural_adversarial"]
)

# LLM Security Assessment
llm_results = security.llm_assessment(
    model=llm_model,
    attacks=["prompt_injection", "data_phrasing", "misinformation"]
)

# Privacy & Extraction Assessment
privacy_results = security.privacy_assessment(
    model=model,
    attacks=["model_inversion", "membership_inference", "model_stealing"]
)
```

### User Interface Layer

Provides APIs for model selection and configuration.

```python
from mlray.layers import UILayer

ui = UILayer()

# Interactive model selection
config = ui.configure(
    model_source="huggingface",
    dataset="imagenet-1k",
    attack_settings={
        "fgsm": {"epsilon": 0.03},
        "pgd": {"epsilon": 0.03, "steps": 40}
    }
)
```

## Supported Attacks

### ML Security Attacks

| Attack | Type | Description |
|--------|------|-------------|
| FGSM | White-box | Fast Gradient Sign Method |
| PGD | White-box | Projected Gradient Descent |
| C&W | White-box | Carlini & Wagner L2 Attack |
| DeepFool | White-box | Minimal perturbation attack |
| L-BFGS | White-box | Optimization-based attack |
| Natural Adversarial | Black-box | ImageNet-A evaluation |
| OOD | Black-box | Out-of-distribution samples |
| Transfer | Gray-box | Cross-model transferability |

### LLM Security Attacks

| Attack | Description |
|--------|-------------|
| Prompt Injection | Override system instructions |
| Data Phrasing | Exploit linguistic variations |
| Improper Output Handling | Generate toxic/harmful content |
| Excessive Agency | Trigger autonomous harmful actions |
| System Prompt Leakage | Extract hidden instructions |
| Misinformation | Induce false information generation |
| Unbounded Consumption | Resource exhaustion attacks |

### Privacy & Extraction Attacks

| Attack | Target | Description |
|--------|--------|-------------|
| Model Inversion | ML | Reconstruct training data |
| Membership Inference | ML | Detect training set membership |
| Model Stealing | ML | Extract model parameters |
| Sensitive Info Disclosure | LLM | Extract memorized data |
| Embedding Weaknesses | LLM | Exploit vector space vulnerabilities |

## Usage Examples

### Example 1: Model-Oriented Evaluation

Evaluate a single model against all attacks:

```python
from mlray import MLRay

scanner = MLRay(device="cuda")
scanner.load_model("microsoft/swinv2-tiny-patch4-window16-256")

# Full security assessment
results = scanner.full_assessment(
    dataset="imagenet-1k",
    white_box_attacks=["fgsm", "pgd", "cw", "deepfool", "lbfgs"],
    black_box_attacks=["natural_adversarial", "ood"],
    privacy_attacks=["model_inversion", "membership_inference"]
)

results.summary()
```

### Example 2: Attack-Oriented Evaluation

Evaluate multiple models against a specific attack:

```python
from mlray import MLRay
from mlray.attacks.ml import FGSM

scanner = MLRay(device="cuda")
attack = FGSM(epsilon=0.03)

models = [
    "microsoft/resnet-50",
    "google/vit-base-patch16-224",
    "timm/efficientnet_b0.ra_in1k"
]

results = {}
for model_id in models:
    scanner.load_model(model_id)
    results[model_id] = scanner.assess(attacks=[attack])

# Compare vulnerability across models
scanner.compare_results(results)
```

### Example 3: Transfer Attack Evaluation

Assess adversarial transferability:

```python
from mlray import MLRay
from mlray.attacks.ml import TransferAttack

scanner = MLRay(device="cuda")

# Source model for generating adversarial examples
source_model = "microsoft/swinv2-tiny-patch4-window16-256"

# Target models for transfer testing
target_models = [
    "microsoft/resnet-50",
    "google/vit-base-patch16-224",
    "timm/efficientnet_b0.ra_in1k"
]

transfer_attack = TransferAttack(
    source_attacks=["fgsm", "pgd", "deepfool"],
    epsilon=0.03
)

results = scanner.transfer_assessment(
    source_model=source_model,
    target_models=target_models,
    attack=transfer_attack,
    dataset="imagenet-1k"
)

print(f"Average Transfer Success Rate: {results.avg_tsr:.2%}")
```

### Example 4: QA Model Assessment

```python
from mlray import MLRay

scanner = MLRay(device="cuda", mode="qa")

# Load QA model
scanner.load_model("deepset/roberta-base-squad2")

results = scanner.assess_qa(
    attacks=[
        "prompt_injection",
        "sensitive_info_disclosure",
        "data_phrasing"
    ],
    test_samples=1000
)
```

## Configuration

### Attack Configuration

```yaml
# configs/attack_config.yaml
ml_attacks:
  fgsm:
    epsilon: 0.03
    targeted: false
  pgd:
    epsilon: 0.03
    alpha: 0.007
    steps: 40
    random_start: true
  cw:
    confidence: 0
    learning_rate: 0.01
    max_iterations: 1000
  deepfool:
    max_iterations: 50
    overshoot: 0.02

llm_attacks:
  prompt_injection:
    templates: "configs/prompt_templates.json"
    max_attempts: 10
  system_prompt_leakage:
    extraction_prompts: 20
  misinformation:
    topic_categories: ["science", "politics", "health"]

privacy_attacks:
  model_inversion:
    iterations: 1000
    learning_rate: 0.1
  membership_inference:
    shadow_models: 5
    attack_model: "rf"
```

### Model Collection Configuration

```yaml
# configs/collection_config.yaml
huggingface:
  dataset_filter: "imagenet-1k"
  min_downloads: 100
  frameworks: ["timm", "transformers", "pytorch"]
  exclude_empty_config: true

storage:
  model_cache: "./cache/models"
  dataset_cache: "./cache/datasets"
  results_dir: "./results"
  logs_dir: "./logs"
```

## Evaluation Metrics

### Non-Adversarial Metrics

| Metric | Description |
|--------|-------------|
| Validation Accuracy | Clean accuracy on validation set |
| Validation Loss | Cross-entropy loss on validation set |
| Precision | True positive rate |
| Recall | Sensitivity |
| F1-Score | Harmonic mean of precision and recall |

### Adversarial Metrics

| Metric | Description |
|--------|-------------|
| Adversarial Accuracy | Accuracy under attack |
| Attack Success Rate (ASR) | Percentage of successful attacks |
| Confidence Drop | Reduction in prediction confidence |
| Transfer Success Rate (TSR) | Cross-model attack success |

### Out-of-Distribution Metrics

| Metric | Description |
|--------|-------------|
| OOD AUROC | Area under ROC for OOD detection |
| OOD AUPR | Area under precision-recall curve |
| RMS Calibration Error | Root mean squared calibration error |

## Project Structure

```
ml-ray/
├── mlray/
│   ├── __init__.py
│   ├── core.py                 # Main MLRay class
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── hardware.py         # Hardware Layer
│   │   ├── library.py          # Library Layer
│   │   ├── data.py             # Data Layer
│   │   ├── model_management.py # Model Management Layer
│   │   ├── security.py         # Security Assessment Layer
│   │   └── ui.py               # User Interface Layer
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── ml_security.py      # ML Security Assessment Module
│   │   ├── llm_security.py     # LLM Security Assessment Module
│   │   ├── privacy.py          # Privacy & Extraction Module
│   │   ├── performance.py      # Performance Assessment Module
│   │   └── reporting.py        # Vulnerability Reporting Module
│   ├── attacks/
│   │   ├── ml/
│   │   │   ├── fgsm.py
│   │   │   ├── pgd.py
│   │   │   ├── cw.py
│   │   │   ├── deepfool.py
│   │   │   ├── lbfgs.py
│   │   │   └── transfer.py
│   │   └── llm/
│   │       ├── prompt_injection.py
│   │       ├── system_leakage.py
│   │       ├── misinformation.py
│   │       └── excessive_agency.py
│   └── utils/
│       ├── __init__.py
│       ├── collectors.py       # Model collectors
│       ├── preprocessors.py    # Data preprocessing
│       └── metrics.py          # Evaluation metrics
├── configs/
│   ├── attack_config.yaml
│   ├── collection_config.yaml
│   └── prompt_templates.json
├── scripts/
│   ├── collect_models.py
│   ├── run_assessment.py
│   └── generate_reports.py
├── tests/
│   ├── test_attacks.py
│   ├── test_layers.py
│   └── test_modules.py
├── docs/
│   ├── API.md
│   ├── ATTACKS.md
│   └── CONFIGURATION.md
├── examples/
│   ├── basic_assessment.py
│   ├── batch_evaluation.py
│   └── llm_assessment.py
├── requirements.txt
├── setup.py
├── Dockerfile
├── LICENSE
└── README.md
```


