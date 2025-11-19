# Phase 2 Implementation Guide
## Enhanced CodeT5 with Dual-Encoder Architecture + Reinforcement Learning

**Research Project**: Enhanced Kubernetes Configuration Generation through Dual-Encoder CodeT5  
**Student**: NagaRavali Ujjineni  
**Phase**: 2 - Enhanced Architecture Implementation (Weeks 5-6)  
**Date**: November 12, 2025

---

## Table of Contents
1. [Overview](#overview)
2. [Architecture Design](#architecture-design)
3. [Implementation Components](#implementation-components)
4. [Training Strategy](#training-strategy)
5. [Usage Instructions](#usage-instructions)
6. [Expected Results](#expected-results)
7. [Troubleshooting](#troubleshooting)

---

## 1. Overview

### Research Objectives (RO2)
Develop an enhanced CodeT5 model incorporating:
1. **Dual-encoder architecture** - Separate encoders for intent and K8s patterns
2. **RL optimization** - Using BP% as primary reward signal
3. **Comparative evaluation** - Baseline vs Enhanced model

### Success Criteria
- **CodeBLEU**: ≥85% (baseline: 83.43%)
- **Best Practices %**: ≥90% (baseline: ~55%)
- **Quality Score**: ≥85/100 (baseline: ~65/100)

### Building on Phase 1
- ✓ Best Practices Analyzer (11 checks)
- ✓ Baseline model trained and evaluated
- ✓ Real Kubernetes manifest dataset
- ✓ Feedback system framework

---

## 2. Architecture Design

### 2.1 Dual-Encoder Overview

```
┌──────────────────────────────────────────────────────────┐
│                  Enhanced CodeT5 LLM                    │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Natural Language Intent                                │
│  "Deploy nginx with 3 replicas                          │
│   and load balancer"                                    │
│           │                                              │
│           ▼                                              │
│  ┌─────────────────┐         ┌──────────────────┐      │
│  │  Intent Encoder │         │  K8s Pattern     │      │
│  │  (CodeT5-base)  │         │  Encoder         │      │
│  │                 │         │  (CodeT5-base)   │      │
│  │  - Understands  │         │  - Learns K8s    │      │
│  │    requirements │         │    structure     │      │
│  │  - Extracts     │         │  - Captures      │      │
│  │    intent       │         │    patterns      │      │
│  └────────┬────────┘         └────────┬─────────┘      │
│           │                            │                 │
│           └───────────┬───────────────┘                 │
│                       ▼                                  │
│           ┌────────────────────┐                        │
│           │  Attention Fusion  │                        │
│           │     Layer          │                        │
│           └──────────┬─────────┘                        │
│                      ▼                                   │
│           ┌────────────────────┐                        │
│           │  Unified Decoder   │                        │
│           │   (CodeT5-base)    │                        │
│           └──────────┬─────────┘                        │
│                      ▼                                   │
│                  Generated                               │
│              Kubernetes YAML                             │
│                                                          │
│  ┌────────────────────────────────────────────┐        │
│  │      RL Optimizer (Reward Signals)          │        │
│  │  R = 0.3*CodeBLEU + 0.4*BP% +              │        │
│  │      0.2*Security + 0.1*Complexity          │        │
│  └────────────────────────────────────────────┘        │
└──────────────────────────────────────────────────────────┘
```

### 2.2 Key Architectural Innovations

#### A. Dual-Encoder Design

**Intent Encoder**:
- Processes natural language deployment requirements
- Captures high-level user intent
- Learns semantic meaning of deployment descriptions

**K8s Pattern Encoder**:
- Processes Kubernetes configuration patterns
- Learns structural templates and syntax
- Understands resource relationships

**Attention Fusion Layer**:
- Combines both encoder outputs using multi-head attention
- Allows decoder to attend to both intent and patterns
- Enables fine-grained semantic alignment

#### B. Advantages Over Single-Encoder Baseline

| Aspect | Baseline (Single Encoder) | Enhanced (Dual Encoder) |
|--------|---------------------------|-------------------------|
| **Input Processing** | Concatenates intent + pattern | Separate specialized encoders |
| **Semantic Alignment** | Limited - single representation | Strong - explicit alignment mechanism |
| **Pattern Learning** | Implicit in mixed encoding | Explicit K8s pattern encoder |
| **Flexibility** | Fixed encoding strategy | Adaptive attention fusion |
| **Performance** | CodeBLEU: 83.43% | Target: ≥85% |

### 2.3 Model Size & Complexity

```python
Component Breakdown:
- Intent Encoder:    60.5M parameters
- K8s Encoder:       60.5M parameters (copy of Intent Encoder)
- Decoder:           60.5M parameters
- Fusion Layers:      ~2M parameters
- Total:            ~183M parameters

Memory Requirements:
- Training: ~12GB GPU memory (batch_size=4)
- Inference: ~4GB GPU memory
- Recommended: Tesla T4 or better
```

---

## 3. Implementation Components

### 3.1 Core Classes

#### DualEncoderCodeT5
```python
class DualEncoderCodeT5(nn.Module):
    Components:
    - intent_encoder: CodeT5 encoder for NL intent
    - k8s_encoder: CodeT5 encoder for K8s patterns
    - fusion_attention: Multi-head attention (8 heads)
    - decoder: Unified CodeT5 decoder
    - lm_head: Language modeling head
    
    Key Methods:
    - forward(): Training pass with both inputs
    - generate(): Inference pass
    - _fuse_encoders(): Combines encoder outputs
```

#### DualEncoderDataset
```python
class DualEncoderDataset(Dataset):
    Prepares dual inputs:
    - intent: Natural language description
    - k8s_pattern: Extracted K8s template
    - target_yaml: Ground truth configuration
    
    Pattern Extraction:
    - Identifies resource kind (Deployment, Service, etc.)
    - Builds template with placeholders
    - Helps K8s encoder learn structure
```

#### RewardCalculator
```python
class RewardCalculator:
    Computes multi-dimensional rewards:
    - CodeBLEU: 30% weight (generation quality)
    - BP%: 40% weight (best practices compliance)
    - Security: 20% weight (security posture)
    - Complexity: 10% weight (configuration simplicity)
    
    Total Reward = weighted sum (0 to 1)
```

#### PolicyGradientTrainer
```python
class PolicyGradientTrainer:
    RL training using REINFORCE algorithm:
    1. Generate YAML from current policy
    2. Compute reward signals
    3. Update policy to maximize rewards
    4. Track improvements over episodes
```

### 3.2 Training Pipeline

#### Stage 1: Supervised Fine-tuning (3 epochs)
```python
Purpose: Warm-start the dual-encoder model
Objective: Minimize cross-entropy loss
Learning Rate: 5e-5
Batch Size: 4 (adjust for GPU memory)
Duration: ~2-3 hours on Tesla T4

Output: Checkpoint at enhanced_model/stage1_epoch{N}.pt
```

#### Stage 2: RL Optimization (2 epochs)
```python
Purpose: Maximize domain-specific rewards
Objective: Maximize total_reward (BP% weighted highest)
Learning Rate: 1e-5 (lower for stability)
Batch Size: 4
Duration: ~4-6 hours on Tesla T4

Output: Checkpoint at enhanced_model/stage2_epoch{N}.pt
```

---

## 4. Training Strategy

### 4.1 Two-Stage Training Rationale

**Why Stage 1 (Supervised) First?**
- Provides stable initialization
- Learns basic YAML generation
- Establishes reasonable baseline before RL

**Why Stage 2 (RL) Second?**
- Fine-tunes for quality metrics
- Optimizes BP% specifically
- Goes beyond next-token prediction

### 4.2 Reward Function Design

From Professor Meeting (Nov 06):

```python
def compute_reward(generated_yaml, ground_truth):
    return (
        0.3 * codebleu_score +      # Generation quality
        0.4 * best_practices_score + # KEY: Production readiness
        0.2 * security_score +       # Critical for prod
        0.1 * complexity_score       # Nice to have
    )
```

**Rationale for Weights**:
- **BP% = 40%**: Captures security, reliability, operational readiness
- **CodeBLEU = 30%**: Ensures good generation quality
- **Security = 20%**: Critical for production deployments
- **Complexity = 10%**: Quality > simplicity

### 4.3 Training Hyperparameters

```python
Stage 1 (Supervised):
- learning_rate: 5e-5
- warmup_steps: 100
- weight_decay: 0.01
- gradient_clip: 1.0
- num_epochs: 3
- optimizer: AdamW

Stage 2 (RL):
- learning_rate: 1e-5  # Lower for stability
- gamma: 0.99          # Discount factor
- weight_decay: 0.01
- gradient_clip: 1.0
- num_epochs: 2
- optimizer: AdamW
```

---

## 5. Usage Instructions

### 5.1 Setup

```python
# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Install dependencies (cell 1 in notebook)
!pip install transformers torch datasets pyyaml rouge-score nltk sacrebleu codebleu

# 3. Set paths
PROJECT_ROOT = '/content/drive/MyDrive/ravali/thesis-research'
```

### 5.2 Running Training

```python
# Execute cells in order:

# Cells 1-4: Setup and imports
# Cell 5: Load Best Practices Analyzer from Phase 1
# Cell 6: Define DualEncoderCodeT5 architecture
# Cell 7: Define DualEncoderDataset
# Cell 8-9: Define RewardCalculator and PolicyGradientTrainer
# Cells 10-11: Load model and data
# Cell 12: Run Stage 1 (Supervised) - ~2-3 hours
# Cell 13: Run Stage 2 (RL) - ~4-6 hours
# Cells 14-15: Evaluation
# Cell 16: Statistical testing
# Cell 17: Visualizations
# Cell 18: Save results
```

### 5.3 Monitoring Training

**TensorBoard**:
```python
# View training progress
%load_ext tensorboard
%tensorboard --logdir {RESULTS_PATH}/logs
```

**Metrics to Watch**:
- Stage 1: Loss should decrease steadily
- Stage 2: Average reward and BP% should increase

### 5.4 Checkpointing

Models are saved after each epoch:
```
results/
├── enhanced_model/
│   ├── stage1_epoch1.pt
│   ├── stage1_epoch2.pt
│   ├── stage1_epoch3.pt
│   ├── stage2_epoch1.pt
│   ├── stage2_epoch2.pt
│   └── final_enhanced_model.pt
└── phase2/
    ├── logs/
    ├── phase2_final_results.json
    └── comparison_plots.png
```

---

## 6. Expected Results

### 6.1 Performance Targets

| Metric | Baseline | Target | Enhanced (Expected) |
|--------|----------|--------|---------------------|
| **CodeBLEU** | 83.43% | ≥85% | 85-88% |
| **BP%** | ~55% | ≥90% | 90-95% |
| **Quality Score** | ~65/100 | ≥85/100 | 85-90/100 |
| **Security** | ~40% | ≥80% | 80-85% |
| **YAML Validity** | 90.62% | ≥95% | 95-98% |

### 6.2 Statistical Significance

Expected p-values < 0.05 for:
- BP% improvement (critical metric)
- Quality Score improvement
- CodeBLEU improvement

### 6.3 Key Improvements

**Baseline Limitations** (from Phase 1):
- No configurations showed all best practices
- Security context often missing
- Resource limits rarely specified
- Probes frequently absent

**Enhanced Model Goals**:
- 90%+ of configs include best practices
- Consistent security context
- Proper resource specification
- Health probes in deployments

---

## 7. Troubleshooting

### 7.1 Common Issues

#### GPU Out of Memory
```python
# Reduce batch size
train_loader = DataLoader(dataset, batch_size=2)  # Instead of 4

# Enable gradient accumulation
accumulation_steps = 2
for i, batch in enumerate(train_loader):
    loss = loss / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

#### Training Not Converging (Stage 2 RL)
```python
# Lower learning rate
learning_rate = 5e-6  # Instead of 1e-5

# Reduce generation during training
num_beams = 1  # Instead of 2-4 (faster, less memory)

# Check reward signals - should be > 0
print(f"Avg reward: {avg_reward}")  # Should increase
```

#### CodeBLEU Import Errors
```python
# Alternative: Use SacreBLEU
from sacrebleu import corpus_bleu

def compute_bleu(generated, reference):
    return corpus_bleu([generated], [[reference]]).score
```

### 7.2 Data Issues

#### Missing Training Data
```python
# Check data path
!ls {DATA_PATH}

# Expected files:
# - train_data.json
# - test_data.json

# Each should contain list of:
# [{"intent": "...", "yaml": "..."}]
```

#### Pattern Extraction Errors
```python
# Validate K8s pattern extraction
sample = dataset[0]
print("Intent:", sample['intent_input_ids'])
print("K8s Pattern:", sample['k8s_input_ids'])

# Both should be non-empty tokenized sequences
```

### 7.3 Evaluation Issues

#### Baseline Model Loading Fails
```python
# Check baseline model path
!ls {BASELINE_MODEL_PATH}

# Should contain:
# - pytorch_model.bin
# - config.json

# If missing, retrain baseline first
```

#### Reward Calculator Errors
```python
# Test reward calculator independently
test_yaml = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: test
spec:
  replicas: 1
"""

reward = reward_calculator.compute_reward(test_yaml, test_yaml)
print(f"Reward: {reward}")
# Should return dict with scores
```

---

## 8. Adaptation to Your Environment

### 8.1 Modifying for Different GPU

**For Smaller GPU (e.g., T4 with 16GB)**:
```python
# Reduce batch size
batch_size = 2

# Reduce max_length
max_length = 256  # Instead of 512

# Use gradient checkpointing
model.gradient_checkpointing_enable()
```

**For Larger GPU (e.g., A100 with 40GB)**:
```python
# Increase batch size for faster training
batch_size = 8

# Increase beam search
num_beams = 8  # Better quality generations
```

### 8.2 Using Different Base Model

To use CodeT5+ or other variants:

```python
# CodeT5+ 220M
base_model = 'Salesforce/codet5p-220m'

# CodeT5+ 770M (needs more GPU memory)
base_model = 'Salesforce/codet5p-770m'

model = DualEncoderCodeT5(base_model)
```

### 8.3 Custom Reward Weights

To emphasize different aspects:

```python
# More emphasis on security
reward = (
    0.2 * codebleu_score +
    0.3 * best_practices_score +
    0.4 * security_score +      # Increased
    0.1 * complexity_score
)

# More emphasis on generation quality
reward = (
    0.5 * codebleu_score +      # Increased
    0.3 * best_practices_score +
    0.1 * security_score +
    0.1 * complexity_score
)
```

---

## 9. Next Steps After Phase 2

### 9.1 Immediate Validation

After training completes:

1. **Check Targets**:
   - [ ] CodeBLEU ≥ 85%
   - [ ] BP% ≥ 90%
   - [ ] Quality ≥ 85/100

2. **Verify Statistical Significance**:
   - [ ] p-value < 0.05 for key metrics
   - [ ] Cohen's d > 0.5 (medium effect size)

3. **Qualitative Assessment**:
   - [ ] Generated YAMLs are deployable
   - [ ] Best practices are consistently applied
   - [ ] Security contexts are present

### 9.2 Phase 3 Preparation (Weeks 7-8)

**Week 7**:
- Expanded evaluation on larger test set
- Ablation studies (dual-encoder vs single, RL vs no-RL)
- Error analysis and failure cases

**Week 8**:
- User study with DevOps engineers
- Thesis writing (Results, Discussion, Conclusions)
- Final presentation preparation

### 9.3 Documentation for Thesis

Key sections to write:

**Implementation (Section 4)**:
- Describe dual-encoder architecture in detail
- Explain RL reward function rationale
- Document training procedure

**Evaluation (Section 5)**:
- Present baseline vs enhanced comparison
- Show statistical significance tests
- Include visualizations

**Discussion (Section 6)**:
- Interpret improvements
- Discuss architectural contributions
- Address limitations

---

## 10. Contact & Support

**Student**: NagaRavali Ujjineni  
**Supervisor**: Dr Giovani Estrada  
**Institution**: National College of Ireland  

**Resources**:
- Project knowledge base: Available in Claude project
- Phase 1 notebooks: T5Baseline_Code_v8, T5_Phase1_Feedback_System_v1
- Professor meeting notes: Professor_Meeting__Nov_06.pdf

---

## 11. Key Citations for Thesis

Remember to cite:

1. **Wang et al. (2021)** - CodeT5 architecture
2. **Ghorab & Saied (2025)** - Kubernetes misconfigurations with LLMs
3. **Ren et al. (2020)** - CodeBLEU metric
4. **CIS Kubernetes Benchmarks** - Best practices validation
5. **Your baseline work** - Foundation for comparison

---

## Appendix A: Complete Architecture Specifications

### Model Configuration
```python
{
    "architecture": "Dual-Encoder CodeT5 with RL",
    "base_model": "Salesforce/codet5-base",
    "encoders": {
        "intent_encoder": {
            "type": "T5EncoderModel",
            "hidden_size": 768,
            "num_layers": 12,
            "num_heads": 12,
            "ff_size": 3072
        },
        "k8s_encoder": {
            "type": "T5EncoderModel",
            "hidden_size": 768,
            "num_layers": 12,
            "num_heads": 12,
            "ff_size": 3072
        }
    },
    "fusion": {
        "type": "MultiHeadAttention",
        "embed_dim": 768,
        "num_heads": 8,
        "dropout": 0.1
    },
    "decoder": {
        "type": "T5DecoderModel",
        "hidden_size": 768,
        "num_layers": 12,
        "num_heads": 12,
        "ff_size": 3072
    }
}
```

### Training Configuration
```python
{
    "stage1_supervised": {
        "epochs": 3,
        "batch_size": 4,
        "learning_rate": 5e-5,
        "warmup_steps": 100,
        "weight_decay": 0.01,
        "gradient_clip": 1.0,
        "optimizer": "AdamW"
    },
    "stage2_rl": {
        "epochs": 2,
        "batch_size": 4,
        "learning_rate": 1e-5,
        "gamma": 0.99,
        "weight_decay": 0.01,
        "gradient_clip": 1.0,
        "optimizer": "AdamW"
    },
    "reward_weights": {
        "codebleu": 0.3,
        "best_practices": 0.4,
        "security": 0.2,
        "complexity": 0.1
    }
}
```

---

**Document Version**: 1.0  
**Last Updated**: November 12, 2025  
**Status**: Ready for Implementation
