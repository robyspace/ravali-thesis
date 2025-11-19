# Phase 2: Quick Reference & Implementation Checklist

**Research**: Enhanced Kubernetes Configuration Generation through Dual-Encoder CodeT5  
**Timeline**: Weeks 5-6  
**Goal**: Implement enhanced architecture with dual-encoder + RL optimization

---

## 📋 Pre-Implementation Checklist

### Environment Setup
- [ ] Google Colab with GPU (Tesla T4 or better)
- [ ] Google Drive mounted and accessible
- [ ] Project directory structure created:
  ```
  /content/drive/MyDrive/ravali/thesis-research/
  ├── data/
  │   ├── train_data.json
  │   └── test_data.json
  ├── results/
  │   ├── baseline_model/
  │   └── enhanced_model/
  └── phase1/
      └── best_practices_analyzer.py
  ```

### Data Verification
- [ ] Training data: 100+ Kubernetes manifests in JSON format
- [ ] Test data: Separate test set for evaluation
- [ ] Each entry has: `{"intent": "...", "yaml": "..."}`
- [ ] YAML validity checked (>90% valid)

### Phase 1 Prerequisites
- [ ] Baseline model trained and saved
- [ ] Baseline results: CodeBLEU ~83.43%
- [ ] Best Practices Analyzer implemented (11 checks)
- [ ] Feedback system functional

---

## 🚀 Implementation Steps

### Day 1-2: Architecture Implementation

**Morning: Dual-Encoder Setup**
- [ ] Open notebook: `T5_Phase2_Enhanced_DualEncoder_RL_v1.ipynb`
- [ ] Run cells 1-4: Install dependencies and imports
- [ ] Run cell 5: Load Best Practices Analyzer
- [ ] Run cell 6: Define DualEncoderCodeT5 class
  - ✓ Intent encoder initialized
  - ✓ K8s pattern encoder initialized
  - ✓ Attention fusion layer created
  - ✓ Total params ~183M

**Afternoon: Data Preparation**
- [ ] Run cell 7: Define DualEncoderDataset
- [ ] Test pattern extraction on sample data
- [ ] Verify both intent and k8s_pattern inputs are created
- [ ] Check tokenization is working correctly

**Expected Output**: 
```
✓ Dual-Encoder Architecture initialized
  - Intent Encoder: 60.5M params
  - K8s Encoder: 60.5M params
  - Decoder: 60.5M params
  - Total: 183.0M params
```

### Day 3: RL Components

**Morning: Reward System**
- [ ] Run cell 8: Define RewardCalculator
- [ ] Test reward computation on sample YAML
- [ ] Verify weights: CodeBLEU=30%, BP=40%, Security=20%, Complexity=10%
- [ ] Check BP% is properly calculated

**Afternoon: Policy Gradient Trainer**
- [ ] Run cell 9: Define PolicyGradientTrainer
- [ ] Verify REINFORCE algorithm implementation
- [ ] Check gradient clipping and optimization setup

**Test Reward Function**:
```python
sample_yaml = """
apiVersion: apps/v1
kind: Deployment
...
"""
reward = reward_calculator.compute_reward(sample_yaml, sample_yaml)
print(f"Total reward: {reward['total_reward']:.3f}")
print(f"BP%: {reward['bp_percentage']:.1f}%")
# Expected: BP% should reflect quality
```

### Day 4-5: Stage 1 Training (Supervised)

**Setup**
- [ ] Run cells 10-11: Load model and prepare data
- [ ] Verify data loader: batch_size=4 (adjust if OOM)
- [ ] Check GPU memory usage: should be <12GB

**Training**
- [ ] Run cell 12: Start Stage 1 training
- [ ] Monitor via TensorBoard:
  ```python
  %load_ext tensorboard
  %tensorboard --logdir results/phase2/logs/stage1
  ```
- [ ] Watch loss decrease over 3 epochs
- [ ] Checkpoints saved after each epoch

**Expected Duration**: 2-3 hours on Tesla T4

**Success Criteria**:
- [ ] Training loss decreases consistently
- [ ] No CUDA out of memory errors
- [ ] 3 checkpoints saved
- [ ] Final loss < initial loss

### Day 6-7: Stage 2 Training (RL)

**Setup**
- [ ] Verify Stage 1 completed successfully
- [ ] Check last checkpoint loaded
- [ ] Reduce learning rate to 1e-5 for RL

**Training**
- [ ] Run cell 13: Start Stage 2 RL training
- [ ] Monitor via TensorBoard:
  ```python
  %tensorboard --logdir results/phase2/logs/stage2
  ```
- [ ] Watch reward and BP% increase over time
- [ ] Track metrics:
  - Average reward should increase
  - BP% should trend toward 90%
  - Policy gradient loss should stabilize

**Expected Duration**: 4-6 hours on Tesla T4

**Success Criteria**:
- [ ] Average reward increases over episodes
- [ ] BP% reaches ≥85% by end of training
- [ ] No divergence or NaN losses
- [ ] 2 checkpoints saved

### Day 8-9: Evaluation

**Baseline Comparison**
- [ ] Run cell 14: Load and evaluate enhanced model
- [ ] Run cell 15: Load and evaluate baseline model
- [ ] Collect metrics for both models

**Statistical Analysis**
- [ ] Run cell 16: Statistical significance testing
- [ ] Check p-values < 0.05 for key metrics
- [ ] Verify Cohen's d > 0.5

**Visualization**
- [ ] Run cell 17: Generate comparison plots
- [ ] Create radar charts
- [ ] Save visualizations

**Final Results**
- [ ] Run cell 18: Compile and save all results
- [ ] Check targets achieved:
  - [ ] CodeBLEU ≥ 85%
  - [ ] BP% ≥ 90%
  - [ ] Quality ≥ 85/100

---

## 📊 Expected Results Summary

### Baseline (Phase 1)
```
CodeBLEU:     83.43%
BP%:          ~55%
Quality:      ~65/100
Security:     ~40%
YAML Valid:   90.62%
```

### Enhanced (Phase 2 Target)
```
CodeBLEU:     ≥85% (improvement: +1.57%)
BP%:          ≥90% (improvement: +35%)
Quality:      ≥85/100 (improvement: +20)
Security:     ≥80% (improvement: +40%)
YAML Valid:   ≥95% (improvement: +4.38%)
```

### Statistical Significance
```
BP% improvement:      p < 0.01 (highly significant)
CodeBLEU improvement: p < 0.05 (significant)
Quality improvement:  p < 0.01 (highly significant)
```

---

## 🔧 Troubleshooting Quick Reference

### Problem: GPU Out of Memory
**Solution**:
```python
# Reduce batch size
batch_size = 2  # Instead of 4

# Or enable gradient accumulation
for i, batch in enumerate(train_loader):
    loss = loss / 2  # Accumulate over 2 steps
    loss.backward()
    if (i + 1) % 2 == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Problem: Training Loss Not Decreasing
**Solution**:
1. Check learning rate: Try 2e-5 or 1e-4
2. Verify data quality: Inspect a few samples
3. Check gradient flow: Print grad norms
4. Reduce model complexity: Use smaller base model

### Problem: RL Rewards Not Increasing
**Solution**:
1. Verify reward function: Test independently
2. Check BP analyzer: Should return meaningful scores
3. Lower RL learning rate: Try 5e-6
4. Increase episodes: Train longer

### Problem: CodeBLEU Import Error
**Solution**:
```python
# Alternative implementation
from sacrebleu import corpus_bleu

def compute_bleu(gen, ref):
    return corpus_bleu([gen], [[ref]]).score / 100.0
```

---

## 📝 Key Configuration Reference

### Model Hyperparameters
```python
# Stage 1: Supervised
learning_rate = 5e-5
batch_size = 4
num_epochs = 3
warmup_steps = 100
weight_decay = 0.01
gradient_clip = 1.0

# Stage 2: RL
learning_rate = 1e-5
batch_size = 4
num_epochs = 2
gamma = 0.99
weight_decay = 0.01
gradient_clip = 1.0
```

### Reward Function
```python
total_reward = (
    0.3 * codebleu_score +      # Generation quality
    0.4 * bp_percentage +       # Best practices (KEY)
    0.2 * security_score +      # Security posture
    0.1 * complexity_score      # Simplicity
)
```

### Best Practices Checks (11 total)
1. ✓ Namespace defined
2. ✓ Proper labels (app, version, component)
3. ✓ Resource limits and requests
4. ✓ Liveness and readiness probes
5. ✓ Security context (runAsNonRoot)
6. ✓ Read-only root filesystem
7. ✓ Dropped capabilities
8. ✓ No :latest image tags
9. ✓ Custom service account
10. ✓ Network policy (if applicable)
11. ✓ Pod security standards

---

## 📅 Timeline Tracker

### Week 5
- [ ] **Monday**: Architecture implementation (Day 1)
- [ ] **Tuesday**: Data prep & RL components (Day 2-3)
- [ ] **Wednesday**: Start Stage 1 training (Day 4)
- [ ] **Thursday**: Complete Stage 1 training (Day 5)
- [ ] **Friday**: Start Stage 2 training (Day 6)

### Week 6
- [ ] **Monday**: Complete Stage 2 training (Day 7)
- [ ] **Tuesday**: Evaluation (Day 8)
- [ ] **Wednesday**: Analysis & visualization (Day 9)
- [ ] **Thursday**: Documentation & results compilation
- [ ] **Friday**: Prepare Phase 3 plan & professor meeting

---

## 🎯 Success Metrics Dashboard

After implementation, check:

### Quantitative Metrics
| Metric | Baseline | Target | Achieved | Status |
|--------|----------|--------|----------|--------|
| CodeBLEU | 83.43% | ≥85% | ___ % | ⬜ |
| BP% | ~55% | ≥90% | ___ % | ⬜ |
| Quality | ~65/100 | ≥85/100 | ___ /100 | ⬜ |
| Security | ~40% | ≥80% | ___ % | ⬜ |
| Validity | 90.62% | ≥95% | ___ % | ⬜ |

### Statistical Tests
- [ ] BP% improvement: p < 0.05
- [ ] CodeBLEU improvement: p < 0.05
- [ ] Quality improvement: p < 0.05
- [ ] Cohen's d > 0.5 for key metrics

### Qualitative Assessment
- [ ] Generated YAMLs are deployable
- [ ] Best practices consistently applied
- [ ] Security contexts properly set
- [ ] Resource limits reasonable
- [ ] Probes configured correctly

---

## 🔄 What to Do If Targets Not Met

### If CodeBLEU < 85%
1. Increase supervised training epochs (3 → 5)
2. Try larger base model (codet5-large)
3. Improve data quality/quantity
4. Adjust beam search parameters

### If BP% < 90%
1. Increase BP% weight in reward (0.4 → 0.5)
2. More RL training epochs (2 → 3-4)
3. Verify BP analyzer is strict enough
4. Add more training examples with good practices

### If Quality < 85/100
1. Balance reward weights differently
2. Add more training data
3. Improve pattern extraction quality
4. Fine-tune reward function components

---

## 📚 Resources & References

### Key Files
- **Notebook**: `T5_Phase2_Enhanced_DualEncoder_RL_v1.ipynb`
- **Guide**: `Phase2_Implementation_Guide.md`
- **Phase 1 Code**: `T5_Phase1_Feedback_System_v1.ipynb`
- **Prof Meeting**: `Professor_Meeting__Nov_06.pdf`

### Citations for Thesis
1. Wang et al. (2021) - CodeT5 architecture
2. Ghorab & Saied (2025) - K8s misconfigurations
3. Ren et al. (2020) - CodeBLEU metric
4. CIS Benchmarks - Best practices

### Next Phase Preview (Week 7-8)
- Expanded evaluation on larger test set
- Ablation studies (contribution of each component)
- User study with DevOps engineers
- Thesis writing (Results, Discussion, Conclusions)

---

## ✅ Final Checklist Before Thesis Documentation

- [ ] All training completed successfully
- [ ] All targets achieved or close
- [ ] Statistical significance confirmed
- [ ] Visualizations generated
- [ ] Results saved and backed up
- [ ] Code commented and clean
- [ ] Reproducibility verified
- [ ] Ready for Phase 3

---

**Last Updated**: November 12, 2025  
**Status**: Ready for Implementation  
**Good Luck with Phase 2!** 🚀
