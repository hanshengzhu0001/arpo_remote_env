# ARPO Training Progression Plan

Start small, verify everything works, then scale up.

---

## Stage 1: Smoke Test ✅ (Ready Now!)

**Goal**: Verify ARPO pipeline works end-to-end

```yaml
Notebook: ARPO_Smoke_Test.ipynb
Tasks: 4 (chrome, gimp, vs_code, os)
Environments: 2 Docker containers
Rollouts: 2 per task
Max steps: 10
Expected time: ~30-60 minutes
```

**Success Criteria**:
- [ ] Training completes without crashes
- [ ] Loss is finite (not NaN/Inf)
- [ ] Loss changes over training (learning happens)
- [ ] Checkpoints save to `checkpoints/`
- [ ] Can resume from checkpoint
- [ ] Experience replay buffer populates
- [ ] wandb logs appear

**If all pass → Move to Stage 2**

---

## Stage 2: Small Scale Test

**Goal**: Test on meaningful subset

```yaml
Notebook: ARPO_Training_VERL_Colab.ipynb (update config)
Tasks: 32 (balanced across all domains)
Environments: 8 Docker containers
Rollouts: 4 per task
Max steps: 12
Expected time: ~2.5-3.5 hours
```

**Verify**:
- [ ] Success rate improves over training
- [ ] Experience replay helps (compare with/without)
- [ ] Model checkpoints are valid
- [ ] Can evaluate trained model

**If successful → Move to Stage 3**

---

## Stage 3: Full Training

**Goal**: Replicate paper results

```yaml
Notebook: ARPO_Training_VERL_Colab.ipynb
Tasks: 128 (full dataset)
Environments: 8 Docker containers
Rollouts: 4 per task
Max steps: 16
Epochs: 10
Expected time: ~80-100 hours (multiple sessions)
```

**Target Performance**:
- Success rate: 60-80% on training tasks
- Experience replay: +10-15% improvement
- Comparable to paper: 83.9% on 128 tasks

---

## Quick Reference

| Stage | Tasks | Envs | Time | Purpose |
|-------|-------|------|------|---------|
| **1. Smoke** | 4 | 2 | ~1 hr | Verify pipeline |
| **2. Small** | 32 | 8 | ~3 hrs | Test learning |
| **3. Full** | 128 | 8 | ~80 hrs | Full replication |

---

## Notebook Files

1. **`ARPO_Smoke_Test.ipynb`** ← Start here!
2. **`ARPO_Training_VERL_Colab.ipynb`** (32 or 128 tasks)
3. **`ARPO_OSWorld_Evaluation.ipynb`** (evaluation only)

---

## After Each Stage

### Smoke Test (Stage 1):
```python
# Verify checkpoint
!ls checkpoints/
# Should see: model.safetensors

# Check loss
# wandb should show: loss decreasing

# Test resume (add to smoke test notebook)
!python -m verl.trainer.main config=smoke_test.yaml trainer.load_checkpoint_path=checkpoints/latest
```

### Small Scale (Stage 2):
```python
# Evaluate trained model
!python OSWorld/run_uitars.py \
    --model trained-2b \
    --checkpoint checkpoints/epoch_1/ \
    --test_all_meta_path test_data/osworld_examples/test_chrome_10.json
```

### Full Training (Stage 3):
- Compare with paper benchmarks
- Test on held-out tasks
- Publish results

---

## Recommended Approach

**Week 1**: Run Stage 1 smoke test (1 hour)  
**Week 1**: If pass, run Stage 2 small scale (3 hours)  
**Week 2**: If pass, run Stage 3 full training (80 hours over multiple sessions)

**Total cost**: ~$30-50 in Colab Pro credits

---

**Start with Stage 1 now!** Upload `ARPO_Smoke_Test.ipynb` to Colab A100.

If it passes all checks, you know the full pipeline works! 🎯
