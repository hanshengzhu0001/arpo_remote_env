# ARPO Replication

Complete ARPO (Agentic Replay Policy Optimization) replication with OSWorld integration.

**Status**: ✅ Ready for Evaluation & Training

---

## 🚀 Quick Start

### Evaluation (GPU Inference on 10 Chrome Tasks)

1. **Colab**: Run `notebooks/GPU_Server_for_OSWorld.ipynb` (A100 GPU)
2. **Mac**: Run `notebooks/ARPO_OSWorld_Evaluation.ipynb`
3. **Expected**: ~45 minutes, 10 tasks evaluated

### Training (VERL on GPU Cluster)

1. **Setup SSH**: Follow `REMOTE_GPU_SETUP.md`
2. **Run**: `notebooks/ARPO_Smoke_Test.ipynb` (4 tasks, ~1 hour)
3. **Scale up**: 32 or 128 tasks after verification

---

## 📁 Repository Structure

```
arpo_replica/
├── notebooks/
│   ├── ARPO_Smoke_Test.ipynb           # 4-task VERL training test
│   ├── ARPO_OSWorld_Evaluation.ipynb   # 10-task evaluation
│   ├── GPU_Server_for_OSWorld.ipynb    # GPU inference server
│   └── arpo_training_notebook.ipynb    # Training guide
├── test_data/osworld_examples/
│   ├── train_smoke_4.json              # 4 tasks for testing
│   ├── train_subset_32.json            # 32 tasks
│   ├── train_all_128.json              # Full dataset
│   └── test_chrome_10.json             # 10 Chrome evaluation tasks
├── configs/
│   ├── config_uitars_2b_mac.yaml       # Training configuration
│   └── wandb_config.yaml               # W&B logging
├── osworld_patches/                     # Modified OSWorld files
├── scripts/                             # Training scripts
└── docs/                                # Additional documentation
```

---

## 📊 Performance Results

**Evaluation** (ARPO UITARS 7B on A100):
- 10 Chrome tasks: ~45 minutes
- Per task: ~4.5 minutes
- Success rate: 20% (1/5 initial test)

**Training** (UI-TARS-2B):
- CPU: ~60 min/step (not practical)
- GPU: ~10-30 sec/step (practical!)

---

## 📖 Documentation

- **`README.md`** - This file (start here)
- **`REMOTE_GPU_SETUP.md`** - SSH setup for GPU cluster
- **`TRAINING_PROGRESSION.md`** - Stage 1-3 training plan
- **`SETUP_FOR_NEW_USERS.md`** - Complete setup from scratch
- **`PIPELINE_VERIFICATION.md`** - Confirms same pipeline as paper
- **`docs/`** - Additional guides

---

## 🎯 Dataset

- **128 total tasks** across 10 domains
- **18 Chrome tasks** (+ 18 noisy versions)
- Format: OSWorld evaluation_examples compatible

From: [arpo_replica/data branch](https://github.com/gowathena/arpo_replica/tree/data)

---

## 🔧 Models

- **UI-TARS-2B**: ByteDance-Seed/UI-TARS-2B-SFT (evaluation/training)
- **ARPO UITARS 7B**: Fanbin/ARPO_UITARS1.5_7B (evaluation)

---

## ⚙️ Requirements

- Python 3.10
- For Mac: VMware Fusion
- For Colab/Cluster: Docker
- For training: GPU (A100, A40, or T4)

---

## 🔗 Links

- **Paper**: [arXiv:2505.16282](https://arxiv.org/abs/2505.16282)
- **Original**: [JIA-Lab-research/ARPO](https://github.com/JIA-Lab-research/ARPO)
- **OSWorld**: [xlang-ai/OSWorld](https://github.com/xlang-ai/OSWorld)

---

## 📝 Citation

```bibtex
@article{lu2025arpo,
  title={ARPO: End-to-End Policy Optimization for GUI Agents with Experience Replay},
  author={Fanbin Lu and Zhisheng Zhong and Shu Liu and Chi-Wing Fu and Jiaya Jia},
  journal={arXiv},
  year={2025}
}
```
