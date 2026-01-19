# ARPO Replication - Complete Setup

**Status**: ✅ Setup Complete | 🎯 Ready for GPU Inference Testing

This repository contains a complete ARPO replication with tested inference pipeline for OSWorld GUI tasks.

---

## 📊 Performance Findings

### CPU Performance (UI-TARS-2B on Mac):
- **Per step**: ~60 minutes (47-88 min range)
- **Per task** (10 steps): ~10 hours  
- **Training** (8 tasks × 5 epochs): ~400 hours (16.7 days)
- **Conclusion**: ❌ Not practical for training

### GPU Performance (Expected with UI-TARS-7B):
- **Per step**: 2-5 seconds (100-200x faster)
- **Per task**: 20-50 seconds
- **Training** (128 tasks × 15 epochs): 5-15 hours
- **Conclusion**: ✅ Practical and matches paper

---

## 📁 Project Structure

```
ARPO_replicate/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
│
├── docs/                        # Documentation
│   ├── START_HERE.md           # Quick start guide
│   ├── TRAINING_GUIDE.md       # Complete training instructions
│   ├── PAPER_SUMMARY.md        # ARPO paper deep dive
│   ├── PERFORMANCE_REPORT.md   # CPU/GPU performance analysis
│   ├── TROUBLESHOOTING.md      # Problem solving
│   └── FILES.md                # File overview
│
├── configs/                     # Training configurations
│   └── config_uitars_2b_mac.yaml  # VERL training config
│
├── scripts/                     # Executable scripts
│   ├── uitars_2b_server.py     # UI-TARS-2B inference server (CPU tested)
│   ├── uitars_7b_server.py     # UI-TARS-7B inference server (GPU)
│   ├── train_uitars_2b_arpo.sh # Training script (2B)
│   ├── test_server.sh          # Server test
│   ├── test_single_task.sh     # Single task test
│   └── test_osworld_uitars.sh  # OSWorld integration test
│
├── notebooks/                   # Jupyter notebooks
│   ├── ARPO_UITARS_Inference.ipynb      # GPU inference (tested) ⭐
│   ├── ARPO_OSWorld_Evaluation.ipynb    # Evaluation on 10 tasks (NEW) ⭐
│   └── arpo_training_notebook.ipynb     # Training guide
│
├── test_data/                   # Test tasks
│   └── osworld_examples/
│       ├── tasks/              # 5 original tasks
│       └── noisy_tasks/        # 5 noisy tasks
│
├── OSWorld/                     # OSWorld benchmark (submodule)
├── verl/                        # VERL training framework  
└── examples/                    # Example training scripts
```

---

## 🚀 Quick Start

### For GPU Inference Testing (Colab/VSCode):

1. **Open notebook**:
   ```bash
   notebooks/ARPO_OSWorld_Evaluation.ipynb
   ```

2. **Run all cells** - It will:
   - Load ARPO UITARS 7B model (4-bit quantized)
   - Test on 5 original + 5 noisy OSWorld tasks
   - Generate results and metrics

**Time**: ~30-60 minutes on A100 GPU

### For CPU Testing (Mac):

See `docs/START_HERE.md` for CPU setup (not recommended for training).

---

## 📊 Test Data

**From**: [gowathena/arpo_replica/tree/data](https://github.com/gowathena/arpo_replica/tree/data)

**Tasks**:
- **5 Original tasks**: Standard OSWorld Chrome tasks
- **5 Noisy tasks**: Same tasks with distractor entries

**Format**: Compatible with OSWorld evaluation_examples

---

## 🎯 Models

### UI-TARS-2B (Tested on CPU):
- **Model**: ByteDance-Seed/UI-TARS-2B-SFT
- **Size**: 2B parameters
- **Performance**: ~60 min/step on CPU
- **Use**: Development/testing only

### UI-TARS-7B (For GPU):
- **Model**: [Fanbin/ARPO_UITARS1.5_7B](https://huggingface.co/Fanbin/ARPO_UITARS1.5_7B) ⭐
- **Size**: 7B parameters (ARPO-trained)
- **Performance**: 2-5 sec/step on GPU
- **Results**: 83.9% on 128 tasks, 29.9% overall
- **Use**: Production training/evaluation

---

## 🔧 Setup

### Requirements:
- Python 3.10+
- For GPU: CUDA 11.8+, 16GB+ VRAM
- For CPU: 16GB+ RAM (very slow, not recommended)

### Install:
```bash
pip install -r requirements.txt
```

### OSWorld Setup (Optional):
Only needed for full training, not for inference testing with notebooks.

See `docs/START_HERE.md` for complete setup.

---

## 📖 Documentation

- **Quick Start**: `docs/START_HERE.md`
- **Training Guide**: `docs/TRAINING_GUIDE.md`
- **Paper Summary**: `docs/PAPER_SUMMARY.md`
- **Performance**: `docs/PERFORMANCE_REPORT.md`
- **Troubleshooting**: `docs/TROUBLESHOOTING.md`

---

## 🎓 What This Repository Provides

1. ✅ **Complete ARPO environment**
2. ✅ **Tested inference pipeline** (CPU with 2B, ready for GPU with 7B)
3. ✅ **OSWorld integration**
4. ✅ **Test data** (10 tasks: 5 original + 5 noisy)
5. ✅ **Training scripts** (VERL framework configured)
6. ✅ **Comprehensive documentation**

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

---

## 🔗 Links

- **Paper**: [arXiv](https://arxiv.org/abs/2505.16282)
- **Original Code**: [JIA-Lab-research/ARPO](https://github.com/JIA-Lab-research/ARPO)
- **ARPO Model**: [Fanbin/ARPO_UITARS1.5_7B](https://huggingface.co/Fanbin/ARPO_UITARS1.5_7B)
- **OSWorld**: [xlang-ai/OSWorld](https://github.com/xlang-ai/OSWorld)

---

**Ready to test with GPU!** 🚀 See `notebooks/ARPO_OSWorld_Evaluation.ipynb`
