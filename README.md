# ARPO Replication

**Status**: ✅ Ready for GPU Evaluation

Complete ARPO replication with OSWorld integration for GUI agent training and evaluation.

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/gowathena/arpo_replica.git
cd arpo_replica
git checkout arpo-cpu-replicate

# Initialize OSWorld submodule
git submodule update --init --recursive

# Create conda environment
conda create -n arpo python=3.10 -y
conda activate arpo

# Install dependencies
pip install -r requirements.txt
cd OSWorld && pip install -r requirements.txt && pip install -e . && cd ..
```

### 2. OSWorld Setup (Mac with VMware Fusion)

**VMware runs the execution environment** - a Ubuntu VM where GUI tasks are executed.

```bash
# A. Install VMware Fusion (if not installed)
# Download from: https://www.vmware.com/products/fusion.html

# B. Add vmrun to PATH
echo 'export PATH="/Applications/VMware Fusion.app/Contents/Library:$PATH"' >> ~/.zshrc
source ~/.zshrc

# C. Verify vmrun works
vmrun -T fusion list  # Should show: Total running VMs: 0

# D. OSWorld VM will auto-download on first run (~38GB Ubuntu VM)
# No manual setup needed - happens automatically when you run evaluation!
```

**Troubleshooting**: If you get "vmrun: command not found", VMware Fusion is not installed or not in PATH. Follow steps A-C above.

**What VMware does**:
- Runs Ubuntu desktop environment
- Executes GUI actions (clicks, typing, etc.)
- Captures screenshots for the model
- Runs applications (Chrome, Firefox, etc.)

**VM Configuration Options**:
- **OS**: Ubuntu (default) or Windows
- **Screen**: 1920x1080 (default), configurable
- **Mode**: Headless (background) or visible window
- **Action space**: `pyautogui` or `computer_13`
- **Observation**: Screenshot, accessibility tree, or both

See `docs/VM_CONFIGURATION.md` for full configuration details.

**Note**: For Linux/Docker setup, see [OSWorld documentation](https://github.com/xlang-ai/OSWorld).

### 3. GPU Server (Colab)

1. **Open**: `notebooks/GPU_Server_for_OSWorld.ipynb` in [Google Colab](https://colab.research.google.com)
2. **Runtime** → **Change runtime type** → **A100 GPU**
3. **Run cells 1-5**:
   - Cell 1: Install dependencies
   - Cell 2: Configure ngrok (enter authtoken when prompted)
   - Cell 3: Load model (~2 min)
   - Cell 4: Create Flask server
   - Cell 5: Start server + get public URL
4. **Copy ngrok URL** from Cell 5 output

### 4. Evaluation (Mac)

1. **Open**: `notebooks/ARPO_OSWorld_Evaluation.ipynb` in VSCode
2. **Select kernel**: `arpo` (Python 3.10)
3. **Update Cell 4**: Paste ngrok URL from Colab
4. **Run all cells**: Evaluates 10 original + 10 noisy Chrome tasks

**Expected time**: ~1.5 hours for 20 tasks (based on actual test: 22m 40s for 5 tasks)

---

## 📊 Results

**Current Performance** (ARPO UITARS 7B on A100 GPU):
- **Tested**: 5 Chrome tasks (initial test)
- **Success rate**: 20% (1/5 tasks passed)
- **Time per task**: ~4.5 minutes (includes VM overhead, network, execution)
- **Model inference**: 2-5 seconds per step
- **Estimated for 10 Chrome tasks**: ~45 minutes
- **Estimated for 20 tasks (10 original + 10 noisy)**: ~1.5 hours

---

## 📁 Project Structure

```
ARPO_replicate/
├── notebooks/
│   ├── GPU_Server_for_OSWorld.ipynb      # Colab GPU server
│   └── ARPO_OSWorld_Evaluation.ipynb     # Mac evaluation
├── scripts/
│   ├── uitars_7b_server.py                # Standalone GPU server
│   └── test_osworld_uitars.sh            # OSWorld test script
├── test_data/osworld_examples/           # 10 test tasks
├── OSWorld/                               # OSWorld benchmark (submodule)
└── verl/                                  # VERL training framework
```

---

## 🎯 Models

- **ARPO UITARS 7B**: [Fanbin/ARPO_UITARS1.5_7B](https://huggingface.co/Fanbin/ARPO_UITARS1.5_7B)
  - 7B parameters, ARPO-trained
  - Performance: 83.9% on 128 tasks, 29.9% overall

---

## 📖 Documentation

- **GPU Setup**: `GPU_SETUP_GUIDE.md`
- **Colab Instructions**: `COLAB_INSTRUCTIONS.md`
- **Training Guide**: `docs/TRAINING_GUIDE.md`
- **Paper Summary**: `docs/PAPER_SUMMARY.md`

---

## 🔗 Links

- **Paper**: [arXiv:2505.16282](https://arxiv.org/abs/2505.16282)
- **Original Code**: [JIA-Lab-research/ARPO](https://github.com/JIA-Lab-research/ARPO)
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
