# ARPO Replication - UI-TARS-2B on Mac CPU

**Status**: ✅ Setup Complete | ⚠️ CPU Training Not Practical

This repository contains a complete replication setup for the ARPO paper, adapted for Mac with UI-TARS-2B. All components work correctly, but CPU performance makes training impractical.

---

## 🎯 Key Findings

### ✅ What Works:
- Complete ARPO environment setup (Python 3.10, all dependencies)
- OSWorld with VMware Fusion on macOS
- UI-TARS-2B model inference server
- End-to-end pipeline verified (VM → Screenshot → Model → Action)
- All optimizations applied (image resize, token limits, etc.)

### ⚠️ CPU Performance Results:

**Measured on Apple Silicon Mac with UI-TARS-2B**:

| Metric | Value |
|--------|-------|
| **Input size** | 786 tokens (1 image resized to 800×450) |
| **Tokenization** | ~0.1s ✅ |
| **Generation** | **47-88 min** (avg: ~60 min) ❌ |
| **Output tokens** | ~95 tokens |
| **Decoding** | ~0.03s ✅ |
| **Total per step** | **~60 minutes** |

**Bottleneck**: Model generation (PyTorch forward pass on CPU)

### 📊 Training Time Estimates:

**For ARPO Training**:
- **Steps per task**: 10 (max_steps=10)
- **Tasks**: 8 (training subset)
- **Epochs**: 5
- **Total inferences**: 8 × 10 × 5 = **400 steps**

**Time calculation**:
- 400 steps × 60 min/step = **24,000 minutes**
- **= 400 hours = 16.7 days of continuous running**

**Conclusion**: Not practical for CPU training. GPU required.

---

## 🚀 GPU Performance (Expected):

With NVIDIA GPU (A40/A6000/A100):
- **Generation**: ~2-5 seconds per step (100-200x faster)
- **Total training**: ~5-10 hours (not 400 hours!)

---

# ARPO: End-to-End Policy Optimization for GUI Agents with Experience Replay

This repository contains the replication setup for the paper:

> **ARPO: End-to-End Policy Optimization for GUI Agents with Experience Replay**  
> *Fanbin Lu, Zhisheng Zhong, Shu Liu, Chi-Wing Fu, Jiaya Jia*  
> CUHK, SmartMore, HKUST  
> [[Paper](https://arxiv.org/abs/2505.16282)] • [[Project Page](https://github.com/dvlab-research/ARPO)] • [[Model on HF](https://huggingface.co/Fanbin/ARPO_UITARS1.5_7B)]

## Overview

**ARPO (Agentic Replay Policy Optimization)** is a novel reinforcement learning framework designed to train **vision-language GUI agents** to complete **long-horizon desktop tasks**. It builds upon **Group Relative Policy Optimization (GRPO)** and introduces:

- **Distributed Rollouts**: Scalable task execution across parallel OSWorld environments with docker.  
- **Multi-modal Input Support**: Processes long histories (15 steps) of screenshots + actions in an end-to-end way.

Access our [model](https://huggingface.co/Fanbin/ARPO_UITARS1.5_7B) on huggingface and view [training logs](https://wandb.ai/fanbinlu/arpo) on the Weights & Biases.

<p align="center">
<img src="assets/traj_reward.png" alt="Trajectory reward during trainig" width="500">
</p>

## 📊 Results on OSWorld

| Model                        |  128 training tasks | OSWorld overall|
|-----------------------------|---------|-------|
| UI-Tars-1.5                |68.7% | 23.5%   | 
| UI-Tars-1.5 + GRPO         |72.9% | 26.0%   | 
| **UI-Tars-1.5 + ARPO (Ours)** |83.9% | **29.9%** |

> Evaluated with a max of **15 steps per trajectory**.

---

## 🛠 Installation

### ⚠️ macOS Users: See [SETUP_MACOS.md](SETUP_MACOS.md)

**Important for Mac**: OSWorld requires VMware Fusion (not Docker) on macOS. Docker doesn't support KVM on Mac, and VirtualBox doesn't support Apple Silicon. See the Mac-specific setup guide above.

### 1. Clone the repository and create environment

```bash
git clone --recurse-submodules https://github.com/dvlab-research/ARPO.git
cd ARPO

# Create and activate Conda environment
conda create -n arpo python=3.10
conda activate arpo

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Install OSWorld
Follow the origin installation guide of [OSWorld](https://github.com/xlang-ai/OSWorld) if you only want to evaluate the model. If you want to train with GRPO, you are required to pip install it.
```bash
cd OSWorld
pip install -e .
cd ..
```

> 💡 We strongly recommend running a full evaluation **with Docker** before training to prepare the docker image, Ubuntu VM data, and cache_dir required.


## ⚙️ Setup for Evaluation with OSWorld

To evaluate ARPO on the OSWorld benchmark with the [released model](https://huggingface.co/Fanbin/ARPO_UITARS1.5_7B) using Docker-based virtual environments, follow these steps:

### 1. **Prepare the Environment**

Ensure you have correctly installed [OSWorld](https://github.com/xlang-ai/OSWorld) by following its Docker setup instructions. Once OSWorld is set up:

```bash
nohup bash start_server.sh &
```

### 2. **Run Evaluation Script**

Navigate into the OSWorld directory and execute the evaluation script:

```bash
cd OSWorld

python run_multienv_uitars.py \
    --headless \
    --observation_type screenshot \
    --max_steps 15 \
    --max_trajectory_length 15 \
    --temperature 0.6 \
    --model ui-tars \
    --action_space pyautogui \
    --num_envs 8 \
    --result_dir ./results/ \
    --test_all_meta_path ./evaluation_examples/test_all.json \
    --trial-id 0 \
    --server_ip http://127.0.0.1
```

### ✅ Parameters Explained

- `--headless`: Enables headless mode (no GUI rendering).
- `--observation_type screenshot`: Use visual observations for the agent.
- `--max_steps` / `--max_trajectory_length`: Limit per-task interaction steps.
- `--temperature`: Sampling temperature for model output.
- `--model`: Name of the model.
- `--num_envs`: Number of parallel environments (VMs).
- `--result_dir`: Directory to store evaluation results.
- `--test_all_meta_path`: JSON file with evaluation task metadata.
- `--trial-id`: ID for the evaluation trial.
- `--server_ip`: IP of the evaluation server (usually localhost).

> You will find vmware_vm_data/, docker_vm_data/, and cache/ folders under the OSWorld after evaluation.
---

## ⚙️ Setup for GRPO Training

```bash
# Link evaluation examples and cache
ln -s $(pwd)/OSWorld/evaluation_examples ./
mkdir cache_dirs/
ln -s $(pwd)/OSWorld/cache ./cache_dirs/cache_0
ln -s $(pwd)/OSWorld/vmware_vm_data ./
ln -s $(pwd)/OSWorld/docker_vm_data ./
```

To run Docker without `sudo`:

```bash
sudo usermod -aG docker $USER
newgrp docker
```

---

## Training ARPO/GRPO with OSWorld

### Single Node (subset training: 32 tasks)
If you only have one node, we suggest training on a subset of OSWorld tasks with at most 16 Docker environments.
```bash
RAY_PORT=2468
RAY_HEAD_IP=<Your IP>
ray start --head --port=$RAY_PORT --resources='{"docker:'$RAY_HEAD_IP'": 128}'
bash ./examples/osworld_subset32.sh
```

### Multi-Node Setup with Ray (e.g. 8 nodes, 128 envs)

On **Ray master node**:

```bash
RAY_PORT=2468
RAY_HEAD_IP=<Your IP>
ray start --head --port=$RAY_PORT --resources='{"docker:'$RAY_HEAD_IP'": 128}'
```

On **Ray slave nodes** (with GPU):

```bash
ray start --address=$RAY_HEAD_IP:$RAY_PORT --num-gpus=8 --resources='{"docker:'$CURRENT_IP'": 128}'
```

Or (CPU only):

```bash
ray start --address=$RAY_HEAD_IP:$RAY_PORT --resources='{"docker:'$CURRENT_IP'": 128}'
```

Then run:

```bash
bash ./examples/osworld_full_arpo.sh
```

---

## 🔗 Related Projects

- [OSWorld](https://github.com/FanbinLu/OSWorld) — Realistic GUI environments for multimodal agents modified for GRPO training.
- [EasyR1](https://github.com/hiyouga/EasyR1) An efficient, scalable, multi-modality RL training framework based on veRL, supporting advanced VLMs and algorithms like GRPO.
---

## 📄 Citation

If you find ARPO useful, please consider citing our work.
