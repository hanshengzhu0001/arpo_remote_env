# Pre-Training Checklist

Before starting training, verify everything is ready:

## ✅ Environment Setup

- [ ] Conda environment `arpo` created
- [ ] All dependencies installed: `pip install -r requirements.txt`
- [ ] OSWorld installed: `cd OSWorld && pip install -e .`
- [ ] VMware Fusion installed
- [ ] `vmrun -T fusion list` works
- [ ] Ubuntu VM downloaded (~38GB in `OSWorld/vmware_vm_data/`)

## ✅ Colab GPU Server

- [ ] Open `notebooks/GPU_Server_for_OSWorld.ipynb` in Colab web
- [ ] Runtime → Change runtime type → **GPU** (T4 or A100)
- [ ] Run cells 1-5
- [ ] Copy ngrok URL from Cell 5 output
- [ ] Keep Colab tab open (server must stay running)

## ✅ Configuration

- [ ] Update `configs/wandb_config.yaml` with your entity ✅ (DONE: hanszhu05-university-of-pennsylvania-org)
- [ ] Update `configs/config_uitars_2b_mac.yaml` with ngrok URL
- [ ] Set wandb API key: `export WANDB_API_KEY="your-key"`

## ✅ OSWorld Patches Applied

- [ ] Copied files from `osworld_patches/` to `OSWorld/`:
  ```bash
  cp osworld_patches/uitars_agent.py OSWorld/mm_agents/
  cp osworld_patches/run_uitars.py OSWorld/
  ```
- [ ] Update `OSWorld/mm_agents/uitars_agent.py` line 562 with your ngrok URL

## ✅ Test Data

- [ ] `test_data/osworld_examples/train_all_128.json` exists (128 tasks)
- [ ] `test_data/osworld_examples/examples/` folder has task JSONs

---

## Quick Verification

```bash
cd ~/Desktop/ARPO_replicate

# 1. Check environment
conda activate arpo
python -c "import torch, transformers, wandb; print('✅ All packages installed')"

# 2. Check VMware
vmrun -T fusion list
# Should show: Total running VMs: 0

# 3. Check Colab server
curl https://YOUR-NGROK-URL/health
# Should return: {"status": "healthy"}

# 4. Check wandb
wandb login
# Paste API key when prompted

# 5. Check config
cat configs/config_uitars_2b_mac.yaml | grep "train_files\|max_steps\|total_episodes\|wandb_entity"
# Should show: 128 tasks, 16 steps, 1 epoch, your entity
```

---

## Ready to Train?

If all ✅ checked, you can run:

```bash
# Option 1: Run training (manual)
cd OSWorld
python run_uitars.py \
    --headless \
    --observation_type screenshot \
    --max_steps 16 \
    --model ui-tars-2b \
    --temperature 0.7 \
    --test_config_base_dir ../test_data/osworld_examples \
    --test_all_meta_path ../test_data/osworld_examples/train_all_128.json \
    --result_dir ../results_training_128/

# Option 2: Run with VERL (full ARPO)
bash scripts/train_uitars_2b_arpo.sh

# Option 3: Run with wandb wrapper
python scripts/run_training_with_wandb.py
```

---

## During Training

**Monitor**:
- Colab: Check Cell 5 output for inference logs
- Mac: `tail -f logs/training.log`
- wandb: https://wandb.ai/hanszhu05-university-of-pennsylvania-org/arpo-uitars-training

**Expected Duration**: ~34-68 hours for 1 epoch with 128 tasks

**Save checkpoints**: Every 20-30 tasks (configured in VERL)

---

**You're almost ready! Just need to start Colab server and update the ngrok URL!** 🚀
