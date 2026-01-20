# OSWorld Modified Files

These are the modified OSWorld files for our setup. Copy these over the original OSWorld files after cloning.

## Files Included

1. **`uitars_agent.py`** → `OSWorld/mm_agents/uitars_agent.py`
   - Changed base_url for Colab GPU server
   - Added 30-minute timeout
   
2. **`run_multienv_uitars.py`** → `OSWorld/run_multienv_uitars.py`
   - Changed provider from "docker" to "vmware"
   
3. **`run_uitars.py`** → `OSWorld/run_uitars.py`
   - Changed provider from "docker" to "vmware"
   - Optimized runtime_conf for CPU/GPU
   - Added proper parameter passing

## How to Apply

After cloning the repository:

```bash
# Navigate to repository
cd arpo_replica

# Initialize OSWorld submodule
git submodule update --init --recursive

# Copy modified files
cp osworld_patches/uitars_agent.py OSWorld/mm_agents/
cp osworld_patches/run_multienv_uitars.py OSWorld/
cp osworld_patches/run_uitars.py OSWorld/

echo "✅ Applied OSWorld modifications"
```

## Note

**Before running**: Update the ngrok URL in `uitars_agent.py` (line 562) with your actual Colab server URL.

## Changes Summary

- **VMware provider** instead of Docker (for macOS)
- **Configurable base_url** for remote GPU servers
- **Extended timeout** (30 min) for slow inference
- **CPU optimizations** (reduced history, smaller images, token caps)
