# OSWorld Local Modifications

OSWorld is a git submodule (from [xlang-ai/OSWorld](https://github.com/xlang-ai/OSWorld)). We made local modifications for our setup.

## Changes Made

### 1. `mm_agents/uitars_agent.py`

**Line 562**: Changed base_url
```python
# Original:
base_url="http://10.1.1.3:9000/v1"

# Modified to:
base_url="https://YOUR-NGROK-URL.ngrok-free.dev/v1"  # Update with your Colab URL
```

**Line 596**: Added timeout
```python
self.vlm = OpenAI(
    base_url=base_url,
    api_key="empty",
    timeout=1800.0,  # Added: 30 min timeout for slow inference
)
```

### 2. `run_multienv_uitars.py`

**Line 252**: Changed provider from docker to vmware
```python
# Original:
provider_name="docker"

# Modified to:
provider_name="vmware"  # For macOS compatibility
```

### 3. `run_uitars.py`

**Line 165**: Changed provider from docker to vmware
```python
# Original:
provider_name="docker"

# Modified to:
provider_name="vmware"  # For macOS compatibility
```

**Lines 148-155**: Optimized runtime_conf for CPU
```python
"history_n": 5,  # Reduced from 15 for faster CPU
"max_pixels": 800*600*28*28,  # Reduced for CPU
"max_tokens": min(args.max_tokens, 128)  # Cap at 128 for speed
```

---

## Why These Aren't Pushed

OSWorld is a **submodule** pointing to the upstream repository. We don't own that repo, so we keep changes local.

## How to Apply These Changes

After cloning the repository:

```bash
# 1. Update uitars_agent.py with your Colab URL
cd OSWorld
sed -i '' 's|http://10.1.1.3:9000/v1|https://YOUR-NGROK-URL/v1|g' mm_agents/uitars_agent.py

# 2. Add timeout (already done in our copy)
# Already included in modified files

# 3. Change to vmware provider (already done)
# Already included in modified files
```

Or simply use our modified version (already set up in your local copy).

---

## Alternative: Keep Changes Local

The current setup works because:
- Your local OSWorld copy has the changes
- They're tracked by git (but not pushed to upstream)
- New clones would need to reapply these changes

This is fine for personal use!
