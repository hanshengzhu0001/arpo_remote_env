# Cloning Private Repository on Colab

Since `arpo_replica` is private, you need authentication to clone it on Colab.

## Method 1: Personal Access Token (Recommended)

### Step 1: Generate GitHub Token

1. Go to: https://github.com/settings/tokens/new
2. Note: "Colab ARPO access"
3. Expiration: 90 days (or custom)
4. Scopes: Select **repo** (full control of private repositories)
5. Click "Generate token"
6. **Copy the token** (you won't see it again!)

### Step 2: Use in Colab

```python
from getpass import getpass

# Enter token when prompted
github_token = getpass('Enter GitHub token: ')

# Clone with token
!git clone https://{github_token}@github.com/gowathena/arpo_replica.git
%cd arpo_replica
!git checkout arpo-cpu-replicate
!git submodule update --init --recursive
```

---

## Method 2: Upload Manually

If you don't want to use tokens:

### Step 1: Download Locally

```bash
# On your Mac
cd ~/Desktop
git clone https://github.com/gowathena/arpo_replica.git arpo_for_colab
cd arpo_for_colab
git checkout arpo-cpu-replicate
git submodule update --init --recursive

# Create archive (without large files)
tar -czf arpo_replica.tar.gz \
    --exclude='OSWorld/vmware_vm_data' \
    --exclude='results*' \
    --exclude='checkpoints*' \
    --exclude='.git' \
    .
```

### Step 2: Upload to Google Drive

1. Upload `arpo_replica.tar.gz` to your Google Drive
2. In Colab:

```python
from google.colab import drive
drive.mount('/content/drive')

!tar -xzf /content/drive/MyDrive/arpo_replica.tar.gz
%cd arpo_replica
```

---

## Method 3: Make Repo Public (Temporary)

If you're okay temporarily:

1. Go to: https://github.com/gowathena/arpo_replica/settings
2. Scroll to "Danger Zone"
3. Click "Change visibility" → Public
4. Run training on Colab (regular `git clone` works)
5. Change back to Private when done

---

## Recommended: Method 1 (Token)

- ✅ Fast (no upload time)
- ✅ Always up-to-date
- ✅ Secure (token expires)
- ✅ Works in notebook (getpass hides token)

---

**The Colab notebook is updated to use Method 1 (token-based clone).**
