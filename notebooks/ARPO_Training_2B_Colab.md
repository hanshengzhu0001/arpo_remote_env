# UI-TARS-2B Training Server (Colab)

For ARPO training with UI-TARS-2B, use the same approach as GPU_Server_for_OSWorld.ipynb but with the 2B model.

## Quick Instructions

1. **Copy** `GPU_Server_for_OSWorld.ipynb`
2. **Change Cell 3** to load UI-TARS-2B instead:

```python
MODEL = "ByteDance-Seed/UI-TARS-2B-SFT"  # Instead of 7B

model = AutoModelForImageTextToText.from_pretrained(
    MODEL,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,
)
```

3. **Start server** - get ngrok URL
4. **On Mac**: Run ARPO training pointing to that URL
5. **Dataset**: Use `train_all_128.json` (128 tasks)

## Training Configuration

For UI-TARS-2B training with 128 tasks:

```yaml
Model: UI-TARS-2B
Tasks: 128 (all domains)
Environments: 4-8 VMs (if you have resources)
Epochs: 10-15
Batch size: 4
Expected time: ~40-80 hours on GPU (vs 400+ hours on CPU!)
```

## Note

The `arpo_training_notebook.ipynb` is primarily educational. For actual GPU-accelerated training:
- Use the Colab server approach
- Run VERL training scripts on Mac connected to Colab
- See `scripts/train_uitars_2b_arpo.sh` for training command

Full training implementation requires VERL framework integration (complex setup).
