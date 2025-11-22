# Watermark remover

Python CLI that automatically detects and removes visible watermarks from one or more images. It works best when you provide a batch of images that share the same watermark (e.g., a stock photo set), because the detector can learn the common overlay and inpaint it away without any manual masks.

## Quick start

```bash
conda activate watermark   # ensure you're in the provided env
pip install -r backend/requirements.txt

# process images; results go to backend/outputs by default
python backend/watermark_remover.py path/to/image1.jpg path/to/other/*.png -o backend/outputs
```

Outputs are saved next to the script (or wherever you point `--output-dir`) using the original filename plus `_cleaned`. Optional masks are saved with `_mask.png` when you pass `--save-masks`.

## Notes

- Multiple images with the same watermark yield the most reliable masks (the detector looks for edges that stay constant across the set while backgrounds change).
- Single-image runs still try to find and inpaint watermark-like strokes automatically, but results are less reliable without a shared pattern to learn from.
- Images are never overwritten unless you pass `--overwrite`.
