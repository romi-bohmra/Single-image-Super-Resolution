# Single-Image Super-Resolution ×4 — online tool

Upload an image and compare **Bicubic, SRCNN-Y, VDSR-Y and SRGAN-Y**. Every network
works on the luminance (Y) channel; Cb/Cr are upscaled with bicubic interpolation and
merged back. The page shows a large result image plus two draggable, synchronised zoom
views. A native Gradio interface is also available at `/gradio/`.

## Run locally

```bash
pip install -r requirements.txt
python run.py            # then open http://localhost:7860
```

## Model files (put them in `models/`)

| File | Model |
|---|---|
| `srcnn_y_model.pth` | SRCNN-Y |
| `vdsr_y_model_300_fixed.pth` | VDSR-Y |
| `srgan_y_G.pth` | SRGAN-Y generator (8 residual blocks) |

A missing file is skipped (its button is greyed out). Set `MAX_INPUT_DIM` (default 320)
to change the largest accepted input size; the output is 4× larger.
