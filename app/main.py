import os
import uuid
from collections import OrderedDict

import gradio as gr
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .inference import bicubic_upscale, image_to_bytes, prepare_input, run_model
from .model import MODEL_SPECS, load_available_models

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
STATIC_DIR = os.path.join(ROOT_DIR, "static")
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(ROOT_DIR, "models"))

# The upload is treated as the LR image and enlarged 4x, so the output is
# 4x this size. 320 px in -> 1280 px out keeps CPU inference to a few seconds.
MAX_INPUT_DIM = int(os.environ.get("MAX_INPUT_DIM", 320))
ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
MAX_UPLOAD_BYTES = 8 * 1024 * 1024

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = load_available_models(MODELS_DIR, device)

# Results are kept in memory (newest 6 jobs) and served as plain PNG files,
# so the browser can cache/decode them and the JSON stays tiny.
RESULTS: "OrderedDict[str, dict]" = OrderedDict()
MAX_JOBS = 6

app = FastAPI(title="SISR Super-Resolution Tool")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/health")
def health():
    return {"status": "ok", "device": str(device), "loaded": list(models)}


@app.get("/api/models")
def list_models():
    out = [{"key": "bicubic", "label": "Bicubic", "available": True}]
    for key, (label, *_rest) in MODEL_SPECS.items():
        out.append({"key": key, "label": label, "available": key in models})
    return {"models": out, "max_input_dim": MAX_INPUT_DIM}


@app.post("/api/enhance")
async def enhance(file: UploadFile = File(...), models_to_run: str = Form("all")):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Please upload a PNG, JPEG, or WEBP image.")
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="Image too large (max 8 MB).")
    if not models:
        raise HTTPException(status_code=503, detail="No model checkpoints found in models/.")

    wanted = list(models) if models_to_run == "all" else [k for k in models_to_run.split(",") if k in models]
    if not wanted:
        raise HTTPException(status_code=400, detail="None of the requested models are available.")

    try:
        lr = prepare_input(data, MAX_INPUT_DIM)
        results = {"bicubic": bicubic_upscale(lr)}
        for key in wanted:
            results[key] = run_model(models[key], lr, device)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    job = uuid.uuid4().hex[:12]
    RESULTS[job] = {k: image_to_bytes(v) for k, v in results.items()}
    while len(RESULTS) > MAX_JOBS:
        RESULTS.popitem(last=False)

    labels = {"bicubic": "Bicubic", **{k: v["label"] for k, v in models.items()}}
    return {
        "job": job,
        "input_size": [lr.width, lr.height],
        "output_size": [lr.width * 4, lr.height * 4],
        "results": [{"key": k, "label": labels[k], "url": f"/api/result/{job}/{k}.png"} for k in results],
    }


@app.get("/api/result/{job}/{name}.png")
def get_result(job: str, name: str):
    png = RESULTS.get(job, {}).get(name)
    if png is None:
        raise HTTPException(status_code=404, detail="Result expired; run the image again.")
    return Response(png, media_type="image/png", headers={"Cache-Control": "private, max-age=600"})


# ---- Secondary interface: native Gradio app (also what makes the Space deployable
# under the "Gradio" SDK). Same inference code as the custom UI.
def gradio_infer(pil_image, model_key):
    if pil_image is None or not models:
        return None
    lr = prepare_input(image_to_bytes(pil_image), MAX_INPUT_DIM)
    return bicubic_upscale(lr), run_model(models[model_key], lr, device)


with gr.Blocks(title="SISR Super-Resolution") as gradio_demo:
    gr.Markdown("## Super-Resolution (Bicubic vs. trained model)\nFor the full draggable-zoom tool, open the root URL.")
    inp = gr.Image(type="pil", label="Upload image")
    choice = gr.Dropdown(choices=[(v["label"], k) for k, v in models.items()],
                         value=next(iter(models), None), label="Model")
    btn = gr.Button("Run", variant="primary")
    slider = gr.ImageSlider(label="Bicubic (left) vs model (right)")
    btn.click(gradio_infer, [inp, choice], slider)

app = gr.mount_gradio_app(app, gradio_demo, path="/gradio")


@app.get("/")
def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
