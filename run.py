"""
Entry point for Hugging Face Spaces under the "Gradio" SDK.

Kept as a separate top-level file (rather than naming it app.py) to avoid
any ambiguity with the `app/` package next to it — this file just imports
the already-built FastAPI application (which has the custom draggable-zoom
UI mounted at "/" and a native Gradio interface mounted at "/gradio") and
runs it.

If deploying on Hugging Face Spaces with the Gradio SDK, set in your
Space's README.md YAML front matter:

    sdk: gradio
    app_file: run.py

Running locally:

    python run.py
"""

import os

from app.main import app  # noqa: F401  (Spaces just needs `app` importable; uvicorn.run below covers `python run.py`)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
