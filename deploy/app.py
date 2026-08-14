import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from functools import lru_cache

import gradio as gr
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

from deploy.service import ServiceBusyError, TableInferenceService


logger = logging.getLogger("ocr_table.deploy")
ALLOWED_IMAGE_TYPES = {
    "application/octet-stream",
    "image/bmp",
    "image/jpeg",
    "image/png",
    "image/tiff",
    "image/webp",
}


def _env_bool(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def get_service():
    return TableInferenceService.from_env()


@asynccontextmanager
async def lifespan(_app):
    # Readiness is enabled only after model validation and warm-up.
    await run_in_threadpool(get_service)
    yield
    get_service.cache_clear()


app = FastAPI(
    title="OCR Table to Excel",
    version="1.1.0",
    description="English table recognition using ONNX Runtime.",
    lifespan=lifespan,
    docs_url="/docs" if _env_bool("OCR_ENABLE_DOCS", True) else None,
    redoc_url=None,
)


cors_origins = [
    origin.strip()
    for origin in os.getenv("OCR_CORS_ORIGINS", "").split(",")
    if origin.strip()
]
if cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Request-ID"],
    )


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    started = time.perf_counter()
    try:
        response = await call_next(request)
    finally:
        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "request_id=%s method=%s path=%s elapsed_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            elapsed_ms,
        )
    response.headers["X-Request-ID"] = request_id
    return response


def _service_or_503():
    try:
        service = get_service()
    except Exception as exc:
        logger.exception("Inference service initialization failed")
        raise HTTPException(
            status_code=503,
            detail="Inference service is not ready.",
        ) from exc
    if not service.is_ready:
        raise HTTPException(status_code=503, detail="Inference service is starting.")
    return service


async def _read_upload(file, service):
    content_type = (file.content_type or "application/octet-stream").lower()
    if content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Unsupported image content type.",
        )
    max_bytes = service.settings.max_upload_mb * 1024 * 1024
    content = await file.read(max_bytes + 1)
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail="Upload exceeds {} MB.".format(
                service.settings.max_upload_mb
            ),
        )
    return content


async def _predict(file):
    service = _service_or_503()
    content = await _read_upload(file, service)
    try:
        return service, await run_in_threadpool(service.predict_bytes, content)
    except ServiceBusyError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/live")
async def live():
    return {"status": "alive"}


@app.get("/ready")
async def ready():
    return _service_or_503().health_info()


@app.get("/health")
async def health():
    return _service_or_503().health_info()


@app.post("/v1/table/recognize")
async def recognize(file: UploadFile = File(...)):
    _, payload = await _predict(file)
    return payload


@app.post("/v1/table/excel", response_class=FileResponse)
async def recognize_excel(file: UploadFile = File(...)):
    service, payload = await _predict(file)
    try:
        path = await run_in_threadpool(service.save_excel, payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return FileResponse(
        path=str(path),
        filename="{}.xlsx".format(PathName.safe_stem(file.filename)),
        media_type=(
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ),
        background=BackgroundTask(path.unlink, missing_ok=True),
    )


class PathName:
    @staticmethod
    def safe_stem(filename):
        base = os.path.basename(filename or "table")
        stem = os.path.splitext(base)[0]
        safe = "".join(
            character
            for character in stem
            if character.isalnum() or character in "-_ "
        ).strip()
        return safe or "table"


def build_gradio():
    with gr.Blocks(title="OCR Table to Excel") as demo:
        gr.Markdown(
            "# OCR Table to Excel\nEnglish table recognition with ONNX Runtime."
        )
        with gr.Row():
            input_image = gr.Image(type="numpy", label="Table image")
            output_image = gr.Image(type="numpy", label="Detected cells")
        run_button = gr.Button("Recognize", variant="primary")
        html_source = gr.Code(language="html", label="Generated table HTML")
        json_result = gr.JSON(label="Recognition result and timing")
        excel_file = gr.File(label="Excel output")
        run_button.click(
            fn=lambda image: get_service().gradio_predict(image),
            inputs=input_image,
            outputs=[output_image, html_source, json_result, excel_file],
        )
    concurrency = max(1, int(os.getenv("OCR_MAX_CONCURRENCY", "1")))
    return demo.queue(default_concurrency_limit=concurrency)


if _env_bool("OCR_ENABLE_GRADIO", True):
    app = gr.mount_gradio_app(
        app,
        build_gradio(),
        path="/ui",
        max_file_size="{}mb".format(os.getenv("OCR_MAX_UPLOAD_MB", "20")),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "deploy.app:app",
        host=os.getenv("OCR_HOST", "0.0.0.0"),
        port=int(os.getenv("OCR_PORT", "8000")),
        reload=False,
        workers=1,
    )
