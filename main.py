"""
main.py — FastAPI application entry point.

Route map
---------
GET  /                          → serve index.html
GET  /api/models                → list all models
GET  /api/models/{id}           → single model detail
POST /api/models/{id}/download  → start model download
GET  /api/models/{id}/download/status → SSE stream for download progress

POST /api/ocr                   → submit OCR job
GET  /api/jobs                  → list all jobs
GET  /api/jobs/{job_id}         → job detail
GET  /api/jobs/{job_id}/progress → SSE stream for job progress
GET  /api/jobs/{job_id}/result  → download result file(s)

GET  /api/history               → job history from DB (if enabled)
DELETE /api/history/{job_id}    → delete a history entry

GET  /api/formats               → list supported output formats
GET  /api/health                → health check
"""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import os
import shutil
import tempfile
import time
import uuid
import zipfile
from pathlib import Path
from typing import Optional

import aiofiles
from dotenv import load_dotenv
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

load_dotenv()

# Local imports
from database import ENABLE_DB, OCRJob, get_db, init_db
from formatters import SUPPORTED_FORMATS, export
from ingestor import load_images_from_bytes
from jobs import (
    JobStatus,
    create_job,
    get_download_state,
    get_job,
    list_download_states,
    list_jobs,
    submit_download_job,
    submit_ocr_job,
)
from model_registry import get_model, load_models
from ocr_engine import run_ocr

from ocr_diagnostics import install_diagnostics
install_diagnostics()          # before app = FastAPI(...)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("main")

UPLOADS_DIR = Path(os.getenv("UPLOADS_DIR", "./uploads"))
OUTPUTS_DIR = Path(os.getenv("OUTPUTS_DIR", "./outputs"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", "./models"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "200"))

for d in (UPLOADS_DIR, OUTPUTS_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

init_db()

app = FastAPI(
    title="OCR Converter API",
    description="Multi-model OCR converter supporting images, PDFs, and ZIP archives.",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def serve_index():
    index = Path(__file__).parent / "static" / "index.html"
    if index.exists():
        return HTMLResponse(content=index.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>OCR App — place index.html in ./static/</h1>", status_code=200)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/api/health", tags=["System"])
async def health():
    return {
        "status": "ok",
        "db_enabled": ENABLE_DB,
        "supported_formats": list(SUPPORTED_FORMATS),
        "models_dir": str(MODELS_DIR),
    }


# ---------------------------------------------------------------------------
# Formats
# ---------------------------------------------------------------------------

@app.get("/api/formats", tags=["Config"])
async def list_formats():
    return {
        "formats": [
            {"id": "md", "name": "Markdown", "extension": ".md"},
            {"id": "html", "name": "HTML", "extension": ".html"},
            {"id": "docx", "name": "Word Document", "extension": ".docx"},
        ]
    }


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@app.get("/api/models", tags=["Models"])
async def list_models_route():
    models = load_models()
    dl_states = {s["model_id"]: s for s in list_download_states()}
    result = []
    for m in models:
        d = m.dict()
        d["download_state"] = dl_states.get(m.id)
        result.append(d)
    return {"models": result}


@app.get("/api/models/{model_id}", tags=["Models"])
async def get_model_route(model_id: str):
    model = get_model(model_id)
    if not model:
        raise HTTPException(404, f"Model '{model_id}' not found")
    d = model.dict()
    d["download_state"] = get_download_state(model_id)
    return d


@app.post("/api/models/{model_id}/download", tags=["Models"])
async def start_model_download(model_id: str):
    model = get_model(model_id)
    if not model:
        raise HTTPException(404, f"Model '{model_id}' not found")
    if model.downloaded:
        return {"status": "already_downloaded", "model_id": model_id}

    local_dir = MODELS_DIR / model_id
    submit_download_job(model_id, model.hf_tag, local_dir)
    return {"status": "started", "model_id": model_id, "hf_tag": model.hf_tag}


@app.get("/api/models/{model_id}/download/status", tags=["Models"])
async def download_status_sse(model_id: str):
    """SSE endpoint — streams download progress every 500ms until done/error."""

    async def event_generator():
        while True:
            state = get_download_state(model_id)
            if state is None:
                yield _sse({"error": "No download in progress", "model_id": model_id})
                return
            yield _sse(state)
            if state["status"] in ("done", "error"):
                return
            await asyncio.sleep(0.5)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# OCR jobs
# ---------------------------------------------------------------------------

@app.post("/api/ocr", tags=["OCR"])
async def submit_ocr(
    files: list[UploadFile] = File(...),
    model_id: str = Form(...),
    output_formats: str = Form("md"),   # comma-separated: md,html,docx
    db: Optional[Session] = Depends(get_db),
):
    model = get_model(model_id)
    if not model:
        raise HTTPException(400, f"Unknown model: '{model_id}'")

    fmt_list = [f.strip().lower() for f in output_formats.split(",") if f.strip()]
    unknown = [f for f in fmt_list if f not in SUPPORTED_FORMATS]
    if unknown:
        raise HTTPException(400, f"Unknown output format(s): {unknown}")
    if not fmt_list:
        raise HTTPException(400, "At least one output format is required")

    job_id = create_job()
    job_dir = UPLOADS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded files
    saved: list[tuple[bytes, str]] = []
    for up in files:
        data = await up.read()
        if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(413, f"File '{up.filename}' exceeds {MAX_UPLOAD_MB}MB limit")
        saved.append((data, up.filename or "upload"))
        dest = job_dir / (up.filename or f"file_{len(saved)}")
        dest.write_bytes(data)

    # DB record
    if ENABLE_DB and db:
        filenames = [fn for _, fn in saved]
        db_job = OCRJob(
            job_id=job_id,
            original_filename=", ".join(filenames),
            input_type=_detect_input_type(filenames),
            model_id=model_id,
            model_name=model.name,
            output_formats=fmt_list,
            status="pending",
        )
        db.add(db_job)
        db.commit()

    model_path = MODELS_DIR / model_id if model.downloaded else None

    def task(jid: str, progress_cb):
        all_images = []
        for data, fname in saved:
            imgs = load_images_from_bytes(data, fname)
            all_images.extend(imgs)

        if not all_images:
            raise ValueError("No images could be extracted from the uploaded files")

        progress_cb(0.1, f"Loaded {len(all_images)} page(s)")

        def _ocr_progress(p: float):
            progress_cb(0.1 + p * 0.7, f"OCR page {int(p * len(all_images))}/{len(all_images)}")

        ocr_result = run_ocr(all_images, model_id, model_path, model.adapter, _ocr_progress)

        progress_cb(0.85, "Exporting results…")
        out_dir = OUTPUTS_DIR / jid
        base_name = Path(saved[0][1]).stem if saved else "output"
        outputs = export(ocr_result, fmt_list, out_dir, base_name, title=base_name)

        output_files = {fmt: str(p.relative_to(OUTPUTS_DIR.parent)) for fmt, p in outputs.items()}
        return {
            "output_files": output_files,
            "content_summary": ocr_result.content_summary,
            "page_count": len(ocr_result.pages),
        }

    def on_complete(jid: str, result: dict):
        if ENABLE_DB and db:
            try:
                rec = db.query(OCRJob).filter(OCRJob.job_id == jid).first()
                if rec:
                    rec.status = "done"
                    rec.output_files = result.get("output_files", {})
                    rec.content_summary = result.get("content_summary", {})
                    rec.page_count = result.get("page_count", 0)
                    db.commit()
            except Exception as e:
                logger.error("DB update failed: %s", e)

    submit_ocr_job(job_id, task, on_complete)

    return {"job_id": job_id, "status": "pending", "model": model.name, "formats": fmt_list}


@app.get("/api/jobs", tags=["OCR"])
async def list_jobs_route():
    return {"jobs": list_jobs()}


@app.get("/api/jobs/{job_id}", tags=["OCR"])
async def get_job_route(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job.to_dict()


@app.get("/api/jobs/{job_id}/progress", tags=["OCR"])
async def job_progress_sse(job_id: str):
    """SSE stream — pushes job status every 300ms until done/error."""

    async def gen():
        while True:
            job = get_job(job_id)
            if job is None:
                yield _sse({"error": "Job not found", "job_id": job_id})
                return
            yield _sse(job.to_dict())
            if job.status in (JobStatus.DONE, JobStatus.ERROR):
                return
            await asyncio.sleep(0.3)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/jobs/{job_id}/download/{fmt}", tags=["OCR"])
async def download_result(job_id: str, fmt: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status != JobStatus.DONE:
        raise HTTPException(409, f"Job is not done (status: {job.status.value})")
    if fmt not in job.output_files:
        raise HTTPException(404, f"Format '{fmt}' not in job outputs: {list(job.output_files.keys())}")

    file_path = Path(job.output_files[fmt])
    if not file_path.is_absolute():
        file_path = Path(__file__).parent / file_path

    if not file_path.exists():
        raise HTTPException(404, "Output file not found on disk")

    mt, _ = mimetypes.guess_type(str(file_path))
    return FileResponse(
        path=str(file_path),
        media_type=mt or "application/octet-stream",
        filename=file_path.name,
    )


@app.get("/api/jobs/{job_id}/download-all", tags=["OCR"])
async def download_all_results(job_id: str):
    """Return a ZIP containing all output formats for a job."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status != JobStatus.DONE:
        raise HTTPException(409, "Job is not done")
    if not job.output_files:
        raise HTTPException(404, "No output files")

    import io as _io
    buf = _io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fmt, rel_path in job.output_files.items():
            p = Path(rel_path)
            if not p.is_absolute():
                p = Path(__file__).parent / p
            if p.exists():
                zf.write(p, arcname=p.name)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=ocr_{job_id[:8]}_results.zip"},
    )


# ---------------------------------------------------------------------------
# History (DB)
# ---------------------------------------------------------------------------

@app.get("/api/history", tags=["History"])
async def get_history(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Optional[Session] = Depends(get_db),
):
    if not ENABLE_DB or db is None:
        return {"enabled": False, "message": "Set ENABLE_DB=true in .env to enable history"}
    jobs = (
        db.query(OCRJob)
        .order_by(OCRJob.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    total = db.query(OCRJob).count()
    return {
        "enabled": True,
        "total": total,
        "offset": offset,
        "limit": limit,
        "items": [_job_to_dict(j) for j in jobs],
    }


@app.get("/api/history/{job_id}", tags=["History"])
async def get_history_item(job_id: str, db: Optional[Session] = Depends(get_db)):
    if not ENABLE_DB or db is None:
        raise HTTPException(503, "History is disabled")
    job = db.query(OCRJob).filter(OCRJob.job_id == job_id).first()
    if not job:
        raise HTTPException(404, "History item not found")
    return _job_to_dict(job)


@app.delete("/api/history/{job_id}", tags=["History"])
async def delete_history_item(job_id: str, db: Optional[Session] = Depends(get_db)):
    if not ENABLE_DB or db is None:
        raise HTTPException(503, "History is disabled")
    job = db.query(OCRJob).filter(OCRJob.job_id == job_id).first()
    if not job:
        raise HTTPException(404, "History item not found")
    db.delete(job)
    db.commit()
    return {"deleted": True, "job_id": job_id}


@app.get("/api/history/stats/summary", tags=["History"])
async def history_stats(db: Optional[Session] = Depends(get_db)):
    if not ENABLE_DB or db is None:
        return {"enabled": False}
    total = db.query(OCRJob).count()
    done = db.query(OCRJob).filter(OCRJob.status == "done").count()
    errors = db.query(OCRJob).filter(OCRJob.status == "error").count()
    return {"total": total, "done": done, "errors": errors, "pending": total - done - errors}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


def _detect_input_type(filenames: list[str]) -> str:
    if len(filenames) > 1:
        return "images"
    fn = filenames[0].lower()
    if fn.endswith(".pdf"):
        return "pdf"
    if fn.endswith(".zip"):
        return "zip"
    return "image"


def _job_to_dict(job: OCRJob) -> dict:
    return {
        "job_id": job.job_id,
        "original_filename": job.original_filename,
        "input_type": job.input_type,
        "model_id": job.model_id,
        "model_name": job.model_name,
        "output_formats": job.output_formats,
        "status": job.status,
        "progress": job.progress,
        "page_count": job.page_count,
        "content_summary": job.content_summary,
        "output_files": job.output_files,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "duration_seconds": job.duration_seconds,
        "error_message": job.error_message,
    }


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        # host=os.getenv("APP_HOST", "0.0.0.0"),
        port=int(os.getenv("APP_PORT", "8000")),
        reload=os.getenv("APP_DEBUG", "false").lower() == "true",
    )