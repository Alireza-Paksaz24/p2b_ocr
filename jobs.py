"""
jobs.py — In-process job queue for OCR tasks.

Each OCR job runs in a ThreadPoolExecutor worker.
Progress is tracked in a shared dict so the /jobs/{id}/progress SSE
endpoint can stream updates to the frontend.
"""

from __future__ import annotations

import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Optional

logger = logging.getLogger("jobs")

_executor = ThreadPoolExecutor(max_workers=int(__import__("os").getenv("OCR_WORKERS", "1")))
_jobs: dict[str, "JobState"] = {}
_lock = Lock()


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    ERROR = "error"


@dataclass
class JobState:
    job_id: str
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0          # 0.0 – 1.0
    message: str = "Queued"
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    result: Optional[dict] = None
    error: Optional[str] = None
    output_files: dict[str, str] = field(default_factory=dict)
    content_summary: dict = field(default_factory=dict)
    page_count: int = 0
    duration_seconds: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "progress": self.progress,
            "message": self.message,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "output_files": self.output_files,
            "content_summary": self.content_summary,
            "page_count": self.page_count,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
        }


def create_job() -> str:
    job_id = str(uuid.uuid4())
    with _lock:
        _jobs[job_id] = JobState(job_id=job_id)
    return job_id


def get_job(job_id: str) -> Optional[JobState]:
    return _jobs.get(job_id)


def list_jobs() -> list[dict]:
    with _lock:
        return [j.to_dict() for j in sorted(_jobs.values(), key=lambda j: j.created_at, reverse=True)]


def _update(job_id: str, **kwargs) -> None:
    job = _jobs.get(job_id)
    if job:
        for k, v in kwargs.items():
            setattr(job, k, v)
        job.updated_at = datetime.utcnow()


def submit_ocr_job(
    job_id: str,
    task_fn: Callable,
    on_complete: Optional[Callable] = None,
) -> None:
    """Submit task_fn to the thread pool. task_fn receives (job_id, progress_cb)."""

    def _run():
        start = time.time()
        _update(job_id, status=JobStatus.PROCESSING, message="Starting OCR…")

        def progress_cb(value: float, msg: str = ""):
            _update(
                job_id,
                progress=max(0.0, min(1.0, value)),
                message=msg or f"Processing… {int(value * 100)}%",
            )

        try:
            result = task_fn(job_id, progress_cb)
            duration = time.time() - start
            _update(
                job_id,
                status=JobStatus.DONE,
                progress=1.0,
                message="Done",
                result=result,
                output_files=result.get("output_files", {}),
                content_summary=result.get("content_summary", {}),
                page_count=result.get("page_count", 0),
                duration_seconds=round(duration, 2),
            )
            if on_complete:
                on_complete(job_id, result)
        except Exception as e:
            logger.exception("Job %s failed", job_id)
            _update(
                job_id,
                status=JobStatus.ERROR,
                message=f"Error: {e}",
                error=str(e),
                duration_seconds=round(time.time() - start, 2),
            )

    _executor.submit(_run)


# ---------------------------------------------------------------------------
# Download job (model weights)
# ---------------------------------------------------------------------------

_dl_jobs: dict[str, dict] = {}   # model_id → progress state


def get_download_state(model_id: str) -> Optional[dict]:
    return _dl_jobs.get(model_id)


def list_download_states() -> list[dict]:
    return list(_dl_jobs.values())


def submit_download_job(model_id: str, hf_tag: str, local_dir: Path) -> None:
    if model_id in _dl_jobs and _dl_jobs[model_id].get("status") == "downloading":
        return  # already in progress

    _dl_jobs[model_id] = {
        "model_id": model_id,
        "status": "downloading",
        "progress": 0.0,
        "message": "Initialising download…",
        "error": None,
    }

    def _dl():
        try:
            from huggingface_hub import snapshot_download
            import threading

            state = _dl_jobs[model_id]
            state["message"] = f"Downloading {hf_tag} from Hugging Face…"

            # huggingface_hub doesn't expose per-file progress easily,
            # so we poll the partial directory size to estimate progress.
            local_dir.mkdir(parents=True, exist_ok=True)

            def _size_watcher():
                import os
                while state["status"] == "downloading":
                    try:
                        total = sum(
                            f.stat().st_size
                            for f in local_dir.rglob("*")
                            if f.is_file()
                        )
                        state["progress"] = min(0.95, total / (1024 ** 3))  # rough GB estimate
                    except Exception:
                        pass
                    time.sleep(1)

            watcher = threading.Thread(target=_size_watcher, daemon=True)
            watcher.start()

            snapshot_download(
                repo_id=hf_tag,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
            )

            state["status"] = "done"
            state["progress"] = 1.0
            state["message"] = "Download complete."

            # Mark in YAML
            from model_registry import mark_downloaded
            mark_downloaded(model_id, str(local_dir))

        except Exception as e:
            logger.exception("Download failed for %s", model_id)
            _dl_jobs[model_id]["status"] = "error"
            _dl_jobs[model_id]["error"] = str(e)
            _dl_jobs[model_id]["message"] = f"Download failed: {e}"

    _executor.submit(_dl)
