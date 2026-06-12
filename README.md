# OCR Converter

A multi-model OCR web application with a FastAPI backend and a clean browser-based frontend.

## Features

- **8 OCR models** configurable via `models.yaml` (DeepSeek, GOT-OCR 2.0, Florence-2, Nougat, Surya, KOSMOS-2, PaddleOCR-VL)
- **Smart content classification** — text, tables, diagrams, images, formulas, headers, footers
- **Multiple input types** — single image, multiple images, ZIP archive, PDF
- **Multiple output formats** — Markdown, HTML, Word (.docx) — select any combination
- **Real-time progress** via Server-Sent Events (SSE) for both downloads and OCR jobs
- **Job history** persisted in SQLite (optional, enable via `.env`)
- **Modular architecture** — swap the DB to Postgres/MySQL by changing `DATABASE_URL`

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# (Optional) Install poppler for full PDF support
# Ubuntu/Debian:  sudo apt install poppler-utils
# macOS:          brew install poppler

# 2. Configure
cp .env .env.local   # edit as needed

# 3. Run
python main.py
# → http://localhost:8000
```

---

## Project Structure

```
ocr-app/
├── main.py              ← FastAPI app + all API routes
├── ocr_engine.py        ← OCR adapters, content classifier, pipeline
├── formatters.py        ← MD / HTML / DOCX exporters
├── ingestor.py          ← File → PIL Images (image, ZIP, PDF)
├── jobs.py              ← Async job queue + download manager
├── model_registry.py    ← Load/query models.yaml
├── database.py          ← SQLAlchemy ORM + optional DB init
├── models.yaml          ← Model registry (edit to add/remove models)
├── .env                 ← Environment config
├── requirements.txt
├── static/
│   └── index.html       ← Frontend UI
├── models/              ← Downloaded model weights (auto-created)
├── uploads/             ← Temp upload storage (auto-created)
└── outputs/             ← Generated OCR outputs (auto-created)
```

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/models` | List all models with download state |
| GET | `/api/models/{id}` | Single model detail |
| POST | `/api/models/{id}/download` | Start model download |
| GET | `/api/models/{id}/download/status` | SSE: download progress |
| POST | `/api/ocr` | Submit OCR job (multipart form) |
| GET | `/api/jobs` | List all in-memory jobs |
| GET | `/api/jobs/{id}` | Job detail |
| GET | `/api/jobs/{id}/progress` | SSE: job progress |
| GET | `/api/jobs/{id}/download/{fmt}` | Download result file |
| GET | `/api/jobs/{id}/download-all` | Download all formats as ZIP |
| GET | `/api/history` | DB job history (requires ENABLE_DB=true) |
| GET | `/api/history/{id}` | Single history entry |
| DELETE | `/api/history/{id}` | Delete history entry |
| GET | `/api/history/stats/summary` | Aggregate stats |
| GET | `/api/formats` | List supported output formats |
| GET | `/api/docs` | Swagger UI |

### POST /api/ocr — form fields

| Field | Type | Description |
|-------|------|-------------|
| `files` | `File[]` | One or more image/zip/pdf files |
| `model_id` | `string` | Model ID from models.yaml |
| `output_formats` | `string` | Comma-separated: `md`, `html`, `docx` |

---

## Adding a New Model

Edit `models.yaml`:

```yaml
- id: my-new-model
  name: My Model
  description: Short description
  hf_tag: org/repo-name
  downloaded: false
  size_gb: 2.5
  supports_table: true
  supports_diagram: false
  supports_formula: false
```

Then add a corresponding adapter in `ocr_engine.py` that inherits `BaseOCRAdapter` and call `register_adapter("my-new-model", MyAdapter)`.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_HOST` | `0.0.0.0` | Listen address |
| `APP_PORT` | `8000` | Listen port |
| `ENABLE_DB` | `false` | Persist jobs to DB |
| `DATABASE_URL` | `sqlite:///./db/ocr_history.db` | SQLAlchemy URL |
| `MODELS_DIR` | `./models` | Where weights are stored |
| `UPLOADS_DIR` | `./uploads` | Temp upload directory |
| `OUTPUTS_DIR` | `./outputs` | OCR output directory |
| `MAX_UPLOAD_MB` | `200` | Per-file upload size limit |
| `OCR_WORKERS` | `1` | Parallel OCR jobs |
