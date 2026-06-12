# OCR Frontend (React + Vite)

Modern SPA for an OCR conversion service. Supports:

- Model management with download progress
- File upload (images, PDF, ZIP)
- Job submission with real‑time SSE progress
- Download results (Markdown, HTML, DOCX)
- Job history (if server database is enabled)

## Quick Start

1. **Install dependencies**

   ```bash
   npm install
   ```

2. **Configure environment**

   Create `.env` (or use existing) with the backend URL:

   ```
   VITE_API_BASE_URL=http://localhost:8000/api
   ```

   The Vite dev server proxies `/api` to that target automatically.

3. **Run development server**

   ```bash
   npm run dev
   ```

   The app opens at `http://localhost:3000`.

4. **Production build**

   ```bash
   npm run build
   npm run preview
   ```

## Project Structure

- `src/lib/api.ts` – all API calls and React Query hooks.
- `src/hooks/useSSE.ts` – generic SSE hook.
- `src/components/ui/` – reusable UI primitives (Button, Card, Progress, Dialog, Select, FileUpload).
- `src/pages/` – six pages matching the required routes.
- Tailwind CSS for styling, Radix UI for accessible components (Dialog, Select).

## API Requirements

The backend must serve the routes described in the prompt. The frontend expects base path `/api`. All calls are relative unless overridden by `VITE_API_BASE_URL`.