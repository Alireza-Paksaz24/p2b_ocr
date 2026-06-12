const BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api';

async function request<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const url = `${BASE_URL}${endpoint}`;
  const res = await fetch(url, {
    headers: {
      ...options?.headers,
      // For JSON requests, default header is set by fetch for GET; POST will need explicit Content-Type
    },
    ...options,
  });

  if (!res.ok) {
    const errorBody = await res.text();
    throw new Error(errorBody || `Request failed with status ${res.status}`);
  }

  // Some endpoints return empty body (e.g., 204)
  const text = await res.text();
  return text ? JSON.parse(text) : undefined;
}

// ---------- Types (mirrors backend responses) ----------
export interface Model {
  id: string;
  name: string;
  hf_tag: string;
  type: string;
  description: string;
  downloaded: boolean;
}

export interface Job {
  job_id: string;
  status: string;
  model_id: string;
  output_formats: string;
  progress: number;
  message: string;
  result_paths: Record<string, string>;
  error: string | null;
  created_at: string;
  pages?: { source: string; content: string }[];
  input_files?: string[];
}

export interface JobSummary {
  job_id: string;
  status: string;
  model_id: string;
  output_formats: string;
  progress: number;
  message: string;
  result_paths: Record<string, string>;
  error: string | null;
  created_at: string;
  input_files: string[];
}

export interface HistoryEntry {
  job_id: string;
  status: string;
  model_id: string;
  input_files: string[];
  output_formats: string;
  created_at: string;
  completed_at: string;
  result_paths: Record<string, string>;
  error: string | null;
}

export interface HealthResponse {
  status: string;
  db_enabled: boolean;
  models_dir: string;
  timestamp: string;
}

export interface DownloadProgress {
  status: 'queued' | 'downloading' | 'done' | 'error';
  progress: number;
  message: string;
  model_id: string;
}

// ---------- API functions ----------
export const api = {
  getModels: () => request<{ models: Model[] }>('/models'),

  downloadModel: (modelId: string) =>
    request<{ status: string; model_id: string }>(`/models/${modelId}/download`, {
      method: 'POST',
    }),

  getModelDownloadStatus: (modelId: string) =>
    request<DownloadProgress>(`/models/${modelId}/download/status`),

  submitJob: (formData: FormData) =>
    request<{ job_id: string; status: string }>('/ocr/submit', {
      method: 'POST',
      body: formData,
    }),

  getJob: (jobId: string) => request<Job>(`/ocr/jobs/${jobId}`),

  getJobs: (limit = 50) => request<{ jobs: JobSummary[] }>(`/ocr/jobs?limit=${limit}`),

  deleteJob: (jobId: string) =>
    request<{ deleted: string }>(`/ocr/jobs/${jobId}`, { method: 'DELETE' }),

  getHistory: () => request<{ history: HistoryEntry[] }>('/history'),

  getHealth: () => request<HealthResponse>('/health'),
};

// ---------- React Query Keys ----------
export const queryKeys = {
  models: ['models'] as const,
  modelDownload: (id: string) => ['model-download', id] as const,
  jobs: ['jobs'] as const,
  job: (id: string) => ['job', id] as const,
  history: ['history'] as const,
  health: ['health'] as const,
};