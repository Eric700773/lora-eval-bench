const DEFAULT_API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000";
const API_BASE_STORAGE_KEY = "easytune_api_base_url";
const API_TOKEN_STORAGE_KEY = "easytune_api_token";

export type BenchmarkMetrics = {
  clean_decontaminated?: { total_samples: number; pass_at_1: number; ci_95: number[] };
  raw_unfiltered?: { total_samples: number; pass_at_1: number; ci_95: number[] };
  contamination_inflation?: number;
  by_category_clean?: Record<string, { total: number; passed: number; pass_at_1: number }>;
  variant_mode?: string;
  elapsed_seconds?: number;
  framework_results?: Record<string, Record<string, number | string>>;
  n_samples?: Record<string, { original: number; effective: number }>;
  versions?: Record<string, number | string>;
  n_shot?: Record<string, number>;
  task_hashes?: Record<string, string>;
  lm_eval_version?: string;
};

export type GateCheck = {
  name: string;
  status: "passed" | "failed" | "not_run";
  detail: string | null;
  evidence: Record<string, unknown>;
};

export type IntegrityGateReport = {
  status: "passed" | "failed" | "not_run";
  checks: GateCheck[];
  infrastructure_errors: string[];
  model_failures: number;
  engineering_failures: number;
};

export type BenchmarkRun = {
  id: number;
  name: string;
  framework: string;
  task_name: string;
  benchmark_spec_json: Record<string, unknown> | null;
  run_manifest_json: Record<string, unknown> | null;
  gate_report_json: IntegrityGateReport | null;
  progress_json: {
    mode?: string;
    job_id?: string;
    worker_status?: string;
    expected_items?: number;
    completed_requests?: number;
    percent?: number;
    elapsed_seconds?: number;
    attempt?: number;
    resumable?: boolean;
    last_error?: string;
  } | null;
  raw_result_path: string | null;
  normalized_result_path: string | null;
  model_version_id: number | null;
  base_model: string;
  adapter_path: string | null;
  benchmark_dataset_id: number;
  reference_train_dataset_id: number | null;
  contaminated_count: number;
  use_dynamic_perturbation: number;
  config_hash: string;
  metrics_json: BenchmarkMetrics | null;
  status: string;
  error_message: string | null;
  created_at: string;
};

export type NormalizedBenchmarkItem = {
  sample_id: string;
  sample_index: number;
  category: string;
  source_record: { question?: string; answer?: string } & Record<string, unknown>;
  framework_request: Record<string, unknown>;
  model_input: {
    messages: Array<{ role: string; content: string }>;
    rendered_prompt: string | null;
    rendered_prompt_sha256: string | null;
    chat_template_sha256: string | null;
    prompt_tokens: number | null;
    truncated: boolean | null;
    nested_chat_template_detected: boolean | null;
    render_token_roundtrip_matched: boolean | null;
    input_ids_sha256: string | null;
    input_ids: number[] | null;
  };
  raw_output: { response?: string; worker_raw_generated_text?: string } & Record<string, unknown>;
  parsed_output: unknown;
  reference: unknown;
  score: number;
  passed: boolean;
  error_type: string | null;
  error_reason: string | null;
  evidence: {
    finish_reason?: string;
    response_content_matched?: boolean;
    worker_evidence_status?: string;
    worker_request_id?: string;
    filters?: Record<string, unknown>;
  } & Record<string, unknown>;
};

export type NormalizedBenchmarkRun = {
  schema_version: number;
  spec: {
    framework: string;
    framework_version: string | null;
    framework_revision: string | null;
    task_name: string;
    task_version: string | null;
    dataset_revision: string | null;
    split: string | null;
    few_shot: number;
    limit: number | null;
    mode: string;
    model_id: string;
    generation: Record<string, unknown>;
    task_config: Record<string, unknown>;
    runner_config: Record<string, unknown>;
  };
  manifest: Record<string, unknown>;
  gate_report: IntegrityGateReport;
  metrics: BenchmarkMetrics;
  items: NormalizedBenchmarkItem[];
  raw_result_path: string | null;
};

export type BenchmarkItemRecord = {
  id: number;
  benchmark_run_id: number;
  sample_index: number;
  framework_sample_id: string | null;
  category: string;
  prompt: string;
  expected_answer: string | null;
  rollouts_json: string[];
  score: number;
  is_passed: number;
  error_tag: string | null;
  error_reason: string | null;
  request_json: Record<string, unknown> | null;
  model_input_json: Record<string, unknown> | null;
  raw_output_json: Record<string, unknown> | null;
  parsed_output_json: Record<string, unknown> | null;
  evidence_json: Record<string, unknown> | null;
};

export type BenchmarkItemPage = {
  total: number;
  offset: number;
  limit: number;
  items: BenchmarkItemRecord[];
};

export type FlywheelIteration = {
  id: number;
  name: string;
  baseline_run_id: number;
  synthesized_dataset_id: number | null;
  training_job_id: number | null;
  post_eval_run_id: number | null;
  status: string;
  attribution_summary_json: Record<string, unknown> | null;
  gain_report_json: {
    baseline_pass_at_1?: number;
    post_pass_at_1?: number;
    net_gain_pass_at_1?: number;
    baseline_ci_95?: number[];
    post_ci_95?: number[];
    category_deltas?: Record<string, number>;
    config_matched?: boolean;
    clean_subset_matched?: boolean;
    model_version_id?: number;
  } | null;
  created_at: string;
};

export type SynthesisConfig = {
  configured: boolean;
  base_url: string | null;
  model: string | null;
  api_key_set: boolean;
  require_external: boolean;
};

export type QwenRuntime = {
  available: boolean;
  python: string;
  torch: string;
  torch_cuda: string;
  transformers: string;
  gpu: string;
  model_type: string;
  model_path: string;
  runtime_backend: string;
};
export type Dataset = {
  id: number;
  name: string;
  original_file_path: string;
  converted_file_path: string | null;
  file_type: string;
  file_size: number;
  sample_count: number;
  formatting: string;
  quality_score: number;
  report_json: Record<string, unknown> | null;
  dataset_info_json: Record<string, unknown> | null;
  status: string;
  created_at: string;
};

export type TrainingJob = {
  id: number;
  name: string;
  dataset_id: number;
  model_family: "qwen" | "rwkv";
  base_model: string;
  template: string;
  stage: string;
  finetuning_type: string;
  budget_level: string;
  extra_config_json: Record<string, unknown> | null;
  config_path: string | null;
  output_dir: string | null;
  status: string;
  progress: number;
  command: string | null;
  log_path: string | null;
  error_message: string | null;
  created_at: string;
  updated_at: string;
};

export type ModelVersion = {
  id: number;
  training_job_id: number;
  name: string;
  base_model: string;
  adapter_path: string;
  export_path: string | null;
  status: string;
  created_at: string;
};

export type ValidationRecord = {
  id: number;
  model_version_id: number;
  prompt: string;
  expected_answer: string | null;
  actual_answer: string;
  human_score: number;
  human_note: string | null;
  created_at: string;
};

export type ValidationGenerateResponse = {
  model_version_id: number;
  provider: string;
  model: string;
  actual_answer: string;
  raw_response: Record<string, unknown> | null;
};

export type Health = {
  status: string;
  storage_root: string;
  llamafactory_cli_available: boolean;
  rwkv_finetune_available: boolean;
  inference_provider: string;
  inference_base_url: string | null;
  api_token_required?: boolean;
  cors_origins?: string;
  gpus: Array<{
    name: string;
    memory_used_mb: string;
    memory_total_mb: string;
    utilization_percent: string;
  }>;
  vendor?: Record<string, boolean>;
  dependencies?: Record<
    string,
    {
      python?: string;
      available: boolean;
      required: Record<string, boolean>;
      optional: Record<string, boolean>;
      missing: string[];
      missing_optional: string[];
    }
  >;
};

export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  if (!(init?.body instanceof FormData) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const apiToken = getApiToken();
  if (apiToken && !headers.has("Authorization")) {
    headers.set("Authorization", `Bearer ${apiToken}`);
  }
  const response = await fetch(`${getApiBaseUrl()}${path}`, {
    ...init,
    cache: "no-store",
    headers
  });
  if (!response.ok) {
    let message = response.statusText;
    try {
      const body = await response.json();
      message = body.detail || message;
    } catch {
      // Ignore non-JSON error bodies.
    }
    throw new Error(message);
  }
  return response.json();
}

export async function uploadDataset(file: File, name?: string): Promise<Dataset> {
  const formData = new FormData();
  formData.append("file", file);
  if (name) {
    formData.append("name", name);
  }
  return apiFetch<Dataset>("/api/datasets/upload", {
    method: "POST",
    body: formData
  });
}

export function getApiBaseUrl(): string {
  if (typeof window === "undefined") {
    return DEFAULT_API_BASE_URL;
  }
  return window.localStorage.getItem(API_BASE_STORAGE_KEY) || DEFAULT_API_BASE_URL;
}

export function setApiBaseUrl(value: string): void {
  if (typeof window === "undefined") {
    return;
  }
  window.localStorage.setItem(API_BASE_STORAGE_KEY, value.replace(/\/$/, ""));
}

export function getApiToken(): string {
  if (typeof window === "undefined") {
    return "";
  }
  return window.localStorage.getItem(API_TOKEN_STORAGE_KEY) || "";
}

export function setApiToken(value: string): void {
  if (typeof window === "undefined") {
    return;
  }
  const trimmedValue = value.trim();
  if (trimmedValue) {
    window.localStorage.setItem(API_TOKEN_STORAGE_KEY, trimmedValue);
  } else {
    window.localStorage.removeItem(API_TOKEN_STORAGE_KEY);
  }
}

export { DEFAULT_API_BASE_URL };
