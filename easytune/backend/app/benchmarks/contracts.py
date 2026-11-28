from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class GenerationConfig(BaseModel):
    temperature: float = 0.0
    max_new_tokens: int = 512
    num_rollouts: int = 1
    enable_thinking: bool = False
    load_in_4bit: bool = True
    seed: int = 42
    stop_sequences: list[str] = Field(default_factory=list)


class BenchmarkSpec(BaseModel):
    schema_version: int = 1
    framework: str
    task_name: str
    task_version: str | None = None
    framework_version: str | None = None
    framework_revision: str | None = None
    dataset_revision: str | None = None
    split: str | None = None
    few_shot: int = 0
    limit: int | None = None
    mode: Literal["custom", "gate", "full"] = "custom"
    model_id: str
    model_revision: str | None = None
    tokenizer_revision: str | None = None
    chat_template_sha256: str | None = None
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    task_config: dict[str, Any] = Field(default_factory=dict)
    runner_config: dict[str, Any] = Field(default_factory=dict)

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def spec_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


class GateCheck(BaseModel):
    name: str
    status: Literal["passed", "failed", "not_run"]
    detail: str | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)


class IntegrityGateReport(BaseModel):
    status: Literal["passed", "failed", "not_run"] = "not_run"
    checks: list[GateCheck] = Field(default_factory=list)
    infrastructure_errors: list[str] = Field(default_factory=list)
    model_failures: int = 0
    engineering_failures: int = 0


class RunManifest(BaseModel):
    schema_version: int = 1
    framework: str
    framework_version: str | None = None
    framework_revision: str | None = None
    task_name: str
    task_config_hash: str | None = None
    benchmark_spec_hash: str
    model_identity: dict[str, Any] = Field(default_factory=dict)
    environment: dict[str, Any] = Field(default_factory=dict)
    command: list[str] = Field(default_factory=list)
    output_dir: str | None = None
    raw_result_path: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


class ModelInputEvidence(BaseModel):
    messages: list[dict[str, Any]] = Field(default_factory=list)
    rendered_prompt: str | None = None
    rendered_prompt_sha256: str | None = None
    chat_template_sha256: str | None = None
    prompt_tokens: int | None = None
    truncated: bool | None = None
    model_context_length: int | None = None
    context_exceeded: bool | None = None
    nested_chat_template_detected: bool | None = None
    render_token_roundtrip_matched: bool | None = None
    decoded_input_with_special_tokens: str | None = None
    input_ids_sha256: str | None = None
    input_ids_head: list[int] = Field(default_factory=list)
    input_ids_tail: list[int] = Field(default_factory=list)
    input_ids: list[int] | None = None


class NormalizedItemResult(BaseModel):
    sample_id: str
    sample_index: int
    category: str = "general"
    source_record: dict[str, Any] = Field(default_factory=dict)
    framework_request: dict[str, Any] = Field(default_factory=dict)
    model_input: ModelInputEvidence = Field(default_factory=ModelInputEvidence)
    raw_output: dict[str, Any] = Field(default_factory=dict)
    parsed_output: Any = None
    reference: Any = None
    score: float = 0.0
    passed: bool = False
    error_type: str | None = None
    error_reason: str | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)


class NormalizedRunResult(BaseModel):
    schema_version: int = 1
    spec: BenchmarkSpec
    manifest: RunManifest
    gate_report: IntegrityGateReport = Field(default_factory=IntegrityGateReport)
    metrics: dict[str, Any] = Field(default_factory=dict)
    items: list[NormalizedItemResult] = Field(default_factory=list)
    raw_result_path: str | None = None
