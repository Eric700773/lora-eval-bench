"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import {
  BenchmarkItemPage,
  BenchmarkItemRecord,
  BenchmarkRun,
  FlywheelIteration,
  NormalizedBenchmarkRun,
  QwenRuntime,
  SynthesisConfig,
  apiFetch,
  errorMessage
} from "../../lib/api";
import styles from "../shared.module.scss";
import { statusClass } from "../ui";

function score(value: number | undefined): string {
  return typeof value === "number" ? `${(value * 100).toFixed(2)}%` : "--";
}

function signedScore(value: number | undefined): string {
  if (typeof value !== "number") return "--";
  const prefix = value > 0 ? "+" : "";
  return `${prefix}${(value * 100).toFixed(2)} pp`;
}

function pretty(value: unknown): string {
  return typeof value === "string" ? value : JSON.stringify(value, null, 2);
}

export default function EvaluationPage() {
  const [runs, setRuns] = useState<BenchmarkRun[]>([]);
  const [iterations, setIterations] = useState<FlywheelIteration[]>([]);
  const [runtime, setRuntime] = useState<QwenRuntime | null>(null);
  const [synthesis, setSynthesis] = useState<SynthesisConfig | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [reportMessage, setReportMessage] = useState("");
  const [selectedOfficialRunId, setSelectedOfficialRunId] = useState<number | null>(null);
  const [officialResult, setOfficialResult] = useState<NormalizedBenchmarkRun | null>(null);
  const [officialLoading, setOfficialLoading] = useState(false);
  const [fullPage, setFullPage] = useState<BenchmarkItemPage | null>(null);
  const [fullOffset, setFullOffset] = useState(0);
  const [fullEvidence, setFullEvidence] = useState<Record<number, Record<string, unknown>>>({});
  const [startingFull, setStartingFull] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const [runData, iterationData, runtimeData, synthesisData] = await Promise.all([
        apiFetch<BenchmarkRun[]>("/api/benchmarks/runs"),
        apiFetch<FlywheelIteration[]>("/api/flywheel"),
        apiFetch<QwenRuntime>("/api/health/qwen-runtime"),
        apiFetch<SynthesisConfig>("/api/flywheel/synthesis/config")
      ]);
      setRuns(runData);
      setSelectedOfficialRunId((current) => current ?? runData.find((run) => run.framework === "lm_eval")?.id ?? null);
      setIterations(iterationData);
      setRuntime(runtimeData);
      setSynthesis(synthesisData);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    const selectedRun = runs.find((run) => run.id === selectedOfficialRunId);
    const isFull = selectedRun?.benchmark_spec_json?.mode === "full" || selectedRun?.progress_json?.mode === "full";
    if (!selectedOfficialRunId || isFull || !selectedRun?.normalized_result_path) {
      setOfficialResult(null);
      return;
    }
    let cancelled = false;
    setOfficialLoading(true);
    apiFetch<NormalizedBenchmarkRun>(`/api/benchmarks/runs/${selectedOfficialRunId}/normalized`)
      .then((result) => {
        if (!cancelled) setOfficialResult(result);
      })
      .catch((err) => {
        if (!cancelled) setError(errorMessage(err));
      })
      .finally(() => {
        if (!cancelled) setOfficialLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedOfficialRunId, runs]);

  useEffect(() => {
    const selectedRun = runs.find((run) => run.id === selectedOfficialRunId);
    if (!selectedRun || selectedRun.progress_json?.mode !== "full") {
      setFullPage(null);
      return;
    }
    if (selectedRun.status === "completed" || selectedRun.status === "gate_failed") {
      apiFetch<BenchmarkItemPage>(`/api/benchmarks/runs/${selectedRun.id}/items/page?offset=${fullOffset}&limit=25`)
        .then(setFullPage)
        .catch((err) => setError(errorMessage(err)));
      return;
    }
    if (selectedRun.status !== "running") return;
    const timer = window.setInterval(() => {
      apiFetch<BenchmarkRun>(`/api/benchmarks/runs/${selectedRun.id}/lm-eval/status`)
        .then((updated) => setRuns((current) => current.map((run) => run.id === updated.id ? updated : run)))
        .catch((err) => setError(errorMessage(err)));
    }, 15000);
    return () => window.clearInterval(timer);
  }, [selectedOfficialRunId, runs, fullOffset]);

  const officialRuns = useMemo(() => runs.filter((run) => run.framework === "lm_eval"), [runs]);
  const selectedOfficialRun = runs.find((run) => run.id === selectedOfficialRunId);
  const officialTaskMetrics = useMemo(() => {
    const values = Object.values(officialResult?.metrics.framework_results || {});
    return values[0] || {};
  }, [officialResult]);
  const strictMetric = Number(officialTaskMetrics["exact_match,strict-match"]);
  const flexibleMetric = Number(officialTaskMetrics["exact_match,flexible-extract"]);

  const completedIteration = useMemo(
    () => iterations.find((item) => item.status === "completed" && item.gain_report_json),
    [iterations]
  );
  const gain = completedIteration?.gain_report_json;
  const originalBaseline = runs.find((run) => run.id === completedIteration?.baseline_run_id);
  const originalPost = runs.find((run) => run.id === completedIteration?.post_eval_run_id);
  const dynamicBaseline = runs.find(
    (run) => run.status === "completed" && run.use_dynamic_perturbation === 1 && run.model_version_id === null
  );
  const dynamicPost = runs.find(
    (run) => run.status === "completed" && run.use_dynamic_perturbation === 1 && run.model_version_id === gain?.model_version_id
  );

  async function startFullEvaluation() {
    setStartingFull(true);
    setError("");
    try {
      const run = await apiFetch<BenchmarkRun>("/api/benchmarks/runs/lm-eval/full", {
        method: "POST",
        body: JSON.stringify({ name: "GSM8K lm-eval Full (1319 samples)", max_new_tokens: 512 })
      });
      setRuns((current) => [run, ...current.filter((item) => item.id !== run.id)]);
      setSelectedOfficialRunId(run.id);
      setFullOffset(0);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setStartingFull(false);
    }
  }

  async function resumeFullEvaluation() {
    if (!selectedOfficialRun) return;
    try {
      const run = await apiFetch<BenchmarkRun>(`/api/benchmarks/runs/${selectedOfficialRun.id}/lm-eval/resume`, { method: "POST" });
      setRuns((current) => current.map((item) => item.id === run.id ? run : item));
    } catch (err) {
      setError(errorMessage(err));
    }
  }

  async function loadFullEvidence(item: BenchmarkItemRecord) {
    if (!selectedOfficialRun || fullEvidence[item.id]) return;
    try {
      const evidence = await apiFetch<Record<string, unknown>>(`/api/benchmarks/runs/${selectedOfficialRun.id}/items/${item.id}/evidence`);
      setFullEvidence((current) => ({ ...current, [item.id]: evidence }));
    } catch (err) {
      setError(errorMessage(err));
    }
  }

  async function generateReport() {
    if (!completedIteration || !dynamicBaseline || !dynamicPost) return;
    setReportMessage("");
    try {
      const result = await apiFetch<{ markdown_path: string }>(
        `/api/flywheel/${completedIteration.id}/report?dynamic_baseline_run_id=${dynamicBaseline.id}&dynamic_post_run_id=${dynamicPost.id}`,
        { method: "POST" }
      );
      setReportMessage(result.markdown_path);
    } catch (err) {
      setError(errorMessage(err));
    }
  }

  return (
    <>
      <div className={styles.titleRow}>
        <div>
          <span className={styles.eyebrow}>Evaluation Flywheel</span>
          <h1>去污染评测与数据飞轮</h1>
          <p>Qwen3.5-2B · NF4 QLoRA · DeepSeek synthesis · paired regression</p>
        </div>
        <div className={styles.actions}>
          <button className={styles.secondaryButton} disabled={loading} onClick={() => void load()} type="button">
            刷新
          </button>
          <button
            className={styles.button}
            disabled={!completedIteration || !dynamicBaseline || !dynamicPost}
            onClick={() => void generateReport()}
            type="button"
          >
            生成报告
          </button>
        </div>
      </div>

      {error && <div className={styles.error}>{error}</div>}
      {reportMessage && <div className={styles.success}>报告：{reportMessage}</div>}

      <section className={styles.panel}>
        <div className={styles.panelHeader}>
          <div>
            <h2>Official Benchmark Gate</h2>
            <p className={styles.muted}>lm-evaluation-harness 固定协议、输入输出证据与评分链路</p>
          </div>
          <div className={styles.actions}>
            <button className={styles.button} disabled={startingFull || runs.some((run) => run.status === "running" && run.framework === "lm_eval")} onClick={() => void startFullEvaluation()} type="button">
              {startingFull ? "启动中..." : "启动 GSM8K 全量"}
            </button>
            {selectedOfficialRun?.progress_json?.resumable && selectedOfficialRun.status !== "running" && (
              <button className={styles.secondaryButton} onClick={() => void resumeFullEvaluation()} type="button">从缓存恢复</button>
            )}
            <label className={styles.compactField}>
              <span>Run</span>
              <select
                disabled={!officialRuns.length || officialLoading}
                onChange={(event) => { setSelectedOfficialRunId(Number(event.target.value)); setFullOffset(0); }}
                value={selectedOfficialRunId ?? ""}
              >
                {!officialRuns.length && <option value="">暂无 lm-eval Run</option>}
                {officialRuns.map((run) => <option key={run.id} value={run.id}>#{run.id} · {run.task_name} · {run.progress_json?.mode || "gate"}</option>)}
              </select>
            </label>
          </div>
        </div>
        {selectedOfficialRun?.progress_json?.mode === "full" && (
          <div className={styles.fullProgress}>
            <div>
              <span>Full evaluation</span>
              <strong>{selectedOfficialRun.progress_json.completed_requests ?? 0} / {selectedOfficialRun.progress_json.expected_items ?? 1319}</strong>
              <small>{selectedOfficialRun.progress_json.worker_status || selectedOfficialRun.status} · attempt {selectedOfficialRun.progress_json.attempt || 1}</small>
            </div>
            <div className={styles.progressTrack}><span style={{ width: `${selectedOfficialRun.progress_json.percent || 0}%` }} /></div>
            <strong>{(selectedOfficialRun.progress_json.percent || 0).toFixed(2)}%</strong>
          </div>
        )}
        {officialResult ? (
          <>
            <div className={styles.gateSummary}>
              <div><span>Gate</span><strong className={officialResult.gate_report.status === "passed" ? styles.positiveValue : styles.negativeValue}>{officialResult.gate_report.status}</strong></div>
              <div><span>Strict</span><strong>{Number.isFinite(strictMetric) ? score(strictMetric) : "--"}</strong></div>
              <div><span>Flexible</span><strong>{Number.isFinite(flexibleMetric) ? score(flexibleMetric) : "--"}</strong></div>
              <div><span>Engineering</span><strong>{officialResult.gate_report.engineering_failures}</strong></div>
              <div><span>Model failures</span><strong>{officialResult.gate_report.model_failures}</strong></div>
            </div>
            <div className={styles.protocolStrip}>
              <span>lm-eval {officialResult.spec.framework_version}</span>
              <span>{officialResult.spec.task_name} v{officialResult.spec.task_version}</span>
              <span>{officialResult.spec.split} · {officialResult.spec.few_shot}-shot · limit {officialResult.spec.limit}</span>
              <span title={officialResult.spec.dataset_revision || ""}>dataset {officialResult.spec.dataset_revision?.slice(0, 12)}</span>
            </div>
            <div className={styles.gateChecks}>
              {officialResult.gate_report.checks.map((check) => (
                <div key={check.name}>
                  <span className={statusClass(check.status)}>{check.status}</span>
                  <strong>{check.name}</strong>
                  <small title={check.detail || ""}>{check.detail || "--"}</small>
                </div>
              ))}
            </div>
          </>
        ) : selectedOfficialRun?.progress_json?.mode === "full" ? null : <p className={styles.muted}>{officialLoading ? "正在加载 Gate 证据..." : "导入 lm-eval Gate 后可查看。"}</p>}
      </section>

      {officialResult && (
        <section className={styles.panel}>
          <div className={styles.panelHeader}>
            <div>
              <h2>逐题证据</h2>
              <p className={styles.muted}>官方样本 → messages → rendered prompt/token → raw output → parser → score</p>
            </div>
          </div>
          <div className={styles.evidenceList}>
            {officialResult.items.map((item) => (
              <details key={item.sample_id}>
                <summary>
                  <span>#{item.sample_id}</span>
                  <strong>{item.source_record.question || "Untitled sample"}</strong>
                  <span className={statusClass(item.passed ? "passed" : "failed")}>{item.passed ? "passed" : "failed"}</span>
                </summary>
                <div className={styles.evidenceMeta}>
                  <div><span>Prompt tokens</span><strong>{item.model_input.prompt_tokens ?? "--"}</strong></div>
                  <div><span>Finish</span><strong>{item.evidence.finish_reason || "--"}</strong></div>
                  <div><span>Roundtrip</span><strong>{item.model_input.render_token_roundtrip_matched ? "Yes" : "No"}</strong></div>
                  <div><span>Response aligned</span><strong>{item.evidence.response_content_matched ? "Yes" : "No"}</strong></div>
                  <div><span>Parsed</span><strong>{pretty(item.parsed_output)}</strong></div>
                </div>
                <div className={styles.evidenceGrid}>
                  <div><h3>Messages</h3><pre>{pretty(item.model_input.messages)}</pre></div>
                  <div><h3>Rendered Prompt</h3><pre>{item.model_input.rendered_prompt || "--"}</pre></div>
                  <div><h3>Raw Output</h3><pre>{item.raw_output.response || "--"}</pre></div>
                  <div><h3>Parser / Reference</h3><pre>{pretty({ parsed: item.parsed_output, reference: item.reference, filters: item.evidence.filters })}</pre></div>
                </div>
                <div className={styles.hashLine}>
                  <span>Prompt SHA256</span><code>{item.model_input.rendered_prompt_sha256 || "--"}</code>
                  <span>Input IDs SHA256</span><code>{item.model_input.input_ids_sha256 || "--"}</code>
                </div>
              </details>
            ))}
          </div>
        </section>
      )}

      {fullPage && selectedOfficialRun && (
        <section className={styles.panel}>
          <div className={styles.panelHeader}>
            <div>
              <h2>全量逐题结果</h2>
              <p className={styles.muted}>{fullPage.total} 条 · 每页 25 条 · 展开时按需读取完整 evidence</p>
            </div>
            <div className={styles.actions}>
              <button className={styles.secondaryButton} disabled={fullOffset === 0} onClick={() => setFullOffset(Math.max(0, fullOffset - 25))} type="button">上一页</button>
              <button className={styles.secondaryButton} disabled={fullOffset + 25 >= fullPage.total} onClick={() => setFullOffset(fullOffset + 25)} type="button">下一页</button>
            </div>
          </div>
          <div className={styles.evidenceList}>
            {fullPage.items.map((item) => (
              <details key={item.id} onToggle={(event) => { if (event.currentTarget.open) void loadFullEvidence(item); }}>
                <summary>
                  <span>#{item.sample_index}</span>
                  <strong>{item.prompt}</strong>
                  <span className={statusClass(item.is_passed ? "passed" : "failed")}>{item.is_passed ? "passed" : "failed"}</span>
                </summary>
                <div className={styles.evidenceMeta}>
                  <div><span>Score</span><strong>{item.score}</strong></div>
                  <div><span>Parsed</span><strong>{pretty(item.parsed_output_json?.value)}</strong></div>
                  <div><span>Finish</span><strong>{pretty(item.evidence_json?.finish_reason)}</strong></div>
                  <div><span>Input tokens</span><strong>{pretty(item.evidence_json?.input_ids_count)}</strong></div>
                  <div><span>Evidence</span><strong>{fullEvidence[item.id] ? "Loaded" : "Loading..."}</strong></div>
                </div>
                {fullEvidence[item.id] && (
                  <div className={styles.evidenceGrid}>
                    <div><h3>Messages</h3><pre>{pretty((fullEvidence[item.id].api_request as Record<string, unknown>)?.messages)}</pre></div>
                    <div><h3>Rendered Prompt</h3><pre>{pretty((fullEvidence[item.id].model_input as Record<string, unknown>)?.rendered_prompt)}</pre></div>
                    <div><h3>Raw Output</h3><pre>{pretty((fullEvidence[item.id].model_output as Record<string, unknown>)?.raw_generated_text)}</pre></div>
                    <div><h3>Parser / Reference</h3><pre>{pretty({ parsed: item.parsed_output_json, reference: item.expected_answer })}</pre></div>
                  </div>
                )}
              </details>
            ))}
          </div>
        </section>
      )}

      <section className={styles.metricGrid}>
        <div className={styles.metricCard}>
          <span>Runtime</span>
          <strong>{runtime?.available ? "CUDA Ready" : "Offline"}</strong>
          <small>{runtime ? `${runtime.gpu} · ${runtime.model_type}` : "--"}</small>
        </div>
        <div className={styles.metricCard}>
          <span>Baseline Clean</span>
          <strong>{score(gain?.baseline_pass_at_1)}</strong>
          <small>Run #{originalBaseline?.id ?? "--"} · {originalBaseline?.contaminated_count ?? "--"} contaminated</small>
        </div>
        <div className={styles.metricCard}>
          <span>Post Clean</span>
          <strong>{score(gain?.post_pass_at_1)}</strong>
          <small>Run #{originalPost?.id ?? "--"} · ModelVersion #{gain?.model_version_id ?? "--"}</small>
        </div>
        <div className={styles.metricCard}>
          <span>Net Gain</span>
          <strong className={styles.positiveValue}>{signedScore(gain?.net_gain_pass_at_1)}</strong>
          <small>Clean subset {gain?.clean_subset_matched ? "matched" : "unmatched"}</small>
        </div>
        <div className={styles.metricCard}>
          <span>Synthesis</span>
          <strong>{synthesis?.configured ? "Configured" : "Missing"}</strong>
          <small>{synthesis?.model || "--"}</small>
        </div>
      </section>

      <section className={styles.compareGrid}>
        <div className={styles.panel}>
          <div className={styles.panelHeader}>
            <div>
              <h2>能力切片变化</h2>
              <p className={styles.muted}>Clean Pass@1 delta</p>
            </div>
          </div>
          <div className={styles.configList}>
            {Object.entries(gain?.category_deltas || {}).map(([category, delta]) => (
              <div key={category}>
                <span>{category}</span>
                <strong className={delta >= 0 ? styles.positiveValue : styles.negativeValue}>{signedScore(delta)}</strong>
              </div>
            ))}
            {!Object.keys(gain?.category_deltas || {}).length && <p className={styles.muted}>暂无完成的回归结果。</p>}
          </div>
        </div>

        <div className={styles.panel}>
          <div className={styles.panelHeader}>
            <div>
              <h2>飞轮状态</h2>
              <p className={styles.muted}>Iteration #{completedIteration?.id ?? "--"}</p>
            </div>
          </div>
          <div className={styles.configList}>
            <div><span>状态</span><strong>{completedIteration?.status || "--"}</strong></div>
            <div><span>合成数据集</span><strong>#{completedIteration?.synthesized_dataset_id ?? "--"}</strong></div>
            <div><span>训练任务</span><strong>#{completedIteration?.training_job_id ?? "--"}</strong></div>
            <div><span>配置一致</span><strong>{gain?.config_matched ? "Yes" : "No"}</strong></div>
          </div>
        </div>
      </section>

      <section className={styles.panel}>
        <div className={styles.panelHeader}>
          <div>
            <h2>Benchmark Runs</h2>
            <p className={styles.muted}>原题、动态变体、微调前后统一记录</p>
          </div>
        </div>
        <div className={styles.tableWrap}>
          <table className={styles.table}>
            <thead>
              <tr><th>Run</th><th>Framework / Task</th><th>模式</th><th>模型</th><th>Primary</th><th>Secondary</th><th>状态</th></tr>
            </thead>
            <tbody>
              {runs.map((run) => (
                <tr key={run.id}>
                  <td><strong>#{run.id}</strong><br /><small>{run.name}</small></td>
                  <td><strong>{run.framework || "custom"}</strong><br /><small>{run.task_name || "custom_dataset"}</small></td>
                  <td>{run.framework === "lm_eval" ? "Gate" : run.use_dynamic_perturbation ? "Dynamic" : "Original"}</td>
                  <td>{run.model_version_id ? `ModelVersion #${run.model_version_id}` : "Base"}</td>
                  <td>{run.framework === "lm_eval" ? score(Number(Object.values(run.metrics_json?.framework_results || {})[0]?.["exact_match,strict-match"])) : score(run.metrics_json?.raw_unfiltered?.pass_at_1)}</td>
                  <td>{run.framework === "lm_eval" ? score(Number(Object.values(run.metrics_json?.framework_results || {})[0]?.["exact_match,flexible-extract"])) : score(run.metrics_json?.clean_decontaminated?.pass_at_1)}</td>
                  <td><span className={statusClass(run.status)}>{run.status}</span></td>
                </tr>
              ))}
              {!runs.length && <tr><td colSpan={7}>{loading ? "加载中..." : "暂无评测任务"}</td></tr>}
            </tbody>
          </table>
        </div>
      </section>
    </>
  );
}
