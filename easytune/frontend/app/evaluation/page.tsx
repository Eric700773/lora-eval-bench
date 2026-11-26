"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import {
  BenchmarkRun,
  FlywheelIteration,
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

export default function EvaluationPage() {
  const [runs, setRuns] = useState<BenchmarkRun[]>([]);
  const [iterations, setIterations] = useState<FlywheelIteration[]>([]);
  const [runtime, setRuntime] = useState<QwenRuntime | null>(null);
  const [synthesis, setSynthesis] = useState<SynthesisConfig | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [reportMessage, setReportMessage] = useState("");

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
              <tr><th>Run</th><th>模式</th><th>模型</th><th>Raw</th><th>Clean</th><th>污染</th><th>状态</th></tr>
            </thead>
            <tbody>
              {runs.map((run) => (
                <tr key={run.id}>
                  <td><strong>#{run.id}</strong><br /><small>{run.name}</small></td>
                  <td>{run.use_dynamic_perturbation ? "Dynamic" : "Original"}</td>
                  <td>{run.model_version_id ? `ModelVersion #${run.model_version_id}` : "Base"}</td>
                  <td>{score(run.metrics_json?.raw_unfiltered?.pass_at_1)}</td>
                  <td>{score(run.metrics_json?.clean_decontaminated?.pass_at_1)}</td>
                  <td>{run.contaminated_count}</td>
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
