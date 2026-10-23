import { useEffect, useMemo, useRef } from "react";
import { Database, X } from "lucide-react";
import {
  cacheKindLabels,
  groupCacheRows,
  type CacheReport,
} from "../lib/cache";
import {
  formatBytes,
  formatNumber,
  type ModelConfig,
  type Scenario,
} from "../lib/model";

export default function CacheDialog({
  model,
  name,
  scenario,
  report,
  onChange,
  onClose,
  error,
}: {
  model: ModelConfig;
  name: string;
  scenario: Scenario;
  report: CacheReport;
  onChange: (s: Partial<Scenario>) => void;
  onClose: () => void;
  error: string;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const rows = useMemo(() => groupCacheRows(report.rows), [report]);
  useEffect(() => {
    const element = dialog.current,
      opener = document.activeElement;
    element?.showModal();
    return () => {
      element?.close();
      if (opener instanceof HTMLElement && opener.isConnected) opener.focus();
    };
  }, []);
  const sections = [
    { kind: "kv", bytes: report.kvBytes },
    { kind: "recurrent", bytes: report.recurrentBytes },
    { kind: "auxiliary", bytes: report.auxiliaryBytes },
  ] as const;
  return (
    <dialog
      ref={dialog}
      className="cache-dialog"
      aria-labelledby="cache-title"
      onCancel={onClose}
      onClick={(e) => {
        if (e.target === dialog.current) onClose();
      }}
    >
      <header className="cache-dialog-header">
        <span className="dialog-icon">
          <Database size={23} />
        </span>
        <div>
          <span className="section-eyebrow">INFERENCE CACHE</span>
          <h2 id="cache-title">KV cache 计算</h2>
          <p>{name}</p>
        </div>
        <button
          className="icon-button"
          aria-label="关闭缓存计算"
          onClick={onClose}
        >
          <X size={20} />
        </button>
      </header>
      <div className="cache-scenario" aria-label="缓存计算场景">
        {(
          [
            { key: "batch", label: "批大小 B", min: 1 },
            { key: "sequence", label: "当前 token S", min: 1 },
            { key: "past", label: "历史 token P", min: 0 },
          ] as const
        ).map(({ key, label, min }) => (
          <label key={key}>
            {label}
            <input
              type="number"
              min={min}
              step="1"
              aria-label={`缓存${label}`}
              value={scenario[key]}
              onChange={(e) => {
                const n = Number(e.target.value);
                if (e.target.value && Number.isSafeInteger(n) && n >= min)
                  onChange({ [key]: n });
              }}
            />
          </label>
        ))}
        <label>
          缓存逻辑精度
          <select
            aria-label="缓存逻辑精度"
            value={scenario.dtype}
            onChange={(e) =>
              onChange({ dtype: e.target.value as Scenario["dtype"] })
            }
          >
            <option>bfloat16</option>
            <option>float16</option>
            <option>float32</option>
          </select>
        </label>
      </div>
      {error && (
        <p className="error-message" role="alert">
          {error}
        </p>
      )}
      <div className="cache-dialog-body">
        <div className="cache-summary-grid">
          <div className="cache-main-total">
            <span>
              {report.partial ? "已知缓存与状态小计" : "缓存与状态合计"}
            </span>
            <strong
              data-testid="cache-total"
              title={`${report.totalBytes} bytes`}
            >
              {formatBytes(report.totalBytes)}
            </strong>
            <small>
              T = P + S = {formatNumber(report.totalTokens)} · B ={" "}
              {scenario.batch}
            </small>
          </div>
          <div>
            <span>每个序列</span>
            <strong data-testid="cache-per-sequence">
              {formatBytes(report.perSequenceBytes)}
            </strong>
            <small>合计 ÷ B</small>
          </div>
          <div>
            <span>本次净增</span>
            <strong data-testid="cache-added">
              {formatBytes(report.addedBytes)}
            </strong>
            <small>
              {scenario.past === 0
                ? "从空缓存开始"
                : "处理 S 个 token 前后之差"}
            </small>
          </div>
          <div>
            <span>下一 token 净增</span>
            <strong data-testid="cache-next">
              {formatBytes(report.nextTokenBytes)}
            </strong>
            <small>整个 batch：T → T + 1</small>
          </div>
        </div>
        <div className="cache-composition" aria-label="缓存分类占用">
          <div className="cache-segments" aria-hidden="true">
            {sections
              .filter((s) => s.bytes > 0)
              .map((s) => (
                <span
                  key={s.kind}
                  className={`cache-color-${s.kind}`}
                  style={{ width: `${(s.bytes / report.totalBytes) * 100}%` }}
                />
              ))}
          </div>
          <div className="cache-breakdown">
            {sections.map((s) => (
              <div key={s.kind}>
                <i className={`cache-color-${s.kind}`} />
                <span>{cacheKindLabels[s.kind]}</span>
                <b data-testid={`cache-${s.kind}`} title={`${s.bytes} bytes`}>
                  {formatBytes(s.bytes)}
                </b>
              </div>
            ))}
          </div>
          {report.kvBytes === 0 && (
            <p className="cache-explanation">
              {report.unknownLayers.length
                ? "部分算子的 KV 状态未知，不能把已知小计 0 视为无需缓存。"
                : "该主干没有随 token 增长的传统 K/V 缓存；推理历史由递归或卷积状态保存。"}
            </p>
          )}
        </div>
        {report.packedGlobalBytes !== null && (
          <p className="cache-packed">
            全局紧凑 FP4：
            <strong>{formatBytes(report.packedGlobalBytes)}</strong>
            <span>
              含 scales；仅全局 KV + 索引 K
              的替代存储，不与上方浮点合计重复相加。
            </span>
          </p>
        )}
        <div className="cache-section-heading">
          <h3>按层与类型计算</h3>
          <span>
            {model.layers} 个主干层 · {rows.length} 组状态
          </span>
        </div>
        <div className="cache-rows">
          {rows.map((row) => (
            <details key={row.id} className={`cache-row cache-row-${row.kind}`}>
              <summary>
                <span>
                  <b>{row.name}</b>
                  <small>
                    {row.layers.length} 个来源层 · {row.copies} 份
                    {row.consumers.length > row.layers.length
                      ? ` · 供 ${row.consumers.length} 层使用`
                      : ""}
                  </small>
                </span>
                <strong>{formatBytes(row.bytes)}</strong>
              </summary>
              <div className="cache-row-detail">
                <code className="cache-formula">
                  每份：{row.formula}
                  <br />
                  合计：每份 × {row.copies} = {formatBytes(row.bytes)}
                </code>
                <p>来源层：{row.layers.map((i) => `L${i}`).join("、")}</p>
                {row.consumers.length > row.layers.length && (
                  <p>
                    使用层（共享引用）：
                    {row.consumers.map((i) => `L${i}`).join("、")}
                  </p>
                )}
                <ul>
                  {row.tensors.map((t) => (
                    <li key={t.name}>
                      <span>{t.name}</span>
                      <code>[{t.shape.join(", ")}]</code>
                      <small>{t.dtype}</small>
                    </li>
                  ))}
                </ul>
                <p>{row.note}</p>
                <p>
                  形状为每份状态；按 {row.copies} 份累加，各来源层仅计一次。
                </p>
              </div>
            </details>
          ))}
        </div>
        <details
          className="cache-assumptions"
          open={report.partial || report.cacheDisabledByDefault}
        >
          <summary>计算口径与范围{report.partial ? " · 部分估算" : ""}</summary>
          <ul>
            {report.notes.map((note) => (
              <li key={note}>{note}</li>
            ))}
          </ul>
        </details>
      </div>
      <footer className="cache-dialog-footer">
        <span>静态容量估算 · 非峰值显存</span>
        <button className="secondary-button" onClick={onClose}>
          返回模型
        </button>
      </footer>
    </dialog>
  );
}
