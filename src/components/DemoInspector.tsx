import { useEffect, useMemo, useState } from "react";
import { ArrowLeft, ArrowRight, Copy, Grid2X2, X } from "lucide-react";
import type { ModuleNode } from "../lib/graph";
import {
  compactNumber,
  formatBytes,
  tensorBytes,
  type ModelConfig,
  type Scenario,
} from "../lib/model";
import {
  activationValue,
  explainDemoNode,
  normalizeExample,
  stableSoftmax,
} from "../lib/demo-explanations";
import { tensorVolume } from "../lib/tensor-layout";
import type { TraceSample, TraceToken } from "../lib/demo-trace";
import { recordedValueColor } from "../lib/visual-values";
import ComponentPreview3D from "./ComponentPreview3D";
import {
  tensorAxes,
  tensorHeadCount,
  tensorCoordinates,
  traceValue,
} from "../lib/tensor-semantics";
const EMPTY_SAMPLES: TraceSample[] = [];

export default function DemoInspector({
  node,
  model,
  scenario,
  tokenLabels,
  tokenIndex,
  onToken,
  head,
  onHead,
  onPrevious,
  onNext,
  onClose,
  onMatrix,
  devMode,
  samples = EMPTY_SAMPLES,
  traceTokens,
  traceSource,
}: {
  node: ModuleNode;
  model: ModelConfig;
  scenario: Scenario;
  tokenLabels: string[];
  tokenIndex: number;
  onToken: (index: number) => void;
  head: number;
  onHead: (head: number) => void;
  onPrevious: () => void;
  onNext: () => void;
  onClose: () => void;
  onMatrix: () => void;
  devMode: boolean;
  samples?: TraceSample[];
  traceTokens?: TraceToken[];
  traceSource?: string;
}) {
  const [tab, setTab] = useState<"explain" | "tensor" | "lab">("explain"),
    [tensorIndex, setTensorIndex] = useState(0),
    [x, setX] = useState(0.5),
    [copied, setCopied] = useState(false),
    [cell, setCell] = useState<[number, number] | null>(null);
  const [rowStart, setRowStart] = useState(0),
    [colStart, setColStart] = useState(0);
  const tensors = [...node.data.inputs, ...node.data.outputs],
    tensor = tensors[Math.min(tensorIndex, tensors.length - 1)],
    volume = tensorVolume(tensor);
  const explanation = useMemo(
    () => explainDemoNode(model, node.data),
    [model, node],
  );
  useEffect(() => {
    setTensorIndex(0);
    setCell(null);
    setCopied(false);
    setRowStart(0);
    setColStart(0);
  }, [node.id, scenario.sequence, scenario.past]);
  const copy = async () => {
    const text = [
      model.architecture,
      node.data.title,
      `Token position: ${tokenIndex + 1}; head: ${head + 1}`,
      node.data.formula,
      ...tensors.map((t) => `${t.name}: [${t.shape.join(", ")}] ${t.dtype}`),
      explanation.purpose,
      explanation.reason,
      traceSource
        ? `来源：导入运行记录 ${traceSource}`
        : "来源：模型配置推导；未加载真实激活。",
    ].join("\n");
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
    } catch {
      setCopied(false);
    }
  };
  const values = [x, -0.5, 1.3, 0.2],
    probabilities = stableSoftmax(values),
    norm = normalizeExample(
      values,
      /rms/i.test(node.data.title),
      model.normEps,
    );
  const headCount =
    tensorHeadCount(tensor) ??
    node.data.attention?.heads ??
    model.catalog?.profiles[node.data.layerIndex ?? 0]?.heads ??
    model.heads;
  const selectedHead = Math.max(0, Math.min(head, headCount - 1));
  useEffect(() => {
    if (head !== selectedHead) onHead(selectedHead);
  }, [head, selectedHead, onHead]);
  const sampled = samples.find(
    (s) =>
      s.direction ===
        (tensorIndex < node.data.inputs.length ? "input" : "output") &&
      s.tensorIndex ===
        (tensorIndex < node.data.inputs.length
          ? tensorIndex
          : tensorIndex - node.data.inputs.length),
  );
  const slice = { token: tokenIndex - scenario.past, head: selectedHead };
  const cellIndices = (row: number, col: number) =>
    tensorCoordinates(tensor, slice, { row, column: col });
  const cellValue = (row: number, col: number) =>
    tensor ? traceValue(sampled, tensor, cellIndices(row, col)) : null;
  const axes = tensor ? tensorAxes(tensor) : [];
  const rowAxis = axes.length - 2;
  useEffect(() => {
    if (!tensor) return;
    const row =
      axes[rowAxis] === "query"
        ? tokenIndex - scenario.past
        : axes[rowAxis] === "head" || axes[rowAxis] === "kv-head"
          ? selectedHead
          : null;
    if (row !== null && row >= 0 && row < volume.rows)
      setRowStart(Math.floor(row / 10) * 10);
    setCell(null);
  }, [tokenIndex, selectedHead, tensor, scenario.past]);

  const pageRow = Math.min(rowStart, Math.max(0, volume.rows - 1));
  const pageColumn = Math.min(colStart, Math.max(0, volume.columns - 1));
  const pageRows = Math.max(0, Math.min(volume.rows - pageRow, 10));
  const pageColumns = Math.max(0, Math.min(volume.columns - pageColumn, 16));
  return (
    <aside className="demo-inspector" aria-label="演示组件详解">
      <header>
        <div>
          <span>COMPONENT DETAILS</span>
          <h2>{node.data.title}</h2>
          <small>{node.data.path}</small>
        </div>
        <button
          className="demo-icon"
          aria-label="关闭组件详解"
          onClick={onClose}
        >
          <X size={17} />
        </button>
      </header>
      <div className="demo-detail-navigation">
        <button aria-label="上一个组件" onClick={onPrevious}>
          <ArrowLeft size={14} />
        </button>
        <span>点击 3D 模块查看详情</span>
        <button aria-label="下一个组件" onClick={onNext}>
          <ArrowRight size={14} />
        </button>
      </div>
      <ComponentPreview3D
        node={node}
        model={model}
        scenario={scenario}
        tokenLabels={tokenLabels.slice(scenario.past)}
        samples={samples}
        head={selectedHead}
        recordedMode={!!traceSource}
      />
      <div className="demo-detail-tokens">
        {tokenLabels.map((label, i) => (
          <button
            key={i}
            aria-label={`查看 token ${i + 1}`}
            aria-pressed={i === tokenIndex}
            onClick={() => onToken(i)}
          >
            {label}
          </button>
        ))}
      </div>
      <div className="demo-token-meta">
        <span>位置 {tokenIndex + 1}</span>
        <span>Token ID：{traceTokens?.[tokenIndex]?.id ?? "未提供"}</span>
        <label>
          Head
          <select
            aria-label="解释面板注意力头"
            value={selectedHead}
            onChange={(e) => onHead(Number(e.target.value))}
          >
            {Array.from({ length: Math.min(headCount, 256) }, (_, i) => (
              <option key={i} value={i}>
                Head {i + 1}
              </option>
            ))}
          </select>
        </label>
      </div>
      <nav className="demo-detail-tabs" aria-label="组件详情页签">
        {(
          [
            { id: "explain", label: "原理解释" },
            { id: "tensor", label: "张量 / 矩阵" },
            { id: "lab", label: "数学示例" },
          ] as const
        ).map((t) => (
          <button
            key={t.id}
            aria-pressed={tab === t.id}
            onClick={() => setTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </nav>
      <div
        className="demo-detail-scroll"
        tabIndex={0}
        aria-label="组件解释内容"
      >
        {tab === "explain" && (
          <>
            <div className="demo-detail-formula">
              {node.data.formula || node.data.subtitle}
            </div>
            <h3>这个组件做什么</h3>
            <p>{explanation.purpose}</p>
            <h3>为什么需要它</h3>
            <p>{explanation.reason}</p>
            {explanation.steps.length > 0 && (
              <>
                <h3>计算过程</h3>
                <ol>
                  {explanation.steps.map((s) => (
                    <li key={s}>{s}</li>
                  ))}
                </ol>
              </>
            )}
            <dl className="demo-detail-properties">
              <dt>参数</dt>
              <dd>
                {node.data.parameterStatus === "unknown"
                  ? "未细分"
                  : compactNumber(node.data.params)}
                {model.catalog ? " · 结构估算" : ""}
              </dd>
              {node.data.details.map(([key, value], i) => (
                <div key={i}>
                  <dt>{key}</dt>
                  <dd>{value}</dd>
                </div>
              ))}
            </dl>
            <p className="demo-detail-scope">{explanation.caution}</p>
          </>
        )}
        {tab === "tensor" && (
          <>
            {tensors.length ? (
              <>
                <label className="demo-tensor-picker">
                  输入 / 输出
                  <select
                    aria-label="选择解释张量"
                    value={Math.min(tensorIndex, tensors.length - 1)}
                    onChange={(e) => {
                      setTensorIndex(Number(e.target.value));
                      setRowStart(0);
                      setColStart(0);
                      setCell(null);
                    }}
                  >
                    {tensors.map((t, i) => (
                      <option key={i} value={i}>
                        {i < node.data.inputs.length ? "输入" : "输出"} ·{" "}
                        {t.name}
                      </option>
                    ))}
                  </select>
                </label>
                <code className="demo-detail-formula">
                  [{tensor.shape.join(", ")}] · {tensor.dtype}
                </code>
                <p className="demo-slice-description">
                  行轴：{tensor.symbols.at(-2) ?? "1"}；列轴：
                  {tensor.symbols.at(-1) ?? "1"}。
                  {tensor.shape.length > 2 &&
                    ` 固定轴：${tensor.symbols
                      .slice(0, -2)
                      .map((name, i) => `${name}=${cellIndices(0, 0)[i]}`)
                      .join(", ")}`}
                </p>
                <div className="demo-matrix-pages">
                  <label>
                    起始行
                    <input
                      type="number"
                      aria-label="矩阵起始行"
                      min={0}
                      max={Math.max(0, volume.rows - 1)}
                      value={pageRow}
                      onChange={(e) => {
                        const n = Number(e.target.value);
                        if (
                          Number.isInteger(n) &&
                          n >= 0 &&
                          n < volume.rows
                        ) {
                          setRowStart(n);
                          setCell(null);
                        }
                      }}
                    />
                  </label>
                  <label>
                    起始列
                    <input
                      type="number"
                      aria-label="矩阵起始列"
                      min={0}
                      max={Math.max(0, volume.columns - 1)}
                      value={pageColumn}
                      onChange={(e) => {
                        const n = Number(e.target.value);
                        if (
                          Number.isInteger(n) &&
                          n >= 0 &&
                          n < volume.columns
                        ) {
                          setColStart(n);
                          setCell(null);
                        }
                      }}
                    />
                  </label>
                </div>
                <div
                  className="demo-matrix"
                  role="group"
                  aria-label="张量维度网格"
                  style={{
                    gridTemplateColumns: `repeat(${pageColumns || 1},1fr)`,
                  }}
                >
                  {Array.from(
                    {
                      length: pageRows * pageColumns,
                    },
                    (_, i) => {
                      const row = pageRow + Math.floor(i / pageColumns),
                        col = pageColumn + (i % pageColumns),
                        value = cellValue(row, col);
                      return (
                        <button
                          key={i}
                          aria-label={`行 ${row} 列 ${col}`}
                          onMouseEnter={() => setCell([row, col])}
                          onFocus={() => setCell([row, col])}
                          onClick={() => setCell([row, col])}
                          style={
                            value !== null
                              ? {
                                  background: recordedValueColor(value),
                                }
                              : undefined
                          }
                          className={
                            cell?.[0] === row && cell?.[1] === col
                              ? "selected"
                              : ""
                          }
                        />
                      );
                    },
                  )}
                </div>
                <p>
                  行 {volume.rowSymbol}={volume.rows.toLocaleString()} · 列{" "}
                  {volume.columnSymbol}={volume.columns.toLocaleString()}
                  <br />
                  {volume.sliceSymbol}={volume.slices.toLocaleString()} 个切片
                  · {formatBytes(tensorBytes(tensor))}
                </p>
                <div className="demo-cell-readout">
                  {cell
                    ? `坐标 [${cellIndices(cell[0], cell[1]).join(", ")}] · 行 ${cell[0]} · 列 ${cell[1]}`
                    : "悬停或点选单元格"}
                  <br />
                  {tensor?.role === "probabilities"
                    ? "注意力权重"
                    : tensor?.role === "scores"
                      ? "注意力分数"
                      : "激活值"}
                  ：
                  {cell && cellValue(cell[0], cell[1]) !== null
                    ? cellValue(cell[0], cell[1])!.toPrecision(6)
                    : traceSource
                      ? "此坐标未记录"
                      : "未加载运行数据"}
                </div>
                <p className="demo-value-legend">
                  记录颜色：绿为正值，红为负值，浅灰为零，深灰为未记录；颜色不跨采样坐标插值。
                </p>
                <p>
                  网格仅抽取最多 10×16 个可视位置。
                  {traceSource
                    ? `数值来源：${traceSource}（导入记录）`
                    : "未导入运行数据时仅显示维度。"}
                </p>
                {tensor.note && <p>{tensor.note}</p>}
              </>
            ) : (
              <p>此节点没有张量输出。</p>
            )}
            <button className="demo-detail-action" onClick={onMatrix}>
              <Grid2X2 size={14} />在 2D 中查看当前组件
            </button>
          </>
        )}
        {tab === "lab" && (
          <>
            <span className="demo-example-label">
              独立数学示例 · 不代表模型激活
            </span>
            <label className="demo-lab-input">
              输入 x
              <input
                type="range"
                aria-label="数学示例输入"
                min="-4"
                max="4"
                step="0.1"
                value={x}
                onChange={(e) => setX(Number(e.target.value))}
              />
              <b>{x.toFixed(1)}</b>
            </label>
            {explanation.lab === "softmax" ||
            explanation.lab === "attention" ? (
              <>
                <h3>Softmax：分数 → 权重</h3>
                <p>
                  以 [{values.map((v) => v.toFixed(1)).join(", ")}]
                  为例，减去最大值后归一化。
                </p>
                <div className="demo-probabilities">
                  {probabilities.map((p, i) => (
                    <div key={i}>
                      <span>{i + 1}</span>
                      <i style={{ width: `${p * 100}%` }} />
                      <b>{(p * 100).toFixed(1)}%</b>
                    </div>
                  ))}
                </div>
                <code>
                  Σ weights ={" "}
                  {probabilities.reduce((a, b) => a + b, 0).toFixed(4)}
                </code>
                <h3>因果可见性</h3>
                <div
                  className="demo-causal-grid"
                  style={{
                    gridTemplateColumns: `repeat(${tokenLabels.length},1fr)`,
                  }}
                >
                  {tokenLabels.flatMap((_, q) =>
                    tokenLabels.map((_, k) => (
                      <button
                        key={`${q}:${k}`}
                        aria-label={`Query ${q + 1}, Key ${k + 1}: ${k <= q ? "可访问" : "已遮挡"}`}
                        className={k <= q ? "allowed" : "masked"}
                        onClick={() => onToken(q)}
                      >
                        {k <= q ? "●" : "×"}
                      </button>
                    )),
                  )}
                </div>
                <p>
                  常规因果注意力只允许 Key 位置 ≤ Query
                  位置；滑动窗口及稀疏选择还会进一步限制范围。当前 head
                  的真实分数需要激活记录。
                </p>
              </>
            ) : explanation.lab === "norm" ? (
              <>
                <h3>
                  {/rms/i.test(node.data.title) ? "RMSNorm" : "LayerNorm"}{" "}
                  示例（scale=1）
                </h3>
                <p>输入 [{values.map((v) => v.toFixed(2)).join(", ")}]</p>
                <code className="demo-detail-formula">
                  输出 [{norm.map((v) => v.toFixed(3)).join(", ")}]
                </code>
                <p>ε={model.normEps}，数值由上方输入现场计算。</p>
              </>
            ) : (
              <>
                <h3>{model.activation} 激活曲线</h3>
                <svg
                  className="demo-activation-chart"
                  viewBox="0 0 300 160"
                  role="img"
                  aria-label="激活函数曲线"
                >
                  <line x1="15" y1="125" x2="285" y2="125" />
                  <line x1="150" y1="15" x2="150" y2="145" />
                  <polyline
                    points={Array.from({ length: 81 }, (_, i) => {
                      const v = (i - 40) / 10;
                      return `${150 + v * 32},${125 - activationValue(v, model.activation) * 25}`;
                    }).join(" ")}
                  />
                  <circle
                    cx={150 + x * 32}
                    cy={125 - activationValue(x, model.activation) * 25}
                    r="4"
                  />
                </svg>
                <code>
                  f({x.toFixed(1)}) ={" "}
                  {activationValue(x, model.activation).toFixed(4)}
                </code>
                <p>此曲线是独立函数示例；不需要模型权重。</p>
              </>
            )}
          </>
        )}
        {devMode && (
          <details className="demo-dev-data">
            <summary>开发数据 · 结构元信息</summary>
            <pre>
              {JSON.stringify(
                {
                  path: node.data.path,
                  layer: node.data.layerIndex,
                  inputs: node.data.inputs,
                  outputs: node.data.outputs,
                  activationValues: null,
                },
                null,
                2,
              )}
            </pre>
          </details>
        )}
      </div>
      <footer>
        <button onClick={() => void copy()}>
          <Copy size={14} />
          {copied ? "已复制" : "复制当前上下文"}
        </button>
      </footer>
    </aside>
  );
}
