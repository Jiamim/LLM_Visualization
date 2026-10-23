import { useEffect, useRef, useState } from "react";
import {
  ArrowDown,
  ArrowUpRight,
  Box,
  Check,
  ChevronRight,
  Copy,
  Info,
  X,
} from "lucide-react";
import type { ModuleData, View } from "../lib/graph";
import {
  compactNumber,
  formatBytes,
  formatNumber,
  tensorBytes,
  type Tensor,
} from "../lib/model";
import { dimensionNames } from "../lib/presentation";

function TensorCard({
  tensor,
  direction,
  output = false,
}: {
  tensor: Tensor;
  direction: string;
  output?: boolean;
}) {
  return (
    <div className={`tensor-card ${output ? "tensor-output" : ""}`}>
      <div className="tensor-caption">
        <span>
          <i />
          {direction}
        </span>
        <code>{tensor.dtype}</code>
      </div>
      <strong className="tensor-name">{tensor.name}</strong>
      <div className="tensor-dimensions">
        {tensor.shape.map((dimension, i) => (
          <div className="dimension-pair" key={i}>
            {i > 0 && <span className="dim-times">×</span>}
            <span
              className={`dimension axis-${tensor.symbols[i]}`}
              title={`${tensor.symbols[i]} · ${dimensionNames[tensor.symbols[i]] ?? tensor.symbols[i]}`}
            >
              <b>{formatNumber(dimension)}</b>
              <small>{tensor.symbols[i]}</small>
            </span>
          </div>
        ))}
      </div>
      <div className="tensor-size">
        <span>
          {formatNumber(tensor.shape.reduce((a, b) => a * b, 1))} 个元素
        </span>
        <strong>{formatBytes(tensorBytes(tensor))}</strong>
      </div>
      {tensor.note && <p className="tensor-note">{tensor.note}</p>}
    </div>
  );
}

export default function Inspector({
  data,
  onDrill,
  selectedEdge,
  onClose,
  selectionKey,
}: {
  data: ModuleData | null;
  onDrill: (view: View) => void;
  selectedEdge?: Tensor;
  onClose: () => void;
  selectionKey: string;
}) {
  const [tab, setTab] = useState<"tensors" | "details">("tensors");
  const [copyState, setCopyState] = useState<"idle" | "done" | "error">(
    "idle",
  );
  const scroll = useRef<HTMLDivElement>(null);
  useEffect(() => {
    setTab("tensors");
    setCopyState("idle");
    scroll.current?.scrollTo(0, 0);
  }, [selectionKey]);
  useEffect(() => {
    if (copyState === "idle") return;
    const timer = window.setTimeout(() => setCopyState("idle"), 2200);
    return () => clearTimeout(timer);
  }, [copyState]);
  const tensors = selectedEdge
    ? [selectedEdge]
    : [...(data?.inputs ?? []), ...(data?.outputs ?? [])];
  const symbols = [...new Set(tensors.flatMap((t) => t.symbols))];
  const copy = async () => {
    const text = tensors
      .map((t) => `${t.name}: [${t.shape.join(", ")}] (${t.dtype})`)
      .join("\n");
    try {
      if (window.isSecureContext && navigator.clipboard)
        await navigator.clipboard.writeText(text);
      else {
        const input = document.createElement("textarea");
        input.value = text;
        input.style.cssText = "position:fixed;opacity:0";
        (scroll.current ?? document.body).append(input);
        input.select();
        const copied = document.execCommand("copy");
        input.remove();
        if (!copied) throw new Error("copy failed");
      }
      setCopyState("done");
    } catch {
      setCopyState("error");
    }
  };
  return (
    <aside className="inspector" aria-label="模块检查器">
      <div className="panel-title">
        <span>
          <Box size={16} />
          {selectedEdge ? "张量详情" : "模块详情"}
        </span>
        <button
          className="icon-button"
          aria-label="收起详情面板"
          title="收起详情面板"
          onClick={onClose}
        >
          <X size={17} />
        </button>
      </div>
      <div className="inspector-scroll" ref={scroll}>
        {data || selectedEdge ? (
          <>
            <div className="inspector-heading">
              <span
                className={`category-label category-${data?.category ?? "input"}`}
              >
                {selectedEdge ? "连接张量" : data?.subtitle || data?.category}
              </span>
              <h2>{selectedEdge?.name ?? data?.title}</h2>
              {data?.inactive && (
                <p className="ced-small-note">
                  此分支在当前输入场景不执行；以下显示该模块的示例形状。
                </p>
              )}
              {data && !selectedEdge && (
                <code className="module-path">{data.path}</code>
              )}
            </div>
            {data?.drill && !selectedEdge && (
              <button
                className="inspect-drill"
                onClick={() => onDrill(data.drill!)}
              >
                进入模块内部
                <ArrowUpRight size={17} />
              </button>
            )}
            {!selectedEdge && (
              <div
                className="inspector-tabs"
                role="tablist"
                aria-label="模块信息"
                onKeyDown={(event) => {
                  if (
                    ["ArrowLeft", "ArrowRight", "Home", "End"].includes(
                      event.key,
                    )
                  ) {
                    event.preventDefault();
                    const next =
                      event.key === "Home"
                        ? "tensors"
                        : event.key === "End"
                          ? "details"
                          : tab === "tensors"
                            ? "details"
                            : "tensors";
                    setTab(next);
                    document
                      .getElementById(
                        next === "tensors" ? "tensor-tab" : "details-tab",
                      )
                      ?.focus();
                  }
                }}
              >
                <button
                  id="tensor-tab"
                  role="tab"
                  aria-selected={tab === "tensors"}
                  tabIndex={tab === "tensors" ? 0 : -1}
                  aria-controls="inspector-content"
                  onClick={() => setTab("tensors")}
                >
                  张量维度
                  <span>{data!.inputs.length + data!.outputs.length}</span>
                </button>
                <button
                  id="details-tab"
                  role="tab"
                  aria-selected={tab === "details"}
                  tabIndex={tab === "details" ? 0 : -1}
                  aria-controls="inspector-content"
                  onClick={() => setTab("details")}
                >
                  计算与参数
                </button>
              </div>
            )}
            <div
              id="inspector-content"
              role="tabpanel"
              aria-labelledby={
                selectedEdge
                  ? undefined
                  : tab === "tensors"
                    ? "tensor-tab"
                    : "details-tab"
              }
            >
              {tab === "tensors" || selectedEdge ? (
                <>
                  <div className="inspector-section-title">
                    <span>输入 → 输出</span>
                    <button
                      className="copy-button"
                      onClick={() => void copy()}
                      aria-label="复制张量维度"
                    >
                      {copyState === "done" ? (
                        <Check size={13} />
                      ) : (
                        <Copy size={13} />
                      )}
                      {copyState === "done" ? "已复制" : "复制"}
                    </button>
                  </div>
                  {copyState === "error" && (
                    <p role="status" className="error-message">
                      复制失败，可在卡片中选中文字复制。
                    </p>
                  )}
                  {selectedEdge ? (
                    <TensorCard tensor={selectedEdge} direction="连接张量" />
                  ) : (
                    <>
                      {data!.inputs.map((tensor, i) => (
                        <TensorCard
                          key={`in-${i}`}
                          tensor={tensor}
                          direction={`输入${data!.inputs.length > 1 ? ` ${i + 1}` : ""}`}
                        />
                      ))}
                      {data!.inputs.length > 0 &&
                        data!.outputs.length > 0 && (
                          <div className="tensor-arrow">
                            <ArrowDown size={17} />
                            <span>
                              {data!.inputs.length === 1 &&
                              data!.outputs.length === 1 &&
                              data!.inputs[0].shape.join() ===
                                data!.outputs[0].shape.join()
                                ? "形状保持不变"
                                : "维度变换"}
                            </span>
                          </div>
                        )}
                      {data!.outputs.map((tensor, i) => (
                        <TensorCard
                          key={`out-${i}`}
                          tensor={tensor}
                          direction={`输出${data!.outputs.length > 1 ? ` ${i + 1}` : ""}`}
                          output
                        />
                      ))}
                    </>
                  )}
                  <div className="dimension-key">
                    <div className="inspector-section-title">维度释义</div>
                    <dl>
                      {symbols.map((symbol) => (
                        <div key={symbol}>
                          <dt>{symbol}</dt>
                          <dd>{dimensionNames[symbol] ?? symbol}</dd>
                        </div>
                      ))}
                    </dl>
                  </div>
                </>
              ) : (
                <>
                  <p className="module-description">{data!.description}</p>
                  {data!.parameterStatus === "unknown" && (
                    <p>模块参数量：未细分，不能视为零参数。</p>
                  )}
                  {data!.params > 0 && (
                    <div className="parameter-summary">
                      <span>模块参数量</span>
                      <strong>{compactNumber(data!.params)}</strong>
                      <small>{formatNumber(data!.params)} parameters</small>
                    </div>
                  )}
                  {data!.formula && (
                    <>
                      <div className="inspector-section-title">
                        计算表达式
                      </div>
                      <div className="formula">{data!.formula}</div>
                    </>
                  )}
                  {data!.details.length > 0 && (
                    <>
                      <div className="inspector-section-title">模块配置</div>
                      <dl className="config-details">
                        {data!.details.map(([key, value]) => (
                          <div key={key}>
                            <dt>{key}</dt>
                            <dd>{value}</dd>
                          </div>
                        ))}
                      </dl>
                    </>
                  )}
                </>
              )}
            </div>
          </>
        ) : (
          <div className="inspector-empty">
            <span className="empty-illustration">
              <Box size={34} />
            </span>
            <h3>从一个模块开始</h3>
            <p>点选图中的模块或连线，查看输入输出维度和计算过程。</p>
            <span>
              选择模块
              <ChevronRight size={14} />
              查看张量
              <ChevronRight size={14} />
              深入内部
            </span>
          </div>
        )}
        <div className="inspector-footnote">
          <Info size={14} />
          <p>根据配置静态推导。内存为逻辑张量大小，不代表实际峰值显存。</p>
        </div>
      </div>
    </aside>
  );
}
