import ArchitectureFrame from "./ArchitectureFrame";
import { memo, type CSSProperties } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";
import { ArrowUpRight, Layers3 } from "lucide-react";
import type { ModuleNode } from "../lib/graph";
import { categories } from "../lib/presentation";
import { tensorLabel, tensorVolume } from "../lib/tensor-layout";

export default memo(function TensorNode({
  data,
  selected,
}: NodeProps<ModuleNode>) {
  if (data.frame) return <ArchitectureFrame data={data} />;
  const tensor = data.outputs[0] ?? data.inputs[0];
  const volume = tensorVolume(tensor);
  const color = categories[data.category].color;
  return (
    <div
      className={`tensor-node ${data.residual ? "residual-node" : ""} ${selected ? "is-selected" : ""} ${data.active ? "is-active" : ""} ${data.inactive ? "is-inactive" : ""}`}
      style={{ "--tensor-color": color } as CSSProperties}
    >
      <Handle type="target" position={Position.Top} id="in" />
      <Handle type="source" position={Position.Bottom} id="out" />
      <Handle type="source" position={Position.Left} id="side-out" />
      <Handle type="target" position={Position.Left} id="side-in" />
      <Handle type="source" position={Position.Right} id="right-out" />
      <Handle type="target" position={Position.Right} id="right-in" />
      <div className="tensor-node-heading">
        <span>
          {String(data.order ?? 1).padStart(2, "0")} ·{" "}
          {data.residual ? "RESIDUAL · 残差" : categories[data.category].label}
        </span>
        {data.drill && (
          <button
            className="nodrag"
            aria-label={`展开 ${data.title}`}
            onClick={(e) => {
              e.stopPropagation();
              data.onDrill?.();
            }}
          >
            <ArrowUpRight size={16} />
          </button>
        )}
      </div>
      <strong>{data.title}</strong>
      <div
        className="tensor-matrix"
        aria-label={
          volume.empty
            ? "空张量"
            : `${tensor?.name ?? "张量"}，${volume.rows} 行 × ${volume.columns} 列，${volume.slices} 个切片`
        }
      >
        <div className="matrix-row-axis">
          {volume.rowSymbol}
          <b>{volume.rows.toLocaleString()}</b>
        </div>
        <svg
          viewBox="0 0 220 78"
          role="img"
          aria-label="张量形状示意，不显示激活数值"
        >
          {volume.empty ? (
            <text x="110" y="42" textAnchor="middle">
              空张量 · 0 个元素
            </text>
          ) : (
            <>
              {volume.slices > 1 && (
                <>
                  <rect
                    className="matrix-stack"
                    x="14"
                    y="1"
                    width="197"
                    height="63"
                    rx="3"
                  />
                  <rect
                    className="matrix-stack"
                    x="9"
                    y="6"
                    width="197"
                    height="63"
                    rx="3"
                  />
                </>
              )}
              <rect
                className="matrix-plane"
                x="4"
                y="11"
                width="197"
                height="63"
                rx="3"
              />
              {Array.from(
                { length: Math.max(0, volume.gridColumns - 1) },
                (_, i) => (
                  <line
                    key={`c${i}`}
                    x1={4 + (197 * (i + 1)) / volume.gridColumns}
                    x2={4 + (197 * (i + 1)) / volume.gridColumns}
                    y1="11"
                    y2="74"
                  />
                ),
              )}
              {Array.from(
                { length: Math.max(0, volume.gridRows - 1) },
                (_, i) => (
                  <line
                    key={`r${i}`}
                    y1={11 + (63 * (i + 1)) / volume.gridRows}
                    y2={11 + (63 * (i + 1)) / volume.gridRows}
                    x1="4"
                    x2="201"
                  />
                ),
              )}
            </>
          )}
        </svg>
        <div className="matrix-column-axis">
          {volume.columnSymbol} <b>{volume.columns.toLocaleString()}</b>
        </div>
      </div>
      <div className="tensor-slices">
        <Layers3 size={12} />
        <span>
          {volume.sliceSymbol} = {volume.slices.toLocaleString()} 个切片
        </span>
        <small>{tensor?.dtype}</small>
      </div>
      <div className="tensor-node-shape">
        <span title={tensor?.name}>
          {tensor?.name ?? "模块接口"}
          {data.outputs.length > 1 ? ` · 1/${data.outputs.length}` : ""}
        </span>
        <code>{tensorLabel(tensor, !!data.symbolic)}</code>
      </div>
      {data.repeat && (
        <span
          className="node-repeat-badge"
          title="重复的层结构；不代表共享权重"
        >
          × {data.repeat}
        </span>
      )}
    </div>
  );
});
