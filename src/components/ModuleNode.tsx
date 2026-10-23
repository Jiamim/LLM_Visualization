import ArchitectureFrame from "./ArchitectureFrame";
import { memo } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";
import {
  ArrowDownToLine,
  ArrowUpRight,
  Box,
  Braces,
  CirclePlus,
  Layers3,
  Network,
  Sigma,
  Sparkles,
} from "lucide-react";
import type { ModuleNode as ModuleNodeType } from "../lib/graph";
import { compactNumber } from "../lib/model";
import { categories } from "../lib/presentation";

const icons = {
  input: Braces,
  embedding: Box,
  block: Layers3,
  attention: Network,
  norm: Sigma,
  mlp: Sparkles,
  output: ArrowDownToLine,
  operation: CirclePlus,
};

export default memo(function ModuleNode({
  data,
  selected,
}: NodeProps<ModuleNodeType>) {
  if (data.frame) return <ArchitectureFrame data={data} />;
  const Icon = icons[data.category];
  const output = data.outputs[0];
  return (
    <div
      className={`module-node category-${data.category} ${data.residual ? "residual-node" : ""} ${selected ? "is-selected" : ""} ${data.active ? "is-active" : ""} ${data.inactive ? "is-inactive" : ""}`}
    >
      <Handle type="target" position={Position.Top} id="in" />
      <Handle type="source" position={Position.Bottom} id="out" />
      <Handle
        type="target"
        position={Position.Left}
        id="side-in"
        className="side-handle"
      />
      <Handle
        type="source"
        position={Position.Left}
        id="side-out"
        className="side-handle"
      />
      <Handle
        type="source"
        position={Position.Right}
        id="right-out"
        className="side-handle"
      />
      <Handle
        type="target"
        position={Position.Right}
        id="right-in"
        className="side-handle"
      />
      <div className="node-kicker">
        <span>
          {data.residual
            ? "RESIDUAL · 残差"
            : categories[data.category].label}
        </span>
        <span>{String(data.order).padStart(2, "0")}</span>
      </div>
      <div className="node-heading">
        <span className="node-icon">
          <Icon size={17} strokeWidth={1.7} />
        </span>
        <div className="node-titles">
          <strong>{data.title}</strong>
          <small>{data.subtitle}</small>
        </div>
        {data.drill && (
          <button
            className="node-drill nodrag"
            title={`展开 ${data.title}`}
            aria-label={`展开 ${data.title}`}
            onClick={(event) => {
              event.stopPropagation();
              data.onDrill?.();
            }}
          >
            <ArrowUpRight size={15} />
          </button>
        )}
      </div>
      <div className="node-footer">
        <code title={output ? `[${output.shape.join(", ")}]` : ""}>
          {output
            ? (data.symbolic ? output.symbols : output.shape).join(" × ")
            : "—"}
        </code>
        <span>
          {data.badge ??
            (data.parameterStatus === "unknown"
              ? "参数未细分"
              : data.params > 0
                ? compactNumber(data.params)
                : output?.dtype)}
        </span>
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
