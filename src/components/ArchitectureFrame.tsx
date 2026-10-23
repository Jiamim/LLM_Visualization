import type { ModuleData } from "../lib/graph";
export default function ArchitectureFrame({ data }: { data: ModuleData }) {
  const frame = data.frame!;
  return (
    <div
      className={`architecture-frame ${frame.period > 1 ? "architecture-cycle" : ""}`}
      style={{ width: frame.width, height: frame.height }}
      aria-label={`${data.title}，重复 ${frame.count} 次，Layer ${frame.range[0]} 至 ${frame.range[1]}`}
    >
      <div className="architecture-frame-title">
        <strong>{data.title}</strong>
        <span>
          L{frame.range[0]}–L{frame.range[1]}
        </span>
      </div>
      <div className="architecture-repeat-bracket">
        <strong>× {frame.count}</strong>
        <small>
          {frame.period > 1 ? `${frame.period} 层 / 周期` : "相同结构"}
        </small>
      </div>
    </div>
  );
}
