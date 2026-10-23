import type { DemoEvent } from "./inference-demo";
import type { ModuleData } from "./graph";

export type DemoPace = "guided" | "continuous" | "step";
export const paceLabels: Record<DemoPace, string> = {
  guided: "慢速讲解",
  continuous: "连续浏览",
  step: "逐步学习",
};

/** Pedagogical reading time, independent of runtime FLOPs or measured latency. */
export function demoStepDuration(
  event: Pick<DemoEvent, "nodeId" | "sample" | "durationMs">,
  data: ModuleData | undefined,
  pace: DemoPace,
  speed = 1,
): number {
  let ms = 5200;
  if (data?.category === "input") ms = 4200;
  if (data?.category === "embedding") ms = 6200;
  if (data?.category === "norm") ms = 6400;
  if (data?.residual) ms = 7400;
  if (
    /scores|softmax|attention.core|sparse.attn|state.update|router|dispatch/.test(
      event.nodeId,
    )
  )
    ms = 8200;
  if (event.sample) ms = 7000;
  ms = event.durationMs ?? ms;
  const modeScale = pace === "continuous" ? 0.48 : 1;
  return (ms * modeScale) / Math.max(0.25, Math.min(4, speed));
}

/** Arrival/read hold → input → compute → output/read hold. */
export function demoBeat(progress: number) {
  const p = Math.max(0, Math.min(1, progress));
  return {
    phase: p < 0.3 ? 0 : p < 0.76 ? 1 : 2,
    arrival: p < 0.13,
    transfer:
      p < 0.15
        ? 0
        : p < 0.33
          ? (p - 0.15) / 0.18
          : p < 0.76
            ? 1
            : Math.min(1, (p - 0.76) / 0.16),
    computing: Math.max(0, Math.min(1, (p - 0.33) / 0.43)),
    output: p >= 0.76,
    hold: p >= 0.92,
  };
}
export const demoPhaseStarts = [0, 0.3, 0.76] as const;
export function stageLabel(event: DemoEvent) {
  return event.sample
    ? "输出续写"
    : event.view === "overview"
      ? "输入与输出"
      : event.view === "attention"
        ? "注意力计算"
        : event.view === "mlp"
          ? "前馈与专家"
          : "主干与残差";
}

/** Duration is captured when the plan is built; changing speed never rebuilds graphs. */
export function demoPlanDuration(
  events: DemoEvent[],
  pace: DemoPace,
  speed = 1,
): number {
  return events.reduce(
    (sum, event) => sum + demoStepDuration(event, undefined, pace, speed),
    0,
  );
}
