import type { ModelConfig } from "./model";
import type { DemoEvent } from "./inference-demo";
import { layerSignature } from "./architecture";

export type DemoRoute = "overview" | "full";
export const routeLabels = { overview: "代表层讲解", full: "完整逐层" };
/** Keep real event identities; omitted layers are explicitly presented as skipped. */
export function teachingPlan(
  model: ModelConfig,
  full: DemoEvent[],
  pass: number,
): DemoEvent[] {
  const layerEvents = full.filter(
    (e) => e.view !== "overview" && e.loop === 0,
  );
  const firstLayer = layerEvents[0]?.layer;
  const representative = layerEvents.filter((e) => e.layer === firstLayer);
  const selected = new Set<DemoEvent>();
  // Preserve semantic checkpoints before reducing repeated transformations.
  const detail =
    pass === 0
      ? representative
      : representative.filter((e) => e.view === "layer");
  const limit = pass === 0 ? 28 : 5;
  const priority = (event: DemoEvent, index: number) => {
    if (index === 0 || index === detail.length - 1) return 0;
    if (
      event.isLayerEnd ||
      (event.view === "layer" && event.category === "operation")
    )
      return 1;
    if (
      [
        "scores",
        "softmax",
        "softmax-values",
        "attention-core",
        "sparse-attn",
        "kv-cache",
        "router",
        "state-update",
      ].includes(event.nodeId)
    )
      return 2;
    if (event.category === "norm") return 3;
    return 4;
  };
  detail
    .map((event, index) => ({ event, index }))
    .sort(
      (a, b) =>
        priority(a.event, a.index) - priority(b.event, b.index) ||
        a.index - b.index,
    )
    .slice(0, limit)
    .forEach(({ event }) => selected.add(event));
  const seen = new Set(
    firstLayer === undefined ? [] : [layerSignature(model, firstLayer)],
  );
  let summaries = 0;
  for (const event of layerEvents) {
    const signature = layerSignature(model, event.layer);
    if (event.isLayerEnd && !seen.has(signature) && summaries < 3) {
      selected.add(event);
      seen.add(signature);
      summaries++;
    }
  }
  const boundaries = full.filter((e) => e.view === "overview" && !e.sample);
  for (const event of [...boundaries.slice(0, 2), ...boundaries.slice(-2)])
    selected.add(event);
  const last = full.at(-1);
  if (last) selected.add(last);
  return full.filter((e) => selected.has(e));
}
