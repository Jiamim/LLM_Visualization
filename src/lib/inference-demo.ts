import { MarkerType } from "@xyflow/react";
import { buildGraph, type ModelGraph, type View } from "./graph";
import { cacheReport } from "./cache";
import { demoStepDuration } from "./demo-pacing";
import { hasLMHead, type ModelConfig, type Scenario } from "./model";

export interface DemoSettings {
  promptTokens: number;
  outputTokens: number;
  useCache: boolean;
  speed: number;
}
export interface DemoEvent {
  view: View;
  layer: number;
  nodeId: string;
  segment: string;
  title: string;
  category: string;
  loop: number;
  instance?: string;
  durationMs?: number;
  isLayerEnd?: boolean;
  sample?: boolean;
}
export interface DemoState {
  status: "ready" | "running" | "paused" | "complete";
  pass: number;
  index: number;
  generated: number;
  progress: number;
  run: number;
}
export type DemoAction =
  | { type: "start" }
  | { type: "pause" }
  | { type: "restart" }
  | { type: "reset" }
  | { type: "advance"; steps: number; passes: number }
  | { type: "progress"; value: number }
  | { type: "set-progress"; value: number }
  | { type: "seek"; index: number }
  | { type: "next-pass"; passes: number; steps?: number }
  | { type: "replay"; pass: number };
export const initialDemoState: DemoState = {
  status: "ready",
  pass: 0,
  index: 0,
  generated: 0,
  progress: 0,
  run: 0,
};
export function demoReducer(state: DemoState, action: DemoAction): DemoState {
  switch (action.type) {
    case "set-progress":
      return {
        ...state,
        status: "paused",
        progress: Math.min(1, Math.max(0, action.value)),
      };
    case "start":
      return state.status === "complete"
        ? { ...initialDemoState, status: "running", run: state.run + 1 }
        : { ...state, status: "running" };
    case "pause":
      return state.status === "running"
        ? { ...state, status: "paused" }
        : state;
    case "restart":
      return { ...initialDemoState, status: "running", run: state.run + 1 };
    case "reset":
      return { ...initialDemoState, run: state.run + 1 };
    case "progress":
      return state.status === "running"
        ? { ...state, progress: Math.max(0, Math.min(1, action.value)) }
        : state;
    case "seek":
      return {
        ...state,
        index: action.index,
        generated: state.pass,
        progress: 0,
        status: "paused",
      };
    case "replay":
      return {
        ...state,
        pass: action.pass,
        index: 0,
        generated: action.pass,
        progress: 0,
        status: "paused",
      };
    case "next-pass": {
      if (state.status === "complete") return state;
      const done = state.pass + 1 >= action.passes;
      return {
        ...state,
        pass: done ? state.pass : state.pass + 1,
        index: done ? (action.steps ? action.steps - 1 : state.index) : 0,
        generated: state.pass + 1,
        progress: 0,
        status: done
          ? "complete"
          : state.status === "running"
            ? "running"
            : "paused",
      };
    }
    case "advance":
      if (state.status === "complete") return state;
      if (state.index + 1 < action.steps)
        return {
          ...state,
          index: state.index + 1,
          progress: 0,
          status: state.status === "ready" ? "paused" : state.status,
        };
      return demoReducer(state, { type: "next-pass", passes: action.passes });
  }
}
export function demoHasHead(m: ModelConfig) {
  return m.catalog ? m.catalog.head : hasLMHead(m);
}
export function demoScenario(
  settings: DemoSettings,
  pass: number,
  dtype: Scenario["dtype"],
): Scenario {
  return {
    batch: 1,
    sequence:
      pass > 0 && settings.useCache ? 1 : settings.promptTokens + pass,
    past:
      pass > 0 && settings.useCache ? settings.promptTokens + pass - 1 : 0,
    dtype,
    cedExecution: "reference",
  };
}
export function demoSettingsValid(m: ModelConfig, s: DemoSettings): boolean {
  return (
    Number.isInteger(s.promptTokens) &&
    s.promptTokens >= 1 &&
    s.promptTokens <= 16 &&
    Number.isInteger(s.outputTokens) &&
    s.outputTokens >= 1 &&
    s.outputTokens <= 8 &&
    s.speed >= 0.25 &&
    s.speed <= 4 &&
    (m.maxPosition === null ||
      s.promptTokens + s.outputTokens - 1 <= m.maxPosition)
  );
}
/** Replay every real layer, using the adapter's ordering including residual and state operations. */
export function buildDemoPlan(m: ModelConfig, s: Scenario): DemoEvent[] {
  const events: DemoEvent[] = [];
  const append = (view: View, layer: number, loop = 0) => {
    const graph = buildGraph(m, s, view, layer);
    for (const node of graph.nodes) {
      if (node.data.inactive || node.data.frame) continue;
      if (
        view === "layer" &&
        (node.data.drill === "attention" || node.data.drill === "mlp")
      ) {
        const detailView = node.data.drill;
        const detailed = buildGraph(m, s, detailView, layer);
        for (const child of detailed.nodes)
          if (!child.data.inactive && !child.data.frame)
            events.push({
              view: detailView,
              layer,
              nodeId: child.id,
              segment: `${loop}:${layer}:${detailView}:${node.id}`,
              title: child.data.title,
              category: child.data.category,
              loop,
              instance: node.id,
              durationMs: demoStepDuration(
                { nodeId: child.id },
                child.data,
                "guided",
              ),
            });
        // Return to the layer graph before its residual merge.
        events.push({
          view,
          layer,
          nodeId: node.id,
          segment: `${loop}:${layer}:return:${node.id}`,
          durationMs: demoStepDuration(
            { nodeId: node.id },
            node.data,
            "guided",
          ),
          title: node.data.title,
          category: node.data.category,
          loop,
        });
      } else
        events.push({
          view,
          layer,
          nodeId: node.id,
          segment: `${loop}:${layer}:${view}`,
          durationMs: demoStepDuration(
            { nodeId: node.id },
            node.data,
            "guided",
          ),
          title: node.data.title,
          category: node.data.category,
          loop,
        });
    }
  };
  const overview = buildGraph(m, s, "overview", 0),
    loops = m.catalog?.loops ?? 1;
  for (const node of overview.nodes) {
    if (node.data.inactive || node.data.frame) continue;
    if (
      node.data.category === "block" &&
      ["encoder", "decoder"].includes(node.id)
    ) {
      const start =
        node.id === "decoder" && m.deepseek ? m.deepseek.encoderLayers : 0;
      const end =
        node.id === "encoder" && m.deepseek
          ? m.deepseek.encoderLayers
          : m.layers;
      for (let loop = 0; loop < loops; loop++)
        for (let layer = start; layer < end; layer++) {
          append("layer", layer, loop);
          if (events.length) events[events.length - 1].isLayerEnd = true;
        }
    } else
      events.push({
        view: "overview",
        layer: 0,
        nodeId: node.id,
        segment: `overview:${node.id}`,
        durationMs: demoStepDuration(
          { nodeId: node.id },
          node.data,
          "guided",
        ),
        title: node.data.title,
        category: node.data.category,
        loop: 0,
      });
  }
  if (demoHasHead(m))
    events.push({
      view: "overview",
      layer: m.layers - 1,
      nodeId: "demo-sample",
      segment: "sample",
      title: "选择下一个示例 token",
      category: "output",
      loop: loops - 1,
      sample: true,
      durationMs: 7000,
    });
  return events;
}
export function demoGraph(
  m: ModelConfig,
  s: Scenario,
  event: DemoEvent,
  symbolic = false,
): ModelGraph {
  if (!event.sample) {
    const graph = buildGraph(m, s, event.view, event.layer, symbolic);
    for (const node of graph.nodes) {
      node.data.layerIndex ??= event.layer;
      node.data.sceneView = event.view;
      node.data.sceneSourceId = node.id;
      node.data.sceneParent =
        event.view === "attention" || event.view === "mlp"
          ? event.instance
          : undefined;
    }
    if (event.view === "overview")
      for (const node of graph.nodes)
        if (node.id === "input")
          node.data.description =
            "按当前示例位置表示 token 输入；文本与 token ID 均为教学占位，不是当前模型 tokenizer 的实际结果。";
    return graph;
  }
  const tensor = {
    name: "logits",
    shape: [1, m.vocab],
    symbols: ["B", "V"],
    dtype: m.deepseek ? ("float32" as const) : s.dtype,
  };
  const scalar = {
    name: "example_token",
    shape: [1, 1],
    symbols: ["B", "1"],
    dtype: "int64" as const,
  };
  const base = {
    subtitle: "教学示例",
    path: "demo",
    params: 0,
    details: [] as [string, string][],
    symbolic,
    category: "output" as const,
  };
  return {
    nodes: [
      {
        id: "demo-logits",
        type: "module",
        position: { x: 0, y: 0 },
        data: {
          ...base,
          title: "最后位置 logits",
          inputs: [tensor],
          outputs: [tensor],
          description: "配置确定词表维度；未运行模型，不提供真实 logits。",
          formula: "logits ∈ R^(1 × V)",
          order: 1,
        },
      },
      {
        id: "demo-sample",
        type: "module",
        position: { x: 0, y: 240 },
        data: {
          ...base,
          title: "选择下一个示例 token",
          inputs: [tensor],
          outputs: [scalar],
          description:
            "按照固定教学脚本追加占位 token；不是所选模型的生成结果，不展示虚构概率。",
          formula: "append(scripted_example_token)",
          order: 2,
        },
      },
    ],
    edges: [
      {
        id: "demo-sample-edge",
        source: "demo-logits",
        target: "demo-sample",
        sourceHandle: "out",
        targetHandle: "in",
        data: { tensor },
        markerEnd: { type: MarkerType.ArrowClosed, color: "#9277d7" },
      },
    ],
  };
}
export function readDemoRoute(search: string) {
  const params = new URLSearchParams(search);
  const integer = (
    name: string,
    fallback: number,
    min: number,
    max: number,
  ) => {
    const raw = params.get(name),
      value = raw === null ? fallback : Number(raw);
    return Number.isInteger(value) && value >= min && value <= max
      ? value
      : fallback;
  };
  return {
    promptTokens: integer("token", 4, 1, 16),
    pass: integer("generation", 0, 0, 7),
    useCache: params.get("kvCache") === "1",
  };
}
/** Cache capacity is a pass-boundary estimate, never a fabricated measured allocation trace. */
export function demoCache(
  m: ModelConfig,
  s: Scenario,
  useCache: boolean,
  committed: boolean,
) {
  const report = cacheReport(m, s);
  const tokens = committed ? s.past + s.sequence : s.past;
  const carried =
    useCache && tokens > 0
      ? cacheReport(m, { ...s, past: tokens, sequence: 0 })
      : null;
  return {
    kvBytes: carried?.kvBytes ?? 0,
    otherBytes: carried ? carried.recurrentBytes + carried.auxiliaryBytes : 0,
    totalBytes: carried?.totalBytes ?? 0,
    tokens: useCache ? tokens : 0,
    partial: report.partial,
    targetBytes: report.totalBytes,
  };
}
export const examplePrompt = ["Can", " machines", " think", "?"];
export const exampleContinuation = [
  " Yes",
  ",",
  " they",
  " can",
  " learn",
  " from",
  " examples",
  ".",
];
