import type { ModelConfig } from "./model";
import type { DemoSettings } from "./inference-demo";
import { buildGraph } from "./graph";
import { demoGraph, demoScenario } from "./inference-demo";

export interface TraceToken {
  text: string;
  id: number;
}
export interface TraceSample {
  loop?: number;
  instance?: string;
  pass: number;
  layer: number;
  view: "overview" | "layer" | "attention" | "mlp";
  nodeId: string;
  direction: "input" | "output";
  tensorIndex: number;
  shape: number[];
  indices: number[][];
  values: number[];
}
export interface DemoTrace {
  version: 1;
  modelKey: string;
  source: string;
  settings: Pick<DemoSettings, "promptTokens" | "outputTokens" | "useCache">;
  prompt: TraceToken[];
  generated: TraceToken[];
  samples: TraceSample[];
}
function canonical(value: unknown): string {
  if (Array.isArray(value)) return "[" + value.map(canonical).join(",") + "]";
  if (value && typeof value === "object")
    return (
      "{" +
      Object.entries(value)
        .sort(([a], [b]) => a.localeCompare(b))
        .map(([k, v]) => JSON.stringify(k) + ":" + canonical(v))
        .join(",") +
      "}"
    );
  return JSON.stringify(value);
}
export function demoModelKey(model: ModelConfig): string {
  let hash = 2166136261;
  for (const char of canonical(model.raw)) {
    hash ^= char.charCodeAt(0);
    hash = Math.imul(hash, 16777619);
  }
  return `${model.architecture}:${model.hidden}:${model.layers}:${(hash >>> 0).toString(16)}`;
}
export function traceTemplate(
  model: ModelConfig,
  settings: DemoSettings,
): DemoTrace {
  return {
    version: 1,
    modelKey: demoModelKey(model),
    source: "填写运行记录的来源与版本",
    settings: {
      promptTokens: settings.promptTokens,
      outputTokens: settings.outputTokens,
      useCache: settings.useCache,
    },
    prompt: Array.from({ length: settings.promptTokens }, (_, i) => ({
      text: `token-${i}`,
      id: 0,
    })),
    generated: Array.from({ length: settings.outputTokens }, (_, i) => ({
      text: `output-${i}`,
      id: 0,
    })),
    samples: [],
  };
}
export function parseDemoTrace(text: string, model: ModelConfig): DemoTrace {
  if (text.length > 8 * 1024 * 1024)
    throw new Error("运行记录超过 8 MiB 上限。");
  let data: DemoTrace;
  try {
    data = JSON.parse(text);
  } catch {
    throw new Error("运行记录不是有效 JSON。");
  }
  if (!data || data.version !== 1 || data.modelKey !== demoModelKey(model))
    throw new Error(
      "记录与当前模型配置不匹配，请使用当前配置生成的 modelKey。",
    );
  const s = data.settings;
  if (
    !s ||
    !Number.isInteger(s.promptTokens) ||
    s.promptTokens < 1 ||
    s.promptTokens > 16 ||
    !Number.isInteger(s.outputTokens) ||
    s.outputTokens < 1 ||
    s.outputTokens > 8 ||
    typeof s.useCache !== "boolean"
  )
    throw new Error("记录的提示长度、输出轮数或缓存模式不合法。");
  if (
    model.maxPosition !== null &&
    s.promptTokens + s.outputTokens - 1 > model.maxPosition
  )
    throw new Error("记录超过模型上下文上限。");
  const validToken = (t: TraceToken) =>
    t &&
    typeof t.text === "string" &&
    t.text.length <= 200 &&
    Number.isInteger(t.id) &&
    t.id >= 0 &&
    t.id < model.vocab;
  if (
    !Array.isArray(data.prompt) ||
    data.prompt.length !== s.promptTokens ||
    !data.prompt.every(validToken) ||
    !Array.isArray(data.generated) ||
    data.generated.length !== s.outputTokens ||
    !data.generated.every(validToken)
  )
    throw new Error("记录的 token 文本、ID 或数量不合法。");
  if (
    typeof data.source !== "string" ||
    !data.source.trim() ||
    data.source.length > 500 ||
    !Array.isArray(data.samples) ||
    data.samples.length > 10000
  )
    throw new Error("记录需要来源说明，最多允许 10,000 条张量记录。");
  const graphCache = new Map<string, ReturnType<typeof demoGraph>>();
  const parentCache = new Map<string, string[]>();
  const keys = new Set<string>();
  let valueCount = 0;
  for (const sample of data.samples) {
    if (
      !sample ||
      (sample.loop !== undefined &&
        (!Number.isInteger(sample.loop) ||
          sample.loop < 0 ||
          sample.loop >= (model.catalog?.loops ?? 1))) ||
      !Number.isInteger(sample.pass) ||
      sample.pass < 0 ||
      sample.pass >= s.outputTokens ||
      !Number.isInteger(sample.layer) ||
      sample.layer < 0 ||
      sample.layer >= model.layers ||
      !["overview", "layer", "attention", "mlp"].includes(sample.view) ||
      !["input", "output"].includes(sample.direction) ||
      !Number.isInteger(sample.tensorIndex) ||
      sample.tensorIndex < 0 ||
      typeof sample.nodeId !== "string"
    )
      throw new Error("记录包含无效轮次、层或组件定位。");
    const scenario = demoScenario(
      { ...s, speed: 1 },
      sample.pass,
      model.dtype,
    );
    if (sample.view === "attention" || sample.view === "mlp") {
      const parentKey = `${sample.pass}:${sample.layer}:${sample.view}`;
      let parents = parentCache.get(parentKey);
      if (!parents) {
        parents = buildGraph(model, scenario, "layer", sample.layer)
          .nodes.filter((n) => n.data.drill === sample.view)
          .map((n) => n.id);
        parentCache.set(parentKey, parents);
      }
      if (sample.instance === undefined && parents.length === 1)
        sample.instance = parents[0];
      if (!parents.includes(sample.instance ?? ""))
        throw new Error(
          `记录 ${sample.nodeId} 需要有效的 instance（父模块 ID），以区分重复子层。`,
        );
    } else if (sample.instance !== undefined)
      throw new Error("总览或层记录不应指定子层 instance。");
    const graphKey = `${sample.pass}:${sample.layer}:${sample.view}:${sample.instance ?? ""}`;
    let graph = graphCache.get(graphKey);
    if (!graph) {
      graph = demoGraph(
        model,
        demoScenario({ ...s, speed: 1 }, sample.pass, model.dtype),
        {
          view: sample.view,
          layer: sample.layer,
          nodeId: sample.nodeId,
          title: "",
          category: "",
          loop: 0,
          segment: "",
          instance: sample.instance,
        },
      );
      graphCache.set(graphKey, graph);
    }
    const node = graph.nodes.find((n) => n.id === sample.nodeId);
    const tensor = (
      sample.direction === "input" ? node?.data.inputs : node?.data.outputs
    )?.[sample.tensorIndex];
    if (
      !tensor ||
      JSON.stringify(tensor.shape) !== JSON.stringify(sample.shape)
    )
      throw new Error(`记录 ${sample.nodeId} 的张量维度与当前配置不一致。`);
    if (
      !Array.isArray(sample.indices) ||
      !Array.isArray(sample.values) ||
      sample.values.length !== sample.indices.length ||
      !sample.values.length ||
      sample.values.some(
        (v) => typeof v !== "number" || !Number.isFinite(v),
      ) ||
      sample.indices.some(
        (i) =>
          !Array.isArray(i) ||
          i.length !== tensor.shape.length ||
          i.some(
            (x, axis) =>
              !Number.isInteger(x) || x < 0 || x >= tensor.shape[axis],
          ),
      )
    )
      throw new Error(`记录 ${sample.nodeId} 的采样坐标或数值无效。`);
    valueCount += sample.values.length;
    if (valueCount > 250000)
      throw new Error("记录最多允许 250,000 个采样值。");
    const key = `${graphKey}:${sample.loop ?? 0}:${sample.nodeId}:${sample.direction}:${sample.tensorIndex}`;
    if (keys.has(key)) throw new Error("同一张量存在重复记录。");
    keys.add(key);
    if (
      new Set(sample.indices.map((i) => i.join(","))).size !==
      sample.indices.length
    )
      throw new Error("张量采样坐标重复。");
  }
  return data;
}
