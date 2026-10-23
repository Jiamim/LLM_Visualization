import type { ModuleNode } from "./graph";
import type { ModelConfig, Scenario, Tensor } from "./model";
import type { TraceSample } from "./demo-trace";
import { tensorCoordinates, traceValue } from "./tensor-semantics";

export type VisualKind =
  | "token"
  | "vector"
  | "matrix"
  | "norm"
  | "residual"
  | "attention"
  | "activation"
  | "cache"
  | "router"
  | "experts"
  | "state"
  | "block";
export interface ReferenceVisual {
  kind: VisualKind;
  inputWidth: number;
  outputWidth: number;
  heads: number;
  kvHeads: number;
  experts: number;
  topK: number;
  lanes: number;
  past: number;
  width: number;
  height: number;
  depth: number;
  rows: number;
  columns: number;
  color: string;
  tensor?: Tensor;
  inputTensor?: Tensor;
  causal: boolean;
  activation: string;
  rms: boolean;
  attentionGrid: boolean;
  queryLength: number;
  keyLength: number;
  window: number | null;
  residualMix: boolean;
  keyOffset: number;
  visibility: "causal" | "bidirectional" | "unknown";
  cacheLabel: string;
  recordedMode?: boolean;
}
export const TOKEN_COLORS = [
  "#e7c55c",
  "#53e5ad",
  "#f180bf",
  "#9b8cff",
  "#59c8ec",
  "#efa574",
];
export const REFERENCE_LAYER_HEIGHT = 140;
const bounded = (n: number, fallback = 1) =>
  Number.isFinite(n) && n > 0 ? n : fallback;
export function sampleCoordinates(size: number, limit = 24): number[] {
  if (size <= 0) return [];
  const count = Math.min(size, limit);
  return Array.from({ length: count }, (_, i) =>
    Math.floor((i * size) / count),
  );
}

/** Geometry semantics come from the active adapter's tensors, never GPT-2 defaults. */
export function referenceVisual(
  node: ModuleNode,
  model?: ModelConfig,
  scenario?: Scenario,
): ReferenceVisual {
  const d = node.data,
    text = `${node.id} ${d.title}`.toLowerCase();
  const tensor = d.outputs[0] ?? d.inputs[0];
  const layer = d.layerIndex ?? 0,
    profile = model?.catalog?.profiles[layer];
  const inputWidth = bounded(
    d.linear?.input ??
      (d.category === "embedding"
        ? (model?.vocab ?? 1)
        : (d.inputs[0]?.shape.at(-1) ?? model?.hidden ?? 1)),
  );
  const outputWidth = bounded(
    d.linear?.output ?? tensor?.shape.at(-1) ?? inputWidth,
  );
  const headTensor = [...d.outputs, ...d.inputs].find((t) =>
    t.symbols.some((s) => /^(Nq|Nkv|Nstate|Hq|Hkv)$/.test(s)),
  );
  const headAxis =
    headTensor?.symbols.findIndex((s) =>
      /^(Nq|Nkv|Nstate|Hq|Hkv)$/.test(s),
    ) ?? -1;
  const heads = bounded(
    d.attention?.heads ??
      headTensor?.shape[headAxis] ??
      profile?.heads ??
      model?.heads ??
      1,
  );
  let kind: VisualKind;
  if (d.residual) kind = "residual";
  else if (/cache|kv.read|kv.write|shared.kv|缓存/.test(text)) kind = "cache";
  else if (/router|routing|dispatch|路由/.test(text)) kind = "router";
  else if (/shared.expert/.test(text)) kind = "matrix";
  else if (/expert|专家/.test(text)) kind = "experts";
  else if (
    /state|recurrent|convolution|conv1d|delta.rule|mamba|递归|卷积/.test(text)
  )
    kind = "state";
  else if (d.category === "input" && tensor?.dtype.startsWith("int"))
    kind = "token";
  else if (d.category === "input") kind = "vector";
  else if (d.category === "norm") kind = "norm";
  else if (
    tensor?.role === "scores" ||
    tensor?.role === "probabilities" ||
    (!!d.attention && !d.linear && tensor?.role === "context")
  )
    kind = "attention";
  else if (/activation|gelu|silu|relu|swiglu/.test(text)) kind = "activation";
  else if (
    /rope|rotary|reshape|transpose|repeat|concat|merge.heads/.test(text)
  )
    kind = "vector";
  else if (d.category === "block") kind = "block";
  else if (d.drill === "attention")
    kind =
      profile &&
      [
        "delta",
        "kda",
        "lightning",
        "mamba",
        "conv",
        "mlstm",
        "gdla",
      ].includes(profile.kind)
        ? "state"
        : "attention";
  else if (
    d.category === "embedding" ||
    d.params > 0 ||
    /proj|linear|lm.head/.test(text)
  )
    kind = "matrix";
  else if (
    d.category === "attention" &&
    /attention$|attn$|self_attn$/.test(text)
  )
    kind = "attention";
  else kind = "vector";
  const queryAxis = tensor?.symbols.findIndex((s) => /^(S|Q)$/.test(s)) ?? -1;
  const lanes = Math.min(
    16,
    bounded(scenario?.sequence ?? tensor?.shape[queryAxis] ?? 1),
  );
  const extent = (size: number) =>
    2.8 + Math.min(4.5, Math.log2(Math.max(1, size)) * 0.28);
  const width =
    kind === "attention"
      ? 6.4
      : Math.max(extent(inputWidth), extent(outputWidth));
  const height =
    kind === "token"
      ? 1.2
      : kind === "vector"
        ? 0.6
        : kind === "matrix" || kind === "experts"
          ? 1.65
          : 1.25;
  const color = d.residual
    ? "#edc679"
    : /(^|[._-])q([._-]|$)|query/.test(text)
      ? "#7db4ff"
      : /(^|[._-])k([._-]|$)|key/.test(text)
        ? "#e1a4cf"
        : /(^|[._-])v([._-]|$)|value/.test(text)
          ? "#77dcb6"
          : d.category === "mlp"
            ? "#b695e6"
            : d.category === "norm"
              ? "#aebccf"
              : "#8fa8cf";
  return {
    kind,
    inputWidth,
    outputWidth,
    heads,
    kvHeads: d.attention?.kvHeads ?? profile?.kvHeads ?? model?.kvHeads ?? 1,
    experts: Math.max(1, profile?.experts ?? model?.deepseek?.experts ?? 1),
    topK: Math.max(1, profile?.topK ?? model?.deepseek?.topExperts ?? 1),
    lanes,
    past: scenario?.past ?? 0,
    width,
    height,
    depth: Math.max(1.3, lanes * 0.85),
    rows: tensor?.shape.at(-2) ?? 1,
    columns: tensor?.shape.at(-1) ?? 1,
    color,
    tensor,
    inputTensor: d.inputs[0],
    causal: d.attention?.visibility !== "bidirectional",
    activation: model?.activation ?? "gelu",
    rms: /rms/i.test(d.title),
    attentionGrid:
      tensor?.role === "scores" || tensor?.role === "probabilities",
    queryLength: d.attention?.queryLength ?? scenario?.sequence ?? 1,
    keyLength: d.attention?.keyLength ?? 0,
    keyOffset: d.attention?.keyOffset ?? 0,
    visibility: d.attention?.visibility ?? "unknown",
    cacheLabel: tensor
      ? `${tensor.name} [${tensor.shape.join(", ")}]`
      : "状态尺寸未提供",
    window: d.attention?.window ?? null,
    residualMix: !!d.residual && /mhc|special|mix/i.test(d.title),
  };
}

/** Only exact recorded coordinates are data; unsampled cells stay visibly unknown. */
export function recordedLane(
  sample: TraceSample | undefined,
  tensor: Tensor | undefined,
  lane: number,
  head: number,
  coordinates: number[],
): (number | null)[] {
  if (!sample || !tensor) return coordinates.map(() => null);
  return coordinates.map((col) => {
    const indices = tensorCoordinates(tensor, { token: lane, head });
    indices[indices.length - 1] = col;
    return traceValue(sample, tensor, indices);
  });
}
