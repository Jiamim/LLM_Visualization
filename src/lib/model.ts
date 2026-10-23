import {
  normalizeDeepseek,
  deepseekParameters,
  deepseekMemory,
  type DeepseekConfig,
} from "./deepseek";
import {
  normalizeGemma,
  gemmaParameters,
  gemmaMemory,
  gemmaHasHead,
  type GemmaConfig,
} from "./gemma";
import {
  CATALOG_TYPES,
  catalogText,
  normalizeCatalog,
  catalogParameters,
  catalogMemory,
  type CatalogConfig,
} from "./catalog";
export type ModelFamily =
  | "llama"
  | "qwen2"
  | "qwen3"
  | "mistral"
  | "deepseek_v41"
  | "gemma4"
  | "catalog";
export type DType = "float32" | "float16" | "bfloat16";
export type RawConfig = Record<string, unknown>;

export interface ModelConfig {
  family: ModelFamily;
  architecture: string;
  hidden: number;
  intermediate: number;
  layers: number;
  heads: number;
  kvHeads: number;
  headDim: number;
  vocab: number;
  maxPosition: number | null;
  activation: string;
  normEps: number;
  dtype: DType;
  tiedEmbeddings: boolean;
  qkvBias: boolean;
  outputBias: boolean;
  mlpBias: boolean;
  qkNorm: boolean;
  slidingWindow: number | null;
  slidingLayers: boolean[];
  warnings: string[];
  raw: RawConfig;
  deepseek?: DeepseekConfig;
  gemma?: GemmaConfig;
  catalog?: CatalogConfig;
  gallery?: {
    id: string;
    name: string;
    url: string;
    configUrl: string;
    sourceNote?: string;
    snapshotDate: string;
  };
}

export interface Scenario {
  batch: number;
  sequence: number;
  past: number;
  dtype: DType;
  cedExecution?: "reference" | "bounded";
  image?: { count: number; gridH: number; gridW: number };
  audio?: { enabled: boolean; frames: number };
}

export type AxisRole =
  | "batch"
  | "query"
  | "key"
  | "head"
  | "kv-head"
  | "feature"
  | "other";
export type TensorRole =
  | "activation"
  | "query"
  | "key"
  | "value"
  | "scores"
  | "probabilities"
  | "context"
  | "state";
export interface Tensor {
  name: string;
  shape: number[];
  symbols: string[];
  dtype: DType | "int64" | "int32" | "bool";
  note?: string;
  axes?: AxisRole[];
  role?: TensorRole;
}

export const SUPPORTED_FAMILIES = [
  "llama",
  "qwen2",
  "qwen3",
  "mistral",
  "deepseek_v41",
  "gemma4",
] as const;
const architectureNames: Record<ModelFamily, string> = {
  llama: "LlamaForCausalLM",
  qwen2: "Qwen2ForCausalLM",
  qwen3: "Qwen3ForCausalLM",
  mistral: "MistralForCausalLM",
  deepseek_v41: "DeepseekV41ForCausalLM",
  gemma4: "Gemma4ForConditionalGeneration",
  catalog: "CatalogLanguageBackbone",
};

export function isRecord(value: unknown): value is RawConfig {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function parseConfig(text: string): ModelConfig {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    throw new Error("JSON 格式不正确，请检查引号、逗号和括号。");
  }
  return normalizeConfig(raw);
}

export function normalizeConfig(value: unknown): ModelConfig {
  if (!isRecord(value)) throw new Error("配置必须是一个 JSON 对象。");
  const raw = value;
  const candidate = catalogText(raw);
  if (
    CATALOG_TYPES.has(candidate.type) ||
    CATALOG_TYPES.has(String(raw.model_type))
  )
    return normalizeCatalog(raw);
  if (
    (raw.model_type === "gemma4" || raw.model_type === "gemma4_text") &&
    (candidate.text.enable_moe_block === true ||
      candidate.text.hidden_size_per_layer_input === 0 ||
      candidate.text.attention_k_eq_v === true)
  )
    return normalizeCatalog(raw);
  if (raw.model_type === "gemma4" || raw.model_type === "gemma4_text")
    return normalizeGemma(raw);
  if (
    raw.model_type === "deepseek_v41" ||
    raw.model_type === "deepseek_v41_text"
  )
    return normalizeDeepseek(raw);
  if (
    !(SUPPORTED_FAMILIES as readonly string[]).includes(
      String(raw.model_type),
    )
  ) {
    const nested = isRecord(raw.text_config)
      ? " 检测到 text_config；当前不支持完整多模态图，请单独导入受支持的文本模型配置。"
      : "";
    throw new Error(
      `暂不支持 model_type="${String(raw.model_type ?? "未提供")}"。可从模型库选择已适配架构（含 111 个画廊条目），或使用其标准配置。${nested}`,
    );
  }
  const family = raw.model_type as ModelFamily;
  const warnings: string[] = [];
  const integer = (key: string, fallback?: number): number => {
    const v = raw[key] ?? fallback;
    if (typeof v !== "number" || !Number.isSafeInteger(v) || v <= 0) {
      throw new Error(
        `${key} 必须是正整数${fallback === undefined ? "，且不可省略" : ""}。`,
      );
    }
    return v;
  };
  const boolean = (key: string, fallback: boolean): boolean => {
    const v = raw[key] ?? fallback;
    if (typeof v !== "boolean") throw new Error(`${key} 必须是布尔值。`);
    return v;
  };
  const hidden = integer("hidden_size");
  const heads = integer("num_attention_heads");
  const layers = integer("num_hidden_layers");
  if (layers > 2048)
    throw new Error("num_hidden_layers 超过当前视图上限 2048。");
  const kvHeads = integer("num_key_value_heads", heads);
  if (heads % kvHeads !== 0)
    throw new Error(
      "num_attention_heads 必须能被 num_key_value_heads 整除。",
    );
  const headDim = integer("head_dim", hidden / heads);
  if (headDim % 2 !== 0) throw new Error("RoPE 要求 head_dim 为偶数。");
  const architecture = architectureNames[family];
  if (
    raw.architectures !== undefined &&
    (!Array.isArray(raw.architectures) ||
      !raw.architectures.every((a) => typeof a === "string"))
  ) {
    throw new Error("architectures 必须是字符串数组。");
  }
  const providedArchitectures = raw.architectures as string[] | undefined;
  if (
    providedArchitectures?.length &&
    !providedArchitectures.every(
      (a) =>
        a === architecture ||
        a === architecture.replace("ForCausalLM", "Model"),
    )
  ) {
    throw new Error(
      `当前 ${family} 适配器仅支持 ${architecture} 或基础 Model；该任务头需要单独适配。`,
    );
  }
  if (raw.auto_map)
    throw new Error(
      "该配置包含 auto_map 自定义模型代码，无法仅凭配置确认其结构；请使用标准架构配置。",
    );
  if (
    raw.num_local_experts ||
    raw.num_experts ||
    raw.is_encoder_decoder === true
  ) {
    throw new Error(
      "该配置使用 MoE 或编码器–解码器结构，需要对应的架构适配器。",
    );
  }
  if (raw.quantization_config)
    warnings.push(
      "配置包含量化信息。参数量按逻辑权重计算；内存按所选浮点精度估算，不反映量化存储。",
    );
  const activation = raw.hidden_act ?? "silu";
  if (
    typeof activation !== "string" ||
    !["silu", "gelu", "gelu_new", "relu"].includes(activation)
  )
    throw new Error(
      "暂不支持该 hidden_act；可用：silu、gelu、gelu_new、relu。",
    );
  const normEps = raw.rms_norm_eps ?? 1e-6;
  if (
    typeof normEps !== "number" ||
    !Number.isFinite(normEps) ||
    normEps <= 0
  )
    throw new Error("rms_norm_eps 必须是正数。");
  const requestedDtype = raw.dtype ?? raw.torch_dtype ?? "float32";
  const dtype: DType = ["float32", "float16", "bfloat16"].includes(
    String(requestedDtype),
  )
    ? (requestedDtype as DType)
    : "float32";
  if (dtype !== requestedDtype)
    warnings.push(
      `未识别精度 ${String(requestedDtype)}，默认使用 float32 估算。`,
    );
  const slidingWindow =
    raw.sliding_window == null ? null : integer("sliding_window");
  const useSliding = boolean("use_sliding_window", family === "mistral");
  const maxWindowLayers =
    raw.max_window_layers == null ? layers : integer("max_window_layers");
  let slidingLayers = Array.from({ length: layers }, (_, index) =>
    Boolean(
      slidingWindow &&
        useSliding &&
        (family === "mistral" || index >= maxWindowLayers),
    ),
  );
  if (raw.layer_types != null) {
    if (
      !Array.isArray(raw.layer_types) ||
      raw.layer_types.length !== layers ||
      !raw.layer_types.every(
        (t) => t === "full_attention" || t === "sliding_attention",
      )
    ) {
      throw new Error(
        "layer_types 必须为每层指定 full_attention 或 sliding_attention。",
      );
    }
    slidingLayers = raw.layer_types.map((t) => t === "sliding_attention");
  }
  if (slidingLayers.some(Boolean)) {
    if (!slidingWindow)
      throw new Error("滑动窗口层需要有效的 sliding_window。");
    warnings.push(
      `部分层使用滑动窗口 ${slidingWindow}。图中展示未裁剪 KV cache 的逻辑形状；窗口体现在 attention mask 中，具体后端可能裁剪缓存。`,
    );
  }
  if (raw.rope_scaling)
    warnings.push("检测到 RoPE scaling；旋转位置编码不改变图中张量形状。");
  if (!providedArchitectures?.length)
    warnings.push(
      `未指定 architectures，按 ${architecture}（包含 LM Head）展示。`,
    );
  const model: ModelConfig = {
    family,
    architecture: providedArchitectures?.[0] ?? architecture,
    hidden,
    heads,
    layers,
    kvHeads,
    headDim,
    intermediate: integer("intermediate_size"),
    vocab: integer("vocab_size"),
    maxPosition:
      raw.max_position_embeddings == null
        ? null
        : integer("max_position_embeddings"),
    activation,
    normEps,
    dtype,
    tiedEmbeddings: boolean("tie_word_embeddings", false),
    qkvBias:
      family === "qwen2"
        ? true
        : family === "mistral"
          ? false
          : boolean("attention_bias", false),
    outputBias:
      family === "qwen2" || family === "mistral"
        ? false
        : boolean("attention_bias", false),
    mlpBias: family === "llama" ? boolean("mlp_bias", false) : false,
    qkNorm: family === "qwen3",
    slidingWindow,
    slidingLayers,
    warnings,
    raw,
  };
  if (!Number.isSafeInteger(parameterCounts(model).total))
    throw new Error("配置的参数规模超出可安全计算的整数范围。");
  return model;
}

export function hasLMHead(model: ModelConfig): boolean {
  if (model.gemma) return gemmaHasHead(model);
  return model.architecture.endsWith("ForCausalLM");
}

export function parameterCounts(m: ModelConfig) {
  if (m.catalog) return catalogParameters(m);
  if (m.gemma) return gemmaParameters(m);
  if (m.deepseek) return deepseekParameters(m);
  const q = m.heads * m.headDim;
  const kv = m.kvHeads * m.headDim;
  const qProj = m.hidden * q + (m.qkvBias ? q : 0);
  const kProj = m.hidden * kv + (m.qkvBias ? kv : 0);
  const vProj = kProj;
  const oProj = q * m.hidden + (m.outputBias ? m.hidden : 0);
  const attention =
    qProj + kProj + vProj + oProj + (m.qkNorm ? 2 * m.headDim : 0);
  const gateProj =
    m.hidden * m.intermediate + (m.mlpBias ? m.intermediate : 0);
  const upProj = gateProj;
  const downProj = m.intermediate * m.hidden + (m.mlpBias ? m.hidden : 0);
  const mlp = gateProj + upProj + downProj;
  const layer = attention + mlp + 2 * m.hidden;
  const embedding = m.vocab * m.hidden;
  const lmHead = hasLMHead(m) ? embedding : 0;
  const total =
    embedding + m.layers * layer + m.hidden + (m.tiedEmbeddings ? 0 : lmHead);
  return {
    qProj,
    kProj,
    vProj,
    oProj,
    attention,
    gateProj,
    upProj,
    downProj,
    mlp,
    layer,
    embedding,
    lmHead,
    total,
  };
}

export function validateScenario(s: Scenario): void {
  for (const key of ["batch", "sequence", "past"] as const) {
    if (!Number.isSafeInteger(s[key]) || s[key] < (key === "past" ? 0 : 1))
      throw new Error(
        "Batch 和序列长度必须是正整数，KV cache 长度必须是非负整数。",
      );
  }
  if (s.batch > 65536 || s.sequence + s.past > 10_000_000)
    throw new Error(
      "输入过大：Batch 上限 65,536，总序列长度上限 10,000,000。",
    );
}

export function tensorShapes(m: ModelConfig, s: Scenario) {
  validateScenario(s);
  const b = s.batch,
    t = s.sequence,
    k = s.sequence + s.past;
  return {
    ids: [b, t],
    hidden: [b, t, m.hidden],
    intermediate: [b, t, m.intermediate],
    qProjection: [b, t, m.heads * m.headDim],
    kvProjection: [b, t, m.kvHeads * m.headDim],
    q: [b, m.heads, t, m.headDim],
    kvNew: [b, m.kvHeads, t, m.headDim],
    kv: [b, m.kvHeads, k, m.headDim],
    repeatedKV: [b, m.heads, k, m.headDim],
    scores: [b, m.heads, t, k],
    context: [b, m.heads, t, m.headDim],
    logits: [b, t, m.vocab],
    mask: [b, 1, t, k],
  };
}

export const dtypeBytes = (dtype: Tensor["dtype"]) =>
  dtype === "int64"
    ? 8
    : dtype === "bool"
      ? 1
      : dtype === "float32" || dtype === "int32"
        ? 4
        : 2;
export const tensorBytes = (tensor: Tensor) =>
  tensor.shape.reduce((a, b) => a * b, 1) * dtypeBytes(tensor.dtype);
export const shapeText = (shape: number[]) => `[${shape.join(", ")}]`;
export const formatNumber = (value: number) =>
  new Intl.NumberFormat("en-US").format(value);
export function compactNumber(value: number): string {
  if (value >= 1e9) return `${(value / 1e9).toFixed(2)}B`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(2)}M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(1)}K`;
  return String(value);
}
export function formatBytes(bytes: number): string {
  if (bytes === 0) return "0 B";
  const power = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), 4);
  return `${(bytes / 1024 ** power).toFixed(power > 0 ? 2 : 0)} ${["B", "KiB", "MiB", "GiB", "TiB"][power]}`;
}
export function memoryEstimates(m: ModelConfig, s: Scenario) {
  if (m.catalog) return catalogMemory(m, s);
  if (m.gemma) return gemmaMemory(m, s);
  if (m.deepseek) return deepseekMemory(m, s);
  const size = dtypeBytes(s.dtype);
  return {
    weights: parameterCounts(m).total * size,
    kvCache:
      2 *
      m.layers *
      s.batch *
      m.kvHeads *
      (s.past + s.sequence) *
      m.headDim *
      size,
    logits: hasLMHead(m) ? s.batch * s.sequence * m.vocab * size : 0,
  };
}
