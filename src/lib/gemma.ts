import type { DType, ModelConfig, RawConfig, Scenario } from "./model";
export const GEMMA_REFERENCE = {
  transformers: "d4804f9d76b8c071b993d813e9e87067b780cbee",
  e2b: "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
  e4b: "ee0ef6023621cff504d758262d4e04895a5af4a2",
};
export interface GemmaConfig {
  text: RawConfig;
  layerTypes: ("sliding_attention" | "full_attention")[];
  globalHeadDim: number;
  sharedLayers: number;
  shareStart: number;
  localSource: number | null;
  globalSource: number | null;
  pleDim: number;
  pleVocab: number;
  doubleMlp: boolean;
  softcap: number | null;
  globalRopeFraction: number;
  vision: null | {
    hidden: number;
    intermediate: number;
    layers: number;
    heads: number;
    kvHeads: number;
    headDim: number;
    patch: number;
    pool: number;
    positions: number;
    defaultTokens: number;
  };
  audio: null | {
    hidden: number;
    layers: number;
    heads: number;
    output: number;
    channels: [number, number];
    kernel: number;
    chunk: number;
    left: number;
    right: number;
    residualWeight: number;
  };
}
const record = (v: unknown): v is RawConfig =>
  v !== null && typeof v === "object" && !Array.isArray(v);
export function normalizeGemma(raw: RawConfig): ModelConfig {
  const nested = raw.model_type === "gemma4",
    c = nested ? raw.text_config : raw;
  if (!record(c) || c.model_type !== "gemma4_text")
    throw new Error("Gemma 4 需要有效的 gemma4_text 配置。");
  const n = (key: string, fallback?: number, min = 1, source = c) => {
    const value = source[key] ?? fallback;
    if (
      typeof value !== "number" ||
      !Number.isSafeInteger(value) ||
      value < min
    )
      throw new Error(`Gemma ${key} 必须为 >= ${min} 的安全整数。`);
    return value;
  };
  const bool = (key: string, fallback: boolean, source = c) => {
    const value = source[key] ?? fallback;
    if (typeof value !== "boolean")
      throw new Error(`Gemma ${key} 必须是布尔值。`);
    return value;
  };
  const positive = (key: string, fallback: number, source = c) => {
    const value = source[key] ?? fallback;
    if (typeof value !== "number" || !Number.isFinite(value) || value <= 0)
      throw new Error(`Gemma ${key} 必须是正数。`);
    return value;
  };
  if (raw.auto_map || c.auto_map)
    throw new Error("Gemma 适配器仅支持标准实现，不解释自定义 auto_map。");
  if (bool("enable_moe_block", false) || c.num_experts || c.top_k_experts)
    throw new Error(
      "Gemma 4 当前支持 E2B/E4B 的 dense + PLE 结构；MoE 变体需要单独适配。",
    );
  if (bool("attention_k_eq_v", false))
    throw new Error(
      "当前 Gemma E 系列适配器要求独立 K/V 投影，不支持 attention_k_eq_v 变体。",
    );
  if (c.use_bidirectional_attention != null || c.is_encoder_decoder === true)
    throw new Error("当前 Gemma E 系列语言主干仅支持标准因果注意力。");
  if (c.per_layer_config != null)
    throw new Error(
      "Gemma 自定义 per_layer_config 暂不支持，请使用官方 E2B/E4B 原始配置。",
    );
  if (
    c.hidden_activation != null &&
    c.hidden_activation !== "gelu_pytorch_tanh"
  )
    throw new Error("Gemma E 系列要求 gelu_pytorch_tanh 门控激活。");
  const layers = n("num_hidden_layers"),
    hidden = n("hidden_size"),
    heads = n("num_attention_heads"),
    kvHeads = n("num_key_value_heads"),
    headDim = n("head_dim"),
    globalHeadDim = n("global_head_dim", 512);
  if (layers > 256 || heads % kvHeads || headDim % 2 || globalHeadDim % 2)
    throw new Error("Gemma 层数、Query/KV 头数或 head_dim 不合法。");
  const shared = n("num_kv_shared_layers", 0, 0),
    start = layers - shared;
  if (shared >= layers)
    throw new Error("num_kv_shared_layers 必须小于总层数。");
  const types = c.layer_types;
  if (
    !Array.isArray(types) ||
    types.length !== layers ||
    !types.every((t) => t === "sliding_attention" || t === "full_attention")
  )
    throw new Error(
      "Gemma layer_types 必须逐层指定 sliding_attention 或 full_attention。",
    );
  if (types.at(-1) !== "full_attention")
    throw new Error("Gemma 末层必须使用 full_attention。");
  const before = types.slice(0, start),
    local = before.lastIndexOf("sliding_attention"),
    global = before.lastIndexOf("full_attention");
  if (shared && types.slice(start).some((t) => !before.includes(t)))
    throw new Error("KV 共享区之前缺少相同类型的缓存来源层。");
  const ple = n("hidden_size_per_layer_input", 0, 0);
  if (!ple)
    throw new Error(
      "此适配器面向含 PLE 的 Gemma E2B/E4B；无 PLE 变体尚未适配。",
    );
  let fraction = 0.25;
  if (c.rope_parameters != null) {
    if (
      !record(c.rope_parameters) ||
      !record(c.rope_parameters.full_attention) ||
      !record(c.rope_parameters.sliding_attention)
    )
      throw new Error("Gemma rope_parameters 必须分别定义全局与局部 RoPE。");
    const full = c.rope_parameters.full_attention,
      sliding = c.rope_parameters.sliding_attention;
    if (full.rope_type !== "proportional" || sliding.rope_type !== "default")
      throw new Error("只支持 Gemma E 系列 proportional/default RoPE。");
    fraction = positive("partial_rotary_factor", 0.25, full);
    if (
      fraction > 1 ||
      !Number.isInteger(globalHeadDim * fraction) ||
      (globalHeadDim * fraction) % 2
    )
      throw new Error("全局 RoPE 旋转维度必须为正偶数且不超过 head_dim。");
  }
  if (
    !Number.isInteger(globalHeadDim * fraction) ||
    (globalHeadDim * fraction) % 2
  )
    throw new Error("默认全局 RoPE 旋转维度必须为正偶数。");
  const architectures = raw.architectures;
  const accepted = nested
    ? ["Gemma4ForConditionalGeneration", "Gemma4Model"]
    : ["Gemma4ForCausalLM", "Gemma4TextModel"];
  if (
    architectures != null &&
    (!Array.isArray(architectures) ||
      architectures.length !== 1 ||
      !accepted.includes(String(architectures[0])))
  )
    throw new Error("不支持该 Gemma 4 任务头或配置层级。");
  let vision: GemmaConfig["vision"] = null,
    audio: GemmaConfig["audio"] = null;
  if (raw.vision_config != null) {
    const v = raw.vision_config;
    if (!record(v) || v.model_type !== "gemma4_vision")
      throw new Error("无效的 gemma4_vision 配置。");
    vision = {
      hidden: n("hidden_size", undefined, 1, v),
      intermediate: n("intermediate_size", undefined, 1, v),
      layers: n("num_hidden_layers", undefined, 1, v),
      heads: n("num_attention_heads", undefined, 1, v),
      kvHeads: n("num_key_value_heads", undefined, 1, v),
      headDim: n("head_dim", undefined, 1, v),
      patch: n("patch_size", 16, 1, v),
      pool: n("pooling_kernel_size", 3, 1, v),
      positions: n("position_embedding_size", 10240, 1, v),
      defaultTokens: n("vision_soft_tokens_per_image", 280, 1, raw),
    };
    if (
      vision.layers > 256 ||
      vision.heads % vision.kvHeads ||
      vision.headDim % 4 ||
      v.hidden_activation !== "gelu_pytorch_tanh"
    )
      throw new Error("视觉层数、头数、RoPE 或激活函数不受支持。");
  }
  if (raw.audio_config != null) {
    const a = raw.audio_config;
    if (!record(a) || a.model_type !== "gemma4_audio")
      throw new Error("无效的 gemma4_audio 配置。");
    const channels = a.subsampling_conv_channels;
    if (
      !Array.isArray(channels) ||
      channels.length !== 2 ||
      !channels.every((v) => Number.isSafeInteger(v) && v > 0) ||
      channels[0] % 4
    )
      throw new Error(
        "音频 subsampling_conv_channels 必须为两个正整数，第一项应能被 4 整除。",
      );
    audio = {
      hidden: n("hidden_size", undefined, 1, a),
      layers: n("num_hidden_layers", undefined, 1, a),
      heads: n("num_attention_heads", undefined, 1, a),
      output: n("output_proj_dims", undefined, 1, a),
      channels: channels as [number, number],
      kernel: n("conv_kernel_size", 5, 1, a),
      chunk: n("attention_chunk_size", 12, 1, a),
      left: n("attention_context_left", 13, 1, a) - 1,
      right: n("attention_context_right", 0, 0, a),
      residualWeight: positive("residual_weight", 0.5, a),
    };
    if (
      audio.layers > 256 ||
      audio.hidden % audio.heads ||
      audio.hidden % 2 ||
      a.hidden_act !== "silu"
    )
      throw new Error("音频层数、head 维度或激活函数不受支持。");
  }
  const dtype = raw.dtype ?? c.dtype ?? c.torch_dtype ?? "bfloat16";
  if (!["float32", "float16", "bfloat16"].includes(String(dtype)))
    throw new Error("不支持该 Gemma 浮点精度。");
  const tied = bool("tie_word_embeddings", true);
  if (raw.tie_word_embeddings != null && raw.tie_word_embeddings !== tied)
    throw new Error("外层与 text_config 的 tie_word_embeddings 必须一致。");
  const g: GemmaConfig = {
    text: c,
    layerTypes: types as GemmaConfig["layerTypes"],
    globalHeadDim,
    sharedLayers: shared,
    shareStart: start,
    localSource: local < 0 ? null : local,
    globalSource: global < 0 ? null : global,
    pleDim: ple,
    pleVocab: n("vocab_size_per_layer_input"),
    doubleMlp: bool("use_double_wide_mlp", false),
    softcap:
      c.final_logit_softcapping == null
        ? null
        : positive("final_logit_softcapping", 30),
    globalRopeFraction: fraction,
    vision,
    audio,
  };
  const model: ModelConfig = {
    family: "gemma4",
    architecture:
      (architectures?.[0] as string) ??
      (nested ? "Gemma4ForConditionalGeneration" : "Gemma4ForCausalLM"),
    hidden,
    intermediate: n("intermediate_size"),
    layers,
    heads,
    kvHeads,
    headDim,
    vocab: n("vocab_size"),
    maxPosition: n("max_position_embeddings"),
    activation: "gelu_pytorch_tanh",
    normEps: positive("rms_norm_eps", 1e-6),
    dtype: dtype as DType,
    tiedEmbeddings: tied,
    qkvBias: bool("attention_bias", false),
    outputBias: bool("attention_bias", false),
    mlpBias: false,
    qkNorm: true,
    slidingWindow: n("sliding_window", undefined, 2),
    slidingLayers: types.map((t) => t === "sliding_attention"),
    gemma: g,
    raw,
    warnings: [
      "Gemma 4 E 系列：局部与全局 head_dim 不同；共享区按注意力类型复用最后一个非共享层，不能按总层数重复累计 KV。",
      "总参数包含 PLE 大型嵌入表、文本主干和多模态分支；E2B/E4B 是 effective 规模名称，不是全部权重参数总数。",
      "缓存估算采用 Transformers DynamicCache：局部持久缓存保留最多 window−1 项，本次 attention 可见键长度是 min(P,window−1)+S。共享字典引用本次 K/V，不增加一份持久缓存。",
      "图像以预处理后规则 patch 网格为例，需整除 pooling kernel；音频输入为预处理的特征帧。S 已包含软 token 和模态分隔标记，不执行图像/音频处理。",
      "展示完整 S 个位置的 logits 与 softcapping；仅为逻辑形状、浮点等效权重，非峰值显存或量化文件大小。",
    ],
  };
  if (raw.quantization_config)
    model.warnings.push("检测到量化配置；仍按逻辑权重计数和所选浮点精度估算。");
  if (!Number.isSafeInteger(gemmaParameters(model).total))
    throw new Error("Gemma 参数规模超出安全整数范围。");
  return model;
}

export function gemmaHasHead(m: ModelConfig) {
  return (
    m.architecture === "Gemma4ForCausalLM" ||
    m.architecture === "Gemma4ForConditionalGeneration"
  );
}
export function gemmaLayer(m: ModelConfig, index: number) {
  const g = m.gemma!;
  if (!Number.isInteger(index) || index < 0 || index >= m.layers)
    throw new Error("Gemma 层编号越界。");
  const sliding = g.layerTypes[index] === "sliding_attention",
    shared = index >= g.shareStart;
  return {
    index,
    sliding,
    shared,
    type: g.layerTypes[index],
    headDim: sliding ? m.headDim : g.globalHeadDim,
    owner: shared ? (sliding ? g.localSource! : g.globalSource!) : index,
    source:
      !shared &&
      index === (sliding ? g.localSource : g.globalSource) &&
      g.sharedLayers > 0,
    intermediate: m.intermediate * (shared && g.doubleMlp ? 2 : 1),
  };
}
export function gemmaShapes(m: ModelConfig, s: Scenario, index: number) {
  const l = gemmaLayer(m, index),
    g = m.gemma!,
    b = s.batch,
    q = s.sequence,
    total = q + s.past;
  const keys = l.sliding ? Math.min(s.past, m.slidingWindow! - 1) + q : total;
  const retained = l.sliding ? Math.min(total, m.slidingWindow! - 1) : total;
  return {
    layer: l,
    keys,
    retained,
    hidden: [b, q, m.hidden],
    ids: [b, q],
    qProjection: [b, q, m.heads * l.headDim],
    kvProjection: [b, q, m.kvHeads * l.headDim],
    q: [b, m.heads, q, l.headDim],
    kv: [b, m.kvHeads, keys, l.headDim],
    newKV: [b, m.kvHeads, q, l.headDim],
    cache: [b, m.kvHeads, retained, l.headDim],
    scores: [b, m.heads, q, keys],
    context: [b, q, m.heads * l.headDim],
    intermediate: [b, q, l.intermediate],
    ple: [b, q, m.layers, g.pleDim],
    pleSlice: [b, q, g.pleDim],
    logits: [b, q, m.vocab],
  };
}
export function gemmaParameters(m: ModelConfig) {
  const g = m.gemma!,
    h = m.hidden;
  const layers = Array.from({ length: m.layers }, (_, index) => {
    const l = gemmaLayer(m, index),
      q = m.heads * l.headDim,
      kv = m.kvHeads * l.headDim;
    const qProj = h * q + (m.qkvBias ? q : 0),
      oProj = h * q + (m.outputBias ? h : 0),
      kProj = l.shared ? 0 : h * kv + (m.qkvBias ? kv : 0),
      vProj = kProj;
    const attention =
      qProj + oProj + kProj + vProj + l.headDim * (l.shared ? 1 : 2);
    const mlp = 3 * h * l.intermediate,
      ple = 2 * h * g.pleDim + h,
      norm = 4 * h;
    return {
      qProj,
      kProj,
      vProj,
      oProj,
      attention,
      mlp,
      ple,
      norm,
      total: attention + mlp + ple + norm,
    };
  });
  const embedding = m.vocab * h,
    pleTable = g.pleVocab * m.layers * g.pleDim,
    pleProjection = h * m.layers * g.pleDim + g.pleDim,
    lmHead = gemmaHasHead(m) ? embedding : 0;
  const text =
    embedding +
    pleTable +
    pleProjection +
    h +
    layers.reduce((a, l) => a + l.total, 0) +
    (m.tiedEmbeddings ? 0 : lmHead);
  let vision = 0,
    audio = 0;
  if (g.vision) {
    const v = g.vision;
    vision =
      3 * v.patch ** 2 * v.hidden +
      2 * v.positions * v.hidden +
      v.layers *
        (2 * v.hidden * (v.heads + v.kvHeads) * v.headDim +
          2 * v.headDim +
          3 * v.hidden * v.intermediate +
          4 * v.hidden) +
      v.hidden * h;
  }
  if (g.audio) {
    const a = g.audio,
      H = a.hidden,
      [c0, c1] = a.channels;
    audio =
      9 * c0 +
      c0 +
      9 * c0 * c1 +
      c1 +
      (c0 / 4) * c1 * H +
      a.layers * (24 * H * H + (9 + a.kernel) * H + H / a.heads) +
      H * a.output +
      a.output +
      a.output * h;
  }
  return {
    ...layers[0],
    gateProj: h * m.intermediate,
    upProj: h * m.intermediate,
    downProj: h * m.intermediate,
    layer: layers[0].total,
    embedding,
    lmHead,
    pleTable,
    pleProjection,
    text,
    vision,
    audio,
    layers,
    total: text + vision + audio,
    nonEmbeddingText: text - embedding - pleTable,
  };
}
export function gemmaMemory(m: ModelConfig, s: Scenario) {
  const size = s.dtype === "float32" ? 4 : 2,
    g = m.gemma!;
  const owners = Array.from({ length: g.shareStart }, (_, index) => {
    const sh = gemmaShapes(m, s, index);
    return {
      index,
      sliding: sh.layer.sliding,
      headDim: sh.layer.headDim,
      length: sh.retained,
      bytes: 2 * s.batch * m.kvHeads * sh.retained * sh.layer.headDim * size,
    };
  });
  return {
    weights: gemmaParameters(m).total * size,
    kvCache: owners.reduce((a, o) => a + o.bytes, 0),
    logits: gemmaHasHead(m) ? s.batch * s.sequence * m.vocab * size : 0,
    pleActivations: s.batch * s.sequence * m.layers * g.pleDim * size,
    owners,
  };
}
export function gemmaVisionShapes(m: ModelConfig, s: Scenario) {
  const v = m.gemma!.vision!,
    im = s.image ?? { count: 0, gridH: 42, gridW: 60 };
  const patches = im.gridH * im.gridW,
    tokens = patches / v.pool ** 2;
  return {
    patches,
    tokens,
    images: im.count * s.batch,
    pixels: [s.batch, patches, 3 * v.patch ** 2],
    positions: [s.batch, patches, 2],
    hidden: [s.batch, patches, v.hidden],
    pooled: [s.batch, tokens, v.hidden],
    projected: [s.batch, tokens, m.hidden],
  };
}
export function gemmaAudioShapes(m: ModelConfig, s: Scenario) {
  const a = m.gemma!.audio!,
    frames = s.audio?.frames ?? 128,
    time1 = Math.ceil(frames / 2),
    time2 = Math.ceil(time1 / 2),
    bins = a.channels[0],
    freq1 = Math.ceil(bins / 2),
    freq2 = Math.ceil(freq1 / 2),
    blocks = Math.ceil(time2 / a.chunk);
  return {
    frames,
    time1,
    time2,
    bins,
    blocks,
    input: [s.batch, frames, bins],
    conv1: [s.batch, a.channels[0], time1, freq1],
    conv2: [s.batch, a.channels[1], time2, freq2],
    hidden: [s.batch, time2, a.hidden],
    queryBlocks: [s.batch, blocks, a.chunk, a.heads, a.hidden / a.heads],
    contextBlocks: [
      s.batch,
      blocks,
      a.chunk + a.left + a.right,
      a.heads,
      a.hidden / a.heads,
    ],
    scores: [s.batch, a.heads, blocks, a.chunk, a.chunk + a.left + a.right],
    projected: [s.batch, time2, a.output],
    softTokens: [s.batch, time2, m.hidden],
  };
}
export function validateGemmaScenario(m: ModelConfig, s: Scenario) {
  let needed = 0;
  if (s.image) {
    if (!m.gemma?.vision) throw new Error("当前 Gemma 配置没有视觉分支。");
    const im = s.image,
      v = m.gemma.vision;
    if (im.count !== 0 && im.count !== 1)
      throw new Error("Gemma 图像场景目前支持每样本 0 或 1 张图像。");
    if (
      [im.gridH, im.gridW].some(
        (n) =>
          !Number.isSafeInteger(n) ||
          n < 1 ||
          n > Math.min(v.positions, 384) ||
          n % v.pool,
      )
    )
      throw new Error(
        `图像 patch 网格必须为 pooling_kernel_size=${v.pool} 的正整数倍，最大 384。`,
      );
    if (im.count) needed += gemmaVisionShapes(m, s).tokens + 2;
  }
  if (s.audio) {
    if (!m.gemma?.audio) throw new Error("当前 Gemma 配置没有音频分支。");
    if (typeof s.audio.enabled !== "boolean")
      throw new Error("audio.enabled 必须为布尔值。");
    if (
      !Number.isSafeInteger(s.audio.frames) ||
      s.audio.frames < 1 ||
      s.audio.frames > 100000
    )
      throw new Error("音频特征帧数必须为 1–100000 的整数。");
    if (s.audio.enabled) needed += gemmaAudioShapes(m, s).time2 + 2;
  }
  if (s.past === 0 && s.sequence < needed)
    throw new Error(
      `S 至少需要 ${needed} 个位置容纳图像/音频软 token 及起止标记；S 已包含这些位置。`,
    );
}
