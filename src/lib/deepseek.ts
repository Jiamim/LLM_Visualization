import type { DType, ModelConfig, RawConfig, Scenario } from "./model";

export const DEEPSEEK_REVISION = "314113f8a7dadb388215396ba5fbee2c610a04d6";
export const DEEPSEEK_SOURCE = `https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/${DEEPSEEK_REVISION}`;
export type AttentionMode = "SWA" | "Full" | "Reindex" | "Reuse";
export interface DeepseekConfig {
  text: RawConfig;
  encoderLayers: number;
  qRank: number;
  oRank: number;
  oGroups: number;
  ropeDim: number;
  hc: number;
  hcIterations: number;
  experts: number;
  sharedExperts: number;
  topExperts: number;
  ratios: number[];
  kvSources: number[];
  indexSources: number[];
  indexHeads: number;
  indexDim: number;
  indexTopk: number;
  candidateSource: number;
  candidateBlocks: number;
  candidateBlockSize: number;
  engramLayers: number[];
  engramRows: number[];
  engramHeads: number;
  engramDim: number;
  ngram: number;
  draftLayers: number;
  draftBlock: number;
  draftTargets: number[];
  draftRank: number;
  draftExperts: number;
  draftTopExperts: number;
  vision: null | {
    layers: number;
    hidden: number;
    heads: number;
    intermediate: number;
    patch: number;
    downsample: number;
    maxTokens: number;
  };
}
export interface LayerProfile {
  index: number;
  partition: "Encoder" | "Decoder";
  mode: AttentionMode;
  ratio: number;
  kvSource: number | null;
  indexSource: number | null;
  candidateSource: number | null;
  engram: boolean;
}

const object = (v: unknown): v is RawConfig =>
  v !== null && typeof v === "object" && !Array.isArray(v);
export function normalizeDeepseek(raw: RawConfig): ModelConfig {
  const nested = raw.model_type === "deepseek_v41";
  const text = nested ? raw.text_config : raw;
  if (!object(text) || text.model_type !== "deepseek_v41_text")
    throw new Error(
      "DeepSeek V4.1 配置需要有效的 text_config（deepseek_v41_text）。",
    );
  const n = (
    key: string,
    fallback?: number,
    min = 1,
    source = text,
  ): number => {
    const v = source[key] ?? fallback;
    if (typeof v !== "number" || !Number.isSafeInteger(v) || v < min)
      throw new Error(`DeepSeek ${key} 必须是 >= ${min} 的安全整数。`);
    return v;
  };
  const list = (key: string, fallback?: number[], source = text): number[] => {
    const v = source[key] ?? fallback;
    if (!Array.isArray(v) || !v.every((x) => Number.isSafeInteger(x) && x >= 0))
      throw new Error(`DeepSeek ${key} 必须为非负整数数组。`);
    return v;
  };
  if (raw.auto_map || text.auto_map)
    throw new Error(
      "不执行自定义 auto_map；此适配器仅支持标准 DeepSeek V4.1 架构。",
    );
  const architectures = raw.architectures;
  if (
    architectures !== undefined &&
    (!Array.isArray(architectures) ||
      !architectures.every((x) =>
        [
          "DeepseekV41ForCausalLM",
          "DeepseekV41Model",
          "DeepseekV41TextModel",
        ].includes(String(x)),
      ))
  )
    throw new Error("DeepSeek V4.1 任务头不受支持。");
  const layers = n("num_hidden_layers");
  if (layers < 2 || layers > 256 || layers % 2 !== 0)
    throw new Error(
      "CED 主干层数必须为 2–256 的偶数，Encoder / Decoder 各占一半。",
    );
  const hidden = n("hidden_size"),
    heads = n("num_attention_heads"),
    headDim = n("head_dim");
  if (n("num_key_value_heads", 1) !== 1)
    throw new Error(
      "V4.1 CSA2 使用单一共享 KV latent，num_key_value_heads 必须为 1。",
    );
  const dtype =
    raw.dtype ??
    raw.torch_dtype ??
    text.dtype ??
    text.torch_dtype ??
    "bfloat16";
  if (!["bfloat16", "float16", "float32"].includes(String(dtype)))
    throw new Error("不支持该 DeepSeek 计算精度。");
  const eps = text.rms_norm_eps ?? 1e-20;
  if (typeof eps !== "number" || !Number.isFinite(eps) || eps <= 0)
    throw new Error("rms_norm_eps 必须为正数。");
  if (text.hidden_act !== undefined && text.hidden_act !== "silu")
    throw new Error("V4.1 专家适配器仅支持 SiLU / SwiGLU。");
  if (text.scoring_func !== undefined && text.scoring_func !== "sqrtsoftplus")
    throw new Error("V4.1 当前适配 sqrtsoftplus 路由评分。");
  if (text.topk_method !== undefined && text.topk_method !== "noaux_tc")
    throw new Error("V4.1 当前适配 noaux_tc 路由。");
  if (text.norm_topk_prob !== undefined && text.norm_topk_prob !== true)
    throw new Error("V4.1 当前适配归一化 Top-K 路由权重。");
  if (
    (text.attention_bias != null && text.attention_bias !== false) ||
    (text.mlp_bias != null && text.mlp_bias !== false) ||
    (text.tie_word_embeddings != null && text.tie_word_embeddings !== false)
  )
    throw new Error("V4.1 适配器不支持额外 linear bias 或共享 LM Head 权重。");
  const draftLayers = n("num_nextn_predict_layers", 0, 0);
  const ratios = list("compress_ratios");
  if (ratios.length !== layers && ratios.length !== layers + draftLayers)
    throw new Error(
      "compress_ratios 必须覆盖全部主干层，可额外包含 DSpark 层。",
    );
  if (!ratios.every((r) => r <= 128))
    throw new Error("compress_ratios 超出支持范围（0–128）。");
  if (ratios.slice(layers).some((r) => r !== 0))
    throw new Error("DSpark 参考实现要求其 compress_ratios 为 0。");
  const sourceList = (key: string, fallback?: number[]) => {
    const values = list(key, fallback);
    if (values.some((v, i) => v >= layers || (i > 0 && v <= values[i - 1])))
      throw new Error(`${key} 必须是严格递增且位于主干范围内的层编号。`);
    return values;
  };
  const kvSources = sourceList("kv_source_layer_ids"),
    indexSources = sourceList("index_source_layer_ids");
  const boundary = layers / 2;
  if (!kvSources.includes(boundary) || kvSources.some((i) => i > boundary))
    throw new Error(
      "CED 要求 Decoder 首层生成全局 KV，后续 Decoder 层共享它；请检查 kv_source_layer_ids。",
    );
  if (ratios[boundary] !== 1)
    throw new Error("CED Decoder 的全局 KV 必须采用 ratio=1。");
  for (let i = 0; i < layers; i++) {
    if (kvSources.includes(i) && !indexSources.includes(i))
      throw new Error("每个 KV 来源层都必须同时是索引来源层。");
    const owner = kvSources.filter((v) => v <= i).at(-1);
    const indexOwner = indexSources.filter((v) => v <= i).at(-1);
    if (ratios[i] === 0) {
      if (kvSources.includes(i) || indexSources.includes(i))
        throw new Error("纯 SWA 层不能作为全局缓存来源。");
    } else if (
      owner === undefined ||
      indexOwner === undefined ||
      indexOwner < owner ||
      ratios[owner] !== ratios[i]
    )
      throw new Error(
        `Layer ${i} 缺少兼容的 KV / Top-K 来源，或压缩比与来源不一致。`,
      );
    if (i >= boundary && ratios[i] !== 1)
      throw new Error("CED Decoder 层必须共享 ratio=1 的全局 KV。");
  }
  let vision: DeepseekConfig["vision"] = null;
  if (raw.vision_config !== undefined) {
    if (
      !object(raw.vision_config) ||
      raw.vision_config.model_type !== "deepseek_v41_vision"
    )
      throw new Error("vision_config 必须是标准 deepseek_v41_vision。");
    const v = raw.vision_config;
    vision = {
      layers: n("num_hidden_layers", undefined, 1, v),
      hidden: n("hidden_size", undefined, 1, v),
      heads: n("num_attention_heads", undefined, 1, v),
      intermediate: n("intermediate_size", undefined, 1, v),
      patch: n("patch_size", 14, 1, v),
      downsample: n("downsample_ratio", 3, 1, v),
      maxTokens: n("max_image_tokens", 1024, 1, v),
    };
    if (
      vision.hidden % vision.heads ||
      (vision.hidden / vision.heads) % 4 ||
      vision.layers > 256
    )
      throw new Error("视觉层数或 head 维度不合法。");
  }
  const ds: DeepseekConfig = {
    text,
    encoderLayers: boundary,
    qRank: n("q_lora_rank"),
    oRank: n("o_lora_rank"),
    oGroups: n("o_groups"),
    ropeDim: n("qk_rope_head_dim"),
    hc: n("hc_mult"),
    hcIterations: n("hc_sinkhorn_iters", 20),
    experts: n("n_routed_experts"),
    sharedExperts: n("n_shared_experts", 1),
    topExperts: n("num_experts_per_tok"),
    ratios: ratios.slice(0, layers),
    kvSources,
    indexSources,
    indexHeads: n("index_n_heads"),
    indexDim: n("index_head_dim"),
    indexTopk: n("index_topk"),
    candidateSource: n("candidate_source_layer_id", -1, -1),
    candidateBlocks: n("candidate_topk_blocks", 0, 0),
    candidateBlockSize: n("candidate_block_size", 1),
    engramLayers: sourceList("engram_layer_ids", []),
    engramRows: list("engram_num_embeddings", []),
    engramHeads: n("engram_n_heads", 0, 0),
    engramDim: n("engram_head_dim", 0, 0),
    ngram: n("engram_max_ngram_size", 1),
    draftLayers,
    draftBlock: n("dspark_block_size", 0, 0),
    draftTargets: sourceList("dspark_target_layer_ids", []),
    draftRank: n("dspark_markov_rank", 256),
    draftExperts: n("dspark_n_routed_experts", 1),
    draftTopExperts: n("dspark_num_experts_per_tok", 1),
    vision,
  };
  if (
    heads % ds.oGroups ||
    ds.ropeDim > Math.min(headDim, ds.indexDim) ||
    ds.ropeDim % 4 ||
    headDim % 16 ||
    ds.indexDim % 32 ||
    ds.hc > 32 ||
    ds.experts > 8192 ||
    ds.topExperts > ds.experts ||
    ds.sharedExperts !== 1
  )
    throw new Error("V4.1 head 分组、RoPE 维度或 MoE 专家配置不兼容。");
  if (
    ds.candidateSource >= 0 &&
    (ds.candidateSource !== boundary ||
      !indexSources.includes(ds.candidateSource) ||
      ds.candidateBlocks === 0)
  )
    throw new Error("分层索引候选来源必须是 Decoder 首层且候选块数为正。");
  if (
    ds.engramLayers.length !== ds.engramRows.length ||
    (ds.engramLayers.length > 0 &&
      (!ds.engramHeads ||
        !ds.engramDim ||
        ds.ngram < 2 ||
        ds.engramRows.some((r) => r === 0)))
  )
    throw new Error("Engram 层、嵌入表行数与 n-gram 配置不完整。");
  if (ds.engramLayers.some((i) => i >= boundary))
    throw new Error(
      "当前 CED 适配要求 Engram 位于 Encoder；Decoder Engram 需要额外的重放调度规则。",
    );
  if (
    draftLayers > 16 ||
    (draftLayers > 0 &&
      (!ds.draftBlock ||
        !ds.draftTargets.length ||
        ds.draftTopExperts > ds.draftExperts))
  )
    throw new Error("DSpark 草稿层配置不完整或超出支持范围。");
  const model: ModelConfig = {
    family: "deepseek_v41",
    architecture:
      typeof architectures?.[0] === "string"
        ? architectures[0]
        : nested
          ? "DeepseekV41ForCausalLM"
          : "DeepseekV41TextModel",
    hidden,
    intermediate: n("moe_intermediate_size"),
    layers,
    heads,
    kvHeads: 1,
    headDim,
    vocab: n("vocab_size"),
    maxPosition: n("max_position_embeddings"),
    activation: "silu",
    normEps: eps,
    dtype: dtype as DType,
    tiedEmbeddings: false,
    qkvBias: false,
    outputBias: false,
    mlpBias: false,
    qkNorm: false,
    slidingWindow: n("sliding_window"),
    slidingLayers: Array(layers).fill(true),
    raw,
    deepseek: ds,
    warnings: [
      "DeepSeek V4.1 专用 CED 适配器：前半为因果 Encoder，后半为 Decoder；不能套用普通 GQA 缓存公式。",
      "默认展示官方参考实现的完整前向逻辑。可切换 CED bounded replay，后者仅为生产调度示意，重放 SWA 与完整前向不严格等价。",
      "参数按参考实现的逻辑权重计算，分别列出主干、Engram、视觉与 DSpark；不包含量化 scales、并行 padding 或优化器状态。",
      "主 KV 为 K/V 共享 latent；全局缓存按来源层计一次。FP4 紧凑估算包含块 scales，与参考代码用浮点张量模拟量化的内存不同。",
      "MoE 实际选中专家及每专家 token 数取决于输入和权重；图中仅展示 Top-K 逻辑维度，不模拟路由结果。",
      "视觉尺寸使用预处理后的 patch 网格；S 包含文字、图像占位与分隔 token。图像内容、分词和实际 resize 不由配置确定。",
    ],
  };
  if (!Number.isSafeInteger(deepseekParameters(model).total))
    throw new Error("DeepSeek 参数规模超出可安全计算范围。");
  return model;
}

export function layerProfile(m: ModelConfig, index: number): LayerProfile {
  const d = m.deepseek!;
  if (!Number.isInteger(index) || index < 0 || index >= m.layers)
    throw new Error("CED 层编号超出范围。");
  const ratio = d.ratios[index];
  const kvSource = ratio ? d.kvSources.filter((v) => v <= index).at(-1)! : null;
  const indexSource = ratio
    ? d.indexSources.filter((v) => v <= index).at(-1)!
    : null;
  return {
    index,
    partition: index < d.encoderLayers ? "Encoder" : "Decoder",
    ratio,
    kvSource,
    indexSource,
    mode:
      ratio === 0
        ? "SWA"
        : kvSource === index
          ? "Full"
          : indexSource === index
            ? "Reindex"
            : "Reuse",
    candidateSource:
      d.candidateSource >= 0 && index > d.candidateSource
        ? d.candidateSource
        : null,
    engram: d.engramLayers.includes(index),
  };
}

export function validateDeepseekScenario(m: ModelConfig, s: Scenario) {
  if (s.past > 0 && s.sequence !== 1)
    throw new Error(
      "V4.1 参考实现的 Decode 每次输入 1 个 token；请设置 S=1，或将 P=0 查看 Prefill。",
    );
  if (s.cedExecution && !["reference", "bounded"].includes(s.cedExecution))
    throw new Error("未知 CED 执行模式。");
  const image = s.image;
  if (image) {
    if (!m.deepseek?.vision) throw new Error("此文本配置不包含视觉编码器。");
    for (const v of [image.gridH, image.gridW])
      if (!Number.isSafeInteger(v) || v < 1 || v > 512)
        throw new Error("图像 patch 网格必须为 1–512 的整数。");
    if (
      !Number.isSafeInteger(image.count) ||
      image.count < 0 ||
      image.count > 16
    )
      throw new Error("每样本图像数量必须为 0–16 的整数。");
    const down = m.deepseek.vision.downsample;
    const h = Math.ceil(image.gridH / down),
      w = Math.ceil(image.gridW / down);
    if (h * w + h + 2 > m.deepseek.vision.maxTokens)
      throw new Error(
        `图像网格含行/边界标记后超出 max_image_tokens=${m.deepseek.vision.maxTokens}。`,
      );
    if (
      image.count &&
      s.past === 0 &&
      image.count * (h * w + h + 2) > s.sequence
    )
      throw new Error(
        `S 必须至少容纳 ${image.count * (h * w + h + 2)} 个图像及分隔 token（S 已包含图像）。`,
      );
  }
}

export function deepseekShapes(m: ModelConfig, s: Scenario, index: number) {
  validateDeepseekScenario(m, s);
  const d = m.deepseek!,
    p = layerProfile(m, index),
    b = s.batch,
    total = s.past + s.sequence;
  const queryLength =
    s.cedExecution === "bounded" && s.past === 0 && index >= d.encoderLayers
      ? Math.min(s.sequence, m.slidingWindow!)
      : s.sequence;
  const compressed = p.ratio ? Math.floor(total / p.ratio) : 0;
  const newCompressed = p.ratio ? compressed - Math.floor(s.past / p.ratio) : 0;
  const topk = Math.min(d.indexTopk, compressed),
    window = Math.min(m.slidingWindow!, total);
  return {
    profile: p,
    queryLength,
    total,
    compressed,
    newCompressed,
    topk,
    window,
    stream: [b, queryLength, d.hc, m.hidden],
    hidden: [b, queryLength, m.hidden],
    qr: [b, queryLength, d.qRank],
    q: [b, queryLength, m.heads, m.headDim],
    localKV: [b, queryLength, m.headDim],
    windowKV: [b, window, m.headDim],
    mainKV: [b, compressed, m.headDim],
    indexK: [b, compressed, d.indexDim],
    indexQ: [b, queryLength, d.indexHeads, d.indexDim],
    indexScores: [b, queryLength, compressed],
    indices: [b, queryLength, topk],
    sparseScores: [b, m.heads, queryLength, window + topk],
    context: [b, queryLength, m.heads, m.headDim],
    grouped: [b, queryLength, d.oGroups, (m.heads * m.headDim) / d.oGroups],
    outputRank: [b, queryLength, d.oGroups, d.oRank],
  };
}

export function deepseekParameters(m: ModelConfig) {
  const d = m.deepseek!,
    h = m.hidden,
    dim = m.headDim,
    q = m.heads * dim;
  const expert = 3 * h * m.intermediate,
    router = (experts: number) => experts * h + experts * (d.vision ? 2 : 1);
  const hc = 2 * ((2 + d.hc) * d.hc * (d.hc * h) + (2 + d.hc) * d.hc + 3);
  const baseAttention =
    h * d.qRank +
    d.qRank +
    d.qRank * q +
    h * dim +
    dim +
    q * d.oRank +
    d.oGroups * d.oRank * h +
    m.heads;
  const compressor = (i: number) => h * dim * (d.ratios[i] > 1 ? 2 : 1) + dim;
  const indexer = (i: number) =>
    d.qRank * d.indexHeads * d.indexDim +
    h * d.indexHeads +
    (d.kvSources.includes(i) ? dim * d.indexDim + d.indexDim : 0);
  const engramAt = (i: number) => {
    const j = d.engramLayers.indexOf(i);
    return j < 0
      ? 0
      : d.engramRows[j] * d.engramDim +
          (d.ngram - 1) * d.engramHeads * d.engramDim * h * (d.hc + 1) +
          2 * d.hc * h;
  };
  const moe = (d.experts + d.sharedExperts) * expert + router(d.experts);
  const layers = Array.from({ length: m.layers }, (_, i) => ({
    attention:
      baseAttention +
      (d.kvSources.includes(i) ? compressor(i) : 0) +
      (d.indexSources.includes(i) ? indexer(i) : 0),
    moe,
    hc,
    norm: 2 * h,
    engram: engramAt(i),
  }));
  const layerTotals = layers.map((l) => l.attention + l.moe + l.hc + l.norm);
  const embedding = m.vocab * h,
    lmHead = m.architecture.endsWith("ForCausalLM") ? embedding : 0;
  const backbone =
    embedding + lmHead + h + layerTotals.reduce((a, b) => a + b, 0);
  const engram = layers.reduce((sum, l) => sum + l.engram, 0);
  let vision = 0;
  if (d.vision) {
    const v = d.vision;
    vision =
      3 * v.patch ** 2 * v.hidden +
      v.hidden +
      v.layers *
        (4 * v.hidden ** 2 +
          4 * v.hidden +
          3 * v.hidden * v.intermediate +
          2 * v.hidden) +
      v.hidden +
      v.hidden * v.downsample ** 2 * h +
      h +
      h * h +
      h +
      3 * h;
  }
  const draft = d.draftLayers
    ? d.draftLayers *
        (baseAttention +
          (d.draftExperts + 1) * expert +
          router(d.draftExperts) +
          hc +
          2 * h) +
      h * h * d.draftTargets.length +
      h +
      h +
      2 * m.vocab * d.draftRank +
      h +
      d.draftRank
    : 0;
  const total = backbone + engram + vision + draft;
  return {
    qProj: h * d.qRank + d.qRank * q,
    kProj: h * dim,
    vProj: 0,
    oProj: q * d.oRank + d.oGroups * d.oRank * h,
    attention: baseAttention,
    gateProj: h * m.intermediate,
    upProj: h * m.intermediate,
    downProj: h * m.intermediate,
    mlp: moe,
    layer: layerTotals[0],
    embedding,
    lmHead,
    total,
    backbone,
    engram,
    vision,
    draft,
    layers,
    layerTotals,
    expert,
    router: router(d.experts),
    hc,
    activeExpertsPerToken: (d.topExperts + d.sharedExperts) * expert,
  };
}

export function deepseekMemory(m: ModelConfig, s: Scenario) {
  const d = m.deepseek!,
    bytes = s.dtype === "float32" ? 4 : 2,
    total = s.past + s.sequence;
  const sources = d.kvSources.map((layer) => ({
    layer,
    entries: Math.floor(total / d.ratios[layer]),
    ratio: d.ratios[layer],
  }));
  const entries = sources.reduce((a, b) => a + b.entries, 0) * s.batch;
  const mainLogical = entries * m.headDim * bytes,
    indexLogical = entries * d.indexDim * bytes;
  const globalPacked =
    entries *
    (m.headDim / 2 +
      Math.ceil(m.headDim / 16) +
      d.indexDim / 2 +
      Math.ceil(d.indexDim / 32));
  const windowLogical =
    s.batch * m.layers * Math.min(total, m.slidingWindow!) * m.headDim * bytes;
  const partialState = d.kvSources
    .filter((i) => d.ratios[i] > 1)
    .reduce((sum, i) => sum + s.batch * d.ratios[i] * m.headDim * 2 * 4, 0);
  return {
    weights: deepseekParameters(m).total * bytes,
    kvCache: mainLogical + indexLogical + windowLogical + partialState,
    logits: m.architecture.endsWith("ForCausalLM") ? s.batch * m.vocab * 4 : 0,
    globalLogical: mainLogical + indexLogical,
    globalPacked,
    windowLogical,
    partialState,
    sources,
  };
}

export function visionShapes(m: ModelConfig, s: Scenario) {
  const v = m.deepseek!.vision!;
  const image = s.image ?? { count: 0, gridH: 42, gridW: 42 };
  const n = image.gridH * image.gridW,
    h = Math.ceil(image.gridH / v.downsample),
    w = Math.ceil(image.gridW / v.downsample);
  return {
    n,
    outputTokens: h * w,
    spanTokens: h * w + h + 2,
    gridH: image.gridH,
    gridW: image.gridW,
    images: s.batch * image.count,
    patches: [n, 3 * v.patch * v.patch],
    embedded: [n, v.hidden],
    downsampled: [h * w, v.hidden * v.downsample ** 2],
    aligned: [h * w, m.hidden],
  };
}
