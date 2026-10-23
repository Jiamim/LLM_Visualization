import type { ModelConfig, RawConfig, Scenario, Tensor } from "./model";

export type Mixer =
  | "attention"
  | "mla"
  | "dsa"
  | "csa"
  | "delta"
  | "kda"
  | "lightning"
  | "mamba"
  | "conv"
  | "mlstm"
  | "ffn"
  | "cca"
  | "gdla"
  | "custom";
export interface CatalogLayer {
  index: number;
  kind: Mixer;
  heads: number;
  kvHeads: number;
  headDim: number;
  valueDim: number;
  window: number | null;
  nope: boolean;
  gated: boolean;
  moe: boolean;
  intermediate: number;
  expertWidth: number;
  expertHidden: number;
  experts: number;
  topK: number;
  sharedWidth: number;
  ratio: number;
  label: string;
  source: number;
  repeats: number;
}
export interface CatalogConfig {
  type: string;
  text: RawConfig;
  profiles: CatalogLayer[];
  norm: "pre" | "post" | "parallel" | "sandwich" | "special";
  normName: string;
  qkNorm: "head" | "projection" | "none";
  kvRank: number;
  qRank: number;
  ropeDim: number;
  hc: number;
  loops: number;
  recurrent: {
    heads: number;
    keyDim: number;
    valueDim: number;
    kernel: number;
    groups: number;
    stateSize: number;
  } | null;
  activation: string;
  gatedMlp: boolean;
  attentionBias: boolean;
  mlpBias: boolean;
  routerBias: boolean;
  sink: boolean;
  tied: boolean;
  head: boolean;
  embeddingScale: number;
  mtp: number;
  softcap: number | null;
  extras: string[];
  special: boolean;
  partialParams: boolean;
  partialCache: boolean;
  mediaKeys: string[];
}
const rec = (v: unknown): v is RawConfig =>
  v !== null && typeof v === "object" && !Array.isArray(v);
export const CATALOG_TYPES = new Set([
  "kolibri1",
  "gpt2",
  "olmo2",
  "olmo3",
  "smollm3",
  "deepseek_v3",
  "deepseek_v32",
  "deepseek_v4",
  "qwen3_moe",
  "qwen3_next",
  "qwen3_5",
  "qwen3_5_text",
  "qwen3_5_moe",
  "qwen3_5_moe_text",
  "qwen4_exp",
  "qwen4_exp_text",
  "kimi_k2",
  "kimi_k25",
  "kimi_k3",
  "kimi_linear",
  "glm4_moe",
  "glm_moe_dsa",
  "glm5_next",
  "glm5_next_text",
  "gpt_oss",
  "git",
  "grok1",
  "mistral3",
  "mistral4",
  "mistral_native",
  "minimax_m2",
  "minimax_m3_vl",
  "nemotron_h",
  "nemotron_h_omni",
  "mimo_v2",
  "mimo_v2_flash",
  "afmoe",
  "step3p5",
  "step5",
  "step5_vl",
  "bailing_hybrid",
  "sarvam_moe",
  "sarvam_mla",
  "phi3",
  "xlstm",
  "ouro",
  "nanbeige",
  "longcat_flash",
  "laguna",
  "zaya",
  "hy_v3",
  "hy_v4",
  "granite",
  "granitemoehybrid",
  "cohere2",
  "cohere2_moe",
  "cohere2_vision",
  "lfm2",
  "lfm2_moe",
  "mellum",
  "inkling_mm_model",
  "Motif",
  "solar_open2",
  "muse_glimmer",
  "muse_glimmer_text",
  "muse_spark",
  "muse_spark_text",
  "naive_n05_flash",
  "gemma3",
  "gemma3_text",
  "gemma4_unified",
  "gemma4_unified_text",
  "llama4",
  "llama4_text",
]);
export function catalogText(raw: RawConfig): {
  text: RawConfig;
  type: string;
} {
  let c = raw;
  for (const key of ["text_config", "language_config", "llm_config"])
    if (rec(raw[key])) {
      c = raw[key];
      break;
    }
  let type = String(c.model_type ?? raw.model_type ?? "");
  if (
    !type &&
    Array.isArray(raw.architectures) &&
    String(raw.architectures[0]).includes("Longcat")
  )
    type = "longcat_flash";
  if (!type && typeof raw.dim === "number" && rec(raw.moe))
    type = "mistral_native";
  if (type === "nemotron_h" && Array.isArray(c.layers_block_type))
    c = {
      ...c,
      num_hidden_layers: c.layers_block_type.length,
      hybrid_override_pattern: c.layers_block_type
        .map((t) =>
          t === "mamba"
            ? "M"
            : t === "attention"
              ? "*"
              : t === "moe"
                ? "E"
                : "-",
        )
        .join(""),
    };
  if (type === "mistral_native")
    c = {
      ...raw,
      hidden_size: raw.dim,
      intermediate_size: raw.hidden_dim,
      num_hidden_layers: raw.n_layers,
      num_attention_heads: raw.n_heads,
      num_key_value_heads: raw.n_kv_heads,
      ...(raw.moe as RawConfig),
      moe_intermediate_size: (raw.moe as RawConfig).expert_hidden_dim,
    };
  return { text: c, type };
}
export function normalizeCatalog(raw: RawConfig): ModelConfig {
  const { text: c, type } = catalogText(raw);
  const architectures = raw.architectures ?? c.architectures;
  if (
    architectures !== undefined &&
    (!Array.isArray(architectures) ||
      !architectures.every(
        (a) =>
          typeof a === "string" &&
          (/ForCausalLM$|ForConditionalGeneration$|Model$/.test(a) ||
            a === "NemotronH_Omni_Reasoning_V3"),
      ))
  )
    throw new Error(
      "architectures 包含不支持的任务头；仅支持语言主干和生成模型。",
    );
  for (const key of [
    "rms_norm_eps",
    "norm_eps",
    "layer_norm_epsilon",
    "sliding_window",
    "sliding_window_size",
    "kv_lora_rank",
    "q_lora_rank",
  ])
    if (
      c[key] != null &&
      (typeof c[key] !== "number" ||
        !Number.isFinite(c[key]) ||
        (Number(c[key]) < 0 &&
          !(
            type === "git" &&
            key === "sliding_window_size" &&
            c[key] === -1
          )) ||
        (/eps|epsilon/.test(key) && Number(c[key]) === 0))
    )
      throw new Error(`${key} 不是有效的非负数或正归一化常数。`);
  const supportedLayers = new Set([
    "attention",
    "conv",
    "mamba",
    "full_attention",
    "sliding_attention",
    "hybrid_sliding",
    "linear_attention",
    "deepseek_sparse_attention",
    "hybrid",
    "window_attention",
  ]);
  if (
    c.layer_types != null &&
    (!Array.isArray(c.layer_types) ||
      !c.layer_types.every((t) => supportedLayers.has(String(t))))
  )
    throw new Error("layer_types 包含未支持的层类型。");
  if (
    c.mlp_layer_types != null &&
    (!Array.isArray(c.mlp_layer_types) ||
      !c.mlp_layer_types.every((t) => t === "dense" || t === "sparse"))
  )
    throw new Error("mlp_layer_types 包含未支持的前馈类型。");
  if (c.is_encoder_decoder === true)
    throw new Error(
      "此通用适配器不支持 encoder-decoder；请使用专用架构适配器。",
    );
  if (type === "git" && !String(raw.architectures).includes("Grok"))
    throw new Error(
      "仅适配 Grok 发布配置中的 model_type=git，不适配同名视觉 GIT 模型。",
    );
  for (const key of ["tie_word_embeddings", "attention_bias", "mlp_bias"])
    if (c[key] != null && typeof c[key] !== "boolean")
      throw new Error(`${key} 必须为布尔值。`);
  if (
    !CATALOG_TYPES.has(type) &&
    !["gemma4_text", "llama", "qwen2", "qwen3", "mistral"].includes(type)
  )
    throw new Error(`尚无 ${type} 架构规则。`);
  const n = (keys: string | string[], fallback?: number, min = 1): number => {
    for (const key of typeof keys === "string" ? [keys] : keys)
      if (c[key] != null) {
        const v = c[key];
        if (typeof v !== "number" || !Number.isSafeInteger(v) || v < min)
          throw new Error(`${key} 必须是 >=${min} 的整数。`);
        return v;
      }
    if (fallback === undefined)
      throw new Error(
        `缺少结构字段 ${(typeof keys === "string" ? [keys] : keys).join(" / ")}。`,
      );
    return fallback;
  };
  const num = (key: string, fallback: number) =>
    typeof c[key] === "number" ? (c[key] as number) : fallback;
  const arr = (key: string): unknown[] =>
    Array.isArray(c[key]) ? (c[key] as unknown[]) : [];
  const flag = (...keys: string[]) => keys.some((k) => c[k] === true);
  const h = n(["hidden_size", "n_embd", "dim", "embedding_dim"]),
    baseLayers = n([
      "num_hidden_layers",
      "n_layer",
      "n_layers",
      "num_blocks",
      "num_layers",
    ]);
  if (baseLayers > 512) throw new Error("最多支持 512 个声明层/块。");
  const isLongcat = type === "longcat_flash",
    layers = baseLayers;
  const heads = n(
      ["num_attention_heads", "n_head", "n_heads", "num_heads"],
      1,
    ),
    kv = n(
      ["num_key_value_heads", "n_kv_heads", "num_attention_groups"],
      heads,
    );
  const mla =
    typeof c.kv_lora_rank === "number" && Number(c.kv_lora_rank) > 0;
  const ropeDim = mla ? n("qk_rope_head_dim", 0, 0) : 0,
    kvRank = mla ? n("kv_lora_rank") : 0;
  const baseD = mla
    ? num("qk_nope_head_dim", num("qk_head_dim", 128) - ropeDim) + ropeDim
    : n("head_dim", h / heads, 0) || h / heads;
  if (!Number.isInteger(baseD) || baseD < 1 || heads % kv)
    throw new Error("head_dim 或 Query/KV heads 不合法。");
  let intermediate = n(
    [
      "intermediate_size",
      "ffn_hidden_size",
      "ffn_dim",
      "hidden_dim",
      "block_ff_dim",
    ],
    type === "gpt2" ? 4 * h : h * 4,
  );
  if (type === "xlstm")
    intermediate =
      Math.ceil(
        (h * num("ffn_proj_factor", 2.667)) /
          num("ffn_round_up_to_multiple_of", 64),
      ) * num("ffn_round_up_to_multiple_of", 64);
  if (type === "lfm2" && c.block_auto_adjust_ff_dim === true)
    intermediate =
      Math.ceil(
        (Math.floor((2 * intermediate) / 3) *
          num("block_ffn_dim_multiplier", 1)) /
          num("block_multiple_of", 256),
      ) * num("block_multiple_of", 256);
  const experts = n(
      [
        "n_routed_experts",
        "num_experts",
        "num_local_experts",
        "moe_num_experts",
      ],
      rec(c.moe) ? Number(c.moe.num_experts) : 0,
      0,
    ),
    topK = experts
      ? n(
          [
            "num_experts_per_tok",
            "num_experts_per_token",
            "moe_top_k",
            "moe_topk",
            "experts_top_k",
            "top_k_experts",
          ],
          1,
        )
      : 0;
  if ((type === "qwen3_moe" || c.enable_moe_block === true) && experts === 0)
    throw new Error("MoE 配置必须明确提供有效专家数。");
  if (
    type.startsWith("gemma4") &&
    (Number(c.num_kv_shared_layers) > 0 ||
      Number(c.hidden_size_per_layer_input) > 0)
  )
    throw new Error(
      "共享 KV / PLE 的 Gemma E 系列应使用专用适配器，不能与大模型模式混用。",
    );
  if (topK > experts + num("zero_expert_num", 0))
    throw new Error("Top-K 大于可选专家数。");
  const eWidth = n(
    [
      "moe_intermediate_size",
      "expert_intermediate_size",
      "expert_ffn_hidden_size",
      "expert_hidden_dim",
    ],
    intermediate,
  );
  const sharedCount = n(["n_shared_experts", "num_shared_experts"], 0, 0);
  const sharedWidth = n(
    [
      "shared_expert_intermediate_size",
      "moe_shared_expert_intermediate_size",
      "share_expert_dim",
      "shared_intermediate_size",
    ],
    sharedCount * eWidth,
    0,
  );
  const densePrefix = n(
    ["first_k_dense_replace", "num_dense_layers", "n_dense_first_layers"],
    0,
    0,
  );
  const moeIds = arr("moe_layers").length
    ? arr("moe_layers")
    : arr("moe_layer_list").length
      ? arr("moe_layer_list")
      : typeof c.moe_layers_enum === "string"
        ? c.moe_layers_enum.split(",").filter(Boolean).map(Number)
        : null;
  const pattern =
    typeof c.hybrid_override_pattern === "string"
      ? c.hybrid_override_pattern
      : null;
  if (pattern && pattern.length !== layers)
    throw new Error("hybrid_override_pattern 长度与层数不一致。");
  const types = arr("layer_types"),
    mlpTypes = arr("mlp_layer_types"),
    mixPattern = arr("hybrid_layer_pattern");
  if (types.length && types.length < layers)
    throw new Error("layer_types 未覆盖全部主干层。");
  const lin = rec(c.linear_attn_config) ? c.linear_attn_config : {};
  const oneBasedKda = type === "kimi_linear";
  const kdaIds = Array.isArray(lin.kda_layers)
    ? lin.kda_layers.map((x) => Number(x) - (oneBasedKda ? 1 : 0))
    : [];
  let recurrent: CatalogConfig["recurrent"] = null;
  if (pattern?.includes("M") || type === "granitemoehybrid")
    recurrent = {
      heads: n(
        ["mamba_num_heads", "mamba_n_heads"],
        Math.floor((2 * h) / 64),
      ),
      keyDim: n(["mamba_head_dim", "mamba_d_head"], 64),
      valueDim: n(["mamba_head_dim", "mamba_d_head"], 64),
      kernel: n(["conv_kernel", "mamba_d_conv"], 4),
      groups: n(["n_groups", "mamba_n_groups"], 1),
      stateSize: n(["ssm_state_size", "mamba_d_state"], 128),
    };
  else if (
    Object.keys(lin).length ||
    c.linear_key_head_dim ||
    type === "bailing_hybrid"
  )
    recurrent = {
      heads: n(["linear_num_value_heads"], Number(lin.num_heads) || heads),
      keyDim: n("linear_key_head_dim", Number(lin.head_dim) || 128),
      valueDim: n("linear_value_head_dim", Number(lin.head_dim) || 128),
      kernel: n(
        ["linear_conv_kernel_dim", "short_conv_kernel_size"],
        Number(lin.short_conv_kernel_size) || 4,
      ),
      groups: n("linear_num_key_heads", Number(lin.num_heads) || heads),
      stateSize: 0,
    };
  else if (type === "xlstm")
    recurrent = {
      heads,
      keyDim: Math.round((h * num("qk_dim_factor", 0.5)) / heads),
      valueDim: Math.round((h * num("v_dim_factor", 1)) / heads),
      kernel: 0,
      groups: heads,
      stateSize: 0,
    };
  if (
    type === "bailing_hybrid" &&
    !String(c.architectures).includes("V3") &&
    recurrent
  )
    recurrent.kernel = 0;
  const rawWindow = num("sliding_window", num("sliding_window_size", 0));
  const window = rawWindow > 0 ? rawWindow : null;
  const anySparse =
    type.includes("dsa") ||
    ["deepseek_v32", "hy_v4", "glm5_next_text", "naive_n05_flash"].includes(
      type,
    ) ||
    flag("use_dsa");
  const globalLayerIds = arr("gqa_layers");
  const profiles: CatalogLayer[] = Array.from({ length: layers }, (_, i) => {
    let kind: Mixer = mla ? (anySparse ? "dsa" : "mla") : "attention";
    let sliding =
      types[i] === "sliding_attention" || types[i] === "hybrid_sliding";
    if (mixPattern.length) sliding = mixPattern[i] === 1;
    if (!types.length && !mixPattern.length && window) {
      const period = num(
        "global_attn_every_n_layers",
        num(
          "sliding_window_period",
          num("_sliding_window_pattern", num("layer_switch", 0)),
        ),
      );
      if (period) sliding = (i + 1) % period !== 0;
    }
    if (Array.isArray(c.local_layer_ids))
      sliding = c.local_layer_ids.includes(i);
    if (pattern) {
      kind =
        pattern[i] === "M"
          ? "mamba"
          : pattern[i] === "*"
            ? "attention"
            : "ffn";
      sliding = false;
    } else if (types[i] === "conv") kind = "conv";
    else if (types[i] === "mamba") kind = "mamba";
    else if (types[i] === "linear_attention") kind = "delta";
    else if (kdaIds.length && kdaIds.includes(i)) kind = "kda";
    else if (type === "qwen3_next" && !types.length)
      kind =
        (i + 1) % num("full_attention_interval", 4) ? "delta" : "attention";
    else if (type === "bailing_hybrid")
      kind =
        (i + 1) % num("layer_group_size", 8)
          ? flag("use_kda_lora", "no_kda_lora") ||
            String(c.architectures).includes("V3")
            ? "kda"
            : "lightning"
          : "mla";
    else if (type === "solar_open2")
      kind = globalLayerIds.includes(i) ? "attention" : "kda";
    else if (type === "xlstm") kind = "mlstm";
    else if (type === "zaya") kind = "cca";
    else if (type === "Motif") kind = "gdla";
    else if (type === "deepseek_v4") kind = "csa";
    if (["qwen4_exp_text"].includes(type) && kind === "attention")
      kind = "dsa";
    const sparse = rec(c.sparse_attention_config)
      ? c.sparse_attention_config
      : rec(c.sparse_config)
        ? c.sparse_config
        : null;
    if (
      sparse &&
      (sparse.enabled === true || sparse.use_sparse_attention === true) &&
      !sliding
    ) {
      const freq = Array.isArray(sparse.sparse_attention_freq)
        ? sparse.sparse_attention_freq
        : [];
      if (!freq.length || freq[i]) kind = "dsa";
    }
    let localHeads = Number(arr("num_attention_heads_per_layer")[i]) || heads,
      localKv = kv,
      dim = baseD,
      vDim = n("v_head_dim", baseD);
    if (sliding) {
      localHeads = n(
        ["swa_num_attention_heads", "num_sliding_attention_heads"],
        localHeads,
      );
      localKv = n(
        ["swa_num_key_value_heads", "num_attention_groups_sliding_attention"],
        localKv,
      );
      dim = n(["swa_head_dim", "head_dim_sliding_attention"], dim);
      vDim = n("swa_v_head_dim", vDim);
    }
    if (
      rec(c.attention_other_setting) &&
      c.attention_other_setting.attention_type === types[i]
    ) {
      const a = c.attention_other_setting;
      localHeads = Number(a.num_attention_heads) || localHeads;
      localKv = Number(a.num_attention_groups) || localKv;
      dim = Number(a.head_dim) || dim;
    }
    if (type === "gemma4_text" || type === "gemma4_unified_text") {
      dim = sliding ? n("head_dim") : n("global_head_dim", 512);
      if (!sliding && c.attention_k_eq_v)
        localKv = n("num_global_key_value_heads", kv);
      vDim = dim;
    }
    if (type === "llama4_text") {
      sliding = arr("no_rope_layers")[i] === 1;
    }
    const moe = pattern
      ? pattern[i] === "E"
      : experts > 0 &&
        (moeIds
          ? moeIds.includes(i)
          : mlpTypes.length
            ? mlpTypes[i] !== "dense"
            : Array.isArray(c.moe_layer_freq)
              ? Boolean(c.moe_layer_freq[i])
              : i >= densePrefix &&
                (i - densePrefix) %
                  num("moe_layer_freq", num("decoder_sparse_step", 1)) ===
                  0);
    const localI =
      type === "llama4_text" && !moe
        ? n("intermediate_size_mlp", intermediate)
        : !moe && c.dense_intermediate_size
          ? n("dense_intermediate_size")
          : !moe && c.prefix_dense_intermediate_size
            ? n("prefix_dense_intermediate_size")
            : intermediate;
    const ropeArray = arr("layer_rope_theta"),
      nope =
        type === "gpt2" ||
        pattern !== null ||
        flag("mla_use_nope", "use_mla_nope") ||
        c.use_rope === false ||
        (["cohere2", "cohere2_moe"].includes(type) && !sliding) ||
        ropeArray[i] === 0 ||
        (type === "llama4_text" && arr("no_rope_layers")[i] === 0) ||
        (type === "smollm3" &&
          (i + 1) % num("no_rope_layer_interval", 4) === 0);
    const gated =
      flag(
        "gating",
        "attn_output_gate",
        "use_head_wise_attn_gate",
        "mla_use_output_gate",
        "use_gqa_gate",
        "gated_mla",
        "elementwise_attn_output_gate",
        "attention_output_gate",
      ) ||
      type.startsWith("qwen3_5") ||
      type === "qwen3_next" ||
      type.startsWith("muse_");
    const ratio = Number(arr("compress_ratios")[i]) || 0;
    let source = i;
    if (
      ["gemma4_text", "gemma4_unified_text"].includes(type) &&
      i >= layers - num("num_kv_shared_layers", 0)
    ) {
      const before = types.slice(0, layers - num("num_kv_shared_layers", 0));
      source = before.lastIndexOf(types[i]);
    }
    if (
      !Number.isSafeInteger(localHeads) ||
      !Number.isSafeInteger(localKv) ||
      localHeads < 1 ||
      localKv < 1 ||
      localHeads % localKv ||
      !Number.isSafeInteger(dim) ||
      dim < 1 ||
      !Number.isSafeInteger(vDim) ||
      vDim < 1
    )
      throw new Error(`Layer ${i} 的 Query/KV 头维度不合法。`);
    return {
      index: i,
      kind,
      heads: localHeads,
      kvHeads: localKv,
      headDim: dim,
      valueDim: vDim,
      window: sliding
        ? type === "llama4_text"
          ? n("attention_chunk_size", 8192)
          : window
        : null,
      nope,
      gated,
      moe,
      intermediate: localI,
      expertWidth: eWidth,
      expertHidden: n(
        ["moe_latent_size", "routed_expert_hidden_size", "moe_hidden_size"],
        h,
      ),
      experts,
      topK,
      sharedWidth,
      ratio,
      label:
        kind === "attention"
          ? sliding
            ? "Sliding GQA"
            : localHeads === localKv
              ? "MHA"
              : "GQA"
          : kind.toUpperCase(),
      source,
      repeats: isLongcat ? 2 : 1,
    };
  });
  const loops = n(["total_ut_steps", "num_loops"], 1),
    hc = n(["hc_mult", "mhc_expansion_rate", "hc_count"], 1),
    mtp = n(["num_nextn_predict_layers", "mtp_num_hidden_layers"], 0, 0);
  if (loops > 32 || loops * layers > 4096)
    throw new Error("主干循环超出可视化上限：最多 32 轮、4096 次层调用。");
  const qkNorm =
    type === "olmo2" || type === "olmo3" || type === "minimax_m2"
      ? "projection"
      : flag("use_qk_norm", "qk_norm") ||
          [
            "qwen3_moe",
            "minimax_m2",
            "lfm2",
            "lfm2_moe",
            "gemma3_text",
            "gemma4_text",
            "gemma4_unified_text",
            "muse_glimmer_text",
            "muse_spark_text",
          ].includes(type) ||
          type.startsWith("qwen3_5") ||
          type === "qwen3_next"
        ? "head"
        : "none";
  const normName =
    type === "gpt2" || type.startsWith("cohere2") ? "LayerNorm" : "RMSNorm";
  const norm: CatalogConfig["norm"] =
    type === "olmo2" || type === "olmo3"
      ? "post"
      : type.startsWith("cohere2")
        ? "parallel"
        : type.startsWith("gemma") || type === "muse_glimmer_text"
          ? "sandwich"
          : hc > 1 ||
              isLongcat ||
              c.attn_res_block_size ||
              type.startsWith("muse_")
            ? "special"
            : "pre";
  const special =
    profiles.some((p) =>
      ["custom", "csa", "gdla", "cca", "lightning", "mlstm"].includes(p.kind),
    ) ||
    norm === "special" ||
    !!raw._atlas_provenance;
  const completeTypes = [
    "gpt2",
    "qwen3_moe",
    "olmo2",
    "olmo3",
    "phi3",
    "smollm3",
    "glm4_moe",
    "gpt_oss",
    "deepseek_v3",
    "kimi_k2",
    "minimax_m2",
    "lfm2",
    "lfm2_moe",
    "nemotron_h",
    "qwen3_next",
    "qwen3_5_text",
    "qwen3_5_moe_text",
  ];
  const mediaKeys = Object.keys(raw).filter(
    (k) => /vision|audio|video/.test(k) && rec(raw[k]),
  );
  const extras: string[] = [];
  if (c.attn_res_block_size)
    extras.push(
      `Attention Residual 分块 ${c.attn_res_block_size}；残差权重需运行时计算。`,
    );
  if (hc > 1)
    extras.push(
      `Hyper-Connections 残差轴 HC=${hc}；混合子算子保留结构说明，不套用普通 residual add。`,
    );
  if (
    c.ngram_vocab_size_ratio ||
    c.ngram_vocab_size_base ||
    c.engram_layer_ids
  )
    extras.push(
      "包含 N-gram / Engram 条件记忆；表布局与哈希索引在独立辅助模块中列出，不计入部分主干权重。",
    );
  if (raw._atlas_adapter)
    extras.push(
      "LoRA 派生：主图使用其声明的基座配置；adapter 权重与训练结果不等于基座权重。",
    );
  if (isLongcat)
    extras.push(
      `原始 config num_layers=${baseLayers}，每块有两次 MLA、两次 dense MLP 与快捷 MoE。采用官方代码的块语义，不沿用画廊标题的层数。`,
    );
  if (c.use_cache === false)
    extras.push(
      "配置默认 use_cache=false。缓存页仅展示开启推理缓存时的理论状态，不宣称此配置默认会分配。",
    );
  if (types.length > layers)
    extras.push(
      `layer_types 的额外 ${types.length - layers} 项属于辅助层；主干只使用 num_hidden_layers=${layers} 项。`,
    );
  const cap =
    typeof c.final_logit_softcapping === "number"
      ? c.final_logit_softcapping
      : typeof c.output_logit_soft_cap === "number"
        ? c.output_logit_soft_cap
        : null;
  const dtype = String(
    raw.dtype ?? c.dtype ?? c.torch_dtype ?? raw.torch_dtype ?? "bfloat16",
  );
  const catalog: CatalogConfig = {
    type,
    text: c,
    profiles,
    norm,
    normName,
    qkNorm,
    kvRank,
    qRank: n("q_lora_rank", 0, 0),
    ropeDim,
    hc,
    loops,
    recurrent,
    activation: String(
      c.mlp_hidden_act ??
        c.hidden_act ??
        c.hidden_activation ??
        c.activation_function ??
        "silu",
    ),
    gatedMlp: type !== "gpt2" && type !== "nemotron_h",
    attentionBias: flag("attention_bias", "use_qkv_bias") || type === "gpt2",
    mlpBias: flag("mlp_bias") || type === "gpt2" || type === "gpt_oss",
    routerBias: type === "gpt_oss",
    sink:
      type === "gpt_oss" ||
      flag("learnable_sink", "sink", "add_swa_attention_sink_bias"),
    tied:
      c.tie_word_embeddings === true ||
      c.tie_embedding === true ||
      c.tied_embeddings === true ||
      type === "gpt2",
    head:
      (type === "gpt2" && String(architectures).includes("LMHeadModel")) ||
      !String(
        (architectures as string[] | undefined)?.[0] ?? "ForCausalLM",
      ).endsWith("Model"),
    embeddingScale: num("embedding_multiplier", 1),
    mtp,
    softcap: cap,
    extras,
    special,
    partialParams:
      !completeTypes.includes(type) ||
      special ||
      mediaKeys.length > 0 ||
      mtp > 0,
    partialCache:
      special ||
      mtp > 0 ||
      anySparse ||
      profiles.some((p) => p.kind === "dsa"),
    mediaKeys,
  };
  const model: ModelConfig = {
    family: "catalog",
    architecture: String(
      (raw.architectures as string[] | undefined)?.[0] ??
        (c.architectures as string[] | undefined)?.[0] ??
        type,
    ),
    hidden: h,
    intermediate,
    layers,
    heads,
    kvHeads: kv,
    headDim: baseD,
    vocab: n("vocab_size"),
    maxPosition: n(
      [
        "max_position_embeddings",
        "max_seq_len",
        "model_max_length",
        "n_positions",
      ],
      null as unknown as number,
    ),
    activation: catalog.activation,
    normEps: num(
      "rms_norm_eps",
      num("norm_eps", num("layer_norm_epsilon", 1e-5)),
    ),
    dtype: ["bfloat16", "float16", "float32"].includes(dtype)
      ? (dtype as ModelConfig["dtype"])
      : "bfloat16",
    tiedEmbeddings: catalog.tied,
    qkvBias: catalog.attentionBias,
    outputBias: catalog.attentionBias,
    mlpBias: catalog.mlpBias,
    qkNorm: qkNorm !== "none",
    slidingWindow: window,
    slidingLayers: profiles.map((p) => !!p.window),
    catalog,
    raw,
    warnings: [
      `画廊语言主干适配（${type}）：显示配置可确定的模块与逻辑张量，不执行远程模型代码。`,
      "复杂算子的图为逻辑结构；动态专家、稀疏索引、门控数值、实际激活/峰值显存不能由配置推断。",
      ...(catalog.partialParams
        ? [
            "参数栏为语言主干结构估算，不是完整或经权重文件验证的精确参数；专用算子、媒体与辅助预测器可能未计入。",
          ]
        : []),
      ...(catalog.partialCache
        ? [
            "缓存栏显示可推导的主干状态部分；未包含专用算子、辅助预测器、临时索引和内核工作区。",
          ]
        : []),
      ...(mediaKeys.length
        ? [
            `此画廊视图聚焦语言主干。原始配置包含 ${mediaKeys.join(", ")}，媒体编码器内部不纳入本图。`,
          ]
        : []),
      ...extras,
    ],
  };
  catalogParameters(model);
  for (const p of profiles) {
    for (const value of [p.intermediate, p.expertWidth, p.expertHidden])
      if (!Number.isSafeInteger(value) || value <= 0)
        throw new Error("层维度超出安全整数范围。");
    for (const state of catalogState(
      model,
      { batch: 1, sequence: 1, past: 0, dtype: model.dtype },
      p,
    ))
      if (state.shape.some((v) => !Number.isSafeInteger(v) || v < 0))
        throw new Error("缓存状态维度不合法。");
  }
  return model;
}

export function catalogLayerParams(m: ModelConfig, p: CatalogLayer) {
  const a = m.catalog!,
    h = m.hidden,
    query = p.heads * p.headDim,
    key = p.kvHeads * p.headDim,
    val = p.kvHeads * p.valueDim;
  let attention = 0;
  if (["attention", "dsa"].includes(p.kind) && !a.kvRank) {
    attention =
      h * (query + key + val + p.heads * p.valueDim) +
      (a.attentionBias ? query + key + val + h : 0);
    if (a.qkNorm === "head") attention += 2 * p.headDim;
    else if (a.qkNorm === "projection") attention += query + key;
    if (p.gated)
      attention +=
        h *
        (a.text.use_head_wise_attn_gate || a.text.gating_type === "headwise"
          ? p.heads
          : p.heads * p.valueDim);
    if (
      a.type.startsWith("gemma4") &&
      !p.window &&
      a.text.attention_k_eq_v === true
    )
      attention -= h * val + (a.attentionBias ? val : 0);
    if (a.type.startsWith("muse_") && a.qkNorm === "head")
      attention -= 2 * p.headDim;
    if (a.sink) attention += p.heads;
    if (a.type === "glm4_moe" && a.attentionBias) attention -= h;
  } else if (["mla", "dsa"].includes(p.kind) && a.kvRank) {
    attention =
      (a.qRank ? h * a.qRank + a.qRank + a.qRank * query : h * query) +
      h * (a.kvRank + a.ropeDim) +
      a.kvRank +
      a.kvRank * p.heads * (p.headDim - a.ropeDim + p.valueDim) +
      p.heads * p.valueDim * h;
  } else if (p.kind === "csa") {
    const rank = Number(a.text.q_lora_rank) || 0,
      groups = Number(a.text.o_groups) || 1,
      outRank = Number(a.text.o_lora_rank) || 0;
    attention =
      h * rank +
      rank +
      rank * p.heads * p.headDim +
      h * p.headDim +
      p.headDim +
      p.heads * p.headDim * outRank +
      groups * outRank * h +
      p.heads;
  } else if (p.kind === "gdla") {
    const signals = p.heads - Number(a.text.num_noise_heads || 0);
    attention =
      h * a.qRank +
      a.qRank +
      a.qRank * p.heads * p.headDim +
      h * (a.kvRank + a.ropeDim) +
      a.kvRank +
      a.kvRank * p.kvHeads * (p.headDim - a.ropeDim + p.valueDim) +
      signals * p.valueDim * h;
    if (a.text.elementwise_attn_output_gate)
      attention += a.qRank * signals * p.valueDim;
  } else if (p.kind === "cca") {
    const channels = (p.heads + p.kvHeads) * p.headDim,
      k0 = Number(a.text.cca_time0) || 2,
      k1 = Number(a.text.cca_time1) || 2;
    attention =
      h * (query + key + val + p.heads * p.valueDim) +
      channels * k0 +
      channels +
      channels * p.headDim * k1 +
      channels +
      p.kvHeads;
  } else if (p.kind === "mlstm" && a.recurrent) {
    const r = a.recurrent;
    attention =
      h * r.heads * (2 * r.keyDim + 2 * r.valueDim) +
      2 * h * r.heads +
      2 * r.heads +
      r.heads * r.valueDim * h +
      r.heads * r.valueDim;
  } else if (p.kind === "lightning" && a.recurrent) {
    const r = a.recurrent;
    attention =
      h * r.heads * (2 * r.keyDim + 2 * r.valueDim) +
      r.heads * r.valueDim * h;
  } else if (p.kind === "conv")
    attention =
      4 * h * h +
      h * (a.recurrent?.kernel ?? (Number(a.text.conv_L_cache) || 3));
  else if (p.kind === "mamba" && a.recurrent) {
    const r = a.recurrent,
      inner = r.heads * r.valueDim,
      conv = inner + 2 * r.groups * r.stateSize;
    attention =
      h * (inner + conv + r.heads) +
      conv * r.kernel +
      3 * r.heads +
      inner +
      inner * h +
      (a.text.use_conv_bias !== false ? conv : 0);
  } else if (["delta", "kda"].includes(p.kind) && a.recurrent) {
    const r = a.recurrent;
    attention =
      h * (2 * r.groups * r.keyDim + 2 * r.heads * r.valueDim + 2 * r.heads) +
      (2 * r.groups * r.keyDim + r.heads * r.valueDim) * r.kernel +
      r.heads * r.valueDim * h +
      2 * r.heads +
      r.valueDim;
  }
  const mlpFactor = a.gatedMlp ? 3 : 2;
  let mlp = p.moe
    ? p.experts *
        (mlpFactor * p.expertHidden * p.expertWidth +
          (a.mlpBias ? 2 * p.expertWidth + p.expertHidden : 0)) +
      h * p.experts +
      (a.routerBias ? p.experts : 0) +
      mlpFactor * h * p.sharedWidth +
      (p.expertHidden !== h ? 2 * h * p.expertHidden : 0)
    : p.kind === "ffn" || !["mamba"].includes(p.kind)
      ? mlpFactor * h * p.intermediate + (a.mlpBias ? p.intermediate + h : 0)
      : 0;
  if (a.type.startsWith("gemma4") && p.moe)
    mlp += 3 * h * p.intermediate + 4 * h + p.experts;
  if (a.type === "llama4_text" && p.moe && !p.sharedWidth)
    mlp += 3 * h * p.intermediate;
  if (a.type === "qwen3_next" && p.sharedWidth) mlp += h;
  if (p.kind === "ffn") attention = 0;
  if (a.text.hybrid_override_pattern && p.kind !== "ffn") mlp = 0;
  const norm =
    (a.text.hybrid_override_pattern
      ? 1
      : a.norm === "sandwich"
        ? 4
        : a.norm === "parallel"
          ? 1
          : 2) *
    h *
    (a.normName === "LayerNorm" && a.type === "gpt2" ? 2 : 1);
  return {
    attention,
    mlp,
    norm,
    total:
      attention * p.repeats +
      mlp +
      norm +
      (a.type === "longcat_flash" ? 6 * h * p.intermediate + 2 * h : 0),
  };
}
export function catalogParameters(m: ModelConfig) {
  const a = m.catalog!,
    perLayer = a.profiles.map((p) => catalogLayerParams(m, p)),
    embedding = m.vocab * m.hidden,
    lmHead = a.head ? embedding : 0,
    position = a.type === "gpt2" ? (m.maxPosition ?? 0) * m.hidden : 0,
    total =
      embedding +
      position +
      (a.tied ? 0 : lmHead) +
      perLayer.reduce((sum, p) => sum + p.total, 0) +
      m.hidden * (a.type === "gpt2" ? 2 : 1);
  if (!Number.isSafeInteger(total))
    throw new Error("结构参数规模超过安全整数。");
  return {
    qProj: 0,
    kProj: 0,
    vProj: 0,
    oProj: 0,
    attention: perLayer[0].attention,
    gateProj: 0,
    upProj: 0,
    downProj: 0,
    mlp: perLayer[0].mlp,
    layer: perLayer[0].total,
    embedding,
    lmHead,
    total,
    perLayer,
    position,
    partial: a.partialParams,
  };
}
export function catalogState(
  m: ModelConfig,
  s: Scenario,
  p: CatalogLayer,
): Tensor[] {
  const a = m.catalog!,
    total = s.past + s.sequence,
    length = p.window ? Math.min(total, p.window) : total,
    base = { dtype: s.dtype } as const;
  if (["attention", "dsa"].includes(p.kind) && !a.kvRank)
    return [
      {
        ...base,
        name: "key_cache",
        shape: [s.batch, p.kvHeads, length, p.headDim],
        symbols: ["B", "Nkv", "Tcache", "D"],
      },
      {
        ...base,
        name: "value_cache",
        shape: [s.batch, p.kvHeads, length, p.valueDim],
        symbols: ["B", "Nkv", "Tcache", "Dv"],
      },
    ];
  if (["mla", "dsa"].includes(p.kind) && a.kvRank)
    return [
      {
        ...base,
        name: "compressed_KV",
        shape: [s.batch, total, a.kvRank],
        symbols: ["B", "T", "Rkv"],
        note: "MLA 吸收投影的潜在缓存表示；朴素后端可能保留展开 K/V。",
      },
      ...(a.ropeDim
        ? [
            {
              ...base,
              name: "rotary_key",
              shape: [s.batch, total, a.ropeDim],
              symbols: ["B", "T", "Dr"],
            },
          ]
        : []),
    ];
  if (p.kind === "csa")
    return [
      {
        ...base,
        name: "window_latent",
        shape: [s.batch, Math.min(total, m.slidingWindow ?? 128), p.headDim],
        symbols: ["B", "W", "D"],
      },
      ...(p.ratio
        ? [
            {
              ...base,
              name: "compressed_latent",
              shape: [s.batch, Math.floor(total / p.ratio), p.headDim],
              symbols: ["B", "C", "D"],
              note: "已完成压缩条目；未包含重叠组、indexer 与临时 buffers。",
            },
          ]
        : []),
    ];
  if (p.kind === "gdla")
    return [
      {
        ...base,
        name: "gdla_kv_latent",
        shape: [s.batch, length, a.kvRank + a.ropeDim],
        symbols: ["B", "Tcache", "Rkv+Dr"],
        note: "潜在 KV 表示；展开缓存取决于后端。",
      },
    ];
  if (p.kind === "cca")
    return [
      {
        ...base,
        name: "key_cache",
        shape: [s.batch, p.kvHeads, length, p.headDim],
        symbols: ["B", "Nkv", "Tcache", "D"],
      },
      {
        ...base,
        name: "value_cache",
        shape: [s.batch, p.kvHeads, length, p.valueDim],
        symbols: ["B", "Nkv", "Tcache", "Dv"],
      },
      {
        ...base,
        name: "qk_conv_state",
        shape: [
          s.batch,
          (p.heads + p.kvHeads) * p.headDim,
          Number(a.text.cca_time0) + Number(a.text.cca_time1) - 2,
        ],
        symbols: ["B", "Cconv", "Kconv"],
      },
      {
        ...base,
        name: "delayed_value",
        shape: [s.batch, (p.kvHeads * p.valueDim) / 2],
        symbols: ["B", "Nkv×Dv/2"],
      },
    ];
  const r = a.recurrent;
  if (p.kind === "mamba" && r)
    return [
      {
        name: "ssm_state",
        shape: [s.batch, r.heads, r.valueDim, r.stateSize],
        symbols: ["B", "Nstate", "Dv", "Dstate"],
        dtype: "float32",
      },
      {
        ...base,
        name: "conv_state",
        shape: [
          s.batch,
          r.heads * r.valueDim + 2 * r.groups * r.stateSize,
          r.kernel,
        ],
        symbols: ["B", "Cconv", "Kconv"],
      },
    ];
  if (p.kind === "mlstm" && r)
    return [
      {
        name: "matrix_state",
        shape: [s.batch, r.heads, r.keyDim, r.valueDim],
        symbols: ["B", "Nstate", "Dk", "Dv"],
        dtype: "float32",
      },
      {
        name: "normalizer_state",
        shape: [s.batch, r.heads, r.keyDim],
        symbols: ["B", "Nstate", "Dk"],
        dtype: "float32",
      },
      {
        name: "stabilizer_state",
        shape: [s.batch, r.heads, 1],
        symbols: ["B", "Nstate", "1"],
        dtype: "float32",
      },
    ];
  if (["delta", "kda", "lightning"].includes(p.kind) && r)
    return [
      {
        name: "recurrent_matrix",
        shape: [s.batch, r.heads, r.keyDim, r.valueDim],
        symbols: ["B", "Nstate", "Dk", "Dv"],
        dtype: "float32",
        note: "逻辑固定大小 recurrent state；未包含每个内核的全部辅助标量。",
      },
      ...(r.kernel
        ? [
            {
              ...base,
              name: "short_conv_state",
              shape: [
                s.batch,
                2 * r.groups * r.keyDim + r.heads * r.valueDim,
                r.kernel,
              ],
              symbols: ["B", "Cconv", "Kconv"],
            },
          ]
        : []),
    ];
  if (p.kind === "conv")
    return [
      {
        ...base,
        name: "conv_state",
        shape: [s.batch, m.hidden, Number(a.text.conv_L_cache) || 3],
        symbols: ["B", "H", "Kconv"],
      },
    ];
  return [];
}
export function catalogMemory(m: ModelConfig, s: Scenario) {
  const size = s.dtype === "float32" ? 4 : 2,
    a = m.catalog!;
  let kvCache = 0;
  for (const p of a.profiles)
    for (const state of catalogState(m, s, p))
      kvCache +=
        state.shape.reduce((a, b) => a * b, 1) *
        (state.dtype === "float32" ? 4 : size) *
        p.repeats *
        a.loops;
  return {
    weights: catalogParameters(m).total * size,
    kvCache,
    logits: a.head ? s.batch * s.sequence * m.vocab * size : 0,
  };
}

/** A partition of the existing parameter estimate, with each component counted once. */
export function catalogAttentionParts(m: ModelConfig, p: CatalogLayer) {
  const a = m.catalog!,
    h = m.hidden,
    q = p.heads * p.headDim;
  if (!["attention", "mla", "dsa"].includes(p.kind)) return null;
  let query: number, kv: number;
  if (a.kvRank) {
    query = a.qRank ? h * a.qRank + a.qRank + a.qRank * q : h * q;
    kv = h * (a.kvRank + a.ropeDim) + a.kvRank;
  } else {
    const k = p.kvHeads * p.headDim,
      v = p.kvHeads * p.valueDim;
    const norm = a.type.startsWith("muse_")
      ? 0
      : a.qkNorm === "head"
        ? p.headDim
        : a.qkNorm === "projection"
          ? q
          : 0;
    query = h * q + (a.attentionBias ? q : 0) + norm;
    kv =
      h * (k + v) +
      (a.attentionBias ? k + v : 0) +
      (a.type.startsWith("muse_")
        ? 0
        : a.qkNorm === "head"
          ? p.headDim
          : a.qkNorm === "projection"
            ? k
            : 0);
    if (
      a.type.startsWith("gemma4") &&
      !p.window &&
      a.text.attention_k_eq_v === true
    )
      kv -= h * v + (a.attentionBias ? v : 0);
  }
  const expansion = a.kvRank
    ? a.kvRank * p.heads * (p.headDim - a.ropeDim + p.valueDim)
    : 0;
  const output = catalogLayerParams(m, p).attention - query - kv - expansion;
  return { query, kv, expansion, output };
}
export function catalogMlpParts(m: ModelConfig, p: CatalogLayer) {
  const a = m.catalog!,
    h = m.hidden,
    factor = a.gatedMlp ? 3 : 2;
  const total = catalogLayerParams(m, p).mlp;
  if (!p.moe) {
    const down = h * p.intermediate + (a.mlpBias ? h : 0);
    return {
      up: total - down,
      down,
      router: 0,
      experts: 0,
      shared: 0,
      latent: 0,
      dense: 0,
      combine: 0,
    };
  }
  const router = h * p.experts + (a.routerBias ? p.experts : 0);
  const experts =
    p.experts *
    (factor * p.expertHidden * p.expertWidth +
      (a.mlpBias ? 2 * p.expertWidth + p.expertHidden : 0));
  const shared =
    factor * h * p.sharedWidth +
    (a.type === "qwen3_next" && p.sharedWidth ? h : 0);
  const latent = p.expertHidden !== h ? 2 * h * p.expertHidden : 0;
  const dense =
    a.type.startsWith("gemma4") ||
    (a.type === "llama4_text" && !p.sharedWidth)
      ? 3 * h * p.intermediate
      : 0;
  return {
    up: 0,
    down: 0,
    router,
    experts,
    shared,
    latent,
    dense,
    combine: total - router - experts - shared - latent - dense,
  };
}
