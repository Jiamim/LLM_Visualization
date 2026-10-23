import { MarkerType, type Edge } from "@xyflow/react";
import type { ModelConfig, Scenario, Tensor } from "./model";
import { formatBytes, shapeText, validateScenario } from "./model";
import type { ModelGraph, ModuleData, ModuleNode, View } from "./graph";
import {
  deepseekShapes,
  deepseekParameters,
  deepseekMemory,
  layerProfile,
  visionShapes,
} from "./deepseek";
import { NODE_HEIGHT, NODE_WIDTH } from "./presentation";

export function buildDeepseekGraph(
  m: ModelConfig,
  s: Scenario,
  view: View,
  index: number,
  symbolic: boolean,
): ModelGraph {
  validateScenario(s);
  const d = m.deepseek!,
    sh = deepseekShapes(m, s, index),
    p = deepseekParameters(m),
    mem = deepseekMemory(m, s),
    profile = sh.profile;
  const b = s.batch,
    q = sh.queryLength,
    total = sh.total,
    base = `model.layers.${index}`;
  const nodes: ModuleNode[] = [],
    edges: Edge[] = [];
  const t = (
    name: string,
    shape: number[],
    symbols: string[],
    note?: string,
    dtype: Tensor["dtype"] = s.dtype,
  ): Tensor => ({ name, shape, symbols, dtype, note });
  const hidden = t("hidden_states", sh.hidden, ["B", "Q", "H"]);
  const stream = t(
    "residual_stream",
    sh.stream,
    ["B", "Q", "HC", "H"],
    "Single-Pass mHC 保持 HC 条并行残差流。",
  );
  const ids = t("input_ids", [b, s.sequence], ["B", "S"], undefined, "int64");
  const global = t(
    "main_kv",
    sh.mainKV,
    ["B", "C", "D"],
    `K 与 V 共用同一 latent；C=floor((P+S)/${profile.ratio || 1})，来源 Layer ${profile.kvSource}。`,
  );
  const idx = t(
    "topk_indices",
    sh.indices,
    ["B", "Q", "K"],
    "K=min(index_topk,C)。不可见位置为 -1；具体位置需要真实输入与权重。",
    "int32",
  );
  const add = (
    id: string,
    x: number,
    y: number,
    data: Partial<ModuleData> & Pick<ModuleData, "title" | "category">,
  ) => {
    nodes.push({
      id,
      type: "module",
      width: NODE_WIDTH,
      height: NODE_HEIGHT,
      ariaLabel: data.title,
      position: { x, y },
      data: {
        subtitle: "",
        path: id,
        inputs: [hidden],
        outputs: [hidden],
        params: 0,
        description: "",
        formula: "",
        details: [],
        symbolic,
        order: nodes.length + 1,
        ...data,
      },
    });
  };
  const edge = (
    source: string,
    target: string,
    tensor: Tensor = hidden,
    kind: "flow" | "shared" | "residual" = "flow",
    label?: string,
  ) => {
    edges.push({
      id: `${source}-${target}-${edges.length}`,
      source,
      target,
      type: "smoothstep",
      sourceHandle: kind === "flow" ? "out" : "right-out",
      targetHandle: kind === "flow" ? "in" : "right-in",
      label:
        label ??
        (symbolic ? `[${tensor.symbols.join(", ")}]` : shapeText(tensor.shape)),
      data: {
        tensor,
        residual: kind === "residual",
        shared: kind === "shared",
      },
      style: {
        stroke:
          kind === "shared"
            ? "#398b85"
            : kind === "residual"
              ? "#9878bd"
              : "#9aacc4",
        strokeWidth: kind === "shared" ? 2 : 1.6,
        strokeDasharray: kind === "flow" ? undefined : "6 4",
      },
      markerEnd: {
        type: MarkerType.ArrowClosed,
        color: kind === "shared" ? "#398b85" : "#9aacc4",
      },
      labelStyle: {
        fill: "#4c607a",
        fontSize: 12,
        fontFamily: "ui-monospace, monospace",
      },
      labelBgStyle: { fill: "#fff", fillOpacity: 0.98 },
      labelBgPadding: [7, 5],
      labelBgBorderRadius: 5,
    } as Edge);
  };
  const boundary = (
    id: string,
    x: number,
    y: number,
    title: string,
    tensor: Tensor = hidden,
  ) =>
    add(id, x, y, {
      title,
      subtitle: "Tensor boundary",
      category: "input",
      inputs: [tensor],
      outputs: [tensor],
      description: "当前视图的数据边界。",
    });
  const layerSummary = (i: number) => {
    const lp = layerProfile(m, i);
    return `${lp.partition} · ${lp.mode}${lp.ratio ? ` · r=${lp.ratio}` : ""}`;
  };
  const attnDetails: [string, string][] = [
    ["mode", profile.mode],
    ["partition", profile.partition],
    ["compress_ratio", String(profile.ratio)],
    [
      "global_kv_source",
      profile.kvSource === null ? "无" : `Layer ${profile.kvSource}`,
    ],
    [
      "topk_source",
      profile.indexSource === null ? "无" : `Layer ${profile.indexSource}`,
    ],
    [
      "candidate_source",
      profile.candidateSource === null
        ? "无"
        : `Layer ${profile.candidateSource}`,
    ],
    ["head_dim", String(m.headDim)],
    ["q_lora_rank", String(d.qRank)],
    ["o_groups", String(d.oGroups)],
  ];
  const attention = (id: string, x: number, y: number) =>
    add(id, x, y, {
      title: "CSA2 Attention",
      subtitle: layerSummary(index),
      category: "attention",
      path: `${base}.self_attn`,
      params: p.layers[index].attention,
      drill: "attention",
      badge: profile.mode,
      description:
        "当前层产生 Query 与本地 SWA KV；Full 层产生全局 KV 与索引 K，Reindex 层重算 Top-K，Reuse 层复用最新 Top-K。",
      formula: "SparseAttention(Q, concat(SWA_KV, shared_main_KV), indices)",
      details: attnDetails,
    });
  const moe = (id: string, x: number, y: number) =>
    add(id, x, y, {
      title: "Mixture of Experts",
      subtitle: `${d.experts} routed + ${d.sharedExperts} shared`,
      category: "mlp",
      path: `${base}.mlp`,
      params: p.layers[index].moe,
      drill: "mlp",
      badge: `Top-${d.topExperts}`,
      description:
        "每个 token 激活 Top-K 路由专家与共享专家。实际专家选择、token 分布不能从配置推断。",
      formula: "shared(x) + Σ routing_weight[e] × expert[e](x)",
      details: [
        ["n_routed_experts", String(d.experts)],
        ["num_experts_per_tok", String(d.topExperts)],
        ["moe_intermediate_size", String(m.intermediate)],
      ],
    });
  const norm = (id: string, x: number, y: number) =>
    add(id, x, y, {
      title: "RMS Normalization",
      subtitle: "RMSNorm",
      category: "norm",
      path: `${base}.${id}`,
      params: m.hidden,
      description: "仅沿隐藏维度归一化，形状保持不变。",
      formula: "x / sqrt(mean(x²) + eps) × weight",
      details: [["eps", String(m.normEps)]],
    });
  if (view === "overview") {
    const all = t(
      "embedded_tokens",
      [b, s.sequence, m.hidden],
      ["B", "S", "H"],
    );
    const enc = t(
      "encoder_stream",
      [b, s.sequence, d.hc, m.hidden],
      ["B", "S", "HC", "H"],
    );
    const decLen =
      s.cedExecution === "bounded" && s.past === 0
        ? Math.min(s.sequence, m.slidingWindow!)
        : s.sequence;
    const dec = t(
      "decoder_stream",
      [b, decLen, d.hc, m.hidden],
      ["B", "Q", "HC", "H"],
    );
    add("input", 0, 0, {
      title: "Text & Image Tokens",
      subtitle: "S includes image placeholders",
      category: "input",
      inputs: [],
      outputs: [ids],
      description:
        "S 是合并后的输入序列长度，包含图像占位及分隔 token；Decode 阶段只输入当前 token。",
    });
    add("embedding", 0, 220, {
      title: "Token Embedding",
      subtitle: "Text / image merge",
      category: "embedding",
      inputs: [ids],
      outputs: [all],
      params: p.embedding,
      description: "将文字 embedding 与视觉分支生成的图像特征在预留位置合并。",
    });
    if (d.vision) {
      add("vision", 430, 0, {
        title: "Vision Encoder",
        subtitle: `${d.vision.layers} × ViT + MLP projector`,
        category: "embedding",
        inputs: [],
        outputs: [
          t(
            "image_features",
            visionShapes(m, s).aligned,
            ["Nimg", "H"],
            "每张图的预处理网格示例；未启用图像时此分支不执行。",
          ),
        ],
        params: p.vision,
        drill: "vision",
        badge: s.image?.count ? `${s.image.count} image / sample` : "可选分支",
        description:
          "2D-RoPE ViT → pixel-unshuffle → 两层 MLP projector。此视觉编码器与语言 CED Encoder 是不同模块。",
      });
      nodes.find((n) => n.id === "vision")!.data.inactive =
        !s.image?.count || s.past > 0;
      edge(
        "vision",
        "embedding",
        nodes.find((n) => n.id === "vision")!.data.outputs[0],
        "shared",
        "image token slots",
      );
    }
    add("expand", 0, 440, {
      title: "Expand mHC Streams",
      subtitle: `${d.hc} parallel residual streams`,
      category: "operation",
      inputs: [all],
      outputs: [enc],
      formula: "[B,S,H] → [B,S,HC,H]",
      description: "复制到并行残差流，不能省略 HC 轴。",
    });
    edge("input", "embedding", ids);
    edge("embedding", "expand", all);
    add("encoder", 0, 660, {
      title: "Causal Encoder",
      subtitle: `Layer 0 – ${d.encoderLayers - 1}`,
      category: "block",
      inputs: [enc],
      outputs: [enc],
      params: p.layerTotals
        .slice(0, d.encoderLayers)
        .reduce((a, b) => a + b, 0),
      drill: "encoder",
      badge: `${d.encoderLayers} layers`,
      description:
        "因果编码器处理所有输入 token；前两层通常仅 SWA，后续按来源层共享压缩全局 KV。Engram 参数单独列出。",
      details: [
        [
          "kv_source_layers",
          d.kvSources.filter((i) => i < d.encoderLayers).join(", "),
        ],
        ["engram_layers", d.engramLayers.join(", ")],
      ],
    });
    edge("expand", "encoder", enc);
    add("ced-kv", 430, 660, {
      title: "Encoder → Global KV",
      subtitle: `Projection owned by Layer ${d.encoderLayers}`,
      category: "attention",
      inputs: [t("encoder_final", [b, s.sequence, m.hidden], ["B", "S", "H"])],
      outputs: [t("decoder_main_kv", [b, total, m.headDim], ["B", "T", "D"])],
      drill: "cache",
      badge: "shared across decoder",
      description:
        "Decoder 首个 Full 层从 Encoder 最终隐藏状态生成全局 KV 与索引 K，后续 Decoder 层复用。各 Decoder 层仍独立生成本地 SWA KV。",
    });
    edge(
      "encoder",
      "ced-kv",
      t(
        "encoder_final",
        [b, s.sequence, m.hidden],
        ["B", "S", "H"],
        "从最终 mHC 残差流经 pre-mix / norm 取得边界隐藏状态。",
      ),
      "shared",
      "final encoder states",
    );
    add("decoder", 0, 880, {
      title: "CED Decoder",
      subtitle: `Layer ${d.encoderLayers} – ${m.layers - 1}`,
      category: "block",
      inputs: [enc],
      outputs: [dec],
      params: p.layerTotals.slice(d.encoderLayers).reduce((a, b) => a + b, 0),
      drill: "decoder",
      badge:
        s.cedExecution === "bounded" && s.past === 0
          ? `Replay Q=${decLen}`
          : `${m.layers - d.encoderLayers} layers`,
      description:
        s.cedExecution === "bounded" && s.past === 0
          ? "CED 调度示意：只重放末尾 window 个 token 构造 Decoder SWA；全局 KV 仍覆盖所有 Encoder token。该近似与完整 Decoder 前向不严格等价。"
          : "参考实现对全部本次 token 运行 Decoder；全球 KV 共用 Encoder 边界缓存，Top-K 按来源分组复用。",
    });
    edge("encoder", "decoder", enc);
    edge(
      "ced-kv",
      "decoder",
      t("decoder_main_kv", [b, total, m.headDim], ["B", "T", "D"]),
      "shared",
      "shared global KV",
    );
    add("collapse", 0, 1100, {
      title: "mHC Collapse + Norm",
      subtitle: "Final residual merge",
      category: "norm",
      inputs: [dec],
      outputs: [t("last_hidden_state", [b, decLen, m.hidden], ["B", "Q", "H"])],
      params: m.hidden,
      formula: "sum(pre_mix × stream, axis=HC) → RMSNorm",
    });
    edge("decoder", "collapse", dec);
    add("lm-head", 0, 1320, {
      title: m.architecture.endsWith("ForCausalLM")
        ? "Language Model Head"
        : "Hidden State Output",
      subtitle: "Reference: last-position logits",
      category: "output",
      inputs: [t("last_hidden_state", [b, decLen, m.hidden], ["B", "Q", "H"])],
      outputs: [
        m.architecture.endsWith("ForCausalLM")
          ? t(
              "logits",
              [b, m.vocab],
              ["B", "V"],
              "参考实现只输出最后位置 logits，使用 float32。",
              "float32",
            )
          : t("last_hidden_state", [b, decLen, m.hidden], ["B", "Q", "H"]),
      ],
      params: p.lmHead,
      description:
        "主干参考生成路径只计算最后位置 logits；不会把所有 prompt 位置的 logits 当作实际分配。",
    });
    edge(
      "collapse",
      "lm-head",
      t("last_hidden_state", [b, decLen, m.hidden], ["B", "Q", "H"]),
    );
    if (d.engramLayers.length) {
      add("engram", 860, 660, {
        title: "Engram Memory",
        subtitle: `Layers ${d.engramLayers.join(", ")}`,
        category: "embedding",
        inputs: [ids],
        outputs: [enc],
        params: p.engram,
        drill: "engram",
        badge: "n-gram lookup",
        description:
          "条件记忆表在指定 Encoder 层注入残差流；单独统计，避免混淆主干和 Engram 参数。",
      });
      edge("input", "engram", ids, "shared", "token n-grams");
      edge("engram", "encoder", enc, "shared", "conditional memory");
    }
    if (d.draftLayers) {
      add("dspark", 430, 1100, {
        title: "DSpark Draft Head",
        subtitle: `${d.draftLayers} draft layers · block ${d.draftBlock}`,
        category: "mlp",
        inputs: [
          t(
            "main_hidden",
            [b, decLen, d.draftTargets.length * m.hidden],
            ["B", "Q", "Nt×H"],
          ),
        ],
        outputs: [
          t(
            "draft_logits",
            [b, d.draftBlock, m.vocab],
            ["B", "Ldraft", "V"],
            undefined,
            "float32",
          ),
        ],
        params: p.draft,
        drill: "dspark",
        badge: "可选推测解码",
        description:
          "独立草稿分支读取目标主干层隐藏状态。示意 forward_spec，不模拟接受率或验证循环。",
      });
      edge(
        "decoder",
        "dspark",
        t(
          "main_hidden",
          [b, decLen, d.draftTargets.length * m.hidden],
          ["B", "Q", "Nt×H"],
          "目标主干层 attention 输入的 HC 均值，经拼接后输出。",
        ),
        "shared",
        `target layers ${d.draftTargets.join(", ")}`,
      );
    }
  } else if (view === "encoder" || view === "decoder") {
    const start = view === "encoder" ? 0 : d.encoderLayers,
      end = view === "encoder" ? d.encoderLayers : m.layers;
    const groups: number[][] = [];
    for (let i = start; i < end; i++) {
      const lp = layerProfile(m, i);
      const last = groups.at(-1);
      const prev = last ? layerProfile(m, last[0]) : null;
      if (
        last &&
        prev?.mode === lp.mode &&
        prev?.kvSource === lp.kvSource &&
        prev?.indexSource === lp.indexSource &&
        prev?.engram === lp.engram
      )
        last.push(i);
      else groups.push([i]);
    }
    groups.forEach((group, g) => {
      const i = group[0],
        lp = layerProfile(m, i),
        ls = deepseekShapes(m, s, i);
      add(`group-${i}`, (g % 2) * 440, Math.floor(g / 2) * 250, {
        title:
          group.length === 1 ? `Layer ${i}` : `Layers ${i} – ${group.at(-1)}`,
        subtitle: `${lp.mode} · ${lp.ratio ? `CSA2 r=${lp.ratio}` : "Sliding Window"}`,
        category:
          lp.mode === "Full"
            ? "attention"
            : lp.mode === "SWA"
              ? "norm"
              : "block",
        inputs: [t("stream", ls.stream, ["B", "Q", "HC", "H"])],
        outputs: [t("stream", ls.stream, ["B", "Q", "HC", "H"])],
        params: group.reduce(
          (sum, j) => sum + p.layerTotals[j] + p.layers[j].engram,
          0,
        ),
        drill: "layer",
        layerIndex: i,
        badge: lp.engram ? "Engram" : `× ${group.length}`,
        description:
          "连续同类型层的折叠分组；展开进入首层，也可从层选择器精确选择任意层。",
        details: [
          ["partition", lp.partition],
          ["KV source", String(lp.kvSource ?? "local only")],
          ["Top-K source", String(lp.indexSource ?? "none")],
        ],
      });
      if (g)
        edge(
          `group-${groups[g - 1][0]}`,
          `group-${i}`,
          t("stream", ls.stream, ["B", "Q", "HC", "H"]),
        );
    });
  } else if (view === "layer") {
    boundary(
      "layer-input",
      0,
      0,
      `${profile.partition} Layer ${index}`,
      stream,
    );
    let first = "layer-input";
    if (profile.engram) {
      add("engram", 420, 0, {
        title: "Engram Injection",
        subtitle: "Token-conditioned memory",
        category: "embedding",
        path: `${base}.engram`,
        inputs: [stream, ids],
        outputs: [stream],
        params: p.layers[index].engram,
        drill: "engram",
        description:
          "在块入口处对 mHC 残差流加入 n-gram 条件记忆；图像位置禁用此注入。",
      });
      edge(first, "engram", stream);
      first = "engram";
    }
    add("hc-pre", 0, 230, {
      title: "mHC Pre-mix",
      subtitle: `${d.hc} streams → 1 hidden`,
      category: "operation",
      inputs: [
        stream,
        t("pre_mix", [b, q, d.hc], ["B", "Q", "HC"], undefined, "float32"),
      ],
      outputs: [
        hidden,
        t("next_pre_mix", [b, q, d.hc], ["B", "Q", "HC"], undefined, "float32"),
        t("post_gate", [b, q, d.hc], ["B", "Q", "HC"], undefined, "float32"),
        t(
          "residual_mix",
          [b, q, d.hc, d.hc],
          ["B", "Q", "HC", "HC"],
          undefined,
          "float32",
        ),
      ],
      params: p.hc / 2,
      description:
        "用上一子层输出的 pre_mix 聚合当前残差流。本子层同时从 residual stream 计算供下一子层使用的 mixing 系数。",
      formula: "[B,Q,HC,H] × [B,Q,HC] → [B,Q,H]",
    });
    edge(first, "hc-pre", stream);
    norm("attn-norm", 0, 460);
    edge("hc-pre", "attn-norm");
    attention("attention", 0, 690);
    edge("attn-norm", "attention");
    add("hc-post", 0, 920, {
      title: "mHC Attention Merge",
      subtitle: "Post gate + residual mixing",
      category: "operation",
      inputs: [
        hidden,
        stream,
        t("post_gate", [b, q, d.hc], ["B", "Q", "HC"], undefined, "float32"),
        t(
          "residual_mix",
          [b, q, d.hc, d.hc],
          ["B", "Q", "HC", "HC"],
          undefined,
          "float32",
        ),
      ],
      outputs: [stream],
      description:
        "post gate 扩展 attention 输出，与 Sinkhorn 约束的 HC×HC 残差混合矩阵组合；并非普通 residual add。",
      formula: "post × attn_out + comb @ residual",
    });
    edge("attention", "hc-post");
    edge(first, "hc-post", stream, "residual");
    add("ffn-pre", 420, 230, {
      title: "mHC FFN Pre-mix",
      subtitle: "Attention pre-mix → FFN",
      category: "operation",
      inputs: [
        stream,
        t("attn_pre_mix", [b, q, d.hc], ["B", "Q", "HC"], undefined, "float32"),
      ],
      outputs: [hidden],
      params: p.hc / 2,
      description:
        "使用 attention 子层生成的 pre_mix；FFN 的 pre_mix 传递给下一 Transformer 层。",
    });
    edge("hc-post", "ffn-pre", stream);
    edge(
      "hc-pre",
      "ffn-pre",
      t("attn_pre_mix", [b, q, d.hc], ["B", "Q", "HC"], undefined, "float32"),
      "shared",
      "pre-mix for next sublayer",
    );
    norm("ffn-norm", 420, 460);
    edge("ffn-pre", "ffn-norm");
    moe("mlp", 420, 690);
    edge("ffn-norm", "mlp");
    add("ffn-post", 420, 920, {
      title: "mHC FFN Merge",
      subtitle: "Output residual stream",
      category: "operation",
      inputs: [hidden, stream],
      outputs: [stream],
      description: "恢复并行残差流并传递到下一块。",
    });
    edge("mlp", "ffn-post");
    edge("hc-post", "ffn-post", stream, "residual");
  } else if (view === "attention") {
    boundary("attn-input", 420, 0, "Normalized Input");
    add("q-lora", 0, 220, {
      title: "Query Low-rank",
      subtitle: `H → ${d.qRank} → Nq × D`,
      category: "attention",
      path: `${base}.self_attn.wq_a`,
      outputs: [
        t("query", sh.q, ["B", "Q", "Nq", "D"]),
        t("q_rank", sh.qr, ["B", "Q", "Rq"]),
      ],
      params: m.hidden * d.qRank + d.qRank + d.qRank * m.heads * m.headDim,
      description:
        "低秩 Query 投影、RMSNorm、升维与尾部 RoPE。未旋转的 head_dim-rope_dim 维度保留原样。",
      formula: "wq_b(RMSNorm(wq_a(x))) → reshape → RoPE",
      details: [
        ["q_rank", String(d.qRank)],
        ["rope_dim", String(d.ropeDim)],
      ],
    });
    edge("attn-input", "q-lora");
    add("local-kv", 420, 220, {
      title: "Local SWA KV",
      subtitle: `Layer-local · window ${m.slidingWindow}`,
      category: "attention",
      path: `${base}.self_attn.wkv`,
      outputs: [
        t(
          "window_kv",
          sh.windowKV,
          ["B", "W", "D"],
          "有效窗口长度；参考代码分配固定 window 大小的环形缓存。",
        ),
      ],
      params: m.hidden * m.headDim + m.headDim,
      description:
        "每层独立生成 K/V 共用的 latent。本卡片展示有效窗口长度，固定容量环形缓存为 [B,window,D]；Prefill 计算读取当前 chunk，再由索引屏蔽窗口外位置。",
      formula: "RMSNorm(wkv(x)) → tail RoPE → SWA ring buffer",
    });
    edge("attn-input", "local-kv");
    if (profile.ratio) {
      add("global-kv", 840, 220, {
        title:
          profile.mode === "Full"
            ? "Global KV Projection"
            : "Read Shared Global KV",
        subtitle: `Owner Layer ${profile.kvSource} · r=${profile.ratio}`,
        category: "attention",
        inputs:
          profile.mode !== "Full"
            ? [global, t("index_k", sh.indexK, ["B", "C", "Di"])]
            : [
                profile.partition === "Decoder"
                  ? t(
                      "encoder_final",
                      [b, s.sequence, m.hidden],
                      ["B", "S", "H"],
                      "Decoder global KV 来自 Encoder 最终状态；只由边界 Full 层产生。",
                    )
                  : hidden,
              ],
        outputs: [global, t("index_k", sh.indexK, ["B", "C", "Di"])],
        params:
          profile.mode === "Full"
            ? m.hidden * m.headDim * (profile.ratio > 1 ? 2 : 1) +
              m.headDim +
              m.headDim * d.indexDim +
              d.indexDim
            : 0,
        description:
          profile.mode === "Full"
            ? "Full 层拥有 Compressor 与索引 K 投影。先产生未旋转 latent 供 indexer K 使用，再对 main KV 做 RoPE / FP4。不完整压缩组保留为临时 state。"
            : "复用最近 Full 层的全局 KV 与 indexer K；本层不新增一份全局缓存。",
        formula: `C=floor((P+S)/${profile.ratio}); new C=${sh.newCompressed}`,
        details: attnDetails,
      });
      if (profile.mode === "Full" && profile.partition === "Encoder")
        edge("attn-input", "global-kv", hidden);
      else {
        const input =
          profile.mode === "Full"
            ? t("encoder_final", [b, s.sequence, m.hidden], ["B", "S", "H"])
            : global;
        add("kv-origin", 1260, 0, {
          title:
            profile.mode === "Full"
              ? "Encoder Final States"
              : `Shared Cache · L${profile.kvSource}`,
          subtitle:
            profile.mode === "Full"
              ? "CED boundary projection input"
              : "External state · no duplicated allocation",
          category: "input",
          inputs: [],
          outputs: [
            input,
            ...(profile.mode !== "Full"
              ? [t("index_k", sh.indexK, ["B", "C", "Di"])]
              : []),
          ],
          description:
            "来自当前层外部的缓存或 Encoder 边界，不从当前 Decoder 层重新生成。",
        });
        edge(
          "kv-origin",
          "global-kv",
          input,
          "shared",
          profile.mode === "Full"
            ? "Encoder boundary states"
            : `cache from Layer ${profile.kvSource}`,
        );
      }
      add("indexer", 840, 440, {
        title:
          profile.mode === "Reuse" ? "Reuse Top-K Indices" : "Sparse Indexer",
        subtitle:
          profile.mode === "Reuse"
            ? `from Layer ${profile.indexSource}`
            : `${d.indexHeads} heads · Top-${d.indexTopk}`,
        category: "operation",
        inputs:
          profile.mode === "Reuse"
            ? [idx]
            : [
                t("q_rank", sh.qr, ["B", "Q", "Rq"]),
                hidden,
                t("index_k", sh.indexK, ["B", "C", "Di"]),
              ],
        outputs: [idx],
        params:
          profile.mode === "Reuse"
            ? 0
            : d.qRank * d.indexHeads * d.indexDim + m.hidden * d.indexHeads,
        description:
          profile.mode === "Reuse"
            ? "直接使用最新索引来源层的 Top-K，注意力分数仍用当前层 Query 重算。"
            : `索引 Q 与共享索引 K 计算分数，经 ReLU 和 head 加权后选 Top-K。${profile.candidateSource !== null ? "先限制在边界层构造的候选块池。" : index === d.candidateSource ? "同时产生供后续 Reindex 使用的候选块池。" : ""}`,
        formula:
          profile.mode === "Reuse"
            ? "indices = shared.topk_indices"
            : "TopK(sum_heads(ReLU(Qi @ Kiᵀ) × weights))",
        details: [
          ["candidate_blocks", String(d.candidateBlocks)],
          ["candidate_block_size", String(d.candidateBlockSize)],
          ["logical_score_shape", shapeText(sh.indexScores)],
        ],
      });
      if (profile.mode !== "Reuse") {
        edge(
          "global-kv",
          "indexer",
          t("index_k", sh.indexK, ["B", "C", "Di"]),
          "shared",
        );
        edge("q-lora", "indexer", t("q_rank", sh.qr, ["B", "Q", "Rq"]));
        edge("attn-input", "indexer", hidden);
      } else {
        boundary(
          "index-origin",
          1260,
          440,
          `Top-K Source · L${profile.indexSource}`,
          idx,
        );
        edge("index-origin", "indexer", idx, "shared");
      }
      if (
        index === d.candidateSource ||
        (profile.candidateSource !== null && profile.mode === "Reindex")
      ) {
        const pool = t(
          "candidate_mask",
          [b, q, sh.compressed],
          ["B", "Q", "C"],
          "逻辑布尔 mask；候选上限 candidate_topk_blocks × candidate_block_size，实际由索引分数决定。",
          "bool",
        );
        add("candidate-pool", 1260, 660, {
          title: "Hierarchical Candidate Pool",
          subtitle: `Source Layer ${d.candidateSource} · ${d.candidateBlocks} × ${d.candidateBlockSize}`,
          category: "operation",
          inputs:
            index === d.candidateSource
              ? [
                  t(
                    "index_scores",
                    sh.indexScores,
                    ["B", "Q", "C"],
                    undefined,
                    "float32",
                  ),
                ]
              : [],
          outputs: [pool],
          description:
            index === d.candidateSource
              ? "先按块聚合最大索引分数，选择候选块并固定保留最新未完整块；后续 Reindex 只在此候选集合重打分。"
              : "复用 Decoder 首个 Full 层的候选集合。参考实现用完整逻辑 mask 表示，生产 kernel 可只计算候选位置。",
        });
        if (index === d.candidateSource) {
          nodes
            .find((n) => n.id === "indexer")!
            .data.outputs.push(
              t(
                "index_scores",
                sh.indexScores,
                ["B", "Q", "C"],
                undefined,
                "float32",
              ),
            );
          edge(
            "indexer",
            "candidate-pool",
            t(
              "index_scores",
              sh.indexScores,
              ["B", "Q", "C"],
              undefined,
              "float32",
            ),
          );
        } else {
          nodes.find((n) => n.id === "indexer")!.data.inputs.push(pool);
          edge("candidate-pool", "indexer", pool, "shared");
        }
      }
    }
    add("sparse-attn", 420, 660, {
      title: "Sparse Attention + Sink",
      subtitle: profile.ratio
        ? "Local window + sparse global"
        : "Sliding window only",
      category: "attention",
      inputs: [
        t("query", sh.q, ["B", "Q", "Nq", "D"]),
        t("window_kv", sh.windowKV, ["B", "W", "D"]),
        ...(profile.ratio ? [global, idx] : []),
      ],
      outputs: [t("context", sh.context, ["B", "Q", "Nq", "D"])],
      params: m.heads,
      description:
        "稀疏 gather 后进行带 attention sink 的 softmax。K 与 V 是同一 latent；逻辑分数宽度为 window + selected global，不是完整序列的稠密 GQA 矩阵。输出尾部执行逆 RoPE。",
      formula: "softmax(Q · selected_KVᵀ / √D, sink) @ selected_KV",
      details: [
        ["logical_scores", shapeText(sh.sparseScores)],
        ["selected_global", String(sh.topk)],
        ["visible_window", String(sh.window)],
      ],
    });
    edge("q-lora", "sparse-attn", t("query", sh.q, ["B", "Q", "Nq", "D"]));
    edge(
      "local-kv",
      "sparse-attn",
      t("window_kv", sh.windowKV, ["B", "W", "D"]),
    );
    if (profile.ratio) {
      edge("global-kv", "sparse-attn", global, "shared");
      edge("indexer", "sparse-attn", idx);
    }
    add("o-group", 420, 880, {
      title: "Grouped Output LoRA",
      subtitle: `${d.oGroups} groups × rank ${d.oRank}`,
      category: "attention",
      path: `${base}.self_attn.wo_a`,
      inputs: [t("context", sh.context, ["B", "Q", "Nq", "D"])],
      outputs: [t("grouped_rank", sh.outputRank, ["B", "Q", "G", "Ro"])],
      params: m.heads * m.headDim * d.oRank,
      description:
        "先将 heads 分为 G 组，每组只投影自己的 head 通道，不能按一个全连接大矩阵重复计数。",
      formula: "[B,Q,G,Nq×D/G] × [G,Ro,Nq×D/G] → [B,Q,G,Ro]",
    });
    edge(
      "sparse-attn",
      "o-group",
      t("context", sh.context, ["B", "Q", "Nq", "D"]),
    );
    add("o-proj", 420, 1100, {
      title: "Output Projection",
      subtitle: `G × Ro → H`,
      category: "attention",
      path: `${base}.self_attn.wo_b`,
      inputs: [t("grouped_rank", sh.outputRank, ["B", "Q", "G", "Ro"])],
      params: d.oGroups * d.oRank * m.hidden,
      description: "拼接各组低秩输出并投影回隐藏维度。",
    });
    edge(
      "o-group",
      "o-proj",
      t("grouped_rank", sh.outputRank, ["B", "Q", "G", "Ro"]),
    );
  } else if (view === "mlp") {
    const score = t(
        "expert_scores",
        [b, q, d.experts],
        ["B", "Q", "E"],
        undefined,
        "float32",
      ),
      route = t(
        "selected_expert_ids",
        [b, q, d.topExperts],
        ["B", "Q", "Kmoe"],
        "具体专家 ID 由输入与权重决定。",
        "int64",
      );
    const slots = t(
      "selected_expert_inputs",
      [b, q, d.topExperts, m.hidden],
      ["B", "Q", "Kmoe", "H"],
      "逻辑路由槽位；实际实现按专家打包，不会给每个专家复制所有 token。",
    );
    const inner = t(
      "expert_intermediate",
      [b, q, d.topExperts, m.intermediate],
      ["B", "Q", "Kmoe", "I"],
    );
    boundary("mlp-input", 210, 0, "MoE Input");
    add("router", 0, 220, {
      title: "Router Scores",
      subtitle: String(d.text.scoring_func ?? "sqrtsoftplus"),
      category: "mlp",
      path: `${base}.mlp.gate`,
      outputs: [score],
      params: p.router,
      description:
        "文本与图像使用不同 correction bias 选择专家；最终权重来自未加 bias 的原始分数。",
      formula: "sqrt(softplus(x @ Wᵀ)) + selection_bias",
    });
    edge("mlp-input", "router");
    add("route", 0, 440, {
      title: "Top-K Expert Routing",
      subtitle: `${d.topExperts} of ${d.experts} experts`,
      category: "operation",
      inputs: [score],
      outputs: [
        route,
        t(
          "routing_weights",
          [b, q, d.topExperts],
          ["B", "Q", "Kmoe"],
          undefined,
          "float32",
        ),
      ],
      description:
        "选择与归一化路由权重。具体专家分配未知；展示配置可确定的 Top-K 槽位。",
      formula: "topk(scores+bias) → normalize(raw_scores) × route_scale",
    });
    edge("router", "route", score);
    add("dispatch", 0, 660, {
      title: "Dispatch to Experts",
      subtitle: "Dynamic token packing",
      category: "operation",
      inputs: [hidden, route],
      outputs: [slots],
      description:
        "每个 token 分配到 K 个路由专家。实际每专家 token 数不固定，图中只给出槽位上限表达。",
    });
    edge("route", "dispatch", route);
    edge("mlp-input", "dispatch", hidden);
    add("expert-up", 0, 880, {
      title: "Expert Gate & Up",
      subtitle: `H → ${m.intermediate}`,
      category: "mlp",
      inputs: [slots],
      outputs: [inner],
      params: 2 * m.hidden * m.intermediate * d.experts,
      description:
        "每个路由专家独立 gate/up 权重；SiLU 与配置中的 swiglu_limit clamp 后逐元素相乘。参数是全部专家总量，张量是本批次选中槽位。",
      formula: "SiLU(clamp(gate(x))) × clamp(up(x))",
    });
    edge("dispatch", "expert-up", slots);
    add("expert-down", 0, 1100, {
      title: "Expert Down Projection",
      subtitle: "Weighted expert outputs",
      category: "mlp",
      inputs: [
        inner,
        t(
          "routing_weights",
          [b, q, d.topExperts],
          ["B", "Q", "Kmoe"],
          undefined,
          "float32",
        ),
      ],
      outputs: [slots],
      params: m.hidden * m.intermediate * d.experts,
      description: "在 down projection 前融合路由权重。",
      formula: "down(routing_weight × gated_intermediate)",
    });
    edge("expert-up", "expert-down", inner);
    edge(
      "route",
      "expert-down",
      t(
        "routing_weights",
        [b, q, d.topExperts],
        ["B", "Q", "Kmoe"],
        undefined,
        "float32",
      ),
      "shared",
      "routing weights",
    );
    add("shared-expert", 440, 440, {
      title: "Shared Expert",
      subtitle: "Always active · SwiGLU",
      category: "mlp",
      params: p.expert,
      description: "所有 token 都经过一个共享专家，与路由分支并行。",
      formula: "down(SiLU(gate(x)) × up(x))",
    });
    edge("mlp-input", "shared-expert");
    add("combine", 210, 1320, {
      title: "Combine Expert Outputs",
      subtitle: "Routed sum + shared expert",
      category: "operation",
      inputs: [slots, hidden],
      description:
        "沿选中专家轴求和，再加共享专家输出，恢复 [B,Q,H]。实际累加使用 float32 后转换回输入精度。",
      formula: "Σ routed_outputs + shared_output",
    });
    edge("expert-down", "combine", slots);
    edge("shared-expert", "combine");
  } else if (view === "cache") {
    d.kvSources.forEach((owner, g) => {
      const ratio = d.ratios[owner],
        c = Math.floor(total / ratio),
        kv = t(`main_kv_layer_${owner}`, [b, c, m.headDim], ["B", "C", "D"]),
        ki = t(`index_k_layer_${owner}`, [b, c, d.indexDim], ["B", "C", "Di"]);
      const until = (d.kvSources[g + 1] ?? m.layers) - 1;
      add(`source-${owner}`, g * 420, 0, {
        title: `Full · Layer ${owner}`,
        subtitle:
          owner < d.encoderLayers
            ? "Encoder KV owner"
            : "CED boundary KV owner",
        category: "attention",
        inputs: [
          t(
            owner < d.encoderLayers ? "layer_hidden" : "encoder_final",
            [b, s.sequence, m.hidden],
            ["B", "S", "H"],
          ),
        ],
        outputs: [kv, ki],
        drill: "attention",
        layerIndex: owner,
        badge: `r=${ratio}`,
        description:
          "此层拥有 main KV 与 index K；来源层之后的消费者不重复存储。",
        details: [
          ["main_KV", shapeText(kv.shape)],
          ["index_K", shapeText(ki.shape)],
          [
            "FP4 + scales / entry",
            `${m.headDim / 2 + Math.ceil(m.headDim / 16) + d.indexDim / 2 + Math.ceil(d.indexDim / 32)} B`,
          ],
        ],
      });
      const indexes = d.indexSources.filter((i) => i >= owner && i <= until);
      indexes.forEach((is, j) => {
        const stop = (indexes[j + 1] ?? until + 1) - 1;
        const ts = t(
          "topk_indices",
          [b, deepseekShapes(m, s, is).queryLength, Math.min(d.indexTopk, c)],
          ["B", "Q", "K"],
          undefined,
          "int32",
        );
        add(`consumer-${is}`, g * 420, 260 + j * 240, {
          title: `${is === owner ? "Full" : "Reindex"} → Reuse`,
          subtitle: `Layers ${is} – ${stop}`,
          category: "block",
          inputs: [kv, ki],
          outputs: [ts],
          drill: "attention",
          layerIndex: is,
          badge: `Top-K owner ${is}`,
          description: `Layer ${is} 产生 Top-K，随后到 Layer ${stop} 复用；${owner >= d.encoderLayers ? "Decoder 全部共享 Encoder 边界 KV。" : "下一 Full 层将创建新的缓存。"}`,
        });
        edge(
          `source-${owner}`,
          `consumer-${is}`,
          kv,
          "shared",
          `shared KV · L${owner}`,
        );
      });
    });
    add("cache-summary", 0, 1550, {
      title: "Cache Accounting",
      subtitle: "Unique owners, not layer count",
      category: "operation",
      inputs: [],
      outputs: [],
      description: `全局 KV 按 ${d.kvSources.length} 个来源层累计，K/V latent 不乘 2。全局 FP4 包括 main KV 每 16 通道 1 byte scale、index K 每 32 通道 1 byte scale。另列有效 SWA 窗口与压缩临时状态；不等同峰值 HBM。`,
      details: [
        ["global logical", formatBytes(mem.globalLogical)],
        ["global FP4 + scales", formatBytes(mem.globalPacked)],
        ["SWA logical", formatBytes(mem.windowLogical)],
        ["compression state fp32", formatBytes(mem.partialState)],
        ["all logical", formatBytes(mem.kvCache)],
      ],
    });
  } else if (view === "vision" && d.vision) {
    const v = d.vision,
      vs = visionShapes(m, s),
      patch = t(
        "patches",
        vs.patches,
        ["Npatch", "3p²"],
        "单张图预处理后的 patch 网格；不推断原图 resize。",
      ),
      emb = t("vision_hidden", vs.embedded, ["Npatch", "Hv"]),
      down = t("pixel_unshuffle", vs.downsampled, ["Nimg", "r²Hv"]),
      out = t("visual_embeddings", vs.aligned, ["Nimg", "H"]);
    boundary("vision-input", 0, 0, "Image Patch Grid", patch);
    add("patch-embed", 0, 220, {
      title: "Linear Patch Embedding",
      subtitle: `Patch ${v.patch} × ${v.patch}`,
      category: "embedding",
      inputs: [patch],
      outputs: [emb],
      params: 3 * v.patch * v.patch * v.hidden + v.hidden,
      description: "每个 RGB patch 展平后线性投影；Npatch=gridH×gridW。",
      details: [
        ["grid", `${vs.gridH} × ${vs.gridW}`],
        ["example images", String(vs.images)],
      ],
    });
    edge("vision-input", "patch-embed", patch);
    add("vision-blocks", 0, 440, {
      title: "DeepSeek ViT Blocks",
      subtitle: `${v.layers} × bidirectional transformer`,
      category: "block",
      inputs: [emb],
      outputs: [emb],
      params:
        v.layers *
        (4 * v.hidden * v.hidden +
          6 * v.hidden +
          3 * v.hidden * v.intermediate),
      description:
        "每层含 RMSNorm、全双向 Self Attention（2D-RoPE）、残差与 SwiGLU。视觉每张图独立处理。",
      details: [
        ["Q/K/V", [v.heads, vs.n, v.hidden / v.heads].join(" × ")],
        ["logical attention scores", `[${v.heads}, ${vs.n}, ${vs.n}]`],
        ["MLP intermediate", `[${vs.n}, ${v.intermediate}]`],
      ],
    });
    edge("patch-embed", "vision-blocks", emb);
    add("vision-norm", 0, 660, {
      title: "Final Vision Norm",
      subtitle: "RMSNorm",
      category: "norm",
      inputs: [emb],
      outputs: [emb],
      params: v.hidden,
    });
    edge("vision-blocks", "vision-norm", emb);
    add("pixel-unshuffle", 0, 880, {
      title: "Pixel Unshuffle",
      subtitle: `${v.downsample} × ${v.downsample} neighborhoods`,
      category: "operation",
      inputs: [emb],
      outputs: [down],
      description: "网格不足 r 倍数时补零；空间维度缩小、通道维度扩大 r² 倍。",
      formula: "ceil(gridH/r) × ceil(gridW/r) tokens",
    });
    edge("vision-norm", "pixel-unshuffle", emb);
    add("projector", 0, 1100, {
      title: "MLP Projector",
      subtitle: "Linear → GELU → Linear",
      category: "embedding",
      inputs: [down],
      outputs: [out],
      params:
        v.hidden * v.downsample ** 2 * m.hidden +
        m.hidden +
        m.hidden * m.hidden +
        m.hidden,
      description: "将视觉特征投影到语言隐藏维度。",
    });
    edge("pixel-unshuffle", "projector", down);
    add("image-merge", 0, 1320, {
      title: "Insert Image Embeddings",
      subtitle: "Image / row / boundary tokens",
      category: "operation",
      inputs: [out, ids],
      outputs: [
        t("merged_embeddings", [b, s.sequence, m.hidden], ["B", "S", "H"]),
      ],
      params: 3 * m.hidden,
      description:
        "图像特征替换 input_ids 中预留的 image span；每张图包含开始/结束 token 和每行 newline。S 已包含这些 token，不能再重复相加。Decode 时视觉分支已缓存，不重新运行。",
      details: [
        ["features / image", String(vs.outputTokens)],
        ["span tokens / image", String(vs.spanTokens)],
        ["images / sample", String(s.image?.count ?? 0)],
        ["active now", String(Boolean(s.image?.count) && s.past === 0)],
      ],
    });
    edge("projector", "image-merge", out);
  } else if (view === "engram") {
    const layer = d.engramLayers.includes(index) ? index : d.engramLayers[0],
      cols = (d.ngram - 1) * d.engramHeads;
    if (layer === undefined) {
      boundary("no-engram", 0, 0, "No Engram in this config");
      return { nodes, edges };
    }
    const j = d.engramLayers.indexOf(layer),
      eh = t(
        "hash_ids",
        [b, s.sequence, cols],
        ["B", "S", "Nhash"],
        undefined,
        "int64",
      ),
      look = t(
        "ngram_embeddings",
        [b, s.sequence, cols * d.engramDim],
        ["B", "S", "Nhash×De"],
      );
    boundary("engram-input", 0, 0, "Token N-grams", ids);
    add("ngram-hash", 0, 220, {
      title: "N-gram Hash Lookup",
      subtitle: `Layer ${layer} · 2…${d.ngram}-grams`,
      category: "operation",
      inputs: [ids],
      outputs: [eh],
      description:
        "使用分词器映射和哈希表生成 ID；仅凭配置不能得到具体 hash_ids，图中展示结构形状。图像位置禁止 n-gram 注入。",
    });
    edge("engram-input", "ngram-hash", ids);
    add("engram-table", 0, 440, {
      title: "Engram Embedding Table",
      subtitle: `${d.engramRows[j].toLocaleString()} rows`,
      category: "embedding",
      inputs: [eh],
      outputs: [look],
      params: d.engramRows[j] * d.engramDim,
      description:
        "稀疏索引嵌入表，仅查询命中的行；参数总量与本次激活大小不同。",
    });
    edge("ngram-hash", "engram-table", eh);
    add("engram-project", 0, 660, {
      title: "Memory Key / Value",
      subtitle: "HC keys + shared value",
      category: "embedding",
      inputs: [look],
      outputs: [
        t(
          "memory_keys",
          [b, s.sequence, d.hc, m.hidden],
          ["B", "S", "HC", "H"],
        ),
        t("memory_value", [b, s.sequence, m.hidden], ["B", "S", "H"]),
      ],
      params: cols * d.engramDim * m.hidden * (d.hc + 1),
      description:
        "将拼接的 n-gram embedding 投影成每条残差流的 key 与单一共享 value。",
    });
    edge("engram-table", "engram-project", look);
    add("engram-gate", 0, 880, {
      title: "Gated Memory Injection",
      subtitle: "Normalized dot → signed sqrt → sigmoid",
      category: "operation",
      inputs: [
        t("stream", [b, s.sequence, d.hc, m.hidden], ["B", "S", "HC", "H"]),
        t(
          "memory_keys",
          [b, s.sequence, d.hc, m.hidden],
          ["B", "S", "HC", "H"],
        ),
        t("memory_value", [b, s.sequence, m.hidden], ["B", "S", "H"]),
      ],
      outputs: [
        t("stream", [b, s.sequence, d.hc, m.hidden], ["B", "S", "HC", "H"]),
      ],
      params: 2 * d.hc * m.hidden,
      description:
        "对每个 token、每条 HC 流计算 gate，按 image mask 屏蔽图像位置，再将共享 value 加入残差。",
      formula: "stream + gate[...,None] × memory_value[...,None,:]",
    });
    edge(
      "engram-project",
      "engram-gate",
      t("memory_keys", [b, s.sequence, d.hc, m.hidden], ["B", "S", "HC", "H"]),
    );
  } else if (view === "dspark") {
    const l = d.draftBlock,
      main = t(
        "main_hidden",
        [b, s.sequence, d.draftTargets.length * m.hidden],
        ["B", "S", "Nt×H"],
      ),
      draft = t(
        "draft_stream",
        [b, l, d.hc, m.hidden],
        ["B", "Ldraft", "HC", "H"],
      );
    boundary("draft-input", 0, 0, "Main Hidden Taps", main);
    add("draft-project", 0, 220, {
      title: "Main Hidden Projection",
      subtitle: `Layers ${d.draftTargets.join(", ")}`,
      category: "attention",
      inputs: [main],
      outputs: [t("main_x", [b, s.sequence, m.hidden], ["B", "S", "H"])],
      description:
        "读取目标层 attention 入口的 HC 均值，拼接后投影与 RMSNorm；不是这些层的最终输出。",
    });
    edge("draft-input", "draft-project", main);
    add("draft-embed", 430, 0, {
      title: "Noise-token Draft Block",
      subtitle: `block_size=${l}`,
      category: "embedding",
      inputs: [t("current_token", [b, 1], ["B", "1"], undefined, "int64")],
      outputs: [draft],
      description:
        "当前 token 放在首位，其余为 noise token；共享主干 embedding，并扩展 HC 流。共享权重不重复计入 DSpark 参数。",
    });
    add("draft-blocks", 0, 440, {
      title: "DSpark Draft Layers",
      subtitle: `${d.draftLayers} × SWA + MoE`,
      category: "mlp",
      inputs: [draft, t("main_x", [b, s.sequence, m.hidden], ["B", "S", "H"])],
      outputs: [draft],
      params: p.draft,
      description: `草稿 MoE 使用 ${d.draftExperts} 个专家、Top-${d.draftTopExperts}。Prefill 的 forward_spec 只初始化窗口缓存；Decode 才运行草稿块。此节点参数汇总全部独立草稿权重。`,
      details: [
        ["active stage", s.past === 0 ? "cache seeding only" : "draft forward"],
        ["new draft positions", String(l)],
        ["verification", "不模拟接受率或验证循环"],
      ],
    });
    edge(
      "draft-project",
      "draft-blocks",
      t("main_x", [b, s.sequence, m.hidden], ["B", "S", "H"]),
    );
    edge("draft-embed", "draft-blocks", draft);
    add("draft-head", 0, 660, {
      title: "Shared LM Head + Markov",
      subtitle: "Autoregressive bias within block",
      category: "output",
      inputs: [draft],
      outputs: [
        t(
          "draft_logits",
          [b, l, m.vocab],
          ["B", "Ldraft", "V"],
          s.past === 0
            ? "Prefill 此输出不执行；这里只展示 Decode 分支形状。"
            : undefined,
          "float32",
        ),
        t(
          "hidden+markov",
          [b, l, m.hidden + d.draftRank],
          ["B", "Ldraft", "H+Rm"],
          "主干共享头旁路保留的隐藏状态，与 Markov embedding 拼接。",
        ),
      ],
      description:
        "共享主 LM Head；Markov 分支给每个草稿位置添加词表 bias。LM Head 权重已在主干统计。",
    });
    edge("draft-blocks", "draft-head", draft);
    add("draft-confidence", 0, 880, {
      title: "Confidence Scores",
      subtitle: "Per draft position",
      category: "output",
      inputs: [
        t(
          "hidden+markov",
          [b, l, m.hidden + d.draftRank],
          ["B", "Ldraft", "H+Rm"],
        ),
      ],
      outputs: [t("confidence", [b, l], ["B", "Ldraft"], undefined, "float32")],
      description:
        "独立 confidence 投影供推测解码调度使用；配置不能推导分数数值或接受 token 数。",
    });
    edge(
      "draft-head",
      "draft-confidence",
      t(
        "hidden+markov",
        [b, l, m.hidden + d.draftRank],
        ["B", "Ldraft", "H+Rm"],
      ),
    );
  } else {
    boundary("unavailable", 0, 0, "该配置未启用此分支");
  }
  if (view === "vision" && (!s.image?.count || s.past > 0))
    nodes.forEach((n) => (n.data.inactive = true));
  if (view === "overview") {
    const draft = nodes.find((n) => n.id === "dspark");
    if (draft) draft.data.inactive = true;
  }
  if (view === "dspark" && s.past === 0)
    for (const n of nodes)
      if (["draft-embed", "draft-head", "draft-confidence"].includes(n.id))
        n.data.inactive = true;
  for (const connection of edges) {
    if (!connection.data?.shared) continue;
    const from = nodes.find((n) => n.id === connection.source)!;
    const to = nodes.find((n) => n.id === connection.target)!;
    if (to.position.x > from.position.x) {
      connection.sourceHandle = "right-out";
      connection.targetHandle = "side-in";
    } else if (to.position.x < from.position.x) {
      connection.sourceHandle = "side-out";
      connection.targetHandle = "right-in";
    }
  }
  // Shared-state edges may be inserted after consumers. Sort the DAG for deterministic playback.
  const ordered: ModuleNode[] = [],
    pending = new Map(nodes.map((n) => [n.id, n]));
  while (pending.size) {
    const ready = [...pending.values()].filter(
      (n) => !edges.some((e) => e.target === n.id && pending.has(e.source)),
    );
    if (!ready.length) break;
    for (const n of ready) {
      ordered.push(n);
      pending.delete(n.id);
    }
  }
  if (pending.size) throw new Error("DeepSeek 可视化图包含循环依赖。");
  ordered.forEach((node, i) => (node.data.order = i + 1));
  return { nodes: ordered, edges };
}
