import { MarkerType, type Edge } from "@xyflow/react";
import type { ModelConfig, Scenario, Tensor } from "./model";
import { formatBytes, shapeText, validateScenario } from "./model";
import type { ModelGraph, ModuleData, ModuleNode, View } from "./graph";
import {
  gemmaShapes,
  gemmaLayer,
  gemmaParameters,
  gemmaMemory,
  gemmaVisionShapes,
  gemmaAudioShapes,
  gemmaHasHead,
  validateGemmaScenario,
} from "./gemma";
import { NODE_HEIGHT, NODE_WIDTH } from "./presentation";

export function buildGemmaGraph(
  m: ModelConfig,
  s: Scenario,
  view: View,
  index: number,
  symbolic: boolean,
): ModelGraph {
  validateScenario(s);
  validateGemmaScenario(m, s);
  const g = m.gemma!,
    sh = gemmaShapes(m, s, index),
    l = sh.layer,
    p = gemmaParameters(m),
    mem = gemmaMemory(m, s),
    lp = p.layers[index],
    b = s.batch,
    base = `model.language_model.layers.${index}`;
  const nodes: ModuleNode[] = [],
    edges: Edge[] = [];
  const t = (
    name: string,
    shape: number[],
    symbols: string[],
    note?: string,
    dtype: Tensor["dtype"] = s.dtype,
  ): Tensor => ({ name, shape, symbols, dtype, note });
  const hidden = t("hidden_states", sh.hidden, ["B", "S", "H"]),
    ids = t("input_ids", sh.ids, ["B", "S"], undefined, "int64"),
    ple = t("per_layer_inputs", sh.ple, ["B", "S", "L", "Dple"]),
    pleSlice = t(`per_layer_input_${index}`, sh.pleSlice, ["B", "S", "Dple"]);
  const add = (
    id: string,
    x: number,
    y: number,
    data: Partial<ModuleData> & Pick<ModuleData, "title" | "category">,
  ) =>
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
        ...data,
      },
    });
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
        shared: kind === "shared",
        residual: kind === "residual",
      },
      markerEnd: {
        type: MarkerType.ArrowClosed,
        color: kind === "shared" ? "#398b85" : "#9aacc4",
      },
      style: {
        stroke:
          kind === "shared"
            ? "#398b85"
            : kind === "residual"
              ? "#9475bc"
              : "#9aacc4",
        strokeWidth: kind === "shared" ? 2 : 1.6,
        strokeDasharray: kind === "flow" ? undefined : "6 4",
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
    tensor = hidden,
  ) =>
    add(id, x, y, {
      title,
      category: "input",
      subtitle: "Tensor boundary",
      inputs: [],
      outputs: [tensor],
      description: "当前图的输入边界。",
    });
  const norm = (
    id: string,
    x: number,
    y: number,
    params = m.hidden,
    input = hidden,
  ) =>
    add(id, x, y, {
      title: "RMS Normalization",
      subtitle: id,
      category: "norm",
      path: `${base}.${id}`,
      inputs: [input],
      outputs: [input],
      params,
      formula: "x / sqrt(mean(x²) + ε) × weight",
      description:
        "沿最后一维归一化；Gemma 使用显式 scale=1 初始化的 RMSNorm。",
    });
  const attention = (id: string, x: number, y: number) =>
    add(id, x, y, {
      title: "Gemma Hybrid Attention",
      subtitle: l.sliding
        ? "Sliding-window attention"
        : "Global attention · proportional RoPE",
      category: "attention",
      path: `${base}.self_attn`,
      params: lp.attention,
      drill: "attention",
      badge: l.shared ? `KV ← Layer ${l.owner}` : `D=${l.headDim}`,
      description:
        "局部与全局层的 head_dim 不同，Query/key 做 RMSNorm，value 做无可训练 scale 的 RMSNorm；attention scaling=1.0。共享区只计算 Query 与输出投影。",
      details: [
        ["layer_type", l.type],
        ["head_dim", String(l.headDim)],
        ["kv_source", String(l.owner)],
        ["shared", String(l.shared)],
        ["scaling", "1.0"],
      ],
    });
  const mlp = (id: string, x: number, y: number) =>
    add(id, x, y, {
      title: "Gemma Gated MLP",
      subtitle: "GELU (tanh approximation) × Up",
      category: "mlp",
      path: `${base}.mlp`,
      params: lp.mlp,
      drill: "mlp",
      badge: `I=${l.intermediate}`,
      description:
        l.shared && g.doubleMlp
          ? "此共享 KV 层的 MLP 中间维度加倍。"
          : "标准宽度门控 MLP。",
      formula: "down(GELU_tanh(gate(x)) × up(x))",
      details: [
        ["intermediate_size", String(l.intermediate)],
        ["double_wide", String(l.shared && g.doubleMlp)],
      ],
    });
  if (view === "overview") {
    boundary("input", 0, 0, "Token IDs", ids);
    add("embedding", 0, 220, {
      title: "Scaled Token Embedding",
      subtitle: "Embedding × √H",
      category: "embedding",
      inputs: [ids],
      params: p.embedding,
      description:
        "词嵌入乘 √hidden_size。多模态占位位置使用 PAD 进行 token-identity PLE 查询。",
      formula: "embed_tokens(input_ids) × √H",
    });
    edge("input", "embedding", ids);
    if (g.vision) {
      const vs = gemmaVisionShapes(m, s);
      add("vision", 440, 0, {
        title: "Vision Encoder",
        subtitle: `${g.vision.layers} layers · spatial pooling`,
        category: "embedding",
        inputs: [],
        outputs: [
          t(
            "image_features",
            vs.projected,
            ["B", "Nimg", "H"],
            "按样本分组的逻辑表示；实际视觉接口将有效 soft tokens 合并为 [B×Nimg,H]。",
          ),
        ],
        params: p.vision,
        drill: "vision",
        inactive: !s.image?.count || s.past > 0,
        badge: s.image?.count ? "图像分支" : "未启用 · 示例",
        description:
          "预处理后的规则 patch 网格 → 双向视觉 Transformer → 按位置平均池化 → 无 scale RMSNorm 与线性映射。",
      });
    }
    if (g.audio) {
      const a = gemmaAudioShapes(m, s);
      add("audio", 880, 0, {
        title: "Audio Encoder",
        subtitle: `${g.audio.layers} Conformer-like layers`,
        category: "embedding",
        inputs: [],
        outputs: [t("audio_features", a.softTokens, ["B", "Ta", "H"])],
        params: p.audio,
        drill: "audio",
        inactive: !s.audio?.enabled || s.past > 0,
        badge: s.audio?.enabled ? "音频分支" : "未启用 · 示例",
        description:
          "两层 stride-2 卷积、chunked relative attention、轻量因果卷积与 FFN，最终映射到语言隐藏维度。",
      });
    }
    add("merge", 0, 440, {
      title: "Merge Modality Embeddings",
      subtitle: "Replace reserved soft-token slots",
      category: "operation",
      inputs: [
        hidden,
        ...(g.vision && s.image?.count && s.past === 0
          ? [
              t("image_features", gemmaVisionShapes(m, s).projected, [
                "B",
                "Nimg",
                "H",
              ]),
            ]
          : []),
        ...(g.audio && s.audio?.enabled && s.past === 0
          ? [
              t("audio_features", gemmaAudioShapes(m, s).softTokens, [
                "B",
                "Ta",
                "H",
              ]),
            ]
          : []),
      ],
      description:
        "特征替换预留的软 token 位置；S 已包含图像/音频及起止标记，不额外增加序列长度。Decode 不重复执行媒体编码器。",
    });
    edge("embedding", "merge");
    if (g.vision)
      edge(
        "vision",
        "merge",
        t("image_features", gemmaVisionShapes(m, s).projected, [
          "B",
          "Nimg",
          "H",
        ]),
        "shared",
        "image soft-token slots",
      );
    if (g.audio)
      edge(
        "audio",
        "merge",
        t("audio_features", gemmaAudioShapes(m, s).softTokens, [
          "B",
          "Ta",
          "H",
        ]),
        "shared",
        "audio soft-token slots",
      );
    add("ple", 440, 440, {
      title: "Per-Layer Embeddings",
      subtitle: `${m.layers} × ${g.pleDim} packed channels`,
      category: "embedding",
      inputs: [ids, hidden],
      outputs: [ple],
      params: p.pleTable + p.pleProjection,
      drill: "ple",
      badge: "PLE",
      description:
        "token-identity lookup 与当前输入 embedding 的 context projection 合成每层专属输入。大表只做按 token 查询，需与有效参数规模区别。",
    });
    edge("input", "ple", ids, "shared", "token identity");
    edge("merge", "ple", hidden);
    add("decoder", 0, 660, {
      title: "Hybrid Decoder Blocks",
      subtitle: `${m.layers} layers · ${g.sharedLayers} share KV`,
      category: "block",
      inputs: [hidden, ple],
      params: p.layers.reduce((a, x) => a + x.total, 0),
      drill: "decoder",
      badge: `KV sharing starts L${g.shareStart}`,
      description:
        "语言主干是因果 Decoder，视觉/音频为前端编码器；它与 DeepSeek CED 的 Encoder/Decoder 划分不同。",
    });
    edge("merge", "decoder");
    edge("ple", "decoder", ple, "shared", "per-layer slice");
    norm("final-norm", 0, 880);
    edge("decoder", "final-norm");
    add("lm-head", 0, 1100, {
      title: gemmaHasHead(m) ? "LM Head + Softcap" : "Hidden State Output",
      subtitle: m.tiedEmbeddings
        ? "Tied embedding weight"
        : "Output projection",
      category: "output",
      outputs: [
        gemmaHasHead(m) ? t("logits", sh.logits, ["B", "S", "V"]) : hidden,
      ],
      params: p.lmHead,
      badge: m.tiedEmbeddings ? "共享权重" : "",
      description:
        "默认展示完整 S 个位置 logits，计算精度保持所选 dtype；词表权重与 embedding 共享时总数只计一次。",
      formula: g.softcap
        ? `logits = ${g.softcap} × tanh(logits / ${g.softcap})`
        : "logits = hidden @ weightᵀ",
    });
    edge("final-norm", "lm-head");
  } else if (view === "decoder") {
    for (let i = 0; i < m.layers; i++) {
      const pr = gemmaLayer(m, i),
        id = `layer-${i}`;
      add(id, (i % 5) * 390, Math.floor(i / 5) * 230, {
        title: `Layer ${i}`,
        subtitle: pr.sliding ? "Sliding attention" : "Global attention",
        category: pr.shared ? "block" : "attention",
        params: p.layers[i].total,
        drill: "layer",
        layerIndex: i,
        badge: pr.shared ? `KV ← L${pr.owner}` : `owns KV · D=${pr.headDim}`,
        description:
          "点击进入该层；每层的注意力宽度、MLP 宽度、PLE 均按实际配置推导。",
        details: [
          ["head_dim", String(pr.headDim)],
          ["MLP width", String(pr.intermediate)],
          ["KV owner", String(pr.owner)],
        ],
      });
      if (i) edge(`layer-${i - 1}`, id);
    }
  } else if (view === "layer") {
    boundary("layer-input", 0, 0, `Layer ${index} Input`);
    norm("input-norm", 0, 220);
    attention("attention", 0, 440);
    norm("post-attn-norm", 0, 660);
    add("attn-add", 0, 880, {
      title: "Attention Residual",
      subtitle: "Post-norm then add",
      category: "operation",
      inputs: [hidden, t("residual", sh.hidden, ["B", "S", "H"])],
      formula: "x + post_attention_norm(attn(input_norm(x)))",
    });
    norm("pre-ffn-norm", 440, 220);
    mlp("mlp", 440, 440);
    norm("post-ffn-norm", 440, 660);
    add("ffn-add", 440, 880, {
      title: "MLP Residual",
      subtitle: "Post-norm then add",
      category: "operation",
      inputs: [hidden, t("residual", sh.hidden, ["B", "S", "H"])],
    });
    add("ple-inject", 440, 1100, {
      title: "PLE Gated Injection",
      subtitle: `Per-layer signal · ${g.pleDim} channels`,
      category: "embedding",
      inputs: [hidden, pleSlice],
      params: lp.ple,
      drill: "ple",
      description:
        "当前层主干完成 attention 与 MLP 后，激活 gate 乘本层 PLE 切片，投影并归一化，随后通过独立残差节点加入主干。",
      formula: "norm(project(GELU(gate(x)) × PLE_l))",
    });
    boundary("ple-source", 880, 880, "Prepared Layer Slice", pleSlice);
    add("ple-add", 440, 1320, {
      title: "PLE Residual Add",
      subtitle: "逐层信号残差",
      category: "operation",
      inputs: [hidden, t("residual", sh.hidden, ["B", "S", "H"])],
      formula: "x + normalized_PLE_signal",
      description: "将 PLE 分支输出与 MLP 后的主干隐藏状态相加。",
    });
    add("layer-scale", 440, 1540, {
      title: "Layer Scalar",
      subtitle: "Non-trainable buffer",
      category: "operation",
      description: "最终乘 layer_scalar；该标量是 buffer，不计入可训练参数。",
      formula: "output = x × layer_scalar",
    });
    for (const [a, z] of [
      ["layer-input", "input-norm"],
      ["input-norm", "attention"],
      ["attention", "post-attn-norm"],
      ["post-attn-norm", "attn-add"],
      ["attn-add", "pre-ffn-norm"],
      ["pre-ffn-norm", "mlp"],
      ["mlp", "post-ffn-norm"],
      ["post-ffn-norm", "ffn-add"],
      ["ffn-add", "ple-inject"],
      ["ple-inject", "ple-add"],
      ["ple-add", "layer-scale"],
    ])
      edge(a, z);
    edge("layer-input", "attn-add", hidden, "residual");
    edge("attn-add", "ffn-add", hidden, "residual");
    edge("ple-source", "ple-inject", pleSlice, "shared");
    edge("ffn-add", "ple-add", hidden, "residual");
  } else if (view === "attention") {
    const query = t("query", sh.q, ["B", "Nq", "S", "D"]),
      key = t("key", sh.kv, ["B", "Nkv", "Klen", "D"]),
      value = { ...key, name: "value" };
    boundary("attn-input", 0, 0, "Normalized Hidden");
    add("q-proj", 0, 220, {
      title: "Query Projection + Norm",
      subtitle: `${m.heads} heads × ${l.headDim}`,
      category: "attention",
      path: `${base}.self_attn.q_proj`,
      outputs: [query],
      params: lp.qProj + l.headDim,
      description:
        "线性投影后按每个 head 最后一维 RMSNorm，再应用 RoPE。局部层完整旋转，全局层按 proportional factor 旋转部分维度。",
      formula: "Linear → reshape → RMSNorm(D) → RoPE",
      details: [
        ["weight", `[${m.heads * l.headDim}, ${m.hidden}]`],
        [
          "rotated dims",
          String(l.sliding ? l.headDim : l.headDim * g.globalRopeFraction),
        ],
      ],
    });
    edge("attn-input", "q-proj");
    if (l.shared) {
      add("kv-source", 440, 0, {
        title: `KV Owner · Layer ${l.owner}`,
        subtitle: l.sliding
          ? "Last independent sliding layer"
          : "Last independent global layer",
        category: "input",
        inputs: [],
        outputs: [key, value],
        description:
          "shared_kv_states 按注意力类型传递本次完整可见 K/V；不是重新分配一份持久缓存。",
      });
      add("shared-kv", 440, 220, {
        title: "Reuse Key / Value",
        subtitle: `Shared from Layer ${l.owner}`,
        category: "attention",
        inputs: [key, value],
        outputs: [key, value],
        description:
          "共享层没有 k_proj、v_proj、k_norm 参数。引用来源层已经归一化并应用 RoPE 的 K，以及归一化 V。",
        details: [
          ["kv_owner", String(l.owner)],
          ["k_proj / v_proj", "不存在"],
          ["visible_length", String(sh.keys)],
        ],
      });
      edge("kv-source", "shared-kv", key, "shared");
    } else {
      const fresh = t("current_key", sh.newKV, ["B", "Nkv", "S", "D"]);
      add("kv-project", 440, 220, {
        title: "Key / Value Projections",
        subtitle: "K norm + RoPE · V norm without scale",
        category: "attention",
        inputs: [hidden],
        outputs: [fresh, { ...fresh, name: "current_value" }],
        params: lp.kProj + lp.vProj + l.headDim,
        description:
          "K 和 V 使用独立投影，K 具有可训练 RMSNorm scale，V 的 RMSNorm 无 scale。E 系列 attention_k_eq_v=false。",
        details: [
          ["per projection weight", `[${m.kvHeads * l.headDim}, ${m.hidden}]`],
        ],
      });
      edge("attn-input", "kv-project");
      add("kv-cache", 440, 440, {
        title: "Dynamic KV Cache",
        subtitle: l.sliding
          ? `Sliding retention ≤ ${m.slidingWindow! - 1}`
          : "Full history retention",
        category: "operation",
        inputs: [
          fresh,
          { ...fresh, name: "current_value" },
          t(
            "past_key",
            [
              b,
              m.kvHeads,
              l.sliding ? Math.min(s.past, m.slidingWindow! - 1) : s.past,
              l.headDim,
            ],
            ["B", "Nkv", "Pcache", "D"],
          ),
          t(
            "past_value",
            [
              b,
              m.kvHeads,
              l.sliding ? Math.min(s.past, m.slidingWindow! - 1) : s.past,
              l.headDim,
            ],
            ["B", "Nkv", "Pcache", "D"],
          ),
        ],
        outputs: [
          key,
          value,
          t("retained_key", sh.cache, ["B", "Nkv", "R", "D"]),
          t("retained_value", sh.cache, ["B", "Nkv", "R", "D"]),
        ],
        description:
          "本次参与 attention 的可见 K/V 与调用结束后保留的持久缓存长度不同。Prefill 可以计算完整 chunk，同时只保留局部窗口尾部供下次调用。",
        formula: l.sliding ? "Klen=min(P,W−1)+S; R=min(P+S,W−1)" : "Klen=R=P+S",
        details: [
          ["shares with later layers", String(l.source)],
          ["persistent length", String(sh.retained)],
        ],
      });
      edge("kv-project", "kv-cache", fresh);
    }
    add("scores", 0, 660, {
      title: "QK Scores + Causal Mask",
      subtitle: l.sliding
        ? `Sliding window ${m.slidingWindow}`
        : "Full causal attention",
      category: "attention",
      inputs: [query, key],
      outputs: [
        t(
          "attention_scores",
          sh.scores,
          ["B", "Nq", "S", "Klen"],
          "逻辑矩阵；融合 attention 后端可能不完整分配。",
        ),
      ],
      description:
        "GQA 按组使用 KV。Gemma 已在 Q/K 上归一化，attention scaling=1.0，不使用通用的 1/√D。",
      formula: "scores = Q @ Kᵀ + mask",
      details: [
        ["scaling", "1.0"],
        ["query_length", String(s.sequence)],
        ["key_length", String(sh.keys)],
      ],
    });
    edge("q-proj", "scores", query);
    edge(l.shared ? "shared-kv" : "kv-cache", "scores", key);
    add("softmax-values", 0, 880, {
      title: "Softmax × Value",
      subtitle: "Normalize over visible key axis",
      category: "operation",
      inputs: [
        t("attention_scores", sh.scores, ["B", "Nq", "S", "Klen"]),
        value,
      ],
      outputs: [t("context", sh.context, ["B", "S", "Nq×D"])],
      formula: "softmax(scores) @ V → merge heads",
    });
    edge(
      "scores",
      "softmax-values",
      t("attention_scores", sh.scores, ["B", "Nq", "S", "Klen"]),
    );
    edge(
      l.shared ? "shared-kv" : "kv-cache",
      "softmax-values",
      value,
      "shared",
    );
    add("o-proj", 0, 1100, {
      title: "Output Projection",
      subtitle: `${m.heads * l.headDim} → ${m.hidden}`,
      category: "attention",
      path: `${base}.self_attn.o_proj`,
      inputs: [t("context", sh.context, ["B", "S", "Nq×D"])],
      params: lp.oProj,
      description:
        "合并头后的宽度可能不同于 hidden_size，通过输出投影恢复隐藏维度。",
    });
    edge(
      "softmax-values",
      "o-proj",
      t("context", sh.context, ["B", "S", "Nq×D"]),
    );
  } else if (view === "mlp") {
    const intermediate = t("intermediate", sh.intermediate, ["B", "S", "I"]);
    boundary("mlp-input", 210, 0, "MLP Input");
    for (const [id, x, title] of [
      ["gate-proj", 0, "Gate Projection"],
      ["up-proj", 440, "Up Projection"],
    ] as const) {
      add(id, x, 220, {
        title,
        subtitle: `${m.hidden} → ${l.intermediate}`,
        category: "mlp",
        outputs: [intermediate],
        params: m.hidden * l.intermediate,
        description:
          "Gemma E2B 的 KV 共享层启用 double-wide MLP，E4B 配置不启用。",
        details: [
          ["layer_intermediate", String(l.intermediate)],
          ["base_intermediate", String(m.intermediate)],
          ["double_wide", String(l.shared && g.doubleMlp)],
        ],
      });
      edge("mlp-input", id);
    }
    add("activation", 0, 440, {
      title: "GELU Tanh Approximation",
      subtitle: "gelu_pytorch_tanh",
      category: "operation",
      inputs: [intermediate],
      outputs: [intermediate],
      formula: "0.5x × (1 + tanh(√(2/π) × (x + 0.044715x³)))",
    });
    edge("gate-proj", "activation", intermediate);
    add("multiply", 210, 660, {
      title: "Gate × Up",
      subtitle: "Element-wise product",
      category: "operation",
      inputs: [intermediate, { ...intermediate, name: "up" }],
      outputs: [intermediate],
    });
    edge("activation", "multiply", intermediate);
    edge("up-proj", "multiply", intermediate);
    add("down-proj", 210, 880, {
      title: "Down Projection",
      subtitle: `${l.intermediate} → ${m.hidden}`,
      category: "mlp",
      inputs: [intermediate],
      params: m.hidden * l.intermediate,
    });
    edge("multiply", "down-proj", intermediate);
  } else if (view === "ple") {
    boundary("ple-ids", 0, 0, "Token Identity", ids);
    boundary("ple-context", 440, 0, "Merged Input Embeddings");
    add("ple-table", 0, 220, {
      title: "Packed PLE Embedding",
      subtitle: `Vple × (${m.layers} × ${g.pleDim})`,
      category: "embedding",
      inputs: [ids],
      outputs: [ple],
      params: p.pleTable,
      description:
        "独立于主词嵌入的大表，按 token 查询并乘 √Dple，reshape 为逐层输入。多模态 token-identity 分支在占位处查询 PAD，不通过图像特征倒推 token。",
      formula: "lookup(ids) × √Dple → [B,S,L,Dple]",
    });
    edge("ple-ids", "ple-table", ids);
    add("ple-project", 440, 220, {
      title: "Context PLE Projection",
      subtitle: "Linear / √H → reshape → RMSNorm",
      category: "embedding",
      inputs: [hidden],
      outputs: [ple],
      params: p.pleProjection,
      description:
        "使用合并后的当前输入 embedding（包括媒体特征），一次生成所有层的 context PLE。",
      formula: "norm(reshape(Linear(inputs_embeds) / √H))",
    });
    edge("ple-context", "ple-project");
    add("ple-combine", 210, 440, {
      title: "Combine PLE Components",
      subtitle: "Identity + context, scaled by 1/√2",
      category: "operation",
      inputs: [ple, { ...ple, name: "context_ple" }],
      outputs: [ple],
      formula: "(token_ple + context_ple) / √2",
    });
    edge("ple-table", "ple-combine", ple);
    edge("ple-project", "ple-combine", ple);
    add("ple-slice", 210, 660, {
      title: `Slice Layer ${index}`,
      subtitle: "Select third tensor axis",
      category: "operation",
      inputs: [ple],
      outputs: [pleSlice],
      formula: `per_layer_inputs[:, :, ${index}, :]`,
    });
    edge("ple-combine", "ple-slice", ple);
    boundary("ple-current", 650, 660, "After Layer MLP");
    add("ple-gate", 650, 880, {
      title: "PLE Gate + GELU",
      subtitle: `H → ${g.pleDim}`,
      category: "mlp",
      inputs: [hidden],
      outputs: [t("activated_gate", sh.pleSlice, ["B", "S", "Dple"])],
      params: m.hidden * g.pleDim,
      description: "gate 输入是当前层 attention 和 MLP 残差后的隐藏状态。",
    });
    edge("ple-current", "ple-gate");
    add("ple-multiply", 210, 1100, {
      title: "Gate × Layer PLE",
      subtitle: "Per-token multiplicative conditioning",
      category: "operation",
      inputs: [pleSlice, t("activated_gate", sh.pleSlice, ["B", "S", "Dple"])],
      outputs: [pleSlice],
    });
    edge("ple-slice", "ple-multiply", pleSlice);
    edge("ple-gate", "ple-multiply", pleSlice);
    add("ple-output", 210, 1320, {
      title: "Project + Norm + Residual",
      subtitle: "Return to hidden dimension",
      category: "embedding",
      inputs: [pleSlice, hidden],
      params: g.pleDim * m.hidden + m.hidden,
      formula: "x + RMSNorm(per_layer_projection(gated_ple))",
    });
    edge("ple-multiply", "ple-output", pleSlice);
    edge("ple-current", "ple-output", hidden, "residual");
  } else if (view === "cache") {
    for (const [type, owner, x] of [
      ["sliding_attention", g.localSource, 0],
      ["full_attention", g.globalSource, 460],
    ] as const) {
      if (owner === null) continue;
      const ownerSh = gemmaShapes(m, s, owner),
        key = t("visible_key", ownerSh.kv, ["B", "Nkv", "Klen", "D"]),
        value = { ...key, name: "visible_value" },
        consumers = g.layerTypes
          .map((t, i) => (t === type && i >= g.shareStart ? i : -1))
          .filter((i) => i >= 0);
      add(`owner-${owner}`, x, 0, {
        title: `KV Owner · Layer ${owner}`,
        subtitle: type,
        category: "attention",
        inputs: [hidden],
        outputs: [key, value],
        params:
          p.layers[owner].kProj + p.layers[owner].vProj + ownerSh.layer.headDim,
        drill: "attention",
        layerIndex: owner,
        badge: `D=${ownerSh.layer.headDim}`,
        description:
          "共享区之前最后一个相同注意力类型的独立层，在本次调用中发布归一化 K/V。",
      });
      add(`reuse-${owner}`, x, 260, {
        title: "Shared KV Consumers",
        subtitle: consumers.length
          ? `Layers ${consumers.join(", ")}`
          : "No consumers",
        category: "block",
        inputs: [key, value],
        outputs: [hidden],
        drill: "attention",
        layerIndex: consumers[0] ?? owner,
        badge: `${consumers.length} layers`,
        description:
          "这些层没有 K/V 投影权重、没有独立 DynamicCache 条目。每层仍以自己的 Query 计算 attention。",
      });
      edge(`owner-${owner}`, `reuse-${owner}`, key, "shared");
    }
    add("cache-summary", 0, 550, {
      title: "Persistent Cache Accounting",
      subtitle: `${g.shareStart} owners / ${m.layers} layers`,
      category: "operation",
      inputs: [],
      outputs: [],
      description:
        "前 shareStart 层各自持有 DynamicCache；共享区不新增条目。局部保留 W−1 个历史项；本次可见 KV 包含当前 chunk。共享字典引用本次可见状态，不按层数再次相乘。",
      details: [
        ["persistent KV", formatBytes(mem.kvCache)],
        ["independent owners", String(mem.owners.length)],
        [
          "local retained",
          String(Math.min(s.past + s.sequence, m.slidingWindow! - 1)),
        ],
        ["global retained", String(s.past + s.sequence)],
        ["PLE activations", formatBytes(mem.pleActivations)],
      ],
    });
  } else if (view === "vision" && g.vision) {
    const v = g.vision,
      vs = gemmaVisionShapes(m, s),
      pixels = t("patches", vs.pixels, ["B", "Npatch", "3p²"]),
      positions = t(
        "pixel_position_ids",
        vs.positions,
        ["B", "Npatch", "XY"],
        undefined,
        "int64",
      ),
      vh = t("vision_hidden", vs.hidden, ["B", "Npatch", "Hv"]);
    boundary("vision-input", 0, 0, "Image Patch Grid", pixels);
    add("vision-embed", 0, 220, {
      title: "Patch + Position Embedding",
      subtitle: "Linear + learned 2D table",
      category: "embedding",
      inputs: [pixels, positions],
      outputs: [vh],
      params: 3 * v.patch ** 2 * v.hidden + 2 * v.positions * v.hidden,
      description:
        "patch 值先按 2×(x−0.5) 缩放再线性投影，并分别查询 x/y 位置 embedding 相加。这里采用无 padding 的规则网格。",
    });
    edge("vision-input", "vision-embed", pixels);
    add("vision-layers", 0, 440, {
      title: "Gemma Vision Transformer",
      subtitle: `${v.layers} × bidirectional blocks`,
      category: "block",
      inputs: [vh],
      outputs: [vh],
      params:
        v.layers *
        (2 * v.hidden * (v.heads + v.kvHeads) * v.headDim +
          2 * v.headDim +
          3 * v.hidden * v.intermediate +
          4 * v.hidden),
      description:
        "每层：pre-norm → Q/K/V 投影与 head norm → 多维 RoPE → 双向 attention（scale=1）→ post-norm + residual → gated GELU MLP 与前后 norm。",
      details: [
        ["Q shape", `[${b}, ${v.heads}, ${vs.patches}, ${v.headDim}]`],
        ["score shape", `[${b}, ${v.heads}, ${vs.patches}, ${vs.patches}]`],
        ["MLP intermediate", String(v.intermediate)],
      ],
    });
    edge("vision-embed", "vision-layers", vh);
    add("vision-pool", 0, 660, {
      title: "Spatial Average Pooling",
      subtitle: `${v.pool} × ${v.pool} → one soft token`,
      category: "operation",
      inputs: [vh, positions],
      outputs: [
        t(
          "pooled_features",
          vs.pooled,
          ["B", "Nimg", "Hv"],
          undefined,
          "float32",
        ),
      ],
      description:
        "按 patch 坐标分组平均池化，不是 pixel-unshuffle；缩放 √Hv 在 float32 进行。规则网格必须能被 pooling kernel 整除，实际 padding/裁剪由 processor 决定。",
      formula: `Nimg = Npatch / ${v.pool ** 2}`,
    });
    edge("vision-layers", "vision-pool", vh);
    const flattened = t(
      "valid_vision_tokens",
      [b * vs.tokens, v.hidden],
      ["Nvalid", "Hv"],
      "全部 patch 有效时 Nvalid=B×Nimg；真实 padding 会减少有效 token 数。",
    );
    add("vision-valid", 0, 880, {
      title: "Cast & Select Valid Tokens",
      subtitle: "Float32 pooling → model dtype → flatten",
      category: "operation",
      inputs: [
        t(
          "pooled_features",
          vs.pooled,
          ["B", "Nimg", "Hv"],
          undefined,
          "float32",
        ),
      ],
      outputs: [flattened],
      description:
        "池化缩放在 float32 中进行，可选 standardize 后转回计算精度，再筛选有效 token；视觉接口将 batch 和 soft token 轴合并。",
    });
    edge(
      "vision-pool",
      "vision-valid",
      t(
        "pooled_features",
        vs.pooled,
        ["B", "Nimg", "Hv"],
        undefined,
        "float32",
      ),
    );
    add("vision-project", 0, 1100, {
      title: "Vision → Language Projection",
      subtitle: "Unscaled RMSNorm → Linear",
      category: "embedding",
      inputs: [flattened],
      outputs: [
        t("image_features", [b * vs.tokens, m.hidden], ["Nvalid", "H"]),
      ],
      params: v.hidden * m.hidden,
      description:
        "池化后转回模型 dtype，可选 standardize 不改变形状；无 scale RMSNorm 后线性投影到语言隐藏维度。",
    });
    edge("vision-valid", "vision-project", flattened);
    add("vision-merge", 0, 1320, {
      title: "Insert Image Soft Tokens",
      subtitle: "Image placeholders already included in S",
      category: "operation",
      inputs: [
        hidden,
        t("image_features", [b * vs.tokens, m.hidden], ["Nvalid", "H"]),
      ],
      description:
        "将特征写回图像占位位置；S 至少包含 Nimg 个软 token 与起止标记。实际视频需逐帧 processor 的布局，不从配置猜测时间轴。",
    });
    edge(
      "vision-project",
      "vision-merge",
      t("image_features", [b * vs.tokens, m.hidden], ["Nvalid", "H"]),
    );
  } else if (view === "audio" && g.audio) {
    const a = g.audio,
      as = gemmaAudioShapes(m, s),
      ah = t("audio_hidden", as.hidden, ["B", "Ta", "Ha"]);
    const H = a.hidden,
      [c0, c1] = a.channels,
      ff = 8 * H * H + 2 * H,
      att = 5 * H * H + H / a.heads + 2 * H,
      conv = 3 * H * H + a.kernel * H + 2 * H;
    boundary(
      "audio-input",
      0,
      0,
      "Preprocessed Audio Features",
      t("input_features", as.input, ["B", "F", "Mel"]),
    );
    add("audio-subsample", 0, 220, {
      title: "Two Stride-2 Convolutions",
      subtitle: "Conv2D → LayerNorm → ReLU",
      category: "embedding",
      inputs: [t("input_features", as.input, ["B", "F", "Mel"])],
      outputs: [ah],
      params: 9 * c0 + c0 + 9 * c0 * c1 + c1 + (c0 / 4) * c1 * H,
      description:
        "两次 3×3 padding=1 stride=2 卷积同时缩小时间和频率，展平后投影到音频隐藏维度。帧数是预处理结果，不从原始音频秒数推断。",
      details: [
        ["conv1", shapeText(as.conv1)],
        ["conv2", shapeText(as.conv2)],
        ["time steps", String(as.time2)],
      ],
    });
    edge(
      "audio-input",
      "audio-subsample",
      t("input_features", as.input, ["B", "F", "Mel"]),
    );
    add("audio-ffn1", 0, 440, {
      title: "Audio FFN 1",
      subtitle: `Repeated × ${a.layers} layers`,
      category: "mlp",
      inputs: [ah],
      outputs: [ah],
      params: ff * a.layers,
      description:
        "每个音频块首先执行 pre-norm → Linear(H,4H) → SiLU → Linear → post-norm，以 residual_weight 缩放后相加。本图中间四步按 num_hidden_layers 重复，节点参数显示所有重复层总和。",
      details: [
        ["residual_weight", String(a.residualWeight)],
        ["repeat count", String(a.layers)],
      ],
    });
    edge("audio-subsample", "audio-ffn1", ah);
    add("audio-attention", 0, 660, {
      title: "Chunked Relative Attention",
      subtitle: `Chunk ${a.chunk} · left ${a.left} · right ${a.right}`,
      category: "attention",
      inputs: [ah],
      outputs: [ah],
      params: att * a.layers,
      description:
        "Q 切成非重叠块，K/V 提取重叠上下文；相对位置 bias、per-dimension Q scale 与 logit softcap。不是完整音频的全局 attention。",
      details: [
        ["query blocks", shapeText(as.queryBlocks)],
        ["KV contexts", shapeText(as.contextBlocks)],
        ["logical scores", shapeText(as.scores)],
        ["repeat count", String(a.layers)],
      ],
    });
    edge("audio-ffn1", "audio-attention", ah);
    add("audio-conv", 0, 880, {
      title: "Light Causal Convolution",
      subtitle: "Linear → GLU → depthwise conv",
      category: "operation",
      inputs: [ah],
      outputs: [ah],
      params: conv * a.layers,
      description:
        "沿时间轴左侧 padding 保持长度；再归一化、SiLU、线性投影和残差连接。",
      details: [
        ["kernel_size", String(a.kernel)],
        ["repeat count", String(a.layers)],
      ],
    });
    edge("audio-attention", "audio-conv", ah);
    add("audio-ffn2", 0, 1100, {
      title: "Audio FFN 2 + Final Norm",
      subtitle: `End of each of ${a.layers} audio blocks`,
      category: "mlp",
      inputs: [ah],
      outputs: [ah],
      params: (ff + H) * a.layers,
      description:
        "块尾第二个 FFN 后再 RMSNorm。整个 FFN1 → Attention → Conv → FFN2 结构重复 L_audio 次。",
    });
    edge("audio-conv", "audio-ffn2", ah);
    add("audio-project", 0, 1320, {
      title: "Audio Output Projection",
      subtitle: `${H} → ${a.output}`,
      category: "embedding",
      inputs: [ah],
      outputs: [t("audio_output", as.projected, ["B", "Ta", "Hao"])],
      params: H * a.output + a.output,
      description:
        "有 bias 的输出投影；attention_mask 下采样后标记有效软 token。当前示例全部帧有效。",
    });
    edge("audio-ffn2", "audio-project", ah);
    add("audio-language", 0, 1540, {
      title: "Audio → Language Soft Tokens",
      subtitle: "Unscaled RMSNorm → Linear",
      category: "embedding",
      inputs: [t("audio_output", as.projected, ["B", "Ta", "Hao"])],
      outputs: [t("audio_features", as.softTokens, ["B", "Ta", "H"])],
      params: a.output * m.hidden,
      description:
        "映射后替换语言序列的音频占位 token。S 已包含软 token 与起止标记；Decode 不重新处理音频。",
    });
    edge(
      "audio-project",
      "audio-language",
      t("audio_output", as.projected, ["B", "Ta", "Hao"]),
    );
  } else boundary("unavailable", 0, 0, "此配置未启用该分支");
  if (view === "vision" && (!s.image?.count || s.past > 0))
    nodes.forEach((n) => (n.data.inactive = true));
  if (view === "audio" && (!s.audio?.enabled || s.past > 0))
    nodes.forEach((n) => (n.data.inactive = true));
  for (const e of edges) {
    if (nodes.find((n) => n.id === e.source)?.data.inactive) {
      e.data = { ...e.data, inactive: true };
      e.style = { ...e.style, opacity: 0.45 };
      e.label = `示例 · ${e.label}`;
    }
  }
  for (const e of edges) {
    if (!e.data?.shared) continue;
    const from = nodes.find((n) => n.id === e.source)!,
      to = nodes.find((n) => n.id === e.target)!;
    if (to.position.x > from.position.x) {
      e.sourceHandle = "right-out";
      e.targetHandle = "side-in";
    } else if (to.position.x < from.position.x) {
      e.sourceHandle = "side-out";
      e.targetHandle = "right-in";
    }
  }
  const ordered: ModuleNode[] = [],
    pending = new Map(nodes.map((n) => [n.id, n]));
  while (pending.size) {
    const ready = [...pending.values()].filter(
      (n) => !edges.some((e) => e.target === n.id && pending.has(e.source)),
    );
    if (!ready.length) throw new Error("Gemma 图出现循环依赖。");
    for (const n of ready) {
      ordered.push(n);
      pending.delete(n.id);
    }
  }
  ordered.forEach((n, i) => (n.data.order = i + 1));
  return { nodes: ordered, edges };
}
