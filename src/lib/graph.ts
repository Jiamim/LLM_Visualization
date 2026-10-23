import { MarkerType, type Edge, type Node } from "@xyflow/react";
import { NODE_WIDTH, NODE_HEIGHT } from "./presentation";
import { buildDeepseekGraph } from "./deepseek-graph";
import { buildGemmaGraph } from "./gemma-graph";
import { buildCatalogGraph } from "./catalog-graph";
import { buildArchitecture } from "./architecture";
import { emphasizeResiduals } from "./residuals";
import { annotateGraph } from "./graph-semantics";
import {
  hasLMHead,
  parameterCounts,
  shapeText,
  tensorShapes,
  type ModelConfig,
  type Scenario,
  type Tensor,
} from "./model";

export type View =
  | "architecture"
  | "overview"
  | "layer"
  | "attention"
  | "mlp"
  | "encoder"
  | "decoder"
  | "cache"
  | "vision"
  | "engram"
  | "dspark"
  | "ple"
  | "audio"
  | "auxiliary";
// Gemma's auxiliary inputs have independent graph views.
export type Category =
  | "input"
  | "embedding"
  | "block"
  | "attention"
  | "norm"
  | "mlp"
  | "output"
  | "operation";
export interface ModuleData extends Record<string, unknown> {
  title: string;
  subtitle: string;
  category: Category;
  path: string;
  inputs: Tensor[];
  outputs: Tensor[];
  params: number;
  parameterStatus?: "known" | "estimated" | "unknown";
  linear?: { input: number; output: number };
  attention?: {
    queryLength: number;
    keyLength: number;
    queryOffset: number;
    keyOffset: number;
    heads: number;
    kvHeads: number;
    window: number | null;
    visibility: "causal" | "bidirectional" | "unknown";
  };
  summary?: boolean;
  sceneView?: View;
  sceneSourceId?: string;
  sceneParent?: string;
  description: string;
  formula: string;
  details: [string, string][];
  drill?: View;
  badge?: string;
  active?: boolean;
  order?: number;
  symbolic?: boolean;
  onDrill?: () => void;
  layerIndex?: number;
  inactive?: boolean;
  residual?: boolean;
  repeat?: number;
  worldPosition?: [number, number, number];
  frame?: {
    width: number;
    height: number;
    count: number;
    range: [number, number];
    period: number;
  };
}
export type ModuleNode = Node<ModuleData, "module">;
export interface ModelGraph {
  nodes: ModuleNode[];
  edges: Edge[];
  ports?: { entry: string; exit: string };
  sections?: {
    title: string;
    description: string;
    anchor: string;
    nodes: string[];
    repeat: number;
    period: number;
    start: number;
    length: number;
  }[];
}

export function buildGraph(
  m: ModelConfig,
  s: Scenario,
  view: View,
  layerIndex: number,
  symbolic = false,
): ModelGraph {
  if (view === "architecture")
    return buildArchitecture(
      m,
      s,
      buildGraph(m, s, "overview", layerIndex, symbolic),
      (index) => buildGraph(m, s, "layer", index, symbolic),
      symbolic,
    );
  const graph = emphasizeResiduals(
    buildBaseGraph(m, s, view, layerIndex, symbolic),
  );
  if (view === "overview")
    for (const node of graph.nodes) {
      if (
        node.data.category === "block" &&
        ["encoder", "decoder"].includes(node.id)
      )
        node.data.repeat =
          node.id === "encoder" && m.deepseek
            ? m.deepseek.encoderLayers
            : node.id === "decoder" && m.deepseek
              ? m.layers - m.deepseek.encoderLayers
              : m.layers;
    }
  return annotateGraph(graph, m, s, view, layerIndex);
}

function buildBaseGraph(
  m: ModelConfig,
  s: Scenario,
  view: View,
  layerIndex: number,
  symbolic: boolean,
): ModelGraph {
  if (m.catalog) return buildCatalogGraph(m, s, view, layerIndex, symbolic);
  if (m.gemma) return buildGemmaGraph(m, s, view, layerIndex, symbolic);
  if (m.deepseek) return buildDeepseekGraph(m, s, view, layerIndex, symbolic);
  const sh = tensorShapes(m, s),
    p = parameterCounts(m);
  const path = `model.layers.${layerIndex}`;
  const t = (
    name: string,
    shape: number[],
    symbols: string[],
    note?: string,
  ): Tensor => ({ name, shape, symbols, dtype: s.dtype, note });
  const ids: Tensor = {
    name: "input_ids",
    shape: sh.ids,
    symbols: ["B", "S"],
    dtype: "int64",
  };
  const hidden = t("hidden_states", sh.hidden, ["B", "S", "H"]);
  const intermediate = t("intermediate", sh.intermediate, ["B", "S", "I"]);
  const logits = t("logits", sh.logits, ["B", "S", "V"]);
  const qProjection = t("q_proj", sh.qProjection, ["B", "S", "Nq × D"]);
  const kvProjection = t("kv_proj", sh.kvProjection, ["B", "S", "Nkv × D"]);
  const q = t("query", sh.q, ["B", "Nq", "S", "D"]);
  const kvNew = t("new_key / value", sh.kvNew, ["B", "Nkv", "S", "D"]);
  const kv = t(
    "cached_key / value",
    sh.kv,
    ["B", "Nkv", "T", "D"],
    "T = P + S；展示未裁剪缓存。",
  );
  const expanded = t(
    "key / value",
    sh.repeatedKV,
    ["B", "Nq", "T", "D"],
    "GQA 逻辑扩展；高效 attention 后端可能不实际复制。",
  );
  const scores = t(
    "attention_scores",
    sh.scores,
    ["B", "Nq", "S", "T"],
    "逻辑矩阵；FlashAttention 等融合实现不完整分配此矩阵。",
  );
  const context = t("context", sh.context, ["B", "Nq", "S", "D"]);
  const mask = t(
    "attention_mask",
    sh.mask,
    ["B", "1", "S", "T"],
    "逻辑广播形状；屏蔽未来位置以及滑动窗口以外的位置。",
  );
  const nodes: ModuleNode[] = [],
    edges: Edge[] = [];
  const add = (
    id: string,
    x: number,
    y: number,
    data: Partial<ModuleData> & Pick<ModuleData, "title" | "category">,
  ) => {
    nodes.push({
      id,
      type: "module",
      ariaLabel: data.title,
      width: NODE_WIDTH,
      height: NODE_HEIGHT,
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
  };
  const edge = (
    source: string,
    target: string,
    tensor: Tensor = hidden,
    label?: string,
    residual = false,
  ) => {
    edges.push({
      id: `${source}-${target}-${edges.length}`,
      source,
      target,
      type: "smoothstep",
      sourceHandle: residual ? "side-out" : "out",
      targetHandle: residual ? "side-in" : "in",
      label:
        label ??
        (symbolic
          ? `[${tensor.symbols.join(", ")}]`
          : shapeText(tensor.shape)),
      data: { tensor, residual },
      markerEnd: {
        type: MarkerType.ArrowClosed,
        color: residual ? "#b3a1d8" : "#9da6b7",
        width: 15,
        height: 15,
      },
      style: {
        stroke: residual ? "#b3a1d8" : "#aab3c2",
        strokeWidth: 1.7,
        strokeDasharray: residual ? "5 5" : undefined,
      },
      labelStyle: {
        fill: "#475569",
        fontSize: 12,
        fontFamily: "ui-monospace, monospace",
      },
      labelBgStyle: { fill: "#ffffff", fillOpacity: 0.98 },
      labelBgPadding: [7, 5],
      labelBgBorderRadius: 5,
      pathOptions: { borderRadius: 16 },
    } as Edge);
  };
  const norm = (id: string, x: number, y: number, modulePath: string) =>
    add(id, x, y, {
      title: "RMS Normalization",
      subtitle: "RMSNorm",
      category: "norm",
      path: modulePath,
      params: m.hidden,
      description: "沿隐藏维度归一化，稳定数值尺度；输入输出形状相同。",
      formula: "y = x / √(mean(x²) + ε) ⊙ weight",
      details: [
        ["normalized_shape", String(m.hidden)],
        ["eps", String(m.normEps)],
      ],
    });
  const attention = (id: string, x: number, y: number) =>
    add(id, x, y, {
      title: "Self Attention",
      subtitle:
        m.heads === m.kvHeads
          ? "Multi-head attention"
          : "Grouped-query attention",
      category: "attention",
      path: `${path}.self_attn`,
      params: p.attention,
      drill: "attention",
      badge: `${m.heads} Q · ${m.kvHeads} KV`,
      description:
        "Q、K、V 投影后应用旋转位置编码，结合 KV cache 计算因果注意力，再投影回隐藏空间。",
      formula: "Attention(Q, K, V) = softmax(QKᵀ / √D + mask)V",
      details: [
        ["num_attention_heads", String(m.heads)],
        ["num_key_value_heads", String(m.kvHeads)],
        ["head_dim", String(m.headDim)],
        ["qk_norm", m.qkNorm ? "RMSNorm" : "无"],
        [
          "attention_type",
          m.slidingLayers[layerIndex]
            ? `sliding (${m.slidingWindow})`
            : "full causal",
        ],
      ],
    });
  const mlp = (id: string, x: number, y: number) =>
    add(id, x, y, {
      title: "Feed Forward",
      subtitle:
        m.activation === "silu"
          ? "SwiGLU MLP"
          : `Gated MLP · ${m.activation}`,
      category: "mlp",
      path: `${path}.mlp`,
      params: p.mlp,
      drill: "mlp",
      badge: `${m.hidden} → ${m.intermediate}`,
      description:
        "门控前馈网络：gate 和 up 分支扩展到中间维度，逐元素相乘后由 down 投影回隐藏维度。",
      formula: `y = down_proj(${m.activation}(gate_proj(x)) ⊙ up_proj(x))`,
      details: [
        ["hidden_size", String(m.hidden)],
        ["intermediate_size", String(m.intermediate)],
        ["activation", m.activation],
        ["bias", String(m.mlpBias)],
      ],
    });
  const passthrough = (
    id: string,
    x: number,
    y: number,
    title: string,
    input = hidden,
  ) =>
    add(id, x, y, {
      title,
      subtitle: "Tensor",
      category: "input",
      inputs: [input],
      outputs: [input],
      description: "当前视图的数据流边界，不包含可训练参数。",
      formula: "shape in = shape out",
    });
  if (view === "overview") {
    add("input", 120, 0, {
      title: "Input Tokens",
      subtitle: "Token IDs",
      category: "input",
      path: "input_ids",
      inputs: [],
      outputs: [ids],
      description:
        "分词器输出的 token ID。Batch 和序列长度由左侧输入场景指定。",
      formula: "input_ids ∈ ℤ^(B × S)",
      details: [
        ["dtype", "int64"],
        ["batch_size", String(s.batch)],
        ["sequence_length", String(s.sequence)],
      ],
    });
    add("embedding", 120, 170, {
      title: "Token Embedding",
      subtitle: "nn.Embedding",
      category: "embedding",
      path: "model.embed_tokens",
      inputs: [ids],
      params: p.embedding,
      description: "将离散 token ID 映射到连续隐藏向量。",
      formula: "hidden_states = embedding_table[input_ids]",
      details: [
        ["weight", `[${m.vocab}, ${m.hidden}]`],
        ["tied_with_lm_head", String(m.tiedEmbeddings && hasLMHead(m))],
      ],
    });
    add("decoder", 120, 340, {
      title: "Transformer Blocks",
      subtitle: `${m.layers} × DecoderLayer`,
      category: "block",
      path: "model.layers",
      params: p.layer * m.layers,
      drill: "layer",
      badge: `× ${m.layers}`,
      description: `由 ${m.layers} 层相同结构的 Decoder Layer 顺序组成。每层包含 RMSNorm、Self Attention、MLP 和两条残差连接。点击展开任意一层。`,
      formula: "xₗ₊₁ = DecoderLayerₗ(xₗ)",
      details: [
        ["num_hidden_layers", String(m.layers)],
        ["per_layer_params", p.layer.toLocaleString("en-US")],
      ],
    });
    norm("final-norm", 120, 510, "model.norm");
    edge("input", "embedding", ids);
    edge("embedding", "decoder");
    edge("decoder", "final-norm");
    if (hasLMHead(m)) {
      add("lm-head", 120, 680, {
        title: "Language Model Head",
        subtitle: "Linear · vocabulary projection",
        category: "output",
        path: "lm_head",
        outputs: [logits],
        params: p.lmHead,
        badge: m.tiedEmbeddings ? "共享权重" : undefined,
        description: `将隐藏状态投影到词表，得到每个位置的 logits。${m.tiedEmbeddings ? "权重与 Token Embedding 共享，总参数量只计算一次。" : ""}图中展示所有 S 个位置，未应用 logits_to_keep。`,
        formula: "logits = hidden_states @ Wᵀ",
        details: [
          ["weight", `[${m.vocab}, ${m.hidden}]`],
          ["tie_word_embeddings", String(m.tiedEmbeddings)],
        ],
      });
      edge("final-norm", "lm-head");
      passthrough("output", 120, 850, "Output Logits", logits);
      edge("lm-head", "output", logits);
    } else {
      passthrough("output", 120, 680, "Last Hidden State");
      edge("final-norm", "output");
    }
  } else if (view === "layer") {
    passthrough("layer-input", 140, 0, "Layer Input");
    norm("input-norm", 140, 170, `${path}.input_layernorm`);
    attention("attention", 140, 340);
    add("attention-residual", 140, 510, {
      title: "Residual Add",
      subtitle: "Element-wise addition",
      category: "operation",
      path: `${path}.residual_1`,
      inputs: [hidden, { ...hidden, name: "residual" }],
      description: "将注意力输出与归一化前的输入相加。",
      formula: "x = residual + self_attn(input_layernorm(x))",
    });
    norm("post-norm", 140, 680, `${path}.post_attention_layernorm`);
    mlp("mlp", 140, 850);
    add("mlp-residual", 140, 1020, {
      title: "Residual Add",
      subtitle: "Element-wise addition",
      category: "operation",
      path: `${path}.residual_2`,
      inputs: [hidden, { ...hidden, name: "residual" }],
      description: "将前馈网络输出与第二次归一化前的隐藏状态相加。",
      formula: "output = residual + mlp(post_attention_layernorm(x))",
    });
    for (let i = 0; i < nodes.length - 1; i++)
      edge(nodes[i].id, nodes[i + 1].id);
    const residualLabel = `residual · ${symbolic ? "[B, S, H]" : shapeText(sh.hidden)}`;
    edge("layer-input", "attention-residual", hidden, residualLabel, true);
    edge("attention-residual", "mlp-residual", hidden, residualLabel, true);
  } else if (view === "attention") {
    passthrough("attn-input", 350, 0, "Normalized Input");
    const projections = [
      {
        id: "q-proj",
        x: 0,
        title: "Query Projection",
        params: p.qProj,
        output: qProjection,
        heads: m.heads,
      },
      {
        id: "k-proj",
        x: 350,
        title: "Key Projection",
        params: p.kProj,
        output: { ...kvProjection, name: "k_proj" },
        heads: m.kvHeads,
      },
      {
        id: "v-proj",
        x: 700,
        title: "Value Projection",
        params: p.vProj,
        output: { ...kvProjection, name: "v_proj" },
        heads: m.kvHeads,
      },
    ];
    projections.forEach((proj) => {
      add(proj.id, proj.x, 180, {
        title: proj.title,
        subtitle: "Linear",
        category: "attention",
        path: `${path}.self_attn.${proj.id.replace("-", "_")}`,
        outputs: [proj.output],
        params: proj.params,
        description: `将隐藏维度 ${m.hidden} 投影到 ${proj.heads} × ${m.headDim}。`,
        formula: `y = x @ Wᵀ${m.qkvBias ? " + bias" : ""}`,
        details: [
          ["weight", `[${proj.heads * m.headDim}, ${m.hidden}]`],
          ["bias", String(m.qkvBias)],
        ],
      });
      edge("attn-input", proj.id);
    });
    for (const [branch, x, input, output] of [
      ["q", 0, qProjection, q],
      ["k", 350, kvProjection, kvNew],
      ["v", 700, kvProjection, kvNew],
    ] as const) {
      const withNorm = m.qkNorm && branch !== "v";
      add(`${branch}-reshape`, x, 360, {
        title: withNorm ? "Reshape + RMSNorm" : "Reshape + Transpose",
        subtitle: withNorm
          ? "Per-head QK normalization"
          : "Split attention heads",
        category: withNorm ? "norm" : "operation",
        path: `${path}.self_attn.${withNorm ? `${branch}_norm` : `${branch}_reshape`}`,
        inputs: [input],
        outputs: [{ ...output, name: branch }],
        params: withNorm ? m.headDim : 0,
        description: `拆分 head 维度并交换序列轴。${withNorm ? "Qwen3 在每个 head 的最后一维应用 RMSNorm。" : ""}`,
        formula: `[B, S, N × D] → [B, S, N, D]${withNorm ? " → RMSNorm" : ""} → [B, N, S, D]`,
        details: [
          ["heads", String(branch === "q" ? m.heads : m.kvHeads)],
          ["head_dim", String(m.headDim)],
        ],
      });
      edge(`${branch}-proj`, `${branch}-reshape`, input);
    }
    for (const [branch, x, tensor] of [
      ["q", 0, q],
      ["k", 350, kvNew],
    ] as const) {
      add(`${branch}-rope`, x, 540, {
        title: "Rotary Position",
        subtitle: "RoPE",
        category: "embedding",
        path: `${path}.self_attn.${branch}_rope`,
        inputs: [tensor],
        outputs: [tensor],
        description:
          "根据位置对 Query / Key 成对旋转，保持张量形状。position_ids 逻辑形状为 [B, S]，通常可按 batch 广播。",
        formula: "rotary(x, position_ids)",
        details: [
          ["rope_theta", String(m.raw.rope_theta ?? "架构默认")],
          ["position_ids", `[${s.batch}, ${s.sequence}]`],
        ],
      });
      edge(`${branch}-reshape`, `${branch}-rope`, tensor);
    }
    for (const [branch, x] of [
      ["k", 350],
      ["v", 700],
    ] as const) {
      add(`${branch}-cache`, x, 720, {
        title: "KV Cache Update",
        subtitle: `Past ${s.past} + current ${s.sequence}`,
        category: "operation",
        path: `${path}.self_attn.${branch}_cache`,
        inputs: [
          kvNew,
          t(
            `past_${branch}`,
            [s.batch, m.kvHeads, s.past, m.headDim],
            ["B", "Nkv", "P", "D"],
          ),
        ],
        outputs: [kv],
        description:
          "沿序列轴拼接历史 Key / Value。P 为输入场景中指定的缓存长度；prefill 通常 P=0。",
        formula: "cache = concat(past, current, dim=2)",
        details: [
          ["past_length", String(s.past)],
          ["total_length", String(s.past + s.sequence)],
        ],
      });
      edge(branch === "k" ? "k-rope" : "v-reshape", `${branch}-cache`, kvNew);
      add(`${branch}-repeat`, x, 900, {
        title:
          m.heads === m.kvHeads ? "Multi-head Layout" : "Expand KV Heads",
        subtitle:
          m.heads === m.kvHeads
            ? "Identity · MHA"
            : `GQA · repeat × ${m.heads / m.kvHeads}`,
        category: "operation",
        path: `${path}.self_attn.${branch}_repeat`,
        inputs: [kv],
        outputs: [expanded],
        description:
          "每组 Query heads 使用同一个 Key / Value head。这里展示逻辑扩展形状；实现可以使用广播或融合内核。",
        formula: "[B, Nkv, T, D] → [B, Nq, T, D]",
        details: [["num_key_value_groups", String(m.heads / m.kvHeads)]],
      });
      edge(`${branch}-cache`, `${branch}-repeat`, kv);
    }
    add("scores", 0, 1080, {
      title: "Attention Scores",
      subtitle: "Scaled dot product",
      category: "attention",
      path: `${path}.self_attn.scores`,
      inputs: [q, { ...expanded, name: "key" }],
      outputs: [scores],
      description:
        "Query 与 Key 转置后做矩阵乘法，得到当前序列对完整缓存序列的注意力分数。",
      formula: "scores = Q @ Kᵀ / √D",
      details: [["scale", (1 / Math.sqrt(m.headDim)).toPrecision(4)]],
    });
    edge("q-rope", "scores", q);
    edge("k-repeat", "scores", expanded);
    add("softmax", 0, 1260, {
      title: "Mask + Softmax",
      subtitle: m.slidingLayers[layerIndex]
        ? `Causal · window ${m.slidingWindow}`
        : "Causal attention",
      category: "operation",
      path: `${path}.self_attn.softmax`,
      inputs: [scores, mask],
      outputs: [{ ...scores, name: "attention_weights" }],
      description:
        "先应用因果 mask，再沿最后一维归一化。图中为推理模式，不应用训练时 dropout。内部 softmax 常使用 float32，再转换回输入精度。",
      formula: "weights = softmax(scores + mask, dim=-1)",
      details: [
        ["logical_mask", shapeText(sh.mask)],
        ["causal_offset", String(s.past)],
      ],
    });
    edge("scores", "softmax", scores);
    add("weighted-values", 350, 1440, {
      title: "Weighted Values",
      subtitle: "Attention × Value",
      category: "attention",
      path: `${path}.self_attn.context`,
      inputs: [
        { ...scores, name: "attention_weights" },
        { ...expanded, name: "value" },
      ],
      outputs: [context],
      description:
        "用注意力权重对 Value 加权求和。每个 Query 位置生成一个 head_dim 长度的向量。",
      formula: "context = weights @ V",
    });
    edge("softmax", "weighted-values", scores);
    edge("v-repeat", "weighted-values", expanded);
    add("merge-heads", 350, 1620, {
      title: "Merge Heads",
      subtitle: "Transpose + reshape",
      category: "operation",
      path: `${path}.self_attn.merge_heads`,
      inputs: [context],
      outputs: [qProjection],
      description:
        "合并所有 Query heads。显式 head_dim 可能使合并维度不同于 hidden_size。",
      formula: "[B, Nq, S, D] → [B, S, Nq × D]",
    });
    edge("weighted-values", "merge-heads", context);
    add("o-proj", 350, 1800, {
      title: "Output Projection",
      subtitle: "Linear",
      category: "attention",
      path: `${path}.self_attn.o_proj`,
      inputs: [qProjection],
      params: p.oProj,
      description:
        "将合并后的 Query heads 投影回 hidden_size，供后续残差连接使用。",
      formula: `y = context @ Wᵀ${m.outputBias ? " + bias" : ""}`,
      details: [
        ["weight", `[${m.hidden}, ${m.heads * m.headDim}]`],
        ["bias", String(m.outputBias)],
      ],
    });
    edge("merge-heads", "o-proj", qProjection);
  } else {
    passthrough("mlp-input", 175, 0, "Normalized Input");
    for (const [id, x, title, params] of [
      ["gate-proj", 0, "Gate Projection", p.gateProj],
      ["up-proj", 350, "Up Projection", p.upProj],
    ] as const) {
      add(id, x, 180, {
        title,
        subtitle: "Linear",
        category: "mlp",
        path: `${path}.mlp.${id.replace("-", "_")}`,
        outputs: [intermediate],
        params,
        description: "沿最后一维从隐藏维度扩展到中间维度。",
        formula: `y = x @ Wᵀ${m.mlpBias ? " + bias" : ""}`,
        details: [
          ["weight", `[${m.intermediate}, ${m.hidden}]`],
          ["bias", String(m.mlpBias)],
        ],
      });
      edge("mlp-input", id);
    }
    add("activation", 0, 360, {
      title:
        m.activation === "silu"
          ? "SiLU Activation"
          : `${m.activation} Activation`,
      subtitle: "Gating function",
      category: "operation",
      path: `${path}.mlp.act_fn`,
      inputs: [intermediate],
      outputs: [intermediate],
      description: "对 gate 分支逐元素应用激活函数，形状保持不变。",
      formula:
        m.activation === "silu"
          ? "SiLU(x) = x · sigmoid(x)"
          : `${m.activation}(x)`,
    });
    edge("gate-proj", "activation", intermediate);
    add("multiply", 175, 540, {
      title: "Element-wise Multiply",
      subtitle: "Gate × Up",
      category: "operation",
      path: `${path}.mlp.multiply`,
      inputs: [
        { ...intermediate, name: "activated_gate" },
        { ...intermediate, name: "up" },
      ],
      outputs: [intermediate],
      description: "逐元素相乘，让 gate 分支控制 up 分支的信息通过量。",
      formula: "gated = activation(gate_proj(x)) ⊙ up_proj(x)",
    });
    edge("activation", "multiply", intermediate);
    edge("up-proj", "multiply", intermediate);
    add("down-proj", 175, 720, {
      title: "Down Projection",
      subtitle: "Linear",
      category: "mlp",
      path: `${path}.mlp.down_proj`,
      inputs: [intermediate],
      params: p.downProj,
      description: "从中间维度投影回 hidden_size，供残差连接使用。",
      formula: `y = gated @ Wᵀ${m.mlpBias ? " + bias" : ""}`,
      details: [
        ["weight", `[${m.hidden}, ${m.intermediate}]`],
        ["bias", String(m.mlpBias)],
      ],
    });
    edge("multiply", "down-proj", intermediate);
  }
  if (view === "overview") {
    // A compact reading path keeps the overview legible before zooming in.
    const positions = [
      { x: 0, y: 0 },
      { x: 460, y: 0 },
      { x: 460, y: 175 },
      { x: 0, y: 175 },
      { x: 0, y: 350 },
      { x: 460, y: 350 },
    ];
    nodes.forEach((node, index) => {
      node.position = positions[index];
      if (node.id === "output") node.data.category = "output";
    });
    edges.forEach((connection, index) => {
      if (index === 0 || index === 4) {
        connection.sourceHandle = "right-out";
        connection.targetHandle = "side-in";
      }
      if (index === 2) {
        connection.sourceHandle = "side-out";
        connection.targetHandle = "right-in";
      }
    });
  }
  if (view !== "overview") {
    nodes.forEach((node) => {
      node.position = {
        x: node.position.x * 1.13,
        y: node.position.y * 1.07,
      };
    });
  }
  nodes.forEach((node, index) => {
    node.data.order = index + 1;
  });
  return { nodes, edges };
}
