import { MarkerType, type Edge } from "@xyflow/react";
import type { ModelConfig, Scenario, Tensor } from "./model";
import { formatBytes, shapeText, validateScenario } from "./model";
import type { ModelGraph, ModuleData, ModuleNode, View } from "./graph";
import {
  catalogParameters,
  catalogLayerParams,
  catalogState,
  catalogAttentionParts,
  catalogMlpParts,
} from "./catalog";
import { NODE_WIDTH, NODE_HEIGHT } from "./presentation";

export function buildCatalogGraph(
  m: ModelConfig,
  s: Scenario,
  view: View,
  index: number,
  symbolic: boolean,
): ModelGraph {
  validateScenario(s);
  const a = m.catalog!,
    p = a.profiles[index];
  if (!p) throw new Error("层编号越界。");
  const cp = catalogParameters(m),
    ap = catalogAttentionParts(m, p),
    mp = catalogMlpParts(m, p),
    b = s.batch,
    q = s.sequence,
    total = s.past + q;
  const tensor = (
    name: string,
    shape: number[],
    symbols: string[],
    note?: string,
    dtype: Tensor["dtype"] = s.dtype,
  ): Tensor => ({ name, shape, symbols, note, dtype });
  const hidden = tensor("hidden_states", [b, q, m.hidden], ["B", "S", "H"]),
    stream =
      a.hc > 1
        ? tensor(
            "residual_stream",
            [b, q, a.hc, m.hidden],
            ["B", "S", "HC", "H"],
          )
        : hidden,
    ids = tensor("input_ids", [b, q], ["B", "S"], undefined, "int64");
  const nodes: ModuleNode[] = [],
    edges: Edge[] = [];
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
      position: { x, y },
      ariaLabel: data.title,
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
        parameterStatus:
          data.params === undefined &&
          ["attention", "mlp", "embedding", "norm"].includes(data.category)
            ? "unknown"
            : "estimated",
        ...data,
      },
    });
  const edge = (
    source: string,
    target: string,
    t = hidden,
    kind: "flow" | "shared" | "residual" = "flow",
    label?: string,
  ) =>
    edges.push({
      id: `${source}-${target}-${edges.length}`,
      source,
      target,
      type: "smoothstep",
      sourceHandle: kind === "flow" ? "out" : "right-out",
      targetHandle: kind === "flow" ? "in" : "right-in",
      label:
        label ??
        (symbolic ? `[${t.symbols.join(", ")}]` : shapeText(t.shape)),
      data: {
        tensor: t,
        shared: kind === "shared",
        residual: kind === "residual",
      },
      style: {
        stroke: kind === "shared" ? "#398b85" : "#9cabc0",
        strokeWidth: 1.6,
        strokeDasharray: kind === "flow" ? undefined : "6 4",
      },
      markerEnd: { type: MarkerType.ArrowClosed, color: "#9cabc0" },
      labelStyle: {
        fill: "#485e79",
        fontSize: 12,
        fontFamily: "ui-monospace,monospace",
      },
      labelBgStyle: { fill: "#fff" },
      labelBgPadding: [7, 5],
      labelBgBorderRadius: 5,
    } as Edge);
  const boundary = (
    id: string,
    x: number,
    y: number,
    title: string,
    t = hidden,
  ) =>
    add(id, x, y, {
      title,
      category: "input",
      subtitle: "Logical boundary",
      inputs: [],
      outputs: [t],
    });
  const note =
    "配置驱动的语言主干逻辑视图；参数为结构估算，不代表完整权重/量化文件大小。";
  const details: [string, string][] = [
    ["mixer", p.label],
    ["Q heads", String(p.heads)],
    ["KV heads", String(p.kvHeads)],
    ["QK head dim", String(p.headDim)],
    ["V head dim", String(p.valueDim)],
    [
      "position",
      a.type === "gpt2" ? "learned absolute" : p.nope ? "NoPE" : "RoPE",
    ],
    ["window", String(p.window ?? "full")],
    ["output gate", String(p.gated)],
  ];
  const norm = (id: string, x: number, y: number, t = hidden) =>
    add(id, x, y, {
      title: a.normName,
      subtitle: `${a.norm}-norm placement`,
      category: "norm",
      inputs: [t],
      outputs: [t],
      description: `${a.normName} 的放置依 ${a.type} 架构规则；特殊归一化/缩放请参见配置。`,
      params:
        m.hidden * (a.normName === "LayerNorm" && a.type === "gpt2" ? 2 : 1),
    });
  const mixer = (id: string, x: number, y: number, prof = p) =>
    add(id, x, y, {
      title:
        prof.kind === "ffn"
          ? "Feed-forward-only Entry"
          : `${prof.label} Mixer`,
      subtitle: prof.window
        ? `Local window ${prof.window}`
        : `Layer ${prof.index}`,
      category: "attention",
      params: catalogLayerParams(m, prof).attention,
      drill: "attention",
      layerIndex: prof.index,
      badge: prof.nope ? "NoPE" : prof.gated ? "gated" : undefined,
      details,
      description: `${note} ${["custom", "csa"].includes(prof.kind) ? "该专用算子提供配置与接口级展开；未知的额外投影不伪造精确参数。" : ""}`,
    });
  const ff = (id: string, x: number, y: number, prof = p) =>
    add(id, x, y, {
      title: prof.moe ? "Mixture of Experts" : "Feed Forward",
      subtitle: prof.moe
        ? `${prof.experts} experts · Top-${prof.topK}`
        : a.activation,
      category: "mlp",
      params: catalogLayerParams(m, prof).mlp,
      drill: "mlp",
      layerIndex: prof.index,
      badge: prof.moe
        ? `shared width ${prof.sharedWidth}`
        : `I=${prof.intermediate}`,
      description: note,
      details: [
        ["activation", a.activation],
        ["expert width", String(prof.expertWidth)],
        ["expert input width", String(prof.expertHidden)],
        ["shared FFN width", String(prof.sharedWidth)],
      ],
    });
  if (view === "overview") {
    boundary("input", 0, 0, "Input Tokens", ids);
    add("embedding", 0, 220, {
      title: "Token Embedding",
      subtitle: a.tied ? "Tied with LM head" : "Vocabulary lookup",
      category: "embedding",
      inputs: [ids],
      params: cp.embedding,
      description: `${note} embedding scaling=${a.embeddingScale}。`,
    });
    edge("input", "embedding", ids);
    let prev = "embedding";
    if (a.type === "gpt2") {
      add("positions", 0, 440, {
        title: "Learned Position Embedding",
        subtitle: `${m.maxPosition} positions`,
        category: "embedding",
        params: cp.position,
        description:
          "GPT-2 使用可训练绝对位置 embedding 与 token embedding 相加，不使用 RoPE。",
      });
      edge(prev, "positions");
      prev = "positions";
    }
    if (a.hc > 1 || a.extras.length) {
      add("auxiliary", 440, 220, {
        title: "Architecture-specific Paths",
        subtitle:
          a.hc > 1
            ? `Hyper-connections · HC=${a.hc}`
            : "Auxiliary memory / residuals",
        category: "operation",
        drill: "auxiliary",
        params: 0,
        description: a.extras.join(" "),
        outputs: [stream],
        details: [["scope", "专用混合/记忆参数未纳入主干小计"]],
      });
      edge(prev, "auxiliary");
    }
    add("decoder", 0, 660, {
      title: "Language Backbone",
      subtitle: `${m.layers} ${a.type === "nemotron_h" ? "hybrid entries" : "blocks"}${a.loops > 1 ? ` × ${a.loops} passes` : ""}`,
      category: "block",
      inputs: [stream],
      outputs: [stream],
      params: cp.perLayer.reduce((sum, p) => sum + p.total, 0),
      drill: "decoder",
      badge: a.loops > 1 ? "跨轮共享权重" : undefined,
      description: note,
      details: [
        [
          "block mix",
          Object.entries(
            a.profiles.reduce<Record<string, number>>(
              (sum, p) => ({ ...sum, [p.kind]: (sum[p.kind] ?? 0) + 1 }),
              {},
            ),
          )
            .map(([k, v]) => `${v} ${k}`)
            .join(" / "),
        ],
        ["repeat passes", String(a.loops)],
      ],
    });
    edge(prev, "decoder", stream);
    if (a.hc > 1 || a.extras.length)
      edge("auxiliary", "decoder", stream, "shared");
    norm("final-norm", 0, 880);
    edge("decoder", "final-norm", stream);
    add("lm-head", 0, 1100, {
      title: a.head ? "Language Model Head" : "Hidden States",
      subtitle: a.softcap
        ? `Logit softcap ${a.softcap}`
        : "Output projection",
      category: "output",
      outputs: [
        a.head ? tensor("logits", [b, q, m.vocab], ["B", "S", "V"]) : hidden,
      ],
      params: cp.lmHead,
      description: `展示完整 S 位置的逻辑 logits，不采用生成器 last-token 优化。${a.tied ? "权重与 Embedding 共享，总量只计一次。" : ""}`,
    });
    edge("final-norm", "lm-head");
    if (a.mtp || a.mediaKeys.length)
      add("scope", 440, 880, {
        title: "Optional / External Components",
        subtitle: `${a.mtp} MTP · ${a.mediaKeys.length} media configs`,
        category: "input",
        inputs: [],
        outputs: [],
        drill: "auxiliary",
        inactive: true,
        description: `画廊只展示语言主干。${a.mediaKeys.join(", ")} 和 MTP/draft 的具体执行需对应输入与实现；不将其默认为普通 Decoder。`,
      });
  } else if (view === "decoder") {
    a.profiles.forEach((prof, i) => {
      add(`block-${i}`, (i % 4) * 390, Math.floor(i / 4) * 220, {
        title: `${a.type === "nemotron_h" ? "Entry" : "Layer"} ${i}`,
        subtitle: prof.label,
        category:
          prof.kind === "ffn"
            ? "mlp"
            : [
                  "mamba",
                  "delta",
                  "kda",
                  "conv",
                  "mlstm",
                  "lightning",
                ].includes(prof.kind)
              ? "operation"
              : "attention",
        params: catalogLayerParams(m, prof).total,
        drill: "layer",
        layerIndex: i,
        badge: prof.moe
          ? `${prof.experts}E · K${prof.topK}`
          : prof.nope
            ? "NoPE"
            : undefined,
        inputs: [stream],
        outputs: [stream],
        description: note,
      });
      if (i) edge(`block-${i - 1}`, `block-${i}`, stream);
    });
  } else if (view === "layer") {
    boundary("block-input", 0, 0, `Layer ${index} Input`, stream);
    if (a.norm === "special") {
      add("special-pre", 0, 220, {
        title:
          a.hc > 1
            ? "HC Collapse / Residual Mixing"
            : "Special Residual Transform",
        subtitle: a.type,
        category: "operation",
        inputs: [stream],
        outputs: [hidden],
        description:
          a.extras.join(" ") ||
          "此架构的残差门控/混合需参考具体模型代码，不按普通 pre-norm 精确展开。",
        drill: "auxiliary",
      });
      edge("block-input", "special-pre", stream);
    }
    const start = a.norm === "special" ? "special-pre" : "block-input";
    if (a.type === "longcat_flash") {
      norm("lc-norm0", 0, 220);
      edge("block-input", "lc-norm0");
      mixer("mixer", 0, 440);
      edge("lc-norm0", "mixer");
      add("lc-add0", 0, 660, {
        title: "First Attention Residual",
        category: "operation",
      });
      edge("mixer", "lc-add0");
      edge("block-input", "lc-add0", hidden, "residual");
      norm("lc-ffnorm0", 0, 880);
      edge("lc-add0", "lc-ffnorm0");
      ff("ffn", 440, 880);
      edge("lc-ffnorm0", "ffn");
      add("lc-dense0", 0, 1100, {
        title: "First Dense MLP",
        subtitle: `I=${p.intermediate}`,
        category: "mlp",
        description: "该 dense 分支与快捷 MoE 并行。",
      });
      edge("lc-ffnorm0", "lc-dense0");
      add("lc-add1", 0, 1320, {
        title: "First Dense Residual",
        category: "operation",
      });
      edge("lc-dense0", "lc-add1");
      edge("lc-add0", "lc-add1", hidden, "residual");
      norm("lc-norm1", 0, 1540);
      edge("lc-add1", "lc-norm1");
      mixer("lc-attn1", 0, 1760);
      edge("lc-norm1", "lc-attn1");
      add("lc-add2", 0, 1980, {
        title: "Second Attention Residual",
        category: "operation",
      });
      edge("lc-attn1", "lc-add2");
      edge("lc-add1", "lc-add2", hidden, "residual");
      norm("lc-ffnorm1", 0, 2200);
      edge("lc-add2", "lc-ffnorm1");
      add("lc-dense1", 0, 2420, {
        title: "Second Dense MLP",
        subtitle: `I=${p.intermediate}`,
        category: "mlp",
      });
      edge("lc-ffnorm1", "lc-dense1");
      add("output", 0, 2640, {
        title: "Add Dense + Shortcut MoE",
        category: "operation",
        inputs: [
          hidden,
          { ...hidden, name: "shortcut_moe" },
          { ...hidden, name: "residual" },
        ],
      });
      edge("lc-dense1", "output");
      edge("ffn", "output", hidden, "shared");
      edge("lc-add2", "output", hidden, "residual");
    } else if (a.norm === "post") {
      mixer("mixer", 0, 220);
      norm("mixer-norm", 0, 440);
      add("attn-add", 0, 660, {
        title: "Residual Add",
        category: "operation",
      });
      edge(start, "mixer");
      edge("mixer", "mixer-norm");
      edge("mixer-norm", "attn-add");
      edge(start, "attn-add", hidden, "residual");
      ff("ffn", 440, 220);
      norm("ffn-norm", 440, 440);
      add("output", 440, 660, {
        title: "Residual Add",
        category: "operation",
      });
      edge("attn-add", "ffn");
      edge("ffn", "ffn-norm");
      edge("ffn-norm", "output");
      edge("attn-add", "output", hidden, "residual");
    } else if (a.norm === "parallel") {
      norm("input-norm", 0, 220);
      mixer("mixer", 0, 440);
      ff("ffn", 440, 440);
      add("output", 220, 660, {
        title: "Parallel Residual Add",
        category: "operation",
        inputs: [
          hidden,
          { ...hidden, name: "attention" },
          { ...hidden, name: "ffn" },
        ],
        formula: "residual + attention(norm(x)) + mlp(norm(x))",
      });
      edge(start, "input-norm");
      edge("input-norm", "mixer");
      edge("input-norm", "ffn");
      edge("mixer", "output");
      edge("ffn", "output");
      edge(start, "output", hidden, "residual");
    } else if (a.text.hybrid_override_pattern) {
      norm("input-norm", 0, 220);
      edge(start, "input-norm");
      if (p.kind === "ffn") ff("ffn", 0, 440);
      else mixer("mixer", 0, 440);
      add("output", 0, 660, { title: "Residual Add", category: "operation" });
      edge("input-norm", p.kind === "ffn" ? "ffn" : "mixer");
      edge(p.kind === "ffn" ? "ffn" : "mixer", "output");
      edge(start, "output", hidden, "residual");
    } else {
      norm("input-norm", 0, 440);
      mixer("mixer", 0, 660);
      add("attn-add", 0, 1100, {
        title:
          a.norm === "special"
            ? "Special Residual Merge"
            : "Attention Residual",
        category: "operation",
        inputs: [hidden, stream],
        outputs: [stream],
      });
      edge(start, "input-norm");
      edge("input-norm", "mixer");
      if (a.norm === "sandwich") {
        norm("post-attn", 0, 880);
        edge("mixer", "post-attn");
        edge("post-attn", "attn-add");
      } else edge("mixer", "attn-add");
      edge(start, "attn-add", stream, "residual");
      norm("ffn-norm", 440, 440);
      ff("ffn", 440, 660);
      add("output", 440, 1100, {
        title:
          a.norm === "special" ? "Special Residual Merge" : "FFN Residual",
        category: "operation",
        inputs: [hidden, stream],
        outputs: [stream],
      });
      edge("attn-add", "ffn-norm", stream);
      edge("ffn-norm", "ffn");
      if (a.norm === "sandwich") {
        norm("post-ffn", 440, 880);
        edge("ffn", "post-ffn");
        edge("post-ffn", "output");
      } else edge("ffn", "output");
      edge("attn-add", "output", stream, "residual");
      if (p.repeats > 1)
        add("repeat-note", 880, 660, {
          title: "Repeated Attention Sublayers",
          subtitle: `${p.repeats} MLA per declared block`,
          category: "operation",
          inputs: [hidden],
          outputs: [hidden],
          description:
            "LongCat 每个声明块含两次 MLA/两次 dense FFN 与快捷 MoE。当前块视图为接口级摘要，不能作为逐算子执行顺序。",
          drill: "auxiliary",
        });
    }
  } else if (view === "attention") {
    boundary("mixer-input", 0, 0, `${p.label} Input`);
    if (p.kind === "ffn") {
      ff("ffn", 0, 220);
      edge("mixer-input", "ffn");
    } else if (["attention", "mla", "dsa"].includes(p.kind)) {
      const isMLA = a.kvRank > 0,
        keys = p.window
          ? Math.min(s.past, Math.max(0, p.window - 1)) + q
          : total;
      const query = tensor(
          "query",
          [b, p.heads, q, p.headDim],
          ["B", "Nq", "S", "Dq"],
        ),
        key = tensor(
          "key",
          [b, isMLA ? p.heads : p.kvHeads, keys, p.headDim],
          ["B", "Nkv", "Klen", "Dq"],
        ),
        value = tensor(
          "value",
          [b, isMLA ? p.heads : p.kvHeads, keys, p.valueDim],
          ["B", "Nkv", "Klen", "Dv"],
        );
      add("q-proj", 0, 220, {
        title: isMLA ? "Query Low-rank / Projection" : "Query Projection",
        subtitle:
          isMLA && a.qRank
            ? `H → ${a.qRank} → Nq×Dq`
            : `${p.heads} heads × ${p.headDim}`,
        category: "attention",
        outputs: [query],
        params: ap?.query ?? 0,
        linear: { input: m.hidden, output: p.heads * p.headDim },
        description: `${a.qkNorm === "none" ? "" : `QK norm: ${a.qkNorm}. `}${p.nope ? "NoPE：该层不添加旋转位置编码。" : "RoPE 仅改变位置相位，不改变张量形状。"} QK/V 通道允许不同。`,
        details,
      });
      edge("mixer-input", "q-proj");
      add("kv-proj", 440, 220, {
        title: isMLA ? "Latent KV Projection" : "Key / Value Projections",
        subtitle: isMLA
          ? `Rkv=${a.kvRank}, Dr=${a.ropeDim}`
          : `${p.kvHeads} KV heads`,
        category: "attention",
        outputs: isMLA
          ? catalogState(m, { ...s, past: 0 }, p)
          : [key, value].map((t) => ({
              ...t,
              shape: [b, p.kvHeads, q, t.shape[3]],
              symbols: ["B", "Nkv", "S", t.symbols[3]],
            })),
        params: ap?.kv ?? 0,
        linear: {
          input: m.hidden,
          output: isMLA ? a.kvRank + a.ropeDim : p.kvHeads * p.headDim,
        },
        description: isMLA
          ? "MLA 将隐藏状态压缩为 KV latent 与独立 RoPE key；吸收投影可在低秩空间 attention。图中另列等价展开 K/V 形状。"
          : "GQA/MHA 的 K、V 投影。可见长度由窗口和当前 chunk 决定；持久缓存另见缓存页。",
        details: [
          ["expanded K", shapeText(key.shape)],
          ["expanded V", shapeText(value.shape)],
          [
            "K=V projection",
            String(a.text.attention_k_eq_v === true && !p.window),
          ],
        ],
      });
      edge("mixer-input", "kv-proj");
      const fresh = nodes.find((n) => n.id === "kv-proj")!.data.outputs;
      const previous = isMLA
        ? catalogState(m, { ...s, sequence: 0 }, p)
        : [key, value].map((t) => ({
            ...t,
            name: `past_${t.name}`,
            shape: [b, p.kvHeads, Math.max(0, keys - q), t.shape[3]],
            symbols: ["B", "Nkv", "Pcache", t.symbols[3]],
          }));
      add("past-kv", 880, 0, {
        title: "Previous KV State",
        category: "input",
        inputs: [],
        outputs: previous,
        description: "前一轮保留的状态；不由本次 token 投影生成。",
      });
      add("kv-cache", 440, 440, {
        title: "KV Cache Update",
        category: "operation",
        inputs: [...fresh, ...previous],
        outputs: isMLA ? catalogState(m, s, p) : [key, value],
        params: 0,
        parameterStatus: "known",
        description:
          "合并历史状态与当前投影；本次可见键长度与持久保留长度分别展示。",
        formula: "visible_KV = cache_update(previous_KV, current_KV)",
        details: [
          ["visible keys", String(keys)],
          [
            "retained state",
            catalogState(m, s, p)
              .map((t) => shapeText(t.shape))
              .join(" / "),
          ],
        ],
      });
      edge("kv-proj", "kv-cache", fresh[0]);
      edge("past-kv", "kv-cache", previous[0], "shared");
      if (isMLA) {
        add("kv-expand", 440, 600, {
          title: "Equivalent KV Expansion",
          category: "attention",
          params: ap?.expansion ?? 0,
          inputs: catalogState(m, s, p),
          outputs: [key, value],
          description:
            "逻辑展开 latent 为各 head 的 K/V；吸收投影后端可保持低秩计算。",
        });
        edge("kv-cache", "kv-expand", catalogState(m, s, p)[0]);
      }
      if (p.kind === "dsa") {
        const sc = recLocal(a.text.sparse_attention_config)
          ? a.text.sparse_attention_config
          : recLocal(a.text.sparse_config)
            ? a.text.sparse_config
            : a.text;
        const budget =
          Number(
            sc.index_topk ??
              sc.indexer_budget ??
              sc.sparse_topk_blocks ??
              sc.sparse_topk,
          ) || null;
        add("sparse-indexer", 880, 440, {
          title: "Sparse Index Selection",
          subtitle: budget
            ? `Configured budget ${budget}`
            : "Model-specific indexer",
          category: "operation",
          inputs: [hidden],
          outputs: budget
            ? [
                tensor(
                  "selected_indices",
                  [
                    b,
                    q,
                    Math.min(
                      budget,
                      sc.sparse_topk_blocks
                        ? Math.ceil(
                            total / Number(sc.sparse_block_size ?? 128),
                          )
                        : total,
                    ),
                  ],
                  ["B", "S", "K"],
                  "逻辑候选上限；块粒度/额外窗口依具体模型，最终索引由运行时数据决定。",
                  "int32",
                ),
              ]
            : [],
          description:
            "DSA / QSA / MiniMax sparse attention 的索引维度见配置。不同索引器不假定同一个打分公式或缓存布局。",
          details: Object.entries(sc)
            .filter(
              ([, v]) => typeof v === "number" || typeof v === "boolean",
            )
            .slice(0, 15)
            .map(([k, v]) => [k, String(v)]),
        });
        edge("mixer-input", "sparse-indexer");
      }
      add("attention-core", 220, 840, {
        title: p.kind === "dsa" ? "Sparse Attention Core" : "Attention Core",
        subtitle: `${p.window ? "Local" : "Causal"} · ${p.gated ? "output gate" : "standard output"}`,
        category: "attention",
        inputs: [query, key, value],
        params: 0,
        parameterStatus: "known",
        outputs: [
          tensor(
            "context",
            [b, q, p.heads * p.valueDim],
            ["B", "S", "Nq×Dv"],
          ),
        ],
        description:
          p.kind === "dsa"
            ? "只聚合运行时选中的位置；不伪造完整稠密 score 作为实际分配。"
            : "矩阵乘、mask 与 softmax 的逻辑路径；融合内核可能不分配完整分数。",
        formula:
          p.kind === "dsa"
            ? "attend(Q, selected_K, selected_V)"
            : "softmax(QKᵀ × scale + mask) V",
        details: [
          ["dense-equivalent scores", `[${b}, ${p.heads}, ${q}, ${keys}]`],
          [
            "scale",
            String(
              a.type.startsWith("gemma4")
                ? 1
                : a.text.query_pre_attn_scalar
                  ? Number(a.text.query_pre_attn_scalar) ** -0.5
                  : p.headDim ** -0.5,
            ),
          ],
          ["sink", String(a.sink)],
          ["activation gate", String(p.gated)],
        ],
      });
      edge("q-proj", "attention-core", query);
      edge(isMLA ? "kv-expand" : "kv-cache", "attention-core", key);
      edge(isMLA ? "kv-expand" : "kv-cache", "attention-core", value);
      if (p.kind === "dsa")
        edge(
          "sparse-indexer",
          "attention-core",
          nodes.find((n) => n.id === "sparse-indexer")?.data.outputs[0] ??
            hidden,
          "shared",
          "selected positions",
        );
      add("out-proj", 220, 1060, {
        params: ap?.output ?? 0,
        linear: { input: p.heads * p.valueDim, output: m.hidden },
        title: p.gated ? "Output Gate + Projection" : "Output Projection",
        subtitle: `${p.heads * p.valueDim} → ${m.hidden}`,
        category: "attention",
        inputs: [
          tensor(
            "context",
            [b, q, p.heads * p.valueDim],
            ["B", "S", "Nq×Dv"],
          ),
        ],
        description:
          "门控 granularity / activation 和温度由架构定义；配置保留原始字段。",
      });
      edge(
        "attention-core",
        "out-proj",
        tensor("context", [b, q, p.heads * p.valueDim], ["B", "S", "Nq×Dv"]),
      );
    } else if (
      ["delta", "kda", "lightning", "mamba", "conv", "mlstm"].includes(p.kind)
    ) {
      const states = catalogState(m, s, p),
        r = a.recurrent;
      add("state-in", 440, 0, {
        title: "Previous Recurrent State",
        subtitle: "Fixed-size state, not token KV",
        category: "input",
        inputs: [],
        outputs: states,
        description:
          "这些状态随 token 更新，主体容量不随历史长度线性增加。不是普通 Attention 的 KV cache。",
      });
      add("input-proj", 0, 220, {
        title:
          p.kind === "mamba"
            ? "Input-dependent SSM Parameters"
            : "Input Projections & Gates",
        subtitle: p.label,
        category: "operation",
        outputs: [hidden],
        description:
          "生成 Q/K/V 或 SSM 的 x/B/C/dt 与门控。具体打包布局因模型而异。",
        details: r
          ? [
              ["state heads", String(r.heads)],
              ["key dim", String(r.keyDim)],
              ["value dim", String(r.valueDim)],
              ["conv kernel", String(r.kernel)],
            ]
          : [],
      });
      if (r) {
        const channels =
          p.kind === "mamba"
            ? 2 * r.heads * r.valueDim + 2 * r.groups * r.stateSize + r.heads
            : 2 * r.groups * r.keyDim +
              2 * r.heads * r.valueDim +
              2 * r.heads;
        const projected = tensor(
          "packed_mixer_projection",
          [b, q, channels],
          ["B", "S", "Cmixer"],
          "逻辑拼接接口；不同实现可拆分多个 Linear。",
        );
        nodes.find((n) => n.id === "input-proj")!.data.outputs = [projected];
      }
      edge("mixer-input", "input-proj");
      add("short-conv", 0, 440, {
        title:
          p.kind === "mlstm"
            ? "Input/Forget Gates"
            : "Short Convolution / Gate",
        subtitle: p.kind === "conv" ? "LIV: C × Conv(B × x)" : p.label,
        category: "operation",
        description:
          p.kind === "conv"
            ? "输入分为 B、C、x，先对 B×x 作短卷积，再乘 C 和输出投影。"
            : "局部卷积补充近期信息；衰减/更新门由输入决定。",
      });
      const projected = nodes.find((n) => n.id === "input-proj")!.data
        .outputs[0];
      nodes.find((n) => n.id === "short-conv")!.data.inputs = [projected];
      nodes.find((n) => n.id === "short-conv")!.data.outputs = [projected];
      edge("input-proj", "short-conv", projected);
      add("state-update", 0, 660, {
        title: `${p.label} State Update`,
        subtitle: "Recurrent / chunkwise equivalent",
        category: "attention",
        inputs: [projected, ...states],
        outputs: [
          hidden,
          ...states.map((t) => ({ ...t, name: `next_${t.name}` })),
        ],
        description:
          p.kind === "mamba"
            ? "选择性 SSM 更新状态，再通过 C 读出；Mamba-2 的 state 维度由 heads/head_dim/state_size 确定。"
            : p.kind === "kda" || p.kind === "delta"
              ? "按 Delta rule 更新记忆矩阵；KDA 使用细粒度遗忘门。块扫描与递归解码是同一逻辑序列的不同执行方式。"
              : "门控递归更新；保留模型专用归一化、稳定化状态的说明。",
        details: [
          ["history length", String(total)],
          ["cache growth", "主体状态固定；卷积缓冲固定"],
          ["scope", "标出的状态为可推导部分，不包含全部内核工作区"],
        ],
      });
      edge("short-conv", "state-update", projected);
      edge("state-in", "state-update", states[0] ?? hidden, "shared");
      add("out-proj", 0, 880, {
        title: "Readout / Output Projection",
        subtitle: "Return to hidden stream",
        category: "operation",
      });
      edge("state-update", "out-proj");
    } else if (p.kind === "csa") {
      const local = tensor(
        "window_kv",
        [b, Math.min(total, m.slidingWindow ?? 128), m.headDim],
        ["B", "W", "D"],
      );
      const comp = tensor(
        "compressed_kv",
        [b, p.ratio ? Math.floor(total / p.ratio) : 0, m.headDim],
        ["B", "C", "D"],
        "DeepSeek V4 按层独立压缩；不是 V4.1 的跨层共享缓存。",
      );
      add("q-proj", 0, 220, {
        title: "Query Low-rank Projection",
        subtitle: `Rq=${String(a.text.q_lora_rank)}`,
        category: "attention",
        outputs: [
          tensor("query", [b, q, p.heads, m.headDim], ["B", "S", "Nq", "D"]),
        ],
      });
      edge("mixer-input", "q-proj");
      add("kv-proj", 440, 220, {
        title: p.ratio ? "SWA + Compressed KV" : "SWA KV",
        subtitle: p.ratio
          ? `r=${p.ratio} · ${p.ratio >= 128 ? "HCA" : "CSA"}`
          : "No global compression",
        category: "attention",
        outputs: [local, ...(p.ratio ? [comp] : [])],
        description:
          "CSA 与 HCA 具有不同压缩窗口、索引与缓冲状态。这里给出已完成压缩条目的逻辑宽度；重叠窗口/临时缓存未纳入。",
      });
      edge("mixer-input", "kv-proj");
      add("attention-core", 220, 440, {
        title:
          p.ratio >= 128
            ? "HCA Compressed Attention"
            : "CSA Sparse Attention",
        subtitle: "Local + compressed positions",
        category: "attention",
        inputs: [local, ...(p.ratio ? [comp] : [])],
        description:
          "HCA 对高压缩条目计算注意力，CSA 使用索引选择；均结合局部窗口。",
      });
      edge("q-proj", "attention-core");
      edge("kv-proj", "attention-core", local);
      add("out-proj", 220, 660, {
        title: "Grouped Output Projection",
        subtitle: `groups=${a.text.o_groups}, rank=${a.text.o_lora_rank}`,
        category: "attention",
      });
      edge("attention-core", "out-proj");
    } else if (p.kind === "cca") {
      const query = tensor(
          "q",
          [b, p.heads, q, p.headDim],
          ["B", "Nq", "S", "D"],
        ),
        key = tensor(
          "k",
          [b, p.kvHeads, q, p.headDim],
          ["B", "Nkv", "S", "D"],
        ),
        value = tensor(
          "v",
          [b, p.kvHeads, q, p.valueDim],
          ["B", "Nkv", "S", "Dv"],
        );
      add("q-proj", 0, 220, {
        title: "CCA Q/K Projection",
        subtitle: "Residual Q/K + causal conv",
        category: "attention",
        outputs: [query, key],
        description:
          "Q 与重复后的 K 构造残差，再经过 depthwise 与 grouped causal conv；两者保留各自的投影残差。",
        details: [
          ["depthwise kernel", String(a.text.cca_time0)],
          ["grouped kernel", String(a.text.cca_time1)],
        ],
      });
      edge("mixer-input", "q-proj");
      add("kv-proj", 440, 220, {
        title: "Current + Delayed Value",
        subtitle: "Half current token, half previous token",
        category: "attention",
        outputs: [value],
        description:
          "当前 value 投影与上一 token 的 delayed value 各占一半；Decode 从固定递归缓冲读取上一 token。",
      });
      edge("mixer-input", "kv-proj");
      add("attention-core", 220, 440, {
        title: "QK Normalize + GQA",
        subtitle: "CCA projection → partial RoPE → attention",
        category: "attention",
        inputs: [query, key, value],
        outputs: [
          tensor(
            "context",
            [b, q, p.heads * p.valueDim],
            ["B", "S", "Nq×Dv"],
          ),
        ],
        description:
          "特殊 Q/K L2 归一化与每 KV head 的可学习 scale，再应用部分 RoPE 与 GQA。",
      });
      edge("q-proj", "attention-core", query);
      edge("kv-proj", "attention-core", value);
      add("out-proj", 220, 660, {
        title: "CCA Output Projection",
        category: "attention",
        inputs: [
          tensor(
            "context",
            [b, q, p.heads * p.valueDim],
            ["B", "S", "Nq×Dv"],
          ),
        ],
      });
      edge("attention-core", "out-proj");
    } else if (p.kind === "gdla") {
      const noise = Number(a.text.num_noise_heads),
        signal = p.heads - noise;
      add("q-proj", 0, 220, {
        title: "Low-rank Signal / Noise Queries",
        subtitle: `${signal} signal + ${noise} noise heads`,
        category: "attention",
        outputs: [
          tensor("query", [b, q, p.heads, p.headDim], ["B", "S", "Nq", "D"]),
        ],
        description:
          "GDLA V2 将 query heads 分成信号组与噪声组；query latent 同时产生输出 gate 与差分权重。",
      });
      edge("mixer-input", "q-proj");
      add("kv-proj", 440, 220, {
        title: "Grouped Latent KV",
        subtitle: `Rkv=${a.kvRank} · ${p.kvHeads} KV heads`,
        category: "attention",
        outputs: [
          tensor(
            "key",
            [b, p.kvHeads, total, p.headDim],
            ["B", "Nkv", "T", "D"],
          ),
          tensor(
            "value",
            [b, p.kvHeads, total, p.valueDim],
            ["B", "Nkv", "T", "Dv"],
          ),
        ],
        description:
          "压缩 KV 再展开为 grouped key/value，RoPE 只应用于配置的旋转维度。",
      });
      edge("mixer-input", "kv-proj");
      add("attention-core", 220, 440, {
        title: "Signal − Gated Noise Attention",
        subtitle: "GDLA V2 grouped differential attention",
        category: "attention",
        outputs: [
          tensor(
            "signal_context",
            [b, q, signal, p.valueDim],
            ["B", "S", "Nsignal", "Dv"],
          ),
        ],
        formula:
          "signal_attention − sigmoid(lambda) × repeated_noise_attention",
        description:
          "每个噪声 head 广播到其信号组，差分结果再乘输出 gate；不是将所有 heads 直接拼接。",
      });
      edge("q-proj", "attention-core");
      edge("kv-proj", "attention-core");
      add("out-proj", 220, 660, {
        title: "Signal Head Output Projection",
        subtitle: `${signal} × ${p.valueDim} → ${m.hidden}`,
        category: "attention",
        inputs: [
          tensor(
            "signal_context",
            [b, q, signal, p.valueDim],
            ["B", "S", "Nsignal", "Dv"],
          ),
        ],
      });
      edge("attention-core", "out-proj");
    } else {
      add("special-mixer", 0, 220, {
        title: `${a.type} Custom Mixer`,
        subtitle: p.label,
        category: "attention",
        description:
          "该模型包含 CCA/GDLA 等专用混合运算。支持其层级、已知输入输出接口与原始配置检查；内部动态路由/噪声头/差分打分未伪装成普通 MHA。",
        details: Object.entries(a.text)
          .filter(
            ([k, v]) =>
              /cca|diff|noise|ratio|head|attn/.test(k) &&
              (typeof v === "number" ||
                typeof v === "boolean" ||
                typeof v === "string"),
          )
          .map(([k, v]) => [k, String(v)]),
      });
      edge("mixer-input", "special-mixer");
    }
  } else if (view === "mlp") {
    boundary("ffn-input", 220, 0, "FFN Input");
    if (a.text.hybrid_override_pattern && p.kind !== "ffn") {
      add("no-ffn", 0, 220, {
        title: "No FFN in This Entry",
        subtitle: p.label,
        category: "operation",
        description:
          "此混合模型将 Mamba/Attention 和 FFN 分成独立 entries；选择相邻的 MoE/FFN entry 查看专家网络。",
      });
      edge("ffn-input", "no-ffn");
    } else if (!p.moe) {
      const inner = tensor(
        "intermediate",
        [b, q, p.intermediate],
        ["B", "S", "I"],
      );
      add("up-proj", 0, 220, {
        title: a.gatedMlp ? "Gate & Up Projections" : "Up Projection",
        subtitle: `${m.hidden} → ${p.intermediate}`,
        category: "mlp",
        outputs: [inner],
        params: mp.up,
        linear: { input: m.hidden, output: p.intermediate },
        description: `${a.activation}; ${note}`,
      });
      edge("ffn-input", "up-proj");
      add("activation", 0, 440, {
        title: a.gatedMlp ? "Activation × Up" : "Activation",
        subtitle: a.activation,
        category: "operation",
        inputs: [inner],
        outputs: [inner],
        description:
          a.type === "gpt_oss"
            ? "GPT-OSS uses clamped gate×sigmoid(1.702 gate)×(up+1)."
            : a.activation === "relu2"
              ? "Squared ReLU"
              : a.activation,
      });
      edge("up-proj", "activation", inner);
      add("down-proj", 0, 660, {
        title: "Down Projection",
        category: "mlp",
        params: mp.down,
        inputs: [inner],
      });
      edge("activation", "down-proj", inner);
    } else {
      const slots = tensor(
          "expert_slots",
          [b, q, p.topK, p.expertHidden],
          ["B", "S", "Kmoe", "He"],
          "逻辑选中槽位，真实每专家 token 分布取决于输入与权重。",
        ),
        inner = tensor(
          "expert_intermediate",
          [b, q, p.topK, p.expertWidth],
          ["B", "S", "Kmoe", "Ie"],
        );
      add("router", 0, 220, {
        title: "Router / Group Top-K",
        params: mp.router,
        subtitle: `${p.experts} experts · Top-${p.topK}`,
        category: "mlp",
        outputs: [
          tensor(
            "router_scores",
            [b, q, p.experts],
            ["B", "S", "E"],
            undefined,
            "float32",
          ),
        ],
        description:
          "sigmoid/softmax/修正 bias、分组路由按原始配置列出；不生成假专家 ID。",
        details: [
          [
            "scoring",
            String(
              a.text.scoring_func ??
                a.text.score_func ??
                a.text.moe_router_activation_func ??
                "see config",
            ),
          ],
          [
            "group count",
            String(a.text.n_group ?? a.text.num_expert_group ?? 1),
          ],
          ["group top-k", String(a.text.topk_group ?? "—")],
        ],
      });
      edge("ffn-input", "router");
      add("dispatch", 0, 440, {
        params: mp.latent / 2,
        title:
          p.expertHidden !== m.hidden
            ? "Latent Projection + Dispatch"
            : "Dispatch Selected Tokens",
        subtitle:
          p.expertHidden !== m.hidden
            ? `H → He=${p.expertHidden}`
            : "Dynamic token packing",
        category: "operation",
        outputs: [slots],
        description:
          "LatentMoE 在专家计算前降维，完成后再投影回 H；shared expert 通常直接读取原始隐藏状态。",
      });
      edge("router", "dispatch");
      add("experts", 0, 660, {
        title: "Routed Expert Networks",
        subtitle: `${p.experts} × ${a.activation}`,
        category: "mlp",
        inputs: [slots],
        outputs: [slots],
        params: mp.experts,
        description: `每个专家的中间张量 ${shapeText(inner.shape)}。所示参数仅计 routed experts，不能把 top-K 激活规模当作总参数。`,
        details: [
          ["expert input", String(p.expertHidden)],
          ["expert intermediate", String(p.expertWidth)],
          ["zero / identity experts", String(a.text.zero_expert_num ?? 0)],
          ["clamp", String(a.text.swiglu_limit ?? "see config")],
        ],
      });
      edge("dispatch", "experts", slots);
      if (mp.dense) {
        add("parallel-dense", 440, 220, {
          title: "Parallel Dense MLP",
          params: mp.dense,
          subtitle: `Dense I=${p.intermediate} + routed MoE`,
          category: "mlp",
          description:
            "Gemma4 MoE 同时保留 dense MLP 分支。专家分支与 dense 分支分别归一化后相加；不是仅有 routed experts。",
        });
        edge("ffn-input", "parallel-dense");
      }
      if (p.sharedWidth) {
        add("shared-expert", 440, 440, {
          title: "Shared Expert Branch",
          params: mp.shared,
          subtitle: `I_shared=${p.sharedWidth}`,
          category: "mlp",
          description:
            "共享专家独立于路由选择。不同架构的加权、平均或门控组合在配置中保留。",
        });
        edge("ffn-input", "shared-expert");
      }
      add("combine", 220, 880, {
        title: "Weighted Combine",
        params: mp.combine + mp.latent / 2,
        subtitle:
          p.expertHidden !== m.hidden
            ? "Latent output → H"
            : "Sum selected expert outputs",
        category: "operation",
        inputs: [slots, ...(p.sharedWidth ? [hidden] : [])],
        formula: a.type.startsWith("gemma4")
          ? "norm(dense_output) + norm(routed_output)"
          : "weighted_sum(expert_outputs) + shared_output",
      });
      edge("experts", "combine", slots);
      if (p.sharedWidth) edge("shared-expert", "combine");
      if (mp.dense) edge("parallel-dense", "combine");
    }
  } else if (view === "cache") {
    const owner = Array.from(
      new Set(
        a.profiles.map(
          (p) => `${p.kind}-${p.window}-${p.heads}-${p.headDim}`,
        ),
      ),
    ).map(
      (key) =>
        a.profiles.find(
          (p) => `${p.kind}-${p.window}-${p.heads}-${p.headDim}` === key,
        )!,
    );
    owner.forEach((prof, i) => {
      const states = catalogState(m, s, prof),
        same = a.profiles.filter(
          (p) =>
            p.kind === prof.kind &&
            p.window === prof.window &&
            p.heads === prof.heads &&
            p.headDim === prof.headDim,
        );
      add(`cache-${i}`, (i % 3) * 410, Math.floor(i / 3) * 250, {
        title: `${prof.label} State`,
        subtitle: `${same.length} entries · ${a.loops} passes`,
        category: "attention",
        inputs: [],
        outputs: states,
        drill: "attention",
        layerIndex: prof.index,
        description: states.length
          ? "每张卡片给出一个该类层的状态形状；累计时乘实际层数，循环模型还需按轮区分运行时状态。"
          : "该专用算子的缓存尚无足够规则计算；未知不等于零。",
        details: [
          [
            "examples",
            same
              .slice(0, 12)
              .map((p) => p.index)
              .join(", "),
          ],
          [
            "one layer bytes",
            states.length
              ? formatBytes(
                  states.reduce(
                    (sum, t) =>
                      sum +
                      t.shape.reduce((a, b) => a * b, 1) *
                        (t.dtype === "float32"
                          ? 4
                          : s.dtype === "float32"
                            ? 4
                            : 2),
                    0,
                  ),
                )
              : "未推导",
          ],
        ],
      });
    });
  } else {
    const groups: [string, unknown][] = [
      ["text config", a.text],
      ...a.mediaKeys.map((k) => [k, m.raw[k]] as [string, unknown]),
      ["source notes", a.extras],
    ];
    groups.forEach(([name, value], i) =>
      add(`aux-${i}`, 0, i * 240, {
        title: name,
        subtitle: "Architecture metadata",
        category: "operation",
        inputs: [],
        outputs: [],
        description:
          typeof value === "object"
            ? JSON.stringify(value, null, 2).slice(0, 6000)
            : String(value),
        details: [
          ["scope", "配置检查，不执行远程代码；完整内容见 config.json"],
        ],
      }),
    );
  }
  for (const e of edges) {
    if (e.data?.shared) {
      const f = nodes.find((n) => n.id === e.source)!,
        t = nodes.find((n) => n.id === e.target)!;
      if (t.position.x > f.position.x) {
        e.sourceHandle = "right-out";
        e.targetHandle = "side-in";
      } else if (t.position.x < f.position.x) {
        e.sourceHandle = "side-out";
        e.targetHandle = "right-in";
      }
    }
  }
  const ordered: ModuleNode[] = [],
    pending = new Map(nodes.map((n) => [n.id, n]));
  while (pending.size) {
    const ready = [...pending.values()].filter(
      (n) => !edges.some((e) => e.target === n.id && pending.has(e.source)),
    );
    if (!ready.length) throw new Error("架构图出现循环。");
    for (const n of ready) {
      ordered.push(n);
      pending.delete(n.id);
    }
  }
  ordered.forEach((n, i) => {
    n.data.order = i + 1;
    if (n.data.params > 0)
      n.data.description += " 参数为结构估算；来源/计数范围见配置说明。";
  });
  return { nodes: ordered, edges };
}
function recLocal(v: unknown): v is Record<string, unknown> {
  return v !== null && typeof v === "object" && !Array.isArray(v);
}
