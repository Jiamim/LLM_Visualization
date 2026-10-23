import type { Category, View } from "./graph";

export const NODE_WIDTH = 304;
export const NODE_HEIGHT = 132;
export const categories: Record<Category, { label: string; color: string }> =
  {
    input: { label: "INPUT", color: "#60748b" },
    embedding: { label: "EMBEDDING", color: "#3575c7" },
    block: { label: "TRANSFORMER", color: "#7555c7" },
    attention: { label: "ATTENTION", color: "#3779bc" },
    norm: { label: "NORMALIZATION", color: "#a77525" },
    mlp: { label: "FEED FORWARD", color: "#258473" },
    output: { label: "OUTPUT", color: "#965bb0" },
    operation: { label: "OPERATION", color: "#64748b" },
  };

export interface Stage {
  title: string;
  description: string;
  anchor: string;
  nodes: string[];
}
export const stages: Record<View, Stage[]> = {
  architecture: [],
  overview: [],
  auxiliary: [],
  ple: [],
  audio: [],
  encoder: [],
  decoder: [],
  cache: [],
  vision: [],
  engram: [],
  dspark: [],
  layer: [
    {
      title: "注意力与残差",
      description: "归一化 → Self Attention → 残差连接",
      anchor: "attention",
      nodes: ["layer-input", "input-norm", "attention", "attention-residual"],
    },
    {
      title: "前馈与残差",
      description: "归一化 → Feed Forward → 残差连接",
      anchor: "mlp",
      nodes: ["post-norm", "mlp", "mlp-residual"],
    },
  ],
  attention: [
    {
      title: "Q / K / V 投影",
      description: "从隐藏向量生成 Query、Key 和 Value",
      anchor: "q-proj",
      nodes: [
        "q-proj",
        "k-proj",
        "v-proj",
        "q-reshape",
        "k-reshape",
        "v-reshape",
      ],
    },
    {
      title: "位置与缓存",
      description: "RoPE → 更新历史 KV → 扩展 KV heads",
      anchor: "k-cache",
      nodes: [
        "q-rope",
        "k-rope",
        "k-cache",
        "v-cache",
        "k-repeat",
        "v-repeat",
      ],
    },
    {
      title: "注意力计算",
      description: "QKᵀ → Mask + Softmax → 与 V 加权",
      anchor: "scores",
      nodes: ["scores", "softmax", "weighted-values"],
    },
    {
      title: "输出投影",
      description: "合并所有 heads，投影回隐藏维度",
      anchor: "o-proj",
      nodes: ["merge-heads", "o-proj"],
    },
  ],
  mlp: [
    {
      title: "门控与升维",
      description: "并行投影到 intermediate_size，再应用激活函数",
      anchor: "gate-proj",
      nodes: ["mlp-input", "gate-proj", "up-proj", "activation"],
    },
    {
      title: "融合与降维",
      description: "Gate × Up → Down Projection",
      anchor: "down-proj",
      nodes: ["multiply", "down-proj"],
    },
  ],
};

export const dimensionNames: Record<string, string> = {
  "Rkv+Dr": "压缩 KV 与旋转 key 总宽度",
  "Nkv×Dv/2": "半宽 delayed value",
  Nsignal: "差分注意力信号头数",
  Cmixer: "混合算子投影接口宽度",
  Tcache: "逻辑缓存条目数",
  Rkv: "KV 潜在压缩宽度",
  Dr: "旋转 key 宽度",
  Dq: "Query/Key 单头维度",
  Dv: "Value 单头维度",
  He: "专家输入宽度",
  Ie: "专家中间维度",
  Nstate: "递归状态头数",
  Dk: "递归 Key 维度",
  Dstate: "SSM 状态维度",
  Cconv: "卷积状态通道",
  Kconv: "短卷积宽度",
  "Nq×Dv": "合并 Value heads 宽度",
  Nvalid: "跨样本合并后的有效视觉 token 数",
  L: "主干层数",
  Dple: "每层 PLE 宽度",
  Klen: "本次 attention 可见 Key 长度",
  Pcache: "本次调用前实际保留的历史长度",
  R: "调用后持久缓存长度",
  "Nq×D": "Query 投影宽度",
  XY: "patch 二维位置",
  F: "预处理音频特征帧数",
  Mel: "音频频率特征数",
  Ta: "下采样后音频帧数",
  Ha: "音频隐藏维度",
  Hao: "音频输出投影宽度",
  Q: "当前层查询长度（有界重放可能小于 S）",
  HC: "mHC 并行残差流数",
  C: "已完成的全局压缩条目",
  K: "选中的全局索引数",
  W: "有效滑动窗口长度",
  Rq: "Query 低秩维度",
  Ro: "Output 低秩维度",
  G: "输出投影分组数",
  Ni: "索引器头数",
  Di: "索引器单头维度",
  E: "路由专家总数",
  Kmoe: "每 token 激活的路由专家数",
  Npatch: "单图 patch 数",
  Nimg: "单图下采样后视觉 token 数",
  Hv: "视觉隐藏维度",
  "3p²": "RGB patch 展平宽度",
  "r²Hv": "pixel-unshuffle 通道宽度",
  Nhash: "每 token 的 n-gram 哈希列数",
  "Nhash×De": "拼接记忆 embedding 宽度",
  Ldraft: "草稿 block 长度",
  "Nt×H": "主干目标层拼接宽度",
  "H+Rm": "隐藏与 Markov embedding 拼接",
  B: "批大小",
  S: "当前序列长度",
  P: "历史缓存长度",
  T: "总长度 P + S",
  H: "隐藏维度",
  I: "中间维度",
  V: "词表大小",
  Nq: "Query 头数",
  Nkv: "KV 头数",
  D: "单头维度",
  "1": "广播维度",
  "Nq × D": "Query 投影宽度",
  "Nkv × D": "KV 投影宽度",
};

export function modelStages(
  model: import("./model").ModelConfig,
  view: View,
): Stage[] {
  if (view === "architecture") return [];
  if (model.catalog) return catalogStages(model, view);
  if (model.gemma) return gemmaStages(model, view);
  if (!model.deepseek) return stages[view];
  const d = model.deepseek;
  const stage = (title: string, anchor: string, nodes: string[]): Stage => ({
    title,
    description: title,
    anchor,
    nodes,
  });
  const all: Record<View, Stage[]> = {
    architecture: [],
    auxiliary: [],
    ple: [],
    audio: [],
    overview: [
      stage("编码与缓存", "encoder", ["encoder", "ced-kv", "decoder"]),
      stage("输出与草稿", "lm-head", [
        "collapse",
        "lm-head",
        ...(d.draftLayers ? ["dspark"] : []),
      ]),
    ],
    encoder: [],
    decoder: [],
    layer: [
      stage("mHC 与注意力", "attention", [
        "hc-pre",
        "attn-norm",
        "attention",
      ]),
      stage("MoE 与混合", "mlp", ["ffn-pre", "ffn-norm", "mlp", "ffn-post"]),
    ],
    attention: [
      stage("Query 与本地 KV", "q-lora", [
        "attn-input",
        "q-lora",
        "local-kv",
      ]),
      stage("全局 KV 与索引", "global-kv", ["global-kv", "indexer"]),
      stage("稀疏注意力", "sparse-attn", ["sparse-attn"]),
      stage("分组输出", "o-proj", ["o-group", "o-proj"]),
    ],
    mlp: [
      stage("路由与分发", "router", ["router", "route", "dispatch"]),
      stage("专家计算", "expert-up", [
        "expert-up",
        "expert-down",
        "shared-expert",
      ]),
      stage("合并输出", "combine", ["combine"]),
    ],
    cache: [
      ...d.kvSources.map((owner) =>
        stage(`KV 来源 L${owner}`, `source-${owner}`, [
          `source-${owner}`,
          `consumer-${owner}`,
        ]),
      ),
      stage("缓存统计", "cache-summary", ["cache-summary"]),
    ],
    vision: [
      stage("Patch 与 ViT", "patch-embed", [
        "vision-input",
        "patch-embed",
        "vision-blocks",
      ]),
      stage("下采样与融合", "projector", [
        "pixel-unshuffle",
        "projector",
        "image-merge",
      ]),
    ],
    engram: [
      stage("N-gram 查询", "engram-table", [
        "engram-input",
        "ngram-hash",
        "engram-table",
      ]),
      stage("门控注入", "engram-gate", ["engram-project", "engram-gate"]),
    ],
    dspark: [
      stage("主干到草稿", "draft-project", [
        "draft-input",
        "draft-project",
        "draft-embed",
      ]),
      stage("草稿输出", "draft-head", [
        "draft-blocks",
        "draft-head",
        "draft-confidence",
      ]),
    ],
  };
  return all[view];
}

function gemmaStages(
  model: import("./model").ModelConfig,
  view: View,
): Stage[] {
  const g = model.gemma!;
  const stage = (title: string, anchor: string, nodes: string[]): Stage => ({
    title,
    description: title,
    anchor,
    nodes,
  });
  const mapping: Record<View, Stage[]> = {
    architecture: [],
    auxiliary: [],
    overview: [
      stage("输入与逐层嵌入", "ple", ["embedding", "merge", "ple"]),
      stage("主干与输出", "decoder", ["decoder", "final-norm", "lm-head"]),
    ],
    layer: [
      stage("注意力与归一化", "attention", [
        "input-norm",
        "attention",
        "post-attn-norm",
      ]),
      stage("MLP 与残差", "mlp", [
        "pre-ffn-norm",
        "mlp",
        "post-ffn-norm",
        "ffn-add",
      ]),
      stage("PLE 注入", "ple-inject", [
        "ple-source",
        "ple-inject",
        "ple-add",
        "layer-scale",
      ]),
    ],
    attention: [
      stage("Query 与 Key/Value", "q-proj", [
        "q-proj",
        "kv-project",
        "shared-kv",
      ]),
      stage("缓存状态", "kv-cache", ["kv-cache", "kv-source", "shared-kv"]),
      stage("注意力计算", "scores", ["scores", "softmax-values"]),
      stage("输出投影", "o-proj", ["o-proj"]),
    ],
    mlp: stages.mlp,
    ple: [
      stage("生成逐层输入", "ple-table", [
        "ple-table",
        "ple-project",
        "ple-combine",
      ]),
      stage("切片与门控", "ple-slice", [
        "ple-slice",
        "ple-current",
        "ple-gate",
      ]),
      stage("注入残差", "ple-output", ["ple-multiply", "ple-output"]),
    ],
    cache: [
      ...[g.localSource, g.globalSource]
        .filter((n): n is number => n !== null)
        .map((owner) =>
          stage(`KV 来源 L${owner}`, `owner-${owner}`, [
            `owner-${owner}`,
            `reuse-${owner}`,
          ]),
        ),
      stage("缓存统计", "cache-summary", ["cache-summary"]),
    ],
    decoder: Array.from({ length: Math.ceil(model.layers / 5) }, (_, i) =>
      stage(
        `L${i * 5}–${Math.min(i * 5 + 4, model.layers - 1)}`,
        `layer-${i * 5}`,
        Array.from(
          { length: Math.min(5, model.layers - i * 5) },
          (_, j) => `layer-${i * 5 + j}`,
        ),
      ),
    ),
    vision: [
      stage("视觉编码", "vision-embed", [
        "vision-input",
        "vision-embed",
        "vision-layers",
      ]),
      stage("池化与投影", "vision-project", [
        "vision-pool",
        "vision-valid",
        "vision-project",
        "vision-merge",
      ]),
    ],
    audio: [
      stage("特征下采样", "audio-subsample", [
        "audio-input",
        "audio-subsample",
      ]),
      stage("分块注意力", "audio-attention", [
        "audio-ffn1",
        "audio-attention",
      ]),
      stage("卷积与前馈", "audio-conv", ["audio-conv", "audio-ffn2"]),
      stage("映射软 token", "audio-language", [
        "audio-project",
        "audio-language",
      ]),
    ],
    encoder: [],
    engram: [],
    dspark: [],
  };
  return mapping[view];
}

function catalogStages(
  model: import("./model").ModelConfig,
  view: View,
): Stage[] {
  const stage = (title: string, anchor: string, nodes: string[]): Stage => ({
    title,
    description: title,
    anchor,
    nodes,
  });
  if (view === "overview")
    return [
      stage("输入与主干", "decoder", ["embedding", "decoder", "auxiliary"]),
      stage("输出", "lm-head", ["final-norm", "lm-head"]),
    ];
  if (view === "decoder")
    return Array.from({ length: Math.ceil(model.layers / 4) }, (_, i) =>
      stage(
        `L${i * 4}–${Math.min(i * 4 + 3, model.layers - 1)}`,
        `block-${i * 4}`,
        Array.from(
          { length: Math.min(4, model.layers - i * 4) },
          (_, j) => `block-${i * 4 + j}`,
        ),
      ),
    );
  if (view === "layer")
    return [
      stage("混合算子", "mixer", [
        "input-norm",
        "mixer",
        "mixer-norm",
        "attn-add",
        "special-pre",
      ]),
      stage("前馈与输出", "ffn", ["attn-add", "ffn-norm", "ffn", "output"]),
    ];
  if (view === "attention")
    return [
      stage("输入投影", "q-proj", [
        "q-proj",
        "kv-proj",
        "input-proj",
        "state-in",
        "past-kv",
        "special-mixer",
      ]),
      stage("计算与状态", "attention-core", [
        "kv-cache",
        "kv-expand",
        "attention-core",
        "state-update",
        "short-conv",
      ]),
      stage("输出投影", "out-proj", ["out-proj"]),
    ];
  if (view === "mlp")
    return [
      stage("路由 / 升维", "router", [
        "router",
        "dispatch",
        "up-proj",
        "no-ffn",
      ]),
      stage("专家 / 激活", "experts", [
        "experts",
        "shared-expert",
        "activation",
      ]),
      stage("合并输出", "combine", ["combine", "down-proj"]),
    ];
  if (view === "cache")
    return [
      stage("状态与缓存", "cache-0", ["cache-0", "cache-1", "cache-2"]),
    ];
  if (view === "auxiliary") return [stage("专用字段", "aux-0", ["aux-0"])];
  return [];
}
