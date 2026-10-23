import {
  dtypeBytes,
  tensorBytes,
  type ModelConfig,
  type Scenario,
  type Tensor,
} from "./model";
import { catalogState } from "./catalog";
import { gemmaLayer, gemmaShapes } from "./gemma";
import { layerProfile } from "./deepseek";

export type CacheKind = "kv" | "recurrent" | "auxiliary";
export const cacheKindLabels: Record<CacheKind, string> = {
  kv: "KV cache",
  recurrent: "递归 / 卷积状态",
  auxiliary: "索引 / 压缩状态",
};
export interface CacheRow {
  id: string;
  name: string;
  kind: CacheKind;
  layers: number[];
  consumers: number[];
  copies: number;
  tensors: Tensor[];
  bytes: number;
  formula: string;
  note: string;
}
export interface CacheReport {
  rows: CacheRow[];
  kvBytes: number;
  recurrentBytes: number;
  auxiliaryBytes: number;
  totalBytes: number;
  perSequenceBytes: number;
  addedBytes: number;
  nextTokenBytes: number;
  nextKvBytes: number;
  totalTokens: number;
  partial: boolean;
  unknownLayers: number[];
  cacheDisabledByDefault: boolean;
  packedGlobalBytes: number | null;
  notes: string[];
}
const recurrentNames = new Set([
  "ssm_state",
  "conv_state",
  "matrix_state",
  "normalizer_state",
  "stabilizer_state",
  "recurrent_matrix",
  "short_conv_state",
  "qk_conv_state",
  "delayed_value",
]);
function rowsAt(m: ModelConfig, s: Scenario): CacheRow[] {
  const rows: CacheRow[] = [],
    bytes = dtypeBytes(s.dtype),
    total = s.past + s.sequence;
  const add = (row: Omit<CacheRow, "bytes">) =>
    rows.push({
      ...row,
      bytes:
        row.tensors.reduce((sum, t) => sum + tensorBytes(t), 0) * row.copies,
    });
  const t = (
    name: string,
    shape: number[],
    symbols: string[],
    dtype: Tensor["dtype"] = s.dtype,
  ): Tensor => ({ name, shape, symbols, dtype });
  if (m.deepseek) {
    const d = m.deepseek;
    for (const owner of d.kvSources) {
      const ratio = d.ratios[owner],
        entries = Math.floor(total / ratio);
      const consumers = Array.from({ length: m.layers }, (_, i) => i).filter(
        (i) => layerProfile(m, i).kvSource === owner,
      );
      add({
        id: `global-${owner}`,
        name: "共享压缩 KV",
        kind: "kv",
        layers: [owner],
        consumers,
        copies: 1,
        tensors: [
          t(
            "compressed_KV",
            [s.batch, entries, m.headDim],
            ["B", "⌊T/r⌋", "D"],
          ),
        ],
        formula: `B × ⌊T/${ratio}⌋ × ${m.headDim} × ${bytes} bytes`,
        note: "K/V 共用一个 latent；仅在来源层计入。",
      });
      add({
        id: `index-${owner}`,
        name: "共享索引 K",
        kind: "auxiliary",
        layers: [owner],
        consumers,
        copies: 1,
        tensors: [
          t("index_key", [s.batch, entries, d.indexDim], ["B", "⌊T/r⌋", "Di"]),
        ],
        formula: `B × ⌊T/${ratio}⌋ × ${d.indexDim} × ${bytes} bytes`,
        note: "与共享全局 KV 对应的索引键，不是另一份 value。",
      });
      if (ratio > 1)
        add({
          id: `compressor-${owner}`,
          name: "未完成压缩组",
          kind: "auxiliary",
          layers: [owner],
          consumers: [owner],
          copies: 1,
          tensors: [
            t(
              "kv_state",
              [s.batch, ratio, m.headDim],
              ["B", "r", "D"],
              "float32",
            ),
            t(
              "score_state",
              [s.batch, ratio, m.headDim],
              ["B", "r", "D"],
              "float32",
            ),
          ],
          formula: `2 × B × ${ratio} × ${m.headDim} × 4 bytes`,
          note: "FP32 压缩缓冲；容量固定，不随 dtype 切换。",
        });
    }
    add({
      id: "window",
      name: "各层滑动窗口 latent",
      kind: "kv",
      layers: Array.from({ length: m.layers }, (_, i) => i),
      consumers: [],
      copies: m.layers,
      tensors: [
        t(
          "window_KV",
          [s.batch, Math.min(total, m.slidingWindow!), m.headDim],
          ["B", "min(T,W)", "D"],
        ),
      ],
      formula: `B × min(T,${m.slidingWindow}) × ${m.headDim} × ${bytes} bytes`,
      note: "每层独立的 K/V 共用 latent。统计有效槽位，参考后端可预分配完整环形窗口。",
    });
  } else if (m.gemma) {
    for (let i = 0; i < m.gemma.shareStart; i++) {
      const profile = gemmaLayer(m, i),
        shape = gemmaShapes(m, s, i);
      add({
        id: `owner-${i}`,
        name: profile.sliding ? "局部 K/V" : "全局 K/V",
        kind: "kv",
        layers: [i],
        consumers: Array.from({ length: m.layers }, (_, n) => n).filter(
          (n) => gemmaLayer(m, n).owner === i,
        ),
        copies: 1,
        tensors: [
          t("key_cache", shape.cache, ["B", "Nkv", "R", "D"]),
          t("value_cache", shape.cache, ["B", "Nkv", "R", "D"]),
        ],
        formula: `2 × B × ${m.kvHeads} × ${profile.sliding ? `min(T,${m.slidingWindow! - 1})` : "T"} × ${profile.headDim} × ${bytes} bytes`,
        note: profile.sliding
          ? "DynamicCache 持久保留 W−1 项；共享消费者不重复计数。"
          : "共享消费者复用来源层 K/V，不重复计数。",
      });
    }
  } else if (m.catalog) {
    const a = m.catalog;
    for (const p of a.profiles) {
      if (p.source !== p.index) continue;
      const states = catalogState(m, s, p);
      for (const kind of ["kv", "recurrent"] as const) {
        const tensors = states.filter(
          (t) => recurrentNames.has(t.name) === (kind === "recurrent"),
        );
        if (!tensors.length) continue;
        const copies = p.repeats * a.loops;
        add({
          id: `layer-${p.index}-${kind}`,
          name: `${p.label} · ${kind === "kv" ? "KV" : "状态"}`,
          kind,
          layers: [p.index],
          consumers: a.profiles
            .filter((q) => q.source === p.index)
            .map((q) => q.index),
          copies,
          tensors,
          formula: tensors
            .map((t) => `[${t.shape.join(" × ")}] × ${dtypeBytes(t.dtype)}`)
            .join(" + "),
          note:
            [...new Set(tensors.map((t) => t.note).filter(Boolean))].join(
              " ",
            ) ||
            (p.window
              ? `窗口上限 ${p.window}，按当前适配器的逻辑窗口计数。`
              : "按当前层实际 KV heads / 维度计算；不乘 Query 头数或专家数。"),
        });
      }
    }
  } else {
    const shape = [s.batch, m.kvHeads, total, m.headDim];
    add({
      id: "dense",
      name:
        m.heads === m.kvHeads
          ? "MHA K/V"
          : m.kvHeads === 1
            ? "MQA K/V"
            : "GQA K/V",
      kind: "kv",
      layers: Array.from({ length: m.layers }, (_, i) => i),
      consumers: [],
      copies: m.layers,
      tensors: [
        t("key_cache", shape, ["B", "Nkv", "T", "D"]),
        t("value_cache", shape, ["B", "Nkv", "T", "D"]),
      ],
      formula: `2 × B × ${m.kvHeads} KV heads × T × ${m.headDim} × ${bytes} bytes`,
      note: m.slidingWindow
        ? "沿用当前计算图的未裁剪逻辑缓存；滑动窗口仅限制 mask，实际后端可以裁剪。"
        : "K 和 V 各计一次；GQA / MQA 使用 KV 头数，不使用 Query 头数。",
    });
  }
  return rows;
}
function totals(rows: CacheRow[]) {
  const sum = (kind: CacheKind) =>
    rows.filter((r) => r.kind === kind).reduce((n, r) => n + r.bytes, 0);
  const kvBytes = sum("kv"),
    recurrentBytes = sum("recurrent"),
    auxiliaryBytes = sum("auxiliary");
  return {
    kvBytes,
    recurrentBytes,
    auxiliaryBytes,
    totalBytes: kvBytes + recurrentBytes + auxiliaryBytes,
  };
}
export function cacheReport(m: ModelConfig, s: Scenario): CacheReport {
  const rows = rowsAt(m, s),
    now = totals(rows),
    totalTokens = s.past + s.sequence;
  const before = totals(rowsAt(m, { ...s, sequence: 0 }));
  const next = totals(rowsAt(m, { ...s, sequence: 1, past: totalTokens }));
  const unknownLayers =
    m.catalog?.profiles
      .filter((p) => p.kind !== "ffn" && catalogState(m, s, p).length === 0)
      .map((p) => p.index) ?? [];
  const partial =
    !!m.catalog?.partialCache || !!m.deepseek || unknownLayers.length > 0;
  const text = m.catalog?.text ?? m.gemma?.text ?? m.deepseek?.text ?? m.raw;
  const notes = [
    "按开启推理缓存估算，T = P + S；B 个序列等长、独立存储，不计前缀共享、分页对齐、预分配空位及峰值显存。",
    "当前 dtype 用于浮点 K/V；标为 float32 的递归与压缩状态始终按 4 bytes。权重量化配置不自动改变 KV 精度。",
    ...(m.catalog
      ? [
          "使用当前架构适配器的逻辑缓存表示；窗口、压缩率、K/V 宽度与循环次数逐层计算。",
        ]
      : []),
    ...(m.catalog?.kvRank
      ? [
          "MLA 采用 latent + rotary key 的吸收投影缓存；后端若保存展开 K/V，内存可能不同。",
        ]
      : []),
    ...(m.gemma
      ? [
          "局部持久缓存上限为 W−1；注意力本次可见 K/V 与跨层临时共享张量不额外计入持久缓存。",
        ]
      : []),
    ...(m.deepseek
      ? [
          "DeepSeek V4.1 统计主干有效窗口、共享全局 latent、索引 K 与 FP32 压缩状态；DSpark、Engram 卷积历史、候选索引及工作区未包含。FP4 仅为全局压缩缓存的紧凑格式估算。",
        ]
      : []),
    ...(m.catalog?.partialCache
      ? [
          "部分状态估算：专用算子、稀疏索引辅助项、媒体与辅助预测分支可能未包含，未知项不当作零。",
        ]
      : []),
    ...(unknownLayers.length
      ? [
          `L${unknownLayers.join(", L")} 的状态尚不能由当前规则确定；合计仅含已知部分。`,
        ]
      : []),
    ...(text.use_cache === false
      ? [
          "配置声明 use_cache=false；这里显示开启缓存后的理论占用，并非该配置默认已分配。",
        ]
      : []),
    ...(m.maxPosition !== null && totalTokens > m.maxPosition
      ? [
          `T=${totalTokens} 超过配置上下文上限 ${m.maxPosition}，仅作容量外推，不代表模型支持该长度。`,
        ]
      : []),
    ...(m.gallery?.sourceNote ? [`配置来源说明：${m.gallery.sourceNote}`] : []),
  ];
  const packedGlobalBytes = m.deepseek
    ? m.deepseek.kvSources.reduce(
        (sum, i) =>
          sum +
          s.batch *
            Math.floor(totalTokens / m.deepseek!.ratios[i]) *
            (m.headDim / 2 +
              Math.ceil(m.headDim / 16) +
              m.deepseek!.indexDim / 2 +
              Math.ceil(m.deepseek!.indexDim / 32)),
        0,
      )
    : null;
  return {
    rows,
    ...now,
    perSequenceBytes: now.totalBytes / s.batch,
    addedBytes: now.totalBytes - (s.past === 0 ? 0 : before.totalBytes),
    nextTokenBytes: next.totalBytes - now.totalBytes,
    nextKvBytes: next.kvBytes - now.kvBytes,
    totalTokens,
    partial,
    unknownLayers,
    cacheDisabledByDefault: text.use_cache === false,
    packedGlobalBytes,
    notes,
  };
}

/** Group identical formulas/shapes while retaining the actual owner layer IDs. */
export function groupCacheRows(rows: CacheRow[]): CacheRow[] {
  const groups = new Map<string, CacheRow>();
  for (const row of rows) {
    const key = JSON.stringify([
      row.name,
      row.kind,
      row.tensors,
      row.formula,
      row.note,
    ]);
    const existing = groups.get(key);
    if (existing) {
      existing.layers.push(...row.layers);
      existing.consumers.push(...row.consumers);
      existing.copies += row.copies;
      existing.bytes += row.bytes;
    } else
      groups.set(key, {
        ...row,
        layers: [...row.layers],
        consumers: [...row.consumers],
      });
  }
  return [...groups.values()];
}
