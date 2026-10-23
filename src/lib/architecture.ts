import { MarkerType, type Edge } from "@xyflow/react";
import type { ModelGraph, ModuleNode } from "./graph";
import type { ModelConfig, Scenario, Tensor } from "./model";
import { gemmaLayer } from "./gemma";
import { layerProfile } from "./deepseek";
import { NODE_HEIGHT, NODE_WIDTH } from "./presentation";

export interface LayerRun {
  start: number;
  length: number;
  period: number;
  count: number;
}
export function layerSignature(model: ModelConfig, index: number): string {
  if (model.catalog) {
    const { index: _, source, ...profile } = model.catalog.profiles[index];
    return JSON.stringify({
      ...profile,
      source: source === index ? "self" : source,
    });
  }
  if (model.gemma) {
    const { index: _, owner, ...profile } = gemmaLayer(model, index);
    return JSON.stringify({
      ...profile,
      owner: owner === index ? "self" : owner,
    });
  }
  if (model.deepseek) {
    const {
      index: _,
      kvSource,
      indexSource,
      ...profile
    } = layerProfile(model, index);
    return JSON.stringify({
      ...profile,
      kvSource: kvSource === index ? "self" : kvSource,
      indexSource: indexSource === index ? "self" : indexSource,
    });
  }
  return JSON.stringify({ sliding: model.slidingLayers[index] });
}
/** Compress contiguous repetition only; never regroup layers in a different order. */
export function compressLayerSequence(signatures: string[]): LayerRun[] {
  const runs: LayerRun[] = [];
  for (let start = 0; start < signatures.length;) {
    let best = { start, length: 1, period: 1, count: 1 },
      saved = 0;
    for (
      let period = 1;
      period <= Math.min(64, Math.floor((signatures.length - start) / 2));
      period++
    ) {
      let count = 1;
      while (start + (count + 1) * period <= signatures.length) {
        let same = true;
        for (let j = 0; j < period; j++)
          if (
            signatures[start + j] !== signatures[start + count * period + j]
          ) {
            same = false;
            break;
          }
        if (!same) break;
        count++;
      }
      if ((count - 1) * period > saved) {
        saved = (count - 1) * period;
        best = { start, length: period * count, period, count };
      }
    }
    runs.push(best);
    start += best.length;
  }
  return runs;
}

type LayerBuilder = (index: number) => ModelGraph;
interface Expanded {
  nodes: ModuleNode[];
  edges: Edge[];
  entry: string;
  exit: string;
  width: number;
  height: number;
  members: string[];
}
const frameNode = (
  id: string,
  title: string,
  width: number,
  height: number,
  count: number,
  range: [number, number],
  period: number,
): ModuleNode => ({
  id,
  type: "module",
  position: { x: 0, y: 0 },
  width,
  height,
  style: { width, height },
  selectable: false,
  draggable: false,
  focusable: false,
  zIndex: -1,
  data: {
    title,
    subtitle: "",
    category: "block",
    path: id,
    inputs: [],
    outputs: [],
    params: 0,
    description: "",
    formula: "",
    details: [],
    frame: { width, height, count, range, period },
    inactive: true,
  },
});
function wire(
  id: string,
  source: string,
  target: string,
  tensor: Tensor | undefined,
  symbolic: boolean,
): Edge {
  return {
    id,
    source,
    target,
    type: "smoothstep",
    sourceHandle: "out",
    targetHandle: "in",
    data: { tensor },
    label: tensor
      ? `[${(symbolic ? tensor.symbols : tensor.shape).join(", ")}]`
      : undefined,
    style: { stroke: "#9cabc0", strokeWidth: 1.7 },
    markerEnd: { type: MarkerType.ArrowClosed, color: "#9cabc0" },
    labelStyle: { fontSize: 11, fill: "#52667f" },
    labelBgStyle: { fill: "#fff" },
    labelBgPadding: [7, 5],
  };
}
/** Layer internals are cloned from their actual adapters, including residual branches. */
function expandUnit(
  model: ModelConfig,
  buildLayer: LayerBuilder,
  index: number,
  count: number,
  prefix: string,
  range: [number, number],
): Expanded {
  const graph = buildLayer(index);
  if (!model.catalog && !model.gemma && !model.deepseek) {
    const positions: Record<string, [number, number]> = {
      "layer-input": [0, 0],
      "input-norm": [0, 190],
      attention: [0, 380],
      "attention-residual": [0, 570],
      "post-norm": [430, 190],
      mlp: [430, 380],
      "mlp-residual": [430, 570],
    };
    for (const node of graph.nodes) {
      const p = positions[node.id];
      if (p) node.position = { x: p[0], y: p[1] };
    }
    for (const edge of graph.edges)
      if (edge.source === "attention-residual" && edge.target === "post-norm") {
        edge.sourceHandle = "right-out";
        edge.targetHandle = "side-in";
      }
  }
  const minX = Math.min(...graph.nodes.map((n) => n.position.x)),
    minY = Math.min(...graph.nodes.map((n) => n.position.y));
  const width =
    Math.max(...graph.nodes.map((n) => n.position.x + NODE_WIDTH)) - minX + 135;
  const height =
    Math.max(...graph.nodes.map((n) => n.position.y + NODE_HEIGHT)) -
    minY +
    110;
  const id = (s: string) => `${prefix}/${s}`;
  const frame = frameNode(
    prefix,
    model.deepseek
      ? `${layerProfile(model, index).partition} 代表层`
      : model.catalog
        ? `${model.catalog.profiles[index].label} 代表层`
        : "Transformer 代表层",
    width,
    height,
    count,
    range,
    1,
  );
  const nodes: ModuleNode[] = graph.nodes.map((node) => ({
    ...node,
    id: id(node.id),
    position: {
      x: node.position.x - minX + 58,
      y: node.position.y - minY + 68,
    },
    data: {
      ...node.data,
      layerIndex: index,
      details: [
        ...node.data.details,
        ["代表层", `L${index}`],
        ["展示范围", `L${range[0]}–L${range[1]}`],
        [
          "重复说明",
          `结构展示一次，连续重复 ${count} 次；各层权重不默认共享。`,
        ],
      ],
    },
  }));
  const edges = graph.edges.map((edge) => ({
    ...edge,
    id: id(edge.id),
    source: id(edge.source),
    target: id(edge.target),
  }));
  const entry =
    nodes.find(
      (n) => n.id === id("layer-input") || n.id === id("block-input"),
    ) ?? nodes[0];
  entry.data.repeat = count;
  entry.data.drill = "layer";
  // Side inputs such as PLE slices are not the sequential block output.
  const exitIds = ["ffn-post", "layer-scale", "mlp-residual", "output"];
  const exit =
    exitIds.map((name) => nodes.find((n) => n.id === id(name))).find(Boolean) ??
    nodes.at(-1)!;
  return {
    nodes: [frame, ...nodes],
    edges,
    entry: entry.id,
    exit: exit.id,
    width,
    height,
    members: nodes.map((n) => n.id),
  };
}
function expandRun(
  model: ModelConfig,
  buildLayer: LayerBuilder,
  run: LayerRun,
  startLayer: number,
  prefix: string,
  symbolic: boolean,
): Expanded {
  const first = startLayer + run.start,
    last = first + run.length - 1;
  if (run.period === 1)
    return expandUnit(model, buildLayer, first, run.count, prefix, [
      first,
      last,
    ]);
  const units: Expanded[] = [];
  for (let offset = 0; offset < run.period;) {
    let count = 1;
    while (
      offset + count < run.period &&
      layerSignature(model, first + offset) ===
        layerSignature(model, first + offset + count)
    )
      count++;
    units.push(
      expandUnit(
        model,
        buildLayer,
        first + offset,
        count,
        `${prefix}/unit-${offset}`,
        [first + offset, first + offset + count - 1],
      ),
    );
    offset += count;
  }
  let y = 65;
  const width = Math.max(...units.map((u) => u.width)) + 60;
  const nodes: ModuleNode[] = [],
    edges: Edge[] = [],
    members: string[] = [];
  for (const unit of units) {
    for (const node of unit.nodes)
      nodes.push({
        ...node,
        position: { x: node.position.x + 30, y: node.position.y + y },
      });
    edges.push(...unit.edges);
    members.push(...unit.members);
    y += unit.height + 65;
  }
  for (let i = 1; i < units.length; i++)
    edges.push(
      wire(
        `${prefix}/cycle-${i}`,
        units[i - 1].exit,
        units[i].entry,
        nodes.find((n) => n.id === units[i - 1].exit)?.data.outputs[0],
        symbolic,
      ),
    );
  const frame = frameNode(
    prefix,
    `${run.period} 层重复周期`,
    width,
    y,
    run.count,
    [first, last],
    run.period,
  );
  frame.zIndex = -2;
  return {
    nodes: [frame, ...nodes],
    edges,
    entry: units[0].entry,
    exit: units.at(-1)!.exit,
    width,
    height: y,
    members,
  };
}

export function buildArchitecture(
  model: ModelConfig,
  _scenario: Scenario,
  overview: ModelGraph,
  buildLayer: LayerBuilder,
  symbolic: boolean,
): ModelGraph {
  const nodes: ModuleNode[] = [],
    edges: Edge[] = [],
    ports = new Map<string, { entry: string; exit: string }>();
  const architectureSections: NonNullable<ModelGraph["sections"]> = [];
  let y = 0;
  for (const node of overview.nodes) {
    const trunk =
      node.data.category === "block" &&
      ["encoder", "decoder"].includes(node.id);
    if (!trunk) {
      const clone = {
        ...node,
        position: { x: 180, y },
        data: { ...node.data },
      };
      nodes.push(clone);
      ports.set(node.id, { entry: node.id, exit: node.id });
      y += NODE_HEIGHT + 110;
      continue;
    }
    const start =
      node.id === "decoder" && model.deepseek
        ? model.deepseek.encoderLayers
        : 0;
    const end =
      node.id === "encoder" && model.deepseek
        ? model.deepseek.encoderLayers
        : model.layers;
    const runs = compressLayerSequence(
      Array.from({ length: end - start }, (_, i) =>
        layerSignature(model, start + i),
      ),
    );
    const loopTop = y;
    const loopNodesStart = nodes.length;
    let entry = "",
      exit = "";
    for (const run of runs) {
      const expanded = expandRun(
        model,
        buildLayer,
        run,
        start,
        `arch-${node.id}-${start + run.start}`,
        symbolic,
      );
      for (const n of expanded.nodes)
        nodes.push({
          ...n,
          position: { x: n.position.x + 30, y: n.position.y + y },
        });
      edges.push(...expanded.edges);
      if (exit)
        edges.push(
          wire(
            `arch-link-${exit}`,
            exit,
            expanded.entry,
            nodes.find((n) => n.id === exit)?.data.outputs[0],
            symbolic,
          ),
        );
      if (!entry) entry = expanded.entry;
      exit = expanded.exit;
      architectureSections.push({
        title: `L${start + run.start}–${start + run.start + run.length - 1} · ${run.period > 1 ? `${run.period}层周期 ` : ""}× ${run.count}`,
        description: "按实际顺序折叠重复结构；展开代表层查看内部",
        anchor:
          expanded.members.find((id) => /attention|mixer/.test(id)) ??
          expanded.entry,
        nodes: [expanded.nodes[0].id, ...expanded.members],
        repeat: run.count,
        period: run.period,
        start: start + run.start,
        length: run.length,
      });
      y += expanded.height + 105;
    }
    ports.set(node.id, { entry, exit });
    if (model.catalog && model.catalog.loops > 1) {
      const width =
        Math.max(
          ...nodes
            .slice(loopNodesStart)
            .map((n) => n.position.x + (n.width ?? NODE_WIDTH)),
        ) + 50;
      const frame = frameNode(
        `arch-loops-${node.id}`,
        "主干循环 · 跨轮共享权重",
        width,
        y - loopTop,
        model.catalog.loops,
        [start, end - 1],
        end - start,
      );
      frame.position = { x: 0, y: loopTop - 38 };
      frame.zIndex = -3;
      nodes.push(frame);
    }
  }
  for (const edge of overview.edges) {
    const from = ports.get(edge.source),
      to = ports.get(edge.target);
    if (from && to) {
      if (model.gemma && edge.source === "ple" && edge.target === "decoder") {
        for (const target of nodes.filter((n) =>
          n.id.endsWith("/ple-source"),
        )) {
          target.data.inputs = [edge.data?.tensor as Tensor];
          target.data.formula = `layer_PLE = PLE[:, :, ${target.data.layerIndex}, :]`;
          target.data.description =
            "从总览生成的逐层嵌入张量提取当前代表层切片。";
          edges.push({
            ...edge,
            id: `outer/${edge.id}/${target.id}`,
            source: from.exit,
            target: target.id,
            sourceHandle: "out",
            targetHandle: "in",
          });
        }
        continue;
      }
      edges.push({
        ...edge,
        id: `outer/${edge.id}`,
        source: from.exit,
        target: to.entry,
        sourceHandle: "out",
        targetHandle: "in",
      });
    }
  }
  if (!model.catalog && !model.gemma && !model.deepseek) {
    const frame = nodes.find((n) => n.data.frame)!;
    const shift = frame.position.y - 230;
    for (const node of nodes) {
      if (node.id === "input") node.position = { x: 30, y: 0 };
      else if (node.id === "embedding") node.position = { x: 460, y: 0 };
      else node.position.y -= shift;
    }
    const bottom =
      Math.max(
        ...nodes
          .filter((n) => n.data.frame)
          .map((n) => n.position.y + n.height!),
      ) + 100;
    const positions: Record<string, [number, number]> = {
      "final-norm": [30, bottom],
      "lm-head": [460, bottom],
      output: [460, bottom + 210],
    };
    for (const node of nodes)
      if (positions[node.id]) {
        const [x, y] = positions[node.id];
        node.position = { x, y };
      }
    for (const edge of edges)
      if (
        (edge.source === "input" && edge.target === "embedding") ||
        (edge.source === "final-norm" && edge.target === "lm-head")
      ) {
        edge.sourceHandle = "right-out";
        edge.targetHandle = "side-in";
      }
  }
  // Stable topological order for playback. Frames are decorative and never execute.
  const pending = new Map(
      nodes.filter((n) => !n.data.frame).map((n) => [n.id, n]),
    ),
    ordered = nodes.filter((n) => n.data.frame);
  while (pending.size) {
    const ready = [...pending.values()].filter(
      (n) => !edges.some((e) => e.target === n.id && pending.has(e.source)),
    );
    if (!ready.length) throw new Error("整体架构出现循环依赖。");
    for (const node of ready) {
      ordered.push(node);
      pending.delete(node.id);
    }
  }
  let order = 0;
  for (const node of ordered) if (!node.data.frame) node.data.order = ++order;
  return { nodes: ordered, edges, sections: architectureSections };
}
