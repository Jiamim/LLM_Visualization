import type { ModelConfig, Scenario } from "./model";
import { buildGraph, type ModelGraph, type ModuleNode } from "./graph";
import type { DemoEvent } from "./inference-demo";
import { demoGraph } from "./inference-demo";
import { categories } from "./presentation";
import { REFERENCE_LAYER_HEIGHT, referenceVisual } from "./reference-scene";

export function demoWorldLayers(model: ModelConfig) {
  return Array.from({ length: model.layers }, (_, index) => ({
    parts: buildGraph(
      model,
      { batch: 1, sequence: 1, past: 0, dtype: model.dtype },
      "layer",
      index,
    )
      .nodes.filter((n) => n.data.category !== "input" && !n.data.frame)
      .map((n) => ({
        id: n.id,
        title: n.data.title,
        color: n.data.residual
          ? "#a77343"
          : categories[n.data.category].color,
        residual: !!n.data.residual,
        visual: referenceVisual(n, model),
      })),
    index,
    label: model.deepseek
      ? `${index < model.deepseek.encoderLayers ? "Encoder" : "Decoder"} L${index}`
      : `Layer ${index + 1}`,
    color:
      categories[
        model.catalog?.profiles[index].kind === "ffn" ? "mlp" : "block"
      ].color,
  }));
}
/** Persistent layer: attention, MLP and residuals coexist while the cursor advances. */
export function buildDemoWorld(
  model: ModelConfig,
  scenario: Scenario,
  event: DemoEvent,
  symbolic: boolean,
): ModelGraph {
  const graph = demoGraph(model, scenario, event, symbolic);
  const stationary = (nodes: ModuleNode[]): ModelGraph => ({
    ...graph,
    nodes: nodes.map((n) => ({
      ...n,
      data: {
        ...n.data,
        sceneView: "overview",
        sceneSourceId: n.id,
        worldPosition: [n.position.x / 58, n.position.y / 46, 0],
      },
    })),
  });
  if (event.sample)
    return stationary(
      graph.nodes.map((n) => ({
        ...n,
        position: {
          x: n.position.x,
          y: n.position.y + (model.layers * REFERENCE_LAYER_HEIGHT + 18) * 46,
        },
      })),
    );
  if (event.view === "overview") {
    return stationary(
      graph.nodes.map((n) => ({
        ...n,
        position: {
          x: n.id === "input" ? 0 : n.position.x,
          y: [
            "input",
            "embedding",
            "vision",
            "audio",
            "merge",
            "ple",
            "expand",
          ].includes(n.id)
            ? -850 + n.position.y * 1.1
            : model.layers * REFERENCE_LAYER_HEIGHT * 46 +
              300 +
              n.position.y * 0.7,
        },
      })),
    );
  }
  const layer = demoGraph(
    model,
    scenario,
    { ...event, view: "layer", instance: undefined },
    symbolic,
  );
  const nodes: ModuleNode[] = [],
    edges: ModelGraph["edges"] = [];
  const ports = new Map<string, { entry: string; exit: string }>();
  const used = new Set(layer.nodes.map((n) => n.id));
  const baseY = event.layer * REFERENCE_LAYER_HEIGHT;
  // Expand real children in place; retain their parent summary for the return event.
  const rows = layer.nodes.filter((n) => !n.data.frame);
  const weights = rows.map((n) =>
    n.data.drill === "attention" ? 28 : n.data.drill === "mlp" ? 20 : 5,
  );
  const scale =
    (REFERENCE_LAYER_HEIGHT - 8) / weights.reduce((a, b) => a + b, 0);
  let cursor = baseY;
  for (const [index, n] of rows.entries()) {
    nodes.push({
      ...n,
      data: {
        ...n.data,
        sceneView: "layer",
        sceneSourceId: n.id,
        summary: n.data.drill === "attention" || n.data.drill === "mlp",
        worldPosition: [
          n.data.drill === "attention" || n.data.drill === "mlp" ? -19 : 0,
          cursor,
          0,
        ],
      },
    });
    used.add(n.id);
    if (n.data.drill === "attention" || n.data.drill === "mlp") {
      const detail = demoGraph(
        model,
        scenario,
        { ...event, view: n.data.drill, instance: n.id },
        symbolic,
      );
      const xs = [...new Set(detail.nodes.map((c) => c.position.x))].sort(
        (a, b) => a - b,
      );
      const ys = [...new Set(detail.nodes.map((c) => c.position.y))].sort(
        (a, b) => a - b,
      );
      const remap = new Map<string, string>();
      for (const c of detail.nodes.filter((c) => !c.data.frame)) {
        const id = used.has(c.id) ? `${n.id}:${c.id}` : c.id;
        remap.set(c.id, id);
        used.add(id);
        nodes.push({
          ...c,
          id,
          data: {
            ...c.data,
            sceneView: n.data.drill,
            sceneSourceId: c.id,
            sceneParent: n.id,
            worldPosition: [
              (xs.indexOf(c.position.x) - (xs.length - 1) / 2) * 9,
              cursor +
                ((1 + ys.indexOf(c.position.y)) *
                  (weights[index] * scale - 2)) /
                  (ys.length + 1) +
                1.5,
              0,
            ],
          },
        });
      }
      edges.push(
        ...detail.edges.map((e) => ({
          ...e,
          id: `${n.id}:${e.id}`,
          source: remap.get(e.source)!,
          target: remap.get(e.target)!,
        })),
      );
      if (detail.ports)
        ports.set(n.id, {
          entry: remap.get(detail.ports.entry)!,
          exit: remap.get(detail.ports.exit)!,
        });
    }
    cursor += weights[index] * scale;
  }
  edges.push(
    ...layer.edges.map((edge) => ({
      ...edge,
      source: ports.get(edge.source)?.exit ?? edge.source,
      target: ports.get(edge.target)?.entry ?? edge.target,
    })),
  );
  return { nodes, edges };
}
export function demoSceneNodeId(graph: ModelGraph, event: DemoEvent) {
  const parent = event.instance ?? event.segment.split(":")[3];
  return (
    graph.nodes.find(
      (n) =>
        n.data.sceneView === event.view &&
        n.data.sceneSourceId === event.nodeId &&
        (!n.data.sceneParent || n.data.sceneParent === parent),
    )?.id ?? event.nodeId
  );
}
export function selectedTokenPosition(
  scenario: Scenario,
  progress: number,
): number {
  return (
    scenario.past +
    Math.min(
      scenario.sequence - 1,
      Math.floor(Math.max(0, Math.min(0.999, progress)) * scenario.sequence),
    )
  );
}
export function demoPhaseSteps(
  event: DemoEvent,
  node: ModuleNode | undefined,
): string[] {
  const text = `${event.nodeId} ${node?.data.title}`.toLowerCase();
  if (
    node?.data.outputs.some((t) => t.role === "context") &&
    /softmax/.test(text)
  )
    return ["归一化分数", "加权 Value", "合并 heads"];
  if (/softmax/.test(text)) return ["减去最大值", "指数变换", "归一化"];
  if (/scores|attention.core|sparse.attn/.test(text))
    return ["Query × Key", "缩放 / Mask", "权重汇聚"];
  if (node?.data.category === "norm")
    return ["统计特征", "归一化", "缩放 / 偏置"];
  if (/projection|proj/.test(text))
    return ["读取输入向量", "矩阵乘法", "写入输出"];
  if (node?.data.residual) return ["主分支输出", "残差旁路", "合并结果"];
  if (event.sample) return ["最后位置 logits", "教学 token", "追加上下文"];
  return ["准备输入", "执行变换", "输出传递"];
}
