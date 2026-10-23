import type { ModelGraph } from "./graph";
import type { Tensor } from "./model";

/** Residuals are merge operations, not parameterized layers. Keep each adapter's semantics. */
export function emphasizeResiduals(graph: ModelGraph): ModelGraph {
  const incoming = new Map<string, typeof graph.edges>();
  for (const edge of graph.edges) {
    const list = incoming.get(edge.target) ?? [];
    list.push(edge);
    incoming.set(edge.target, list);
  }
  const residualTargets = new Set(
    graph.edges.filter((e) => e.data?.residual).map((e) => e.target),
  );
  return {
    nodes: graph.nodes.map((node) => {
      if (!residualTargets.has(node.id)) return node;
      const special = /mHC|Special/.test(node.data.title);
      const edges = incoming.get(node.id) ?? [];
      const inputs = edges
        .map((edge) => ({
          ...(edge.data?.tensor as Tensor),
          name: edge.data?.residual ? "residual" : String(edge.source),
        }))
        .filter((t) => Array.isArray(t.shape));
      return {
        ...node,
        data: {
          ...node.data,
          residual: true,
          inputs: node.data.inputs.length < 2 ? inputs : node.data.inputs,
          subtitle: special ? "架构专用残差混合" : "残差相加 · skip connection",
          description:
            node.data.description ||
            (special
              ? "按当前架构的门控与混合规则合并残差流，不能简化为普通逐元素相加。"
              : "将子层输出与跳过子层的原输入相加；保留隐藏维度，不引入额外权重。"),
          formula:
            node.data.formula ||
            (special
              ? "architecture-specific residual merge"
              : "output = branch_output + residual"),
          details: [
            ...node.data.details,
            ["残差类型", special ? "门控 / 混合残差" : "逐元素加法"],
            [
              "旁路来源",
              edges
                .filter((e) => e.data?.residual)
                .map((e) => e.source)
                .join(", "),
            ],
          ],
        },
      };
    }),
    edges: graph.edges.map((edge) =>
      edge.data?.residual
        ? {
            ...edge,
            label: `残差 · ${String(edge.label ?? "").replace(/^residual\s*·\s*/, "")}`,
            style: {
              ...edge.style,
              stroke: "#bd6f35",
              strokeWidth: 2.3,
              strokeDasharray: "7 4",
            },
          }
        : edge,
    ),
  };
}
