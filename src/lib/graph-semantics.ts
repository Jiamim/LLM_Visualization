import type { AxisRole, ModelConfig, Scenario, Tensor } from "./model";
import type { ModelGraph, View } from "./graph";
import { annotateTensor, featureWidth, tensorAxes } from "./tensor-semantics";
import { gemmaLayer } from "./gemma";

/** Enrich the adapter graph once, before any layout or rendering transformation. */
export function annotateGraph(
  graph: ModelGraph,
  model: ModelConfig,
  scenario: Scenario,
  view: View,
  layer: number,
): ModelGraph {
  for (const node of graph.nodes) {
    const d = node.data;
    d.layerIndex ??= layer;
    d.inputs = d.inputs.map(annotateTensor);
    d.outputs = d.outputs.map(annotateTensor);
    d.parameterStatus ??= model.catalog ? "estimated" : "known";
    const profile = model.catalog?.profiles[d.layerIndex];
    const tensors = [...d.outputs, ...d.inputs];
    const score = tensors.find(
      (t) => t.role === "scores" || t.role === "probabilities",
    );
    const query = tensors.find((t) => t.role === "query");
    const key = tensors.find((t) => t.role === "key");
    const axisSize = (t: Tensor | undefined, role: AxisRole) => {
      const i = t ? tensorAxes(t).indexOf(role) : -1;
      return i >= 0 ? t!.shape[i] : undefined;
    };
    const isAttention =
      !!score ||
      !!query ||
      d.drill === "attention" ||
      [
        "attention-core",
        "sparse-attn",
        "weighted-values",
        "softmax-values",
      ].includes(node.id);
    if (isAttention) {
      const window = profile
        ? profile.window
        : model.slidingLayers[d.layerIndex]
          ? model.slidingWindow
          : null;
      const queryLength =
        axisSize(score ?? query, "query") ?? scenario.sequence;
      const keyLength =
        axisSize(score ?? key, "key") ??
        (model.gemma && window
          ? Math.min(scenario.past, window - 1) + queryLength
          : scenario.past + queryLength);
      const unknown =
        !!model.deepseek ||
        (!!profile && !["attention", "mla"].includes(profile.kind));
      d.attention = {
        queryLength,
        keyLength,
        queryOffset: scenario.past,
        keyOffset: Math.max(0, scenario.past + queryLength - keyLength),
        heads: profile?.heads ?? model.heads,
        kvHeads: profile?.kvHeads ?? model.kvHeads,
        window,
        visibility: unknown
          ? "unknown"
          : view === "vision" || view === "audio"
            ? "bidirectional"
            : "causal",
      };
    }
    // Projection IDs belong to the adapter contract; never classify probabilities by display title.
    if (
      !d.linear &&
      d.inputs[0] &&
      d.outputs[0] &&
      /(?:^|[-:])(?:q-proj|k-proj|v-proj|o-proj|out-proj|up-proj|down-proj|gate-proj|q-lora|kv-project)$/.test(
        node.id,
      )
    ) {
      d.linear = {
        input: featureWidth(d.inputs[0]),
        output: featureWidth(d.outputs[0]),
      };
    }
    if (model.gemma && view === "attention" && node.id === "q-proj") {
      d.linear = {
        input: model.hidden,
        output: model.heads * gemmaLayer(model, d.layerIndex).headDim,
      };
    }
  }
  for (const edge of graph.edges) {
    if (edge.data?.tensor)
      edge.data.tensor = annotateTensor(edge.data.tensor as Tensor);
  }
  // External state roots are deliberately not included in the main input port.
  if (view === "attention" || view === "mlp") {
    const entry =
      graph.nodes.find((n) =>
        ["attn-input", "mixer-input", "mlp-input", "ffn-input"].includes(
          n.id,
        ),
      ) ?? graph.nodes[0];
    const exit = [...graph.nodes]
      .reverse()
      .find((n) => !graph.edges.some((e) => e.source === n.id));
    if (entry && exit) graph.ports = { entry: entry.id, exit: exit.id };
  }
  return graph;
}
