import type { ModelGraph, ModuleNode } from "./graph";
import type { DemoEvent } from "./inference-demo";
import type { TraceSample } from "./demo-trace";

/** A semantic address is stable across collapsed, planar and expanded scenes. */
export function eventForNode(node: ModuleNode, cursor: DemoEvent): DemoEvent {
  return {
    ...cursor,
    view: node.data.sceneView ?? cursor.view,
    nodeId: node.data.sceneSourceId ?? node.id,
    instance: node.data.sceneParent,
    layer: node.data.layerIndex ?? cursor.layer,
  };
}
export function sceneTraceSamples(
  graph: ModelGraph,
  samples: TraceSample[],
  pass: number,
  loop: number,
  layer: number,
): TraceSample[] {
  return samples.flatMap((sample) => {
    if (
      sample.pass !== pass ||
      (sample.loop ?? 0) !== loop ||
      sample.layer !== layer
    )
      return [];
    const node = graph.nodes.find(
      (n) =>
        n.data.sceneView === sample.view &&
        n.data.sceneSourceId === sample.nodeId &&
        n.data.sceneParent === sample.instance,
    );
    return node ? [{ ...sample, nodeId: node.id }] : [];
  });
}
