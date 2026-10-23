import { describe, it, expect } from "vitest";
import { modelLibrary, loadLibraryModel } from "../data/model-library";
import { buildDemoPlan, demoScenario } from "./inference-demo";
import { buildDemoWorld, demoSceneNodeId } from "./demo-world";
import {
  referenceVisual,
  recordedLane,
  sampleCoordinates,
} from "./reference-scene";
const scenario = {
  batch: 1,
  sequence: 4,
  past: 0,
  dtype: "bfloat16" as const,
};
describe("model-native reference renderer", () => {
  it.each(modelLibrary)(
    "$id uses finite target-model geometry, exact tensor sizes and valid persistent connections",
    async (entry) => {
      const { model } = await loadLibraryModel({
        entryId: entry.id,
        variantId: entry.variants[0].id,
      });
      const plan = buildDemoPlan(model, scenario);
      for (const layer of [...new Set([0, model.layers - 1])]) {
        const event =
          plan.find((e) => e.layer === layer && e.view !== "overview") ??
          plan[0];
        const graph = buildDemoWorld(model, scenario, event, false),
          ids = new Set(graph.nodes.map((n) => n.id));
        expect(ids.size).toBe(graph.nodes.length);
        for (const step of plan.filter(
          (e) => e.layer === layer && e.view !== "overview",
        )) {
          const target = graph.nodes.find(
            (n) => n.id === demoSceneNodeId(graph, step),
          );
          expect(
            target,
            `${entry.id}: ${step.segment}/${step.nodeId}`,
          ).toBeDefined();
          expect(target?.data.sceneView).toBe(step.view);
          expect(target?.data.sceneSourceId).toBe(step.nodeId);
        }
        for (const e of graph.edges) {
          expect(ids.has(e.source), e.source).toBe(true);
          expect(ids.has(e.target), e.target).toBe(true);
        }
        for (const n of graph.nodes) {
          const s = referenceVisual(n, model, scenario);
          expect(
            [
              s.width,
              s.height,
              s.depth,
              s.inputWidth,
              s.outputWidth,
              s.heads,
            ].every((x) => Number.isFinite(x) && x > 0),
          ).toBe(true);
          expect(s.outputWidth).toBe(
            n.data.linear?.output ??
              ((n.data.outputs[0] ?? n.data.inputs[0])?.shape.at(-1) ||
                s.inputWidth),
          );
          expect(n.data.worldPosition?.every(Number.isFinite)).toBe(true);
        }
      }
    },
  );
  it("matches recorded coordinates without inventing unsampled values", () => {
    const tensor = {
      name: "q",
      shape: [1, 8, 1, 64],
      symbols: ["B", "Nq", "S", "D"],
      dtype: "bfloat16" as const,
    };
    const sample = {
      pass: 1,
      layer: 0,
      view: "attention" as const,
      nodeId: "q",
      direction: "output" as const,
      tensorIndex: 0,
      shape: tensor.shape,
      indices: [
        [0, 3, 0, 0],
        [0, 3, 0, 16],
      ],
      values: [0.25, -0.75],
    };
    expect(recordedLane(sample, tensor, 0, 3, [0, 8, 16])).toEqual([
      0.25,
      null,
      -0.75,
    ]);
    expect(recordedLane(sample, tensor, 0, 2, [0])).toEqual([null]);
    expect(sampleCoordinates(0)).toEqual([]);
    expect(sampleCoordinates(128)).toHaveLength(24);
  });
  it("decode changes physical token lanes without expanding history into query lanes", () => {
    expect(
      demoScenario(
        { promptTokens: 4, outputTokens: 4, useCache: true, speed: 1 },
        2,
        "bfloat16",
      ),
    ).toMatchObject({ sequence: 1, past: 5 });
  });
});
