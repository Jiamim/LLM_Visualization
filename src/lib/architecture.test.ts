import { describe, it, expect } from "vitest";
import { compressLayerSequence, layerSignature } from "./architecture";
import { buildGraph } from "./graph";
import {
  modelLibrary,
  loadLibraryModel,
  selectionForPreset,
} from "../data/model-library";
import { normalizeConfig, type Scenario } from "./model";
import qwen25 from "../../tests/fixtures/qwen2.5-7b.json";

const scenario: Scenario = {
  batch: 2,
  sequence: 1,
  past: 13,
  dtype: "bfloat16",
};
describe("folded complete architecture", () => {
  it("compresses consecutive identical layers or full repeated cycles without reordering", () => {
    expect(compressLayerSequence(Array(28).fill("a"))).toEqual([
      { start: 0, length: 28, period: 1, count: 28 },
    ]);
    expect(
      compressLayerSequence(["a", "a", "a", "b", "a", "a", "a", "b"]),
    ).toEqual([{ start: 0, length: 8, period: 4, count: 2 }]);
    const input = ["a", "b", "b", "a", "c", "a"];
    const expanded = compressLayerSequence(input).flatMap((run) =>
      Array.from({ length: run.count }, () =>
        input.slice(run.start, run.start + run.period),
      ).flat(),
    );
    expect(expanded).toEqual(input);
  });
  it("includes the complete dense path and two residual merges in a single repeated frame", () => {
    const m = normalizeConfig(qwen25),
      graph = buildGraph(m, scenario, "architecture", 0);
    expect(graph.sections).toHaveLength(1);
    expect(graph.sections![0]).toMatchObject({
      repeat: 28,
      period: 1,
      start: 0,
      length: 28,
    });
    expect(graph.nodes.filter((n) => n.data.frame)).toHaveLength(1);
    expect(graph.nodes.filter((n) => n.data.residual)).toHaveLength(2);
    for (const id of ["input", "embedding", "final-norm", "lm-head", "output"])
      expect(graph.nodes.some((n) => n.id === id)).toBe(true);
    const residual = graph.nodes.find((n) =>
      n.id.endsWith("/attention-residual"),
    )!;
    expect(residual.data.inputs).toHaveLength(2);
    expect(residual.data.params).toBe(0);
    expect(
      graph.edges.some((e) => e.target === residual.id && e.data?.residual),
    ).toBe(true);
    expect(residual.data.outputs[0].shape).toEqual([2, 1, 3584]);
  });
  it("distinguishes local/full attention and PLE sharing boundaries", async () => {
    const { model } = await loadLibraryModel(selectionForPreset("gemma4-e2b"));
    expect(layerSignature(model, 0)).not.toBe(layerSignature(model, 4));
    expect(layerSignature(model, 12)).not.toBe(layerSignature(model, 13));
    expect(layerSignature(model, 13)).not.toBe(layerSignature(model, 15));
    const graph = buildGraph(model, scenario, "architecture", 0);
    expect(
      graph.nodes.some((n) => n.id.endsWith("/ple-add") && n.data.residual),
    ).toBe(true);
    expect(
      graph.edges.some(
        (e) => e.source === "ple" && e.target.endsWith("/ple-source"),
      ),
    ).toBe(true);
  });
  it("retains CED partitions and gated mHC residual semantics", async () => {
    const { model } = await loadLibraryModel(
      selectionForPreset("deepseek-v41"),
    );
    const graph = buildGraph(model, scenario, "architecture", 0);
    for (const section of graph.sections!)
      expect(
        section.start < 20
          ? section.start + section.length <= 20
          : section.start >= 20,
      ).toBe(true);
    const merged = graph.nodes.filter(
      (n) => n.data.residual && n.id.endsWith("/hc-post"),
    );
    expect(merged.length).toBeGreaterThan(0);
    expect(
      merged.every((n) => n.data.formula.includes("post × attn_out")),
    ).toBe(true);
    expect(graph.nodes.some((n) => n.id === "ced-kv")).toBe(true);
  });
  it("marks every incoming operand for parallel residuals", async () => {
    const entry = modelLibrary.find((e) => e.modelType === "cohere2")!;
    const { model } = await loadLibraryModel({
      entryId: entry.id,
      variantId: entry.variants[0].id,
    });
    const graph = buildGraph(model, scenario, "layer", 0),
      output = graph.nodes.find((n) => n.id === "output")!;
    expect(output.data.residual).toBe(true);
    expect(output.data.inputs).toHaveLength(3);
    expect(output.data.formula).toContain("residual + attention");
  });
  it.each(modelLibrary)(
    "$id folds all layers with valid residuals, topology and shapes",
    async (entry) => {
      const { model } = await loadLibraryModel({
        entryId: entry.id,
        variantId: entry.variants[0].id,
      });
      const graph = buildGraph(model, scenario, "architecture", 0),
        sections = graph.sections!;
      expect(sections.reduce((sum, s) => sum + s.length, 0)).toBe(model.layers);
      const represented = sections.flatMap((s) =>
        Array.from({ length: s.length }, (_, i) => s.start + i),
      );
      expect(represented).toEqual(
        Array.from({ length: model.layers }, (_, i) => i),
      );
      for (const section of sections) {
        expect(section.period * section.repeat).toBe(section.length);
        for (let i = 0; i < section.length; i++)
          expect(layerSignature(model, section.start + i)).toBe(
            layerSignature(model, section.start + (i % section.period)),
          );
      }
      const positions = new Map(graph.nodes.map((n, i) => [n.id, i]));
      expect(positions.size).toBe(graph.nodes.length);
      for (const edge of graph.edges) {
        expect(positions.has(edge.source)).toBe(true);
        expect(positions.has(edge.target)).toBe(true);
        expect(positions.get(edge.source)!).toBeLessThan(
          positions.get(edge.target)!,
        );
        if (edge.data?.residual)
          expect(
            graph.nodes.find((n) => n.id === edge.target)?.data.residual,
          ).toBe(true);
      }
      for (const node of graph.nodes) {
        expect(
          Number.isFinite(node.position.x) && Number.isFinite(node.position.y),
        ).toBe(true);
        for (const tensor of [...node.data.inputs, ...node.data.outputs])
          expect(tensor.shape.length).toBe(tensor.symbols.length);
      }
    },
  );
});
