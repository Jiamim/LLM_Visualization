import { describe, expect, it } from "vitest";
import { tensorLabel, tensorVolume, spatialBlocks } from "./tensor-layout";
import { buildGraph } from "./graph";
import { galleryModels, loadGalleryModel } from "../data/gallery";
import type { Tensor } from "./model";

const tensor = (
  shape: number[],
  symbols: string[] = shape.map((_, i) => `D${i}`),
): Tensor => ({ name: "test", shape, symbols, dtype: "bfloat16" });
describe("tensor projections", () => {
  it("projects four-dimensional attention into S by T planes with B × Nq slices", () => {
    const input = tensor([2, 28, 7, 20], ["B", "Nq", "S", "T"]);
    expect(tensorVolume(input)).toMatchObject({
      rows: 7,
      columns: 20,
      slices: 56,
      rowSymbol: "S",
      columnSymbol: "T",
      sliceSymbol: "B × Nq",
      gridRows: 7,
      gridColumns: 18,
      empty: false,
    });
    expect(tensorLabel(input, false)).toBe("[2, 28, 7, 20]");
    expect(tensorLabel(input, true)).toBe("[B, Nq, S, T]");
  });
  it("retains scalar, vector, and zero-length cache semantics", () => {
    expect(tensorVolume(tensor([]))).toMatchObject({
      rows: 1,
      columns: 1,
      slices: 1,
      empty: false,
    });
    expect(tensorVolume(tensor([3584], ["H"]))).toMatchObject({
      rows: 1,
      columns: 3584,
      slices: 1,
    });
    expect(tensorVolume(tensor([2, 4, 0, 128]))).toMatchObject({
      rows: 0,
      columns: 128,
      slices: 8,
      gridRows: 0,
      empty: true,
    });
    expect(tensorVolume(tensor([0, 4, 16, 128]))).toMatchObject({
      slices: 0,
      empty: true,
    });
  });
  it("caps geometry independently of tensor dimensions without mutating the tensor", () => {
    const input = tensor([128, 256, 1048576, 262144]);
    const before = JSON.stringify(input),
      volume = tensorVolume(input);
    expect(volume.rows).toBe(1048576);
    expect(volume.columns).toBe(262144);
    expect(volume.slices).toBe(32768);
    expect(volume.gridColumns * volume.gridRows).toBeLessThanOrEqual(180);
    expect(volume.width).toBeLessThanOrEqual(5);
    expect(volume.height).toBeLessThanOrEqual(2.3);
    expect(volume.depth).toBeLessThanOrEqual(1.7);
    expect(JSON.stringify(input)).toBe(before);
  });
  it("preserves every gallery graph identity, branch and shape for both renderers", async () => {
    for (const entry of galleryModels) {
      const model = await loadGalleryModel(entry);
      for (const view of ["overview", "attention", "mlp"] as const) {
        const graph = buildGraph(
          model,
          { batch: 2, sequence: 1, past: 13, dtype: "bfloat16" },
          view,
          0,
        );
        const blocks = spatialBlocks(graph);
        expect(blocks.map((b) => b.id)).toEqual(graph.nodes.map((n) => n.id));
        for (const block of blocks) {
          expect(block.tensor).toEqual(
            block.node.data.outputs[0] ?? block.node.data.inputs[0],
          );
          expect(block.position.every(Number.isFinite)).toBe(true);
          expect(
            block.volume.width * block.volume.height * block.volume.depth,
          ).toBeGreaterThan(0);
        }
        for (const edge of graph.edges) {
          expect(blocks.some((b) => b.id === edge.source)).toBe(true);
          expect(blocks.some((b) => b.id === edge.target)).toBe(true);
        }
      }
    }
  });
});
