import { describe, it, expect } from "vitest";
import { galleryModels, loadGalleryModel } from "../data/gallery";
import { buildGraph, type View } from "./graph";
import { parameterCounts, memoryEstimates, type Scenario } from "./model";
const scenario: Scenario = {
  batch: 1,
  sequence: 128,
  past: 0,
  dtype: "bfloat16",
};
describe("Gallery catalog snapshot", () => {
  it("has every one of the 111 distinct source cards", () => {
    expect(galleryModels).toHaveLength(111);
    expect(new Set(galleryModels.map((m) => m.id)).size).toBe(111);
  });
  it.each(galleryModels)(
    "$id loads and builds all language-backbone views",
    async (entry) => {
      const m = await loadGalleryModel(entry);
      expect(m.hidden).toBeGreaterThan(0);
      expect(m.layers).toBeGreaterThan(0);
      const params = parameterCounts(m),
        memory = memoryEstimates(m, scenario);
      expect(Number.isFinite(params.total)).toBe(true);
      expect(Number.isFinite(memory.kvCache)).toBe(true);
      const views: View[] = m.catalog
        ? [
            "overview",
            "decoder",
            "layer",
            "attention",
            "mlp",
            "cache",
            "auxiliary",
          ]
        : ["overview", "layer", "attention", "mlp"];
      for (const view of views) {
        const graph = buildGraph(m, scenario, view, 0),
          order = new Map(graph.nodes.map((n, i) => [n.id, i]));
        expect(order.size).toBeGreaterThan(0);
        for (const edge of graph.edges) {
          expect(order.has(edge.source)).toBe(true);
          expect(order.has(edge.target)).toBe(true);
        }
        for (const node of graph.nodes)
          for (const tensor of [...node.data.inputs, ...node.data.outputs]) {
            expect(tensor.shape.length).toBe(tensor.symbols.length);
            expect(
              tensor.shape.every((n) => Number.isSafeInteger(n) && n >= 0),
            ).toBe(true);
          }
      }
    },
  );
});

it("uses actual hybrid schedules, recurrent caches and layer-specific widths", async () => {
  const load = (id: string) =>
    loadGalleryModel(galleryModels.find((e) => e.id === id)!);
  const nemotron = await load("nemotron-3-super-120b-a12b");
  expect(
    nemotron.catalog!.profiles.filter((p) => p.kind === "mamba"),
  ).toHaveLength(40);
  expect(
    nemotron.catalog!.profiles.filter((p) => p.kind === "attention"),
  ).toHaveLength(8);
  expect(
    nemotron.catalog!.profiles.filter((p) => p.kind === "ffn" && p.moe),
  ).toHaveLength(40);
  const next = await load("qwen3-next-80b-a3b");
  expect(next.catalog!.profiles.filter((p) => p.kind === "delta")).toHaveLength(
    36,
  );
  const kimi = await load("kimi-linear-48b-a3b");
  expect(kimi.catalog!.profiles.filter((p) => p.kind === "kda")).toHaveLength(
    20,
  );
  expect(kimi.catalog!.profiles[3].kind).toBe("mla");
  const mimo = await load("xiaomi-mimo-v2-5-310b");
  expect(mimo.catalog!.profiles[0]).toMatchObject({
    headDim: 192,
    valueDim: 128,
    kvHeads: 4,
  });
  expect(mimo.catalog!.profiles[1]).toMatchObject({ kvHeads: 8, window: 128 });
  const laguna = await load("laguna-xs-2");
  expect(laguna.catalog!.profiles[0].heads).toBe(48);
  expect(laguna.catalog!.profiles[1].heads).toBe(64);
  const motif = await load("motif-3-beta");
  expect(motif.catalog!.profiles[0].kind).toBe("gdla");
  const zaya = await load("zaya1-8b");
  expect(zaya.catalog!.profiles[0].kind).toBe("cca");
});

it.each(galleryModels)(
  "$id supports every distinct mixer/FFN variant during decode",
  async (entry) => {
    const m = await loadGalleryModel(entry);
    if (!m.catalog) return;
    const unique = new Map(
      m.catalog.profiles.map((p) => [
        `${p.kind}-${p.window}-${p.moe}-${p.heads}`,
        p.index,
      ]),
    );
    for (const index of unique.values())
      for (const view of ["layer", "attention", "mlp", "cache"] as View[]) {
        const graph = buildGraph(
            m,
            { ...scenario, sequence: 1, past: 1024 },
            view,
            index,
          ),
          positions = new Map(graph.nodes.map((n, i) => [n.id, i]));
        for (const e of graph.edges)
          expect(positions.get(e.source)!).toBeLessThan(
            positions.get(e.target)!,
          );
        for (const n of graph.nodes)
          for (const t of [...n.data.inputs, ...n.data.outputs])
            expect(
              t.shape.every((v) => Number.isSafeInteger(v) && v >= 0),
            ).toBe(true);
      }
  },
);

it('caps sparse block indices by both available blocks and configured Top-K blocks',async()=>{
 const model=await loadGalleryModel(galleryModels.find(e=>e.id==='minimax-m3-428b')!);
 const graph=buildGraph(model,{...scenario,sequence:1,past:8191},'attention',3);
 expect(graph.nodes.find(n=>n.id==='sparse-indexer')!.data.outputs[0].shape).toEqual([1,1,16]);
});
