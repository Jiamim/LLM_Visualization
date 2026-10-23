import { describe, expect, it } from "vitest";
import { cacheReport, groupCacheRows } from "./cache";
import {
  loadLibraryModel,
  modelLibrary,
  selectionForPreset,
} from "../data/model-library";
import { memoryEstimates, normalizeConfig, type Scenario } from "./model";
import qwen25 from "../../tests/fixtures/qwen2.5-7b.json";

const scenario: Scenario = {
  batch: 2,
  sequence: 7,
  past: 13,
  dtype: "bfloat16",
};
const load = async (id: string) =>
  (await loadLibraryModel({ entryId: id, variantId: "gallery" })).model;
describe("cache report", () => {
  it("uses KV heads, both K and V, and all independent layers for GQA", () => {
    const model = normalizeConfig(qwen25),
      r = cacheReport(model, scenario);
    expect(r.kvBytes).toBe(2 * 28 * 2 * 4 * 20 * 128 * 2);
    expect(r.recurrentBytes).toBe(0);
    expect(r.auxiliaryBytes).toBe(0);
    expect(r.perSequenceBytes).toBe(r.totalBytes / 2);
    expect(r.addedBytes).toBe(2 * 28 * 2 * 4 * 7 * 128 * 2);
    expect(r.nextTokenBytes).toBe(2 * 28 * 2 * 4 * 128 * 2);
    expect(r.rows[0].tensors[0].shape).toEqual([2, 4, 20, 128]);
    expect(
      cacheReport(model, { ...scenario, dtype: "float32" }).totalBytes,
    ).toBe(2 * r.totalBytes);
  });
  it("does not infer head_dim from hidden_size for Qwen3", async () => {
    const { model } = await loadLibraryModel(selectionForPreset("qwen3"));
    expect(cacheReport(model, scenario).kvBytes).toBe(
      2 * 28 * 2 * 8 * 20 * 128 * 2,
    );
  });
  it("preserves each Gemma owner once and stops local growth at W-1", async () => {
    const { model } = await loadLibraryModel(selectionForPreset("gemma4-e2b"));
    const r = cacheReport(model, { ...scenario, sequence: 1024, past: 0 });
    expect(r.kvBytes).toBe(2 * 2 * 1 * 2 * (12 * 511 * 256 + 3 * 1024 * 512));
    expect(r.nextKvBytes).toBe(2 * 2 * 1 * 3 * 512 * 2);
    expect(r.rows).toHaveLength(15);
    expect(r.rows.flatMap((row) => row.consumers)).toHaveLength(35);
    expect(groupCacheRows(r.rows).reduce((s, row) => s + row.bytes, 0)).toBe(
      r.totalBytes,
    );
  });
  it("handles MLA latent and rotary key without duplicating KV or expert counts", async () => {
    const model = await load("deepseek-v3"),
      r = cacheReport(model, scenario);
    expect(r.kvBytes).toBe(model.layers * 2 * 20 * (512 + 64) * 2);
    expect(
      r.rows
        .flatMap((row) => row.tensors)
        .some((t) => t.name === "compressed_KV"),
    ).toBe(true);
  });
  it("separates hybrid KV from fixed FP32 recurrent state", async () => {
    const model = await load("qwen3-next-80b-a3b"),
      r = cacheReport(model, { ...scenario, sequence: 1, past: 1023 });
    expect(r.kvBytes).toBe(12 * 2 * 1024 * 2 * 2 * 256 * 2);
    expect(r.recurrentBytes).toBeGreaterThan(0);
    const longer = cacheReport(model, { ...scenario, sequence: 1, past: 2047 });
    expect(longer.recurrentBytes).toBe(r.recurrentBytes);
    expect(longer.kvBytes).toBe(2 * r.kvBytes);
    const fp32 = cacheReport(model, {
      ...scenario,
      sequence: 1,
      past: 1023,
      dtype: "float32",
    });
    expect(fp32.kvBytes).toBe(2 * r.kvBytes);
    expect(fp32.recurrentBytes).toBeLessThan(2 * r.recurrentBytes);
  });
  it("reports pure recurrent models as zero traditional KV with nonzero state", async () => {
    const model = await load("xlstm-7b"),
      r = cacheReport(model, scenario);
    expect(r.kvBytes).toBe(0);
    expect(r.recurrentBytes).toBeGreaterThan(0);
    expect(r.nextTokenBytes).toBe(0);
  });
  it("caps hybrid attention windows under the registered logical policy", async () => {
    const model = await load("gpt-oss-20b");
    const a = cacheReport(model, { ...scenario, sequence: 1, past: 1023 }),
      b = cacheReport(model, { ...scenario, sequence: 1, past: 2047 });
    const global = model.catalog!.profiles.filter((p) => !p.window);
    const increase = global.reduce(
      (sum, p) => sum + 2 * 1024 * p.kvHeads * (p.headDim + p.valueDim) * 2,
      0,
    );
    expect(b.kvBytes - a.kvBytes).toBe(increase);
  });
  it("deduplicates CED KV sources, keeps index and compression separate, and handles boundaries", async () => {
    const { model } = await loadLibraryModel(
      selectionForPreset("deepseek-v41"),
    );
    const r = cacheReport(model, {
      ...scenario,
      batch: 1,
      sequence: 1024,
      past: 0,
    });
    expect(r.kvBytes).toBe((512 * 3 + 1024) * 512 * 2 + 40 * 128 * 512 * 2);
    expect(r.auxiliaryBytes).toBe(
      (512 * 3 + 1024) * 128 * 2 + 3 * 2 * 512 * 2 * 4,
    );
    expect(r.packedGlobalBytes).toBe(1024 * 890);
    expect(r.partial).toBe(true);
    expect(r.nextTokenBytes).toBe((512 + 128) * 2);
    expect(
      cacheReport(model, { ...scenario, batch: 1, sequence: 1, past: 1024 })
        .nextTokenBytes,
    ).toBe(4 * (512 + 128) * 2);
    expect(
      cacheReport(model, {
        ...scenario,
        batch: 1,
        sequence: 1024,
        past: 0,
        dtype: "float32",
      }).packedGlobalBytes,
    ).toBe(r.packedGlobalBytes);
  });
  it("distinguishes unknown caches and disabled defaults from a true zero", async () => {
    const model = structuredClone(await load("gpt-oss-20b"));
    model.catalog!.profiles[0].kind = "custom";
    model.catalog!.text.use_cache = false;
    const r = cacheReport(model, scenario);
    expect(r.unknownLayers).toEqual([0]);
    expect(r.partial).toBe(true);
    expect(r.cacheDisabledByDefault).toBe(true);
  });
  it("counts initialization of fixed recurrent state when starting from an empty cache", async () => {
    const model = await load("xlstm-7b"),
      r = cacheReport(model, { ...scenario, past: 0 });
    expect(r.addedBytes).toBe(r.totalBytes);
    expect(r.nextTokenBytes).toBe(0);
  });
  it.each(modelLibrary)(
    "$id provides a finite consistent cache report for all configuration variants",
    async (entry) => {
      for (const variant of entry.variants) {
        const { model } = await loadLibraryModel({
          entryId: entry.id,
          variantId: variant.id,
        });
        const s = { ...scenario, sequence: 1, past: 1023 };
        const r = cacheReport(model, s);
        expect(Number.isFinite(r.totalBytes)).toBe(true);
        expect(r.totalBytes).toBeGreaterThanOrEqual(0);
        expect(r.kvBytes + r.recurrentBytes + r.auxiliaryBytes).toBe(
          r.totalBytes,
        );
        expect(r.nextTokenBytes).toBeGreaterThanOrEqual(0);
        expect(r.totalBytes).toBe(memoryEstimates(model, s).kvCache);
        expect(
          groupCacheRows(r.rows).reduce((s, row) => s + row.bytes, 0),
        ).toBe(r.totalBytes);
        const owners = r.rows.flatMap((row) =>
          row.layers.map((layer) => `${row.kind}:${row.name}:${layer}`),
        );
        expect(new Set(owners).size).toBe(owners.length);
      }
    },
  );
});
