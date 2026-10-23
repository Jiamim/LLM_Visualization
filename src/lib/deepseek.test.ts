import { describe, expect, it } from "vitest";
import raw from "../data/deepseek-v4.1-flash.json";
import {
  normalizeConfig,
  memoryEstimates,
  parameterCounts,
  type Scenario,
} from "./model";
import {
  deepseekMemory,
  deepseekParameters,
  deepseekShapes,
  layerProfile,
  validateDeepseekScenario,
  visionShapes,
} from "./deepseek";
import { buildGraph, type View } from "./graph";
import { modelStages } from "./presentation";
const model = normalizeConfig(raw);
const prefill: Scenario = {
  batch: 2,
  sequence: 1024,
  past: 0,
  dtype: "bfloat16",
};
const decode: Scenario = {
  batch: 2,
  sequence: 1,
  past: 128,
  dtype: "bfloat16",
};
const change = (fields: Record<string, unknown>) => ({
  ...raw,
  text_config: { ...raw.text_config, ...fields },
});

describe("DeepSeek V4.1 configuration", () => {
  it("accepts full and text-only configurations without falling through to dense GQA", () => {
    expect(model.deepseek?.encoderLayers).toBe(20);
    expect(model.layers).toBe(40);
    const text = normalizeConfig(raw.text_config);
    expect(text.deepseek?.vision).toBeNull();
    expect(text.architecture).toBe("DeepseekV41TextModel");
    expect(parameterCounts(text).lmHead).toBe(0);
    expect(memoryEstimates(text, prefill).logits).toBe(0);
  });
  it.each([
    { num_hidden_layers: 39 },
    { num_key_value_heads: 8 },
    { num_experts_per_tok: 385 },
    { qk_rope_head_dim: 513 },
    { kv_source_layer_ids: [2, 8, 14] },
    { kv_source_layer_ids: [2, 8, 14, 20, 24] },
    { index_source_layer_ids: [2, 8, 20] },
    { kv_source_layer_ids: [2, 2, 14, 20] },
    { compress_ratios: Array(43).fill(1) },
    { engram_num_embeddings: [1] },
    { dspark_target_layer_ids: [] },
    { hc_mult: 0 },
    { o_groups: 7 },
    { candidate_source_layer_id: 24 },
    { moe_intermediate_size: 0 },
  ])("rejects incompatible config %j", (fields) =>
    expect(() => normalizeConfig(change(fields))).toThrow(),
  );
  it("rejects a mismatched V4 wrapper around V4.1 text config", () =>
    expect(() =>
      normalizeConfig({ ...raw, model_type: "deepseek_v4" }),
    ).toThrow());
  it("does not require is_encoder_decoder=true to recognize CED", () =>
    expect(model.deepseek).toBeDefined());
  it("accepts backbone-only compress_ratios with DSpark separately described", () =>
    expect(
      normalizeConfig(
        change({
          compress_ratios: raw.text_config.compress_ratios.slice(0, 40),
        }),
      ).deepseek?.ratios,
    ).toHaveLength(40));
});

describe("CED layer ownership", () => {
  it.each([
    [0, "Encoder", "SWA", null, null],
    [2, "Encoder", "Full", 2, 2],
    [7, "Encoder", "Reuse", 2, 2],
    [14, "Encoder", "Full", 14, 14],
    [19, "Encoder", "Reuse", 14, 14],
    [20, "Decoder", "Full", 20, 20],
    [23, "Decoder", "Reuse", 20, 20],
    [24, "Decoder", "Reindex", 20, 24],
    [39, "Decoder", "Reuse", 20, 36],
  ] as const)(
    "Layer %i has the expected owners",
    (i, partition, mode, kv, index) => {
      expect(layerProfile(model, i)).toMatchObject({
        partition,
        mode,
        kvSource: kv,
        indexSource: index,
      });
    },
  );
  it("deduplicates full, reindex, and reuse groups", () => {
    const modes = Array.from(
      { length: 40 },
      (_, i) => layerProfile(model, i).mode,
    );
    expect(modes.filter((x) => x === "SWA")).toHaveLength(2);
    expect(modes.filter((x) => x === "Full")).toHaveLength(4);
    expect(modes.filter((x) => x === "Reindex")).toHaveLength(4);
    expect(modes.filter((x) => x === "Reuse")).toHaveLength(30);
  });
  it("uses boundary candidates only in later decoder layers", () => {
    expect(layerProfile(model, 20).candidateSource).toBeNull();
    expect(layerProfile(model, 24).candidateSource).toBe(20);
  });
});

describe("DeepSeek dimensions and memory", () => {
  it("keeps the mHC axis and independent low-rank dimensions", () => {
    const sh = deepseekShapes(model, prefill, 2);
    expect(sh.stream).toEqual([2, 1024, 4, 5120]);
    expect(sh.qr).toEqual([2, 1024, 1280]);
    expect(sh.q).toEqual([2, 1024, 64, 512]);
    expect(sh.outputRank).toEqual([2, 1024, 8, 1024]);
  });
  it("floors incomplete compressor groups across decode steps", () => {
    const sh = deepseekShapes(model, decode, 2);
    expect(sh.compressed).toBe(64);
    expect(sh.newCompressed).toBe(0);
    expect(
      deepseekShapes(model, { ...decode, past: 129 }, 2).newCompressed,
    ).toBe(1);
    expect(
      deepseekShapes(model, { ...prefill, sequence: 1 }, 2).indices,
    ).toEqual([2, 1, 0]);
  });
  it("distinguishes decoder query length from full global KV under bounded replay", () => {
    const s = { ...prefill, cedExecution: "bounded" as const };
    expect(deepseekShapes(model, s, 19).queryLength).toBe(1024);
    expect(deepseekShapes(model, s, 20).queryLength).toBe(128);
    expect(deepseekShapes(model, s, 24).mainKV).toEqual([2, 1024, 512]);
    expect(deepseekShapes(model, s, 24).indices).toEqual([2, 128, 512]);
  });
  it("counts shared K/V once per owner and reproduces 890 bytes per token", () => {
    const mem = deepseekMemory(model, { ...prefill, batch: 1, sequence: 1024 });
    expect(mem.sources.map((x) => x.entries)).toEqual([512, 512, 512, 1024]);
    expect(mem.globalPacked).toBe(1024 * 890);
    expect(mem.globalLogical).toBe((512 * 3 + 1024) * (512 + 128) * 2);
    expect(mem.windowLogical).toBe(40 * 128 * 512 * 2);
    expect(mem.partialState).toBe(3 * 2 * 512 * 2 * 4);
    expect(mem.logits).toBe(129280 * 4);
  });
  it("keeps fixed FP4 storage independent from selected logical activation dtype", () => {
    expect(
      deepseekMemory(model, { ...prefill, dtype: "float32" }).globalPacked,
    ).toBe(deepseekMemory(model, prefill).globalPacked);
    expect(
      deepseekMemory(model, { ...prefill, dtype: "float32" }).globalLogical,
    ).toBe(2 * deepseekMemory(model, prefill).globalLogical);
  });
  it("rejects unsupported multi-token cached decode and insufficient image slots", () => {
    expect(() =>
      validateDeepseekScenario(model, { ...decode, sequence: 2 }),
    ).toThrow(/S=1/);
    expect(() =>
      validateDeepseekScenario(model, {
        ...prefill,
        sequence: 128,
        image: { count: 1, gridH: 42, gridW: 42 },
      }),
    ).toThrow(/212/);
    expect(() =>
      validateDeepseekScenario(model, {
        ...prefill,
        image: { count: 1, gridH: 96, gridW: 96 },
      }),
    ).toThrow(/max_image_tokens/);
  });
  it("models padded pixel-unshuffle and boundary / newline tokens", () => {
    const vs = visionShapes(model, {
      ...prefill,
      image: { count: 1, gridH: 43, gridW: 42 },
    });
    expect(vs.embedded).toEqual([1806, 1024]);
    expect(vs.downsampled).toEqual([210, 9216]);
    expect(vs.aligned).toEqual([210, 5120]);
    expect(vs.spanTokens).toBe(227);
  });
});

describe("DeepSeek parameters and graph", () => {
  it("separates the known backbone and Engram parameters", () => {
    const p = deepseekParameters(model);
    expect(p.backbone).toBe(551566180464);
    expect(p.engram).toBe(196928504320);
    expect(p.total).toBe(p.backbone + p.engram + p.vision + p.draft);
  });
  it.each([
    "overview",
    "encoder",
    "decoder",
    "layer",
    "attention",
    "mlp",
    "cache",
    "vision",
    "engram",
    "dspark",
  ] as View[])(
    "%s is a DAG with finite shapes and usable stage anchors",
    (view) => {
      const graph = buildGraph(model, prefill, view, 20);
      const order = new Map(graph.nodes.map((n, i) => [n.id, i]));
      expect(order.size).toBe(graph.nodes.length);
      for (const e of graph.edges) {
        expect(order.get(e.source)).toBeDefined();
        expect(order.get(e.target)).toBeDefined();
        expect(order.get(e.source)!).toBeLessThan(order.get(e.target)!);
      }
      for (const n of graph.nodes)
        for (const t of [...n.data.inputs, ...n.data.outputs])
          expect(t.shape.every((x) => Number.isSafeInteger(x) && x >= 0)).toBe(
            true,
          );
      for (const stage of modelStages(model, view))
        expect(graph.nodes.some((n) => stage.nodes.includes(n.id))).toBe(true);
    },
  );
  it.each([0, 2, 3, 20, 24, 39])(
    "Layer %i sums its actual attention and MoE weights",
    (index) => {
      const p = deepseekParameters(model);
      expect(
        buildGraph(model, prefill, "attention", index).nodes.reduce(
          (sum, n) => sum + n.data.params,
          0,
        ),
      ).toBe(p.layers[index].attention);
      expect(
        buildGraph(model, prefill, "mlp", index).nodes.reduce(
          (sum, n) => sum + n.data.params,
          0,
        ),
      ).toBe(p.layers[index].moe);
    },
  );
  it("marks cache reuse edges and decoder boundary sources", () => {
    const graph = buildGraph(model, prefill, "cache", 24);
    expect(graph.edges.filter((e) => e.source === "source-20")).toHaveLength(5);
    expect(graph.edges.every((e) => e.data?.shared === true)).toBe(true);
    expect(
      buildGraph(model, prefill, "attention", 24).nodes.find(
        (n) => n.id === "global-kv",
      )?.data.params,
    ).toBe(0);
  });
});

describe("verified reference parameter totals", () => {
  it("matches official meta-instantiated module totals excluding quantization scales", () => {
    const counts = deepseekParameters(model);
    expect(counts.vision).toBe(485268480);
    expect(counts.draft).toBe(14225362530);
    expect(counts.total).toBe(763205315794);
  });
  it("exposes candidate pool direction for Full and Reindex without recomputing it in Reuse", () => {
    const full = buildGraph(model, prefill, "attention", 20),
      reindex = buildGraph(model, prefill, "attention", 24),
      reuse = buildGraph(model, prefill, "attention", 25);
    expect(
      full.edges.some(
        (e) => e.source === "indexer" && e.target === "candidate-pool",
      ),
    ).toBe(true);
    expect(
      reindex.edges.some(
        (e) => e.source === "candidate-pool" && e.target === "indexer",
      ),
    ).toBe(true);
    expect(reuse.nodes.some((n) => n.id === "candidate-pool")).toBe(false);
  });
});

it("keeps inactive vision and optional draft branches out of the primary playback path", () => {
  const graph = buildGraph(model, prefill, "overview", 0);
  expect(graph.nodes.find((n) => n.id === "vision")?.data.inactive).toBe(true);
  expect(graph.nodes.find((n) => n.id === "dspark")?.data.inactive).toBe(true);
  const images = buildGraph(
    model,
    { ...prefill, image: { count: 1, gridH: 42, gridW: 42 } },
    "overview",
    0,
  );
  expect(images.nodes.find((n) => n.id === "vision")?.data.inactive).toBe(
    false,
  );
});
