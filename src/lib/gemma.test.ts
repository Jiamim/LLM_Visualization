import { describe, expect, it } from "vitest";
import e2b from "../data/gemma4-e2b.json";
import e4b from "../data/gemma4-e4b.json";
import reference2 from "../../docs/gemma/reference-e2b.json";
import reference4 from "../../docs/gemma/reference-e4b.json";
import {
  normalizeConfig,
  parameterCounts,
  memoryEstimates,
  hasLMHead,
  type Scenario,
} from "./model";
import {
  gemmaLayer,
  gemmaShapes,
  gemmaParameters,
  gemmaMemory,
  gemmaVisionShapes,
  gemmaAudioShapes,
  validateGemmaScenario,
} from "./gemma";
import { buildGraph, type View } from "./graph";
import { modelStages } from "./presentation";
const m = normalizeConfig(e2b),
  larger = normalizeConfig(e4b);
const prefill: Scenario = {
  batch: 2,
  sequence: 1024,
  past: 0,
  dtype: "bfloat16",
};
const change = (fields: Record<string, unknown>) => ({
  ...e2b,
  text_config: { ...e2b.text_config, ...fields },
});

describe("Gemma E configuration", () => {
  it("accepts nested conditional-generation and text-only causal LM configs", () => {
    expect(m.gemma?.shareStart).toBe(15);
    expect(hasLMHead(m)).toBe(true);
    const text = normalizeConfig(e2b.text_config);
    expect(text.gemma?.vision).toBeNull();
    expect(text.gemma?.audio).toBeNull();
    expect(hasLMHead(text)).toBe(true);
    const base = normalizeConfig({
      ...e2b.text_config,
      architectures: ["Gemma4TextModel"],
    });
    expect(hasLMHead(base)).toBe(false);
    expect(parameterCounts(base).lmHead).toBe(0);
    expect(memoryEstimates(base, prefill).logits).toBe(0);
  });
  it.each([
    { num_kv_shared_layers: 35 },
    { num_kv_shared_layers: 34 },
    { num_hidden_layers: 0 },
    { num_attention_heads: 7, num_key_value_heads: 2 },
    { head_dim: 255 },
    { layer_types: ["full_attention"] },
    { enable_moe_block: true },
    { attention_k_eq_v: true },
    { use_bidirectional_attention: "all" },
    { hidden_size_per_layer_input: 0 },
    { hidden_activation: "silu" },
    { per_layer_config: { 0: { head_dim: 32 } } },
    { tie_word_embeddings: false },
  ])("rejects unsupported or inconsistent configuration %j", (fields) =>
    expect(() => normalizeConfig(change(fields))).toThrow(),
  );
  it("does not confuse Gemma 3n with Gemma 4", () =>
    expect(() => normalizeConfig({ ...e2b, model_type: "gemma3n" })).toThrow(
      /暂不支持/,
    ));
  it("rejects invalid nested audio and vision metadata", () => {
    expect(() =>
      normalizeConfig({
        ...e2b,
        audio_config: {
          ...e2b.audio_config,
          subsampling_conv_channels: [3, 1],
        },
      }),
    ).toThrow();
    expect(() =>
      normalizeConfig({
        ...e2b,
        vision_config: { ...e2b.vision_config, head_dim: 63 },
      }),
    ).toThrow();
  });
});

describe("Gemma layer-specific rules", () => {
  it("distinguishes local and global projection dimensions", () => {
    expect(gemmaShapes(m, prefill, 0).q).toEqual([2, 8, 1024, 256]);
    expect(gemmaShapes(m, prefill, 4).q).toEqual([2, 8, 1024, 512]);
    expect(gemmaShapes(m, prefill, 4).qProjection).toEqual([2, 1024, 4096]);
  });
  it("E2B shares two owners and doubles only shared-layer MLP width", () => {
    expect(gemmaLayer(m, 13)).toMatchObject({
      shared: false,
      source: true,
      headDim: 256,
      intermediate: 6144,
    });
    expect(gemmaLayer(m, 14)).toMatchObject({
      shared: false,
      source: true,
      headDim: 512,
    });
    expect(gemmaLayer(m, 15)).toMatchObject({
      shared: true,
      owner: 13,
      intermediate: 12288,
    });
    expect(gemmaLayer(m, 19)).toMatchObject({
      shared: true,
      owner: 14,
      intermediate: 12288,
    });
  });
  it("E4B derives its different sharing boundary and standard-width MLP", () => {
    expect(larger.gemma?.shareStart).toBe(24);
    expect(gemmaLayer(larger, 24)).toMatchObject({
      owner: 22,
      intermediate: 10240,
    });
    expect(gemmaLayer(larger, 29).owner).toBe(23);
    expect(larger.kvHeads).toBe(2);
  });
  it("PLE has a distinct packed layer axis", () => {
    expect(gemmaShapes(m, prefill, 20).ple).toEqual([2, 1024, 35, 256]);
    expect(gemmaShapes(m, prefill, 20).pleSlice).toEqual([2, 1024, 256]);
  });
  it("distinguishes attention-visible keys from retained sliding cache", () => {
    const local = gemmaShapes(m, prefill, 0);
    expect(local.keys).toBe(1024);
    expect(local.retained).toBe(511);
    const decoded = gemmaShapes(m, { ...prefill, sequence: 7, past: 1024 }, 15);
    expect(decoded.keys).toBe(518);
    expect(decoded.retained).toBe(511);
    expect(
      gemmaShapes(m, { ...prefill, sequence: 7, past: 1024 }, 19).keys,
    ).toBe(1031);
  });
  it("counts each independent KV owner once", () => {
    const mem = gemmaMemory(m, prefill);
    expect(mem.owners).toHaveLength(15);
    expect(mem.kvCache).toBe(2 * 2 * 1 * 2 * (12 * 511 * 256 + 3 * 1024 * 512));
    expect(mem.pleActivations).toBe(2 * 1024 * 35 * 256 * 2);
    expect(gemmaMemory(m, { ...prefill, dtype: "float32" }).kvCache).toBe(
      2 * mem.kvCache,
    );
  });
});

describe("Gemma reference parameter totals", () => {
  it.each([
    [m, reference2],
    [larger, reference4],
  ] as const)(
    "matches all Transformers meta-instantiated components",
    (model, reference) => {
      const p = gemmaParameters(model);
      expect(p.total).toBe(reference.total);
      expect(p.text).toBe(reference.text);
      expect(p.vision).toBe(reference.vision);
      expect(p.audio).toBe(reference.audio);
      expect(p.pleTable).toBe(reference.ple_table);
      expect(p.pleProjection).toBe(reference.ple_projection);
      p.layers.forEach((l, i) =>
        expect({ attention: l.attention, mlp: l.mlp, all: l.total }).toEqual(
          reference.layers[i],
        ),
      );
    },
  );
  it("does not count tied head twice, nor non-trainable layer_scalar or clipping buffers", () => {
    const untied = normalizeConfig({
      ...e2b,
      tie_word_embeddings: false,
      text_config: { ...e2b.text_config, tie_word_embeddings: false },
    });
    expect(gemmaParameters(untied).total - gemmaParameters(m).total).toBe(
      m.vocab * m.hidden,
    );
    expect(gemmaParameters(m).layers[15].kProj).toBe(0);
  });
});

describe("Gemma media", () => {
  it("uses averaging pooling, keeping channel width unchanged", () => {
    const s = { ...prefill, image: { count: 1, gridH: 42, gridW: 60 } };
    validateGemmaScenario(m, s);
    const sh = gemmaVisionShapes(m, s);
    expect(sh.pixels).toEqual([2, 2520, 768]);
    expect(sh.pooled).toEqual([2, 280, 768]);
    expect(sh.projected).toEqual([2, 280, 1536]);
  });
  it("rounds each stride-2 audio convolution upward and models block padding", () => {
    const sh = gemmaAudioShapes(m, {
      ...prefill,
      audio: { enabled: true, frames: 129 },
    });
    expect(sh.conv1).toEqual([2, 128, 65, 64]);
    expect(sh.conv2).toEqual([2, 32, 33, 32]);
    expect(sh.hidden).toEqual([2, 33, 1024]);
    expect(sh.scores).toEqual([2, 8, 3, 12, 24]);
    expect(sh.softTokens).toEqual([2, 33, 1536]);
  });
  it("validates merged token budget and prepared image grids", () => {
    expect(() =>
      validateGemmaScenario(m, {
        ...prefill,
        sequence: 281,
        image: { count: 1, gridH: 42, gridW: 60 },
      }),
    ).toThrow(/282/);
    expect(() =>
      validateGemmaScenario(m, {
        ...prefill,
        image: { count: 1, gridH: 43, gridW: 60 },
      }),
    ).toThrow(/正整数倍/);
    expect(() =>
      validateGemmaScenario(m, {
        ...prefill,
        sequence: 315,
        image: { count: 1, gridH: 42, gridW: 60 },
        audio: { enabled: true, frames: 128 },
      }),
    ).toThrow(/316/);
    expect(() =>
      validateGemmaScenario(m, {
        ...prefill,
        sequence: 1,
        past: 1024,
        image: { count: 1, gridH: 42, gridW: 60 },
      }),
    ).not.toThrow();
  });
});

describe("Gemma graph topology and semantics", () => {
  it.each([
    "overview",
    "decoder",
    "layer",
    "attention",
    "mlp",
    "ple",
    "cache",
    "vision",
    "audio",
  ] as View[])("%s is acyclic and has usable stage anchors", (view) => {
    const graph = buildGraph(m, prefill, view, 19),
      order = new Map(graph.nodes.map((n, i) => [n.id, i]));
    expect(order.size).toBe(graph.nodes.length);
    for (const e of graph.edges) {
      expect(order.has(e.source)).toBe(true);
      expect(order.has(e.target)).toBe(true);
      expect(order.get(e.source)!).toBeLessThan(order.get(e.target)!);
    }
    for (const n of graph.nodes)
      for (const t of [...n.data.inputs, ...n.data.outputs]) {
        expect(t.shape.length).toBe(t.symbols.length);
        expect(t.shape.every((n) => Number.isSafeInteger(n) && n >= 0)).toBe(
          true,
        );
      }
    for (const stage of modelStages(m, view))
      expect(graph.nodes.some((n) => stage.nodes.includes(n.id))).toBe(true);
  });
  it.each([0, 4, 13, 14, 15, 19, 34])(
    "Layer %i sums real parameters from the subgraphs",
    (index) => {
      const p = gemmaParameters(m).layers[index];
      for (const [view, expected] of [
        ["attention", p.attention],
        ["mlp", p.mlp],
        ["layer", p.total],
      ] as const)
        expect(
          buildGraph(m, prefill, view, index).nodes.reduce(
            (sum, n) => sum + n.data.params,
            0,
          ),
        ).toBe(expected);
    },
  );
  it("audio and vision grouped graph nodes sum to their model-wide component counts", () => {
    const p = gemmaParameters(m);
    expect(
      buildGraph(m, prefill, "audio", 0).nodes.reduce(
        (sum, n) => sum + n.data.params,
        0,
      ),
    ).toBe(p.audio);
    expect(
      buildGraph(m, prefill, "vision", 0).nodes.reduce(
        (sum, n) => sum + n.data.params,
        0,
      ),
    ).toBe(p.vision);
  });
  it("marks disabled modalities inactive and omits K/V projection from shared layers", () => {
    expect(
      buildGraph(m, prefill, "vision", 0).nodes.every((n) => n.data.inactive),
    ).toBe(true);
    const graph = buildGraph(m, prefill, "attention", 19);
    expect(graph.nodes.some((n) => n.id === "kv-project")).toBe(false);
    expect(graph.nodes.find((n) => n.id === "shared-kv")?.data.params).toBe(0);
  });
  it("makes the pooled float32 to flattened working-dtype interface explicit", () => {
    const graph = buildGraph(
      m,
      { ...prefill, image: { count: 1, gridH: 42, gridW: 60 } },
      "vision",
      0,
    );
    const valid = graph.nodes.find((n) => n.id === "vision-valid")!.data;
    expect(valid.inputs[0].dtype).toBe("float32");
    expect(valid.outputs[0].shape).toEqual([560, 768]);
    expect(valid.outputs[0].dtype).toBe("bfloat16");
    expect(
      graph.nodes.find((n) => n.id === "vision-project")!.data.outputs[0].shape,
    ).toEqual([560, 1536]);
  });
});
