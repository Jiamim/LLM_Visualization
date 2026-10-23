import { describe, expect, it } from "vitest";
import { presets } from "../data/presets";
import qwen25 from "../../tests/fixtures/qwen2.5-7b.json";
import mistral from "../../tests/fixtures/mistral-7b.json";
import qwen3 from "../data/qwen3-0.6b.json";
import llama from "../data/llama3.2-1b.json";
import { buildGraph, type View } from "./graph";
import { configURL } from "./huggingface";
import {
  memoryEstimates,
  normalizeConfig,
  parameterCounts,
  parseConfig,
  tensorShapes,
  validateScenario,
} from "./model";

const qwen = normalizeConfig(qwen25);
const scenario = {
  batch: 2,
  sequence: 7,
  past: 13,
  dtype: "bfloat16" as const,
};

describe("config validation", () => {
  it.each(presets)("accepts the real $name config", (preset) => {
    expect(normalizeConfig(preset.raw).layers).toBeGreaterThan(0);
  });
  it.each(["null", "[]", "{", '"llama"'])(
    "rejects invalid JSON config %s",
    (json) => {
      expect(() => parseConfig(json)).toThrow();
    },
  );
  it.each([
    { model_type: "bert" },
    { model_type: "qwen3_moe" },
    { num_attention_heads: 27 },
    { head_dim: 7 },
    { num_hidden_layers: 0 },
    { hidden_size: -1 },
    { head_dim: 1.5 },
    { intermediate_size: undefined },
    { tie_word_embeddings: "false" },
    { architectures: ["Qwen2ForSequenceClassification"] },
    { auto_map: { AutoModel: "custom.Model" } },
    { num_experts: 8 },
    { layer_types: ["full_attention"] },
    { vocab_size: Number.MAX_SAFE_INTEGER },
  ])("rejects incompatible or invalid fields %j", (fields) => {
    expect(() => normalizeConfig({ ...qwen.raw, ...fields })).toThrow();
  });
  it("does not silently interpret a nested multimodal config as the full model", () => {
    expect(() =>
      normalizeConfig({ model_type: "qwen2_vl", text_config: qwen.raw }),
    ).toThrow(/多模态/);
  });
  it("records quantization as an estimate limitation", () => {
    const m = normalizeConfig({
      ...qwen.raw,
      quantization_config: { quant_method: "gptq" },
    });
    expect(m.warnings.join()).toContain("量化");
  });
});

describe("dimension inference", () => {
  it("handles GQA, batch, non-square decode attention, and past KV cache", () => {
    const shapes = tensorShapes(qwen, scenario);
    expect(shapes.hidden).toEqual([2, 7, 3584]);
    expect(shapes.q).toEqual([2, 28, 7, 128]);
    expect(shapes.kvNew).toEqual([2, 4, 7, 128]);
    expect(shapes.kv).toEqual([2, 4, 20, 128]);
    expect(shapes.repeatedKV).toEqual([2, 28, 20, 128]);
    expect(shapes.scores).toEqual([2, 28, 7, 20]);
    expect(shapes.logits).toEqual([2, 7, 152064]);
  });
  it("honors explicit Qwen3 head_dim instead of hidden_size / heads", () => {
    const m = normalizeConfig(qwen3),
      shapes = tensorShapes(m, scenario);
    expect(m.hidden / m.heads).toBe(64);
    expect(m.headDim).toBe(128);
    expect(shapes.qProjection).toEqual([2, 7, 2048]);
    expect(shapes.hidden).toEqual([2, 7, 1024]);
  });
  it("handles MHA and multi-query attention", () => {
    const mha = normalizeConfig({ ...qwen.raw, num_key_value_heads: 28 });
    const mqa = normalizeConfig({ ...qwen.raw, num_key_value_heads: 1 });
    expect(tensorShapes(mha, scenario).kv).toEqual([2, 28, 20, 128]);
    expect(tensorShapes(mqa, scenario).kv).toEqual([2, 1, 20, 128]);
  });
  it.each([
    { batch: 0 },
    { sequence: 0 },
    { past: -1 },
    { sequence: 1.5 },
    { past: Infinity },
    { sequence: 10_000_001 },
  ])("rejects invalid scenario %j", (change) => {
    expect(() => validateScenario({ ...scenario, ...change })).toThrow();
  });
});

describe("parameter and memory estimates", () => {
  it("accounts for Qwen2 QKV biases in its known 7B parameter count", () => {
    expect(parameterCounts(qwen).total).toBe(7_615_616_512);
  });
  it("accounts for Qwen3 per-head RMSNorm and tied embeddings", () => {
    const m = normalizeConfig(qwen3);
    expect(parameterCounts(m).total).toBe(596_049_920);
    const untied = normalizeConfig({ ...m.raw, tie_word_embeddings: false });
    expect(parameterCounts(untied).total - parameterCounts(m).total).toBe(
      m.vocab * m.hidden,
    );
  });
  it("matches Llama 3.2 1B and Mistral 7B reference counts", () => {
    expect(parameterCounts(normalizeConfig(llama)).total).toBe(
      1_235_814_400,
    );
    expect(parameterCounts(normalizeConfig(mistral)).total).toBe(
      7_241_732_096,
    );
  });
  it("omits the task head for the base model", () => {
    const base = normalizeConfig({
      ...qwen.raw,
      architectures: ["Qwen2Model"],
    });
    expect(parameterCounts(qwen).total - parameterCounts(base).total).toBe(
      qwen.vocab * qwen.hidden,
    );
    expect(memoryEstimates(base, scenario).logits).toBe(0);
    expect(
      buildGraph(base, scenario, "overview", 0).nodes.some(
        (n) => n.id === "lm-head",
      ),
    ).toBe(false);
  });
  it("includes both K and V across every layer and scales precision", () => {
    const mem = memoryEstimates(qwen, scenario);
    expect(mem.kvCache).toBe(2 * 28 * 2 * 4 * 20 * 128 * 2);
    expect(
      memoryEstimates(qwen, { ...scenario, dtype: "float32" }).kvCache,
    ).toBe(2 * mem.kvCache);
  });
});

describe("graph topology and data flow", () => {
  it.each(["overview", "layer", "attention", "mlp"] as View[])(
    "%s is connected, acyclic and ordered for playback",
    (view) => {
      const graph = buildGraph(qwen, scenario, view, 3);
      const order = new Map(graph.nodes.map((node, index) => [node.id, index]));
      expect(order.size).toBe(graph.nodes.length);
      for (const edge of graph.edges) {
        expect(order.has(edge.source)).toBe(true);
        expect(order.has(edge.target)).toBe(true);
        expect(order.get(edge.source)!).toBeLessThan(order.get(edge.target)!);
        expect(edge.data?.tensor).toBeDefined();
      }
      for (const node of graph.nodes.slice(1))
        expect(graph.edges.some((edge) => edge.target === node.id)).toBe(true);
    },
  );
  it("preserves both residual connections", () => {
    const graph = buildGraph(qwen, scenario, "layer", 0);
    expect(
      graph.edges
        .filter((edge) => edge.data?.residual)
        .map((edge) => [edge.source, edge.target]),
    ).toEqual([
      ["layer-input", "attention-residual"],
      ["attention-residual", "mlp-residual"],
    ]);
  });
  it("aggregates every attention and MLP parameter exactly once", () => {
    for (const preset of presets.filter(
      (p) => p.id !== "deepseek-v41" && !p.id.startsWith("gemma4"),
    )) {
      const model = normalizeConfig(preset.raw),
        counts = parameterCounts(model);
      expect(
        buildGraph(model, scenario, "attention", 0).nodes.reduce(
          (sum, node) => sum + node.data.params,
          0,
        ),
      ).toBe(counts.attention);
      expect(
        buildGraph(model, scenario, "mlp", 0).nodes.reduce(
          (sum, node) => sum + node.data.params,
          0,
        ),
      ).toBe(counts.mlp);
    }
  });
  it("shows Qwen3 QK normalization and the output projection dimensions", () => {
    const m = normalizeConfig(qwen3);
    const graph = buildGraph(m, scenario, "attention", 5);
    expect(graph.nodes.find((n) => n.id === "q-reshape")?.data.params).toBe(
      128,
    );
    expect(
      graph.nodes.find((n) => n.id === "o-proj")?.data.inputs[0].shape,
    ).toEqual([2, 7, 2048]);
    expect(
      graph.nodes.find((n) => n.id === "o-proj")?.data.outputs[0].shape,
    ).toEqual([2, 7, 1024]);
  });
  it("keeps Mistral sliding window mask visible without silently cropping logical cache", () => {
    const m = normalizeConfig(mistral);
    const graph = buildGraph(m, { ...scenario, past: 5000 }, "attention", 0);
    expect(
      graph.nodes.find((n) => n.id === "softmax")?.data.subtitle,
    ).toContain("4096");
    expect(
      graph.nodes.find((n) => n.id === "k-cache")?.data.outputs[0].shape[2],
    ).toBe(5007);
  });
});

describe("Hugging Face URLs", () => {
  it("escapes revision names and fixes requests to Hugging Face", () => {
    expect(configURL("Qwen/Qwen3-0.6B", "refs/pr/42")).toBe(
      "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/refs%2Fpr%2F42/config.json",
    );
  });
  it.each([
    "https://example.com/x",
    "../config",
    "Qwen/Qwen/x",
    "Qwen",
    "x/../../y",
  ])("rejects invalid repo %s", (repo) => {
    expect(() => configURL(repo)).toThrow();
  });
});
