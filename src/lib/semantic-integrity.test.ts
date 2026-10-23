import { describe, expect, it } from "vitest";
import { loadLibraryModel, modelLibrary } from "../data/model-library";
import { normalizeConfig, parameterCounts, type Scenario } from "./model";
import { buildGraph, type ModelGraph } from "./graph";
import { buildDemoPlan, demoGraph, demoScenario } from "./inference-demo";
import { buildDemoWorld, demoSceneNodeId } from "./demo-world";
import { referenceVisual, recordedLane } from "./reference-scene";
import { tensorCoordinates, traceValue } from "./tensor-semantics";
import {
  parseDemoTrace,
  traceTemplate,
  type TraceSample,
} from "./demo-trace";
import { sceneTraceSamples, eventForNode } from "./demo-selection";
import { teachingPlan } from "./demo-route";
import { demoStepDuration } from "./demo-pacing";
import qwen from "../data/gallery/qwen3-30b-a3b.json";

const scenario: Scenario = {
  batch: 1,
  sequence: 4,
  past: 0,
  dtype: "bfloat16",
};
const load = async (id: string) =>
  (await loadLibraryModel({ entryId: id, variantId: "gallery" })).model;
function reachable(graph: ModelGraph, from: string, to: string) {
  const pending = [from],
    seen = new Set<string>();
  while (pending.length) {
    const id = pending.pop()!;
    if (id === to) return true;
    if (seen.has(id)) continue;
    seen.add(id);
    pending.push(
      ...graph.edges.filter((e) => e.source === id).map((e) => e.target),
    );
  }
  return false;
}

describe("semantic contracts across adapters and views", () => {
  it.each([
    { vocab_size: Number.MAX_SAFE_INTEGER },
    { rms_norm_eps: -1 },
    { total_ut_steps: 1_000_000 },
    { layer_types: Array(48).fill("garbage") },
    { architectures: ["Qwen3MoeForSequenceClassification"] },
    { kv_lora_rank: -1 },
  ])(
    "rejects invalid imported configuration before state commit: %j",
    (change) => {
      expect(() => normalizeConfig({ ...qwen, ...change })).toThrow();
    },
  );
  it("preserves parallel branch independence through 3D expansion", async () => {
    const model = await load("command-a-218b-a25b");
    const event = buildDemoPlan(model, scenario).find(
      (e) => e.view === "layer",
    )!;
    const world = buildDemoWorld(model, scenario, event, false);
    expect(reachable(world, "out-proj", "ffn-input")).toBe(false);
    expect(reachable(world, "input-norm", "mixer-input")).toBe(true);
    expect(reachable(world, "input-norm", "ffn-input")).toBe(true);
    expect(reachable(world, "out-proj", "output")).toBe(true);
    expect(reachable(world, "combine", "output")).toBe(true);
    expect(
      world.edges.some((e) => e.source === "mixer" || e.target === "mixer"),
    ).toBe(false);
  });
  it("keeps each Gemma layer's window and tensor-local key positions", async () => {
    const model = await load("gemma-4-e2b");
    const s = { ...scenario, sequence: 1, past: 1024 };
    const local = buildGraph(model, s, "attention", 0).nodes.find(
      (n) => n.id === "scores",
    )!;
    const global = buildGraph(model, s, "attention", 4).nodes.find(
      (n) => n.id === "scores",
    )!;
    expect(referenceVisual(local, model, s)).toMatchObject({
      keyLength: 512,
      keyOffset: 513,
      window: 512,
    });
    expect(referenceVisual(global, model, s)).toMatchObject({
      keyLength: 1025,
      keyOffset: 0,
      window: null,
    });
    expect(global.data.layerIndex).toBe(4);
  });
  it("does not infer a dense mask for DeepSeek sparse candidates", async () => {
    const model = await load("deepseek-v4-1-flash");
    const node = buildGraph(model, scenario, "attention", 2).nodes.find(
      (n) => n.id === "sparse-attn",
    )!;
    expect(referenceVisual(node, model, scenario).visibility).toBe("unknown");
  });
  it("separates projection, past state and cache update during decode", () => {
    const model = normalizeConfig(qwen);
    const s = { ...scenario, sequence: 1, past: 1024 };
    const graph = buildGraph(model, s, "attention", 0);
    const find = (id: string) => graph.nodes.find((n) => n.id === id)!;
    expect(find("kv-proj").data.outputs[0].shape).toEqual([1, 4, 1, 128]);
    expect(find("past-kv").data.outputs[0].shape).toEqual([1, 4, 1024, 128]);
    expect(find("kv-cache").data.outputs[0].shape).toEqual([1, 4, 1025, 128]);
    expect(reachable(graph, "past-kv", "attention-core")).toBe(true);
    expect(referenceVisual(find("q-proj"), model, s)).toMatchObject({
      inputWidth: 2048,
      outputWidth: 4096,
    });
    expect(find("q-proj").data.params).toBe(2048 * 4096 + 128);
    expect(graph.nodes.reduce((sum, n) => sum + n.data.params, 0)).toBe(
      parameterCounts(model).attention,
    );
  });
  it.each([
    "gpt-2-xl-1-5b",
    "qwen3-30b-a3b",
    "deepseek-v3",
    "gemma-4-26b-a4b",
  ])(
    "partitions known attention and FFN parameters exactly once: %s",
    async (id) => {
      const model = await load(id);
      for (const view of ["attention", "mlp"] as const) {
        const graph = buildGraph(model, scenario, view, 0);
        expect(graph.nodes.every((n) => n.data.params >= 0)).toBe(true);
        expect(graph.nodes.reduce((sum, n) => sum + n.data.params, 0)).toBe(
          parameterCounts(model)[view === "attention" ? "attention" : "mlp"],
        );
      }
    },
  );
  it("looks up token/head coordinates consistently for both axis orders", async () => {
    const model = await load("deepseek-v4-1-flash");
    const tensor = buildGraph(model, scenario, "attention", 2).nodes.find(
      (n) => n.id === "q-lora",
    )!.data.outputs[0];
    const sample = {
      indices: [
        [0, 0, 3, 0],
        [0, 2, 3, 0],
      ],
      values: [0.11, 0.73],
    } as TraceSample;
    const indices = tensorCoordinates(
      tensor,
      { token: 2, head: 3 },
      { row: 3, column: 0 },
    );
    expect(indices).toEqual([0, 2, 3, 0]);
    expect(traceValue(sample, tensor, indices)).toBe(0.73);
    expect(recordedLane(sample, tensor, 2, 3, [0])).toEqual([0.73]);
    expect(recordedLane(sample, tensor, 1, 3, [0])).toEqual([null]);
    const standard = {
      ...tensor,
      shape: [1, 64, 4, 512],
      symbols: ["B", "Nq", "S", "D"],
      axes: undefined,
    };
    expect(
      tensorCoordinates(
        standard,
        { token: 2, head: 3 },
        { row: 2, column: 0 },
      ),
    ).toEqual([0, 3, 2, 0]);
  });
  it("never binds a merged Context tensor as an attention probability grid", async () => {
    const model = await load("gemma-4-e2b");
    const graph = buildGraph(model, scenario, "attention", 0);
    const context = graph.nodes.find((n) => n.id === "softmax-values")!;
    const scores = graph.nodes.find((n) => n.id === "scores")!;
    expect(context.data.outputs[0].role).toBe("context");
    expect(referenceVisual(context, model, scenario).attentionGrid).toBe(
      false,
    );
    expect(referenceVisual(scores, model, scenario).attentionGrid).toBe(true);
  });
  it("distinguishes records of repeated attention instances and rejects ambiguous legacy records", async () => {
    const model = await load("longcat-flash-lite-68-5b-a3b");
    const plan = buildDemoPlan(model, scenario);
    const events = plan.filter(
      (e) => e.layer === 0 && e.view === "attention" && e.nodeId === "q-proj",
    );
    expect(events.map((e) => e.instance)).toEqual(["mixer", "lc-attn1"]);
    const trace = traceTemplate(model, {
      promptTokens: 4,
      outputTokens: 1,
      useCache: false,
      speed: 1,
    });
    trace.samples = events.map((e, i) => ({
      pass: 0,
      layer: 0,
      view: "attention",
      nodeId: "q-proj",
      instance: e.instance,
      direction: "output",
      tensorIndex: 0,
      shape: demoGraph(model, scenario, e).nodes.find(
        (n) => n.id === "q-proj",
      )!.data.outputs[0].shape,
      indices: [[0, 0, 0, 0]],
      values: [i + 0.1],
    }));
    expect(parseDemoTrace(JSON.stringify(trace), model).samples).toHaveLength(
      2,
    );
    const graph = buildDemoWorld(model, scenario, events[0], false);
    expect(events.map((e) => demoSceneNodeId(graph, e))).toEqual([
      "q-proj",
      "lc-attn1:q-proj",
    ]);
    const recorded = parseDemoTrace(JSON.stringify(trace), model).samples;
    expect(
      sceneTraceSamples(graph, recorded, 0, 0, 0).map((s) => [
        s.nodeId,
        s.values[0],
      ]),
    ).toEqual([
      ["q-proj", 0.1],
      ["lc-attn1:q-proj", 1.1],
    ]);
    const second = graph.nodes.find((n) => n.id === "lc-attn1:q-proj")!;
    const target = eventForNode(second, events[0]);
    const plane = demoGraph(model, scenario, target);
    expect(
      sceneTraceSamples(plane, recorded, 0, 0, 0).map((s) => s.values[0]),
    ).toEqual([1.1]);
    delete trace.samples[0].instance;
    expect(() => parseDemoTrace(JSON.stringify(trace), model)).toThrow(
      /instance/,
    );
  });
  it.each(modelLibrary)(
    "bounds the default lesson and retains real event identity: $id",
    async (entry) => {
      const { model } = await loadLibraryModel({
        entryId: entry.id,
        variantId: entry.variants[0].id,
      });
      let budget = 0;
      for (let pass = 0; pass < 2; pass++) {
        const s = demoScenario(
          { promptTokens: 4, outputTokens: 2, useCache: false, speed: 1 },
          pass,
          model.dtype,
        );
        const full = buildDemoPlan(model, s),
          lesson = teachingPlan(model, full, pass);
        expect(lesson.every((e) => full.includes(e))).toBe(true);
        expect(lesson.at(-1)).toBe(full.at(-1));
        expect(lesson.length).toBeLessThanOrEqual(pass === 0 ? 36 : 13);
        for (const e of lesson)
          budget += demoStepDuration(
            e,
            demoGraph(model, s, e).nodes.find((n) => n.id === e.nodeId)?.data,
            "guided",
          );
      }
      expect(budget).toBeLessThanOrEqual(5 * 60 * 1000);
    },
  );
});
