import { describe, it, expect } from "vitest";
import {
  activationValue,
  explainDemoNode,
  normalizeExample,
  stableSoftmax,
} from "./demo-explanations";
import { demoModelKey, parseDemoTrace, traceTemplate } from "./demo-trace";
import {
  buildDemoWorld,
  demoWorldLayers,
  selectedTokenPosition,
} from "./demo-world";
import {
  buildDemoPlan,
  demoReducer,
  initialDemoState,
  readDemoRoute,
} from "./inference-demo";
import { modelLibrary, loadLibraryModel } from "../data/model-library";
import qwen25 from "../../tests/fixtures/qwen2.5-7b.json";
import { normalizeConfig } from "./model";
const model = normalizeConfig(qwen25),
  settings = { promptTokens: 4, outputTokens: 2, useCache: true, speed: 1 };
describe("reference-style explanation and trace", () => {
  it("computes stable masked softmax and norm from explicit teaching inputs", () => {
    const p = stableSoftmax([1000, 1001, 1002], [false, true, false]);
    expect(p[1]).toBe(0);
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1);
    expect(p[2]).toBeGreaterThan(p[0]);
    expect(stableSoftmax([1, 2], [true, true])).toEqual([0, 0]);
    expect(normalizeExample([3, 3, 3], false)).toEqual([0, 0, 0]);
    expect(
      normalizeExample([3, 4], true).reduce((s, x) => s + x * x, 0) / 2,
    ).toBeCloseTo(1);
    expect(activationValue(0, "silu")).toBe(0);
    expect(activationValue(-1, "relu2")).toBe(0);
    expect(activationValue(2, "relu2")).toBe(4);
  });
  it("parses bounded routes and preserves local substep pause", () => {
    expect(readDemoRoute("?token=4&generation=2&kvCache=1")).toEqual({
      promptTokens: 4,
      pass: 2,
      useCache: true,
    });
    expect(readDemoRoute("?token=-10&generation=999")).toMatchObject({
      promptTokens: 4,
      pass: 0,
    });
    expect(
      demoReducer(initialDemoState, { type: "set-progress", value: 0.66 }),
    ).toMatchObject({ status: "paused", progress: 0.66, index: 0 });
    expect(
      selectedTokenPosition(
        { batch: 1, sequence: 1, past: 4, dtype: "bfloat16" },
        0.5,
      ),
    ).toBe(4);
  });
  it("validates token identity, shape, finite values, coordinates and model compatibility", () => {
    const trace = traceTemplate(model, settings);
    trace.source = "unit test fixture";
    trace.samples.push({
      pass: 0,
      layer: 0,
      view: "overview",
      nodeId: "embedding",
      direction: "output",
      tensorIndex: 0,
      shape: [1, 4, 3584],
      indices: [
        [0, 0, 0],
        [0, 0, 1],
      ],
      values: [1.25, -0.5],
    });
    expect(
      parseDemoTrace(JSON.stringify(trace), model).samples[0].values,
    ).toEqual([1.25, -0.5]);
    for (const mutate of [
      (t: typeof trace) => {
        t.modelKey = "wrong";
      },
      (t: typeof trace) => {
        t.prompt[0].id = model.vocab;
      },
      (t: typeof trace) => {
        t.samples[0].indices[0][1] = 4;
      },
      (t: typeof trace) => {
        t.samples[0].shape[1] = 8;
      },
      (t: typeof trace) => {
        t.samples[0].values[0] = Infinity;
      },
      (t: typeof trace) => {
        t.samples.push(t.samples[0]);
      },
    ]) {
      const invalid = structuredClone(trace);
      mutate(invalid);
      expect(() => parseDemoTrace(JSON.stringify(invalid), model)).toThrow();
    }
    const reordered = {
      ...model,
      raw: Object.fromEntries(Object.entries(model.raw).reverse()),
    };
    expect(demoModelKey(reordered)).toBe(demoModelKey(model));
  });
  it("keeps a boundary input separate from an actual normalization operation", () => {
    const scenario = {
        batch: 1,
        sequence: 4,
        past: 0,
        dtype: "bfloat16" as const,
      },
      plan = buildDemoPlan(model, scenario),
      event = plan.find((e) => e.nodeId === "attn-input")!;
    const graph = buildDemoWorld(model, scenario, event, false),
      node = graph.nodes.find((n) => n.id === "attn-input")!;
    expect(explainDemoNode(model, node.data).lab).toBeNull();
  });
  it.each(modelLibrary)(
    "$id has a complete stable world and nonempty component explanations",
    async (entry) => {
      const { model } = await loadLibraryModel({
        entryId: entry.id,
        variantId: entry.variants[0].id,
      });
      const scenario = {
          batch: 1,
          sequence: 4,
          past: 0,
          dtype: "bfloat16" as const,
        },
        plan = buildDemoPlan(model, scenario);
      expect(demoWorldLayers(model)).toHaveLength(model.layers);
      const event = plan.find((e) => e.view === "attention") ?? plan[0],
        graph = buildDemoWorld(model, scenario, event, false);
      for (const node of graph.nodes) {
        expect(node.data.worldPosition?.every(Number.isFinite)).toBe(true);
        const e = explainDemoNode(model, node.data);
        expect(e.purpose.length).toBeGreaterThan(10);
        expect(e.reason.length).toBeGreaterThan(10);
      }
    },
  );
});
