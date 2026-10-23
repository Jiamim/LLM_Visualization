import { describe, expect, it } from "vitest";
import {
  buildDemoPlan,
  demoCache,
  demoGraph,
  demoHasHead,
  demoReducer,
  demoScenario,
  demoSettingsValid,
  initialDemoState,
  type DemoSettings,
} from "./inference-demo";
import {
  modelLibrary,
  loadLibraryModel,
  selectionForPreset,
} from "../data/model-library";
import qwen25 from "../../tests/fixtures/qwen2.5-7b.json";
import { normalizeConfig } from "./model";
const settings: DemoSettings = {
  promptTokens: 4,
  outputTokens: 3,
  useCache: true,
  speed: 1,
};

describe("example inference state machine", () => {
  it("pauses at the exact fractional step, resumes and restarts from empty state", () => {
    let s = demoReducer(initialDemoState, { type: "start" });
    s = demoReducer(s, { type: "progress", value: 0.42 });
    s = demoReducer(s, { type: "pause" });
    expect(s).toMatchObject({ status: "paused", index: 0, progress: 0.42 });
    expect(demoReducer(s, { type: "progress", value: 0.9 })).toBe(s);
    s = demoReducer(s, { type: "start" });
    expect(s.progress).toBe(0.42);
    s = demoReducer(s, { type: "next-pass", passes: 3 });
    expect(s).toMatchObject({ pass: 1, index: 0, generated: 1 });
    s = demoReducer(s, { type: "restart" });
    expect(s).toMatchObject({
      status: "running",
      pass: 0,
      index: 0,
      generated: 0,
      progress: 0,
      run: 1,
    });
  });
  it("stops at completion, rewinds generated tokens when seeking, and does not wrap automatically", () => {
    let s = demoReducer(initialDemoState, { type: "start" });
    for (let i = 0; i < 6; i++)
      s = demoReducer(s, { type: "advance", steps: 2, passes: 3 });
    expect(s).toMatchObject({
      status: "complete",
      pass: 2,
      index: 1,
      generated: 3,
    });
    expect(demoReducer(s, { type: "advance", steps: 2, passes: 3 })).toBe(s);
    s = demoReducer(s, { type: "seek", index: 0 });
    expect(s).toMatchObject({ status: "paused", generated: 2 });
    s = demoReducer(s, { type: "replay", pass: 0 });
    expect(s).toMatchObject({
      pass: 0,
      generated: 0,
      index: 0,
      status: "paused",
    });
  });
  it("distinguishes cache-on single-token decode from full prefix recomputation", () => {
    expect(demoScenario(settings, 0, "bfloat16")).toMatchObject({
      batch: 1,
      sequence: 4,
      past: 0,
    });
    expect(demoScenario(settings, 1, "bfloat16")).toMatchObject({
      sequence: 1,
      past: 4,
    });
    expect(demoScenario(settings, 2, "bfloat16")).toMatchObject({
      sequence: 1,
      past: 5,
    });
    expect(
      demoScenario({ ...settings, useCache: false }, 2, "bfloat16"),
    ).toMatchObject({ sequence: 6, past: 0 });
  });
  it("counts carried state only at completed forward boundaries and uses the actual KV heads", () => {
    const m = normalizeConfig(qwen25),
      s = demoScenario(settings, 0, "bfloat16");
    expect(demoCache(m, s, true, false).totalBytes).toBe(0);
    expect(demoCache(m, s, true, true).kvBytes).toBe(2 * 28 * 4 * 4 * 128 * 2);
    const next = demoScenario(settings, 1, "bfloat16");
    expect(demoCache(m, next, true, false).tokens).toBe(4);
    expect(demoCache(m, next, true, true).tokens).toBe(5);
    expect(demoCache(m, next, false, true).totalBytes).toBe(0);
  });
  it("does not invent sampling for a base model without an LM head", () => {
    const m = normalizeConfig({
      ...qwen25,
      architectures: ["Qwen2Model"],
    });
    expect(demoHasHead(m)).toBe(false);
    expect(
      buildDemoPlan(m, demoScenario(settings, 0, "bfloat16")).some(
        (e) => e.sample,
      ),
    ).toBe(false);
  });
  it("keeps every real dense layer and both residuals in execution order", () => {
    const m = normalizeConfig(qwen25),
      s = demoScenario(settings, 0, "bfloat16"),
      plan = buildDemoPlan(m, s);
    expect(plan.filter((e) => e.isLayerEnd).map((e) => e.layer)).toEqual(
      Array.from({ length: 28 }, (_, i) => i),
    );
    const first = plan.filter((e) => e.view !== "overview" && e.layer === 0);
    expect(first.findIndex((e) => e.nodeId === "o-proj")).toBeLessThan(
      first.findIndex((e) => e.nodeId === "attention-residual"),
    );
    expect(first.findIndex((e) => e.nodeId === "down-proj")).toBeLessThan(
      first.findIndex((e) => e.nodeId === "mlp-residual"),
    );
    expect(plan.at(-1)?.sample).toBe(true);
  });
  it("handles CED cache decode, Gemma PLE residual and hybrid state operations", async () => {
    for (const id of ["deepseek-v41", "gemma4-e2b"]) {
      const { model } = await loadLibraryModel(selectionForPreset(id));
      const s = demoScenario(settings, 1, "bfloat16"),
        plan = buildDemoPlan(model, s);
      const node = plan.find(
        (e) => e.nodeId === (id === "deepseek-v41" ? "hc-post" : "ple-add"),
      )!;
      expect(
        demoGraph(model, s, node).nodes.find((n) => n.id === node.nodeId)?.data
          .residual,
      ).toBe(true);
      expect(plan.filter((e) => e.isLayerEnd).map((e) => e.layer)).toEqual(
        Array.from({ length: model.layers }, (_, i) => i),
      );
    }
  });
  it("validates demo bounds without silently exceeding model context", () => {
    const m = normalizeConfig(qwen25);
    expect(demoSettingsValid(m, settings)).toBe(true);
    expect(demoSettingsValid(m, { ...settings, promptTokens: 0 })).toBe(false);
    expect(demoSettingsValid({ ...m, maxPosition: 4 }, settings)).toBe(false);
  });
  it.each(modelLibrary)(
    "$id supports a complete model-specific demo plan",
    async (entry) => {
      const { model } = await loadLibraryModel({
        entryId: entry.id,
        variantId: entry.variants[0].id,
      });
      for (const pass of [0, 1]) {
        const s = demoScenario(settings, pass, "bfloat16"),
          plan = buildDemoPlan(model, s);
        expect(plan.length).toBeGreaterThan(0);
        expect(plan.filter((e) => e.isLayerEnd)).toHaveLength(
          model.layers * (model.catalog?.loops ?? 1),
        );
        const unique = new Map(
          plan.map((e) => [`${e.view}/${e.layer}/${e.sample}`, e]),
        );
        for (const e of unique.values()) {
          const graph = demoGraph(model, s, e);
          const ids = new Set(graph.nodes.map((n) => n.id));
          expect(ids.has(e.nodeId)).toBe(true);
          for (const t of graph.nodes.flatMap((n) => [
            ...n.data.inputs,
            ...n.data.outputs,
          ]))
            expect(t.shape.every((n) => Number.isFinite(n) && n >= 0)).toBe(
              true,
            );
        }
        expect(
          Number.isFinite(demoCache(model, s, true, true).totalBytes),
        ).toBe(true);
      }
    },
    20000,
  );
});
