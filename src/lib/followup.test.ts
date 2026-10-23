import { describe, it, expect, vi } from "vitest";
import { tensorHeadCount, traceValue } from "./tensor-semantics";
import { recordedValueColor, UNKNOWN_VALUE_COLOR } from "./visual-values";
import { normalizeConfig, type Tensor } from "./model";
import { buildDemoPlan, demoScenario } from "./inference-demo";
import { demoPlanDuration, demoStepDuration } from "./demo-pacing";
import { demoGraph } from "./inference-demo";
import type { TraceSample } from "./demo-trace";
import qwen from "../data/qwen3-0.6b.json";

const tensor: Tensor = {
  name: "key",
  shape: [1, 1, 4, 256],
  symbols: ["B", "Nkv", "S", "D"],
  dtype: "bfloat16",
};
describe("recorded tensor boundaries and repeated work", () => {
  it("uses the tensor's actual head axis for GQA and either axis order", () => {
    expect(tensorHeadCount(tensor)).toBe(1);
    expect(
      tensorHeadCount({
        ...tensor,
        shape: [1, 8, 4, 256],
        symbols: ["B", "Nq", "S", "D"],
      }),
    ).toBe(8);
    expect(
      tensorHeadCount({
        ...tensor,
        shape: [1, 4, 64, 512],
        symbols: ["B", "Q", "Nq", "D"],
      }),
    ).toBe(64);
    expect(
      tensorHeadCount({
        ...tensor,
        shape: [1, 4, 256],
        symbols: ["B", "S", "H"],
      }),
    ).toBeUndefined();
  });
  it("indexes coordinates once and shares that index across scene remapping", () => {
    const indices = [
      [0, 0, 0, 0],
      [0, 0, 3, 255],
    ];
    const sample = { indices, values: [-0.5, 0.8] } as TraceSample;
    const spy = vi.spyOn(indices, "map");
    expect(traceValue(sample, tensor, [0, 0, 3, 255])).toBe(0.8);
    expect(
      traceValue({ ...sample, nodeId: "remapped" }, tensor, [0, 0, 0, 0]),
    ).toBe(-0.5);
    expect(traceValue(sample, tensor, [0, 0, 2, 7])).toBeNull();
    expect(traceValue(sample, tensor, [0, 7, 0, 0])).toBeNull();
    expect(spy).toHaveBeenCalledTimes(1);
    spy.mockRestore();
  });
  it("keeps positive, negative, zero and unknown visual values distinguishable", () => {
    expect(recordedValueColor(0.7)).not.toBe(recordedValueColor(-0.7));
    expect(recordedValueColor(0)).not.toBe(UNKNOWN_VALUE_COLOR);
    expect(recordedValueColor(null)).toBe(UNKNOWN_VALUE_COLOR);
    expect(recordedValueColor(undefined)).toBe(UNKNOWN_VALUE_COLOR);
  });
  it("captures timing in the plan so tempo changes only scale durations", () => {
    const model = normalizeConfig(qwen);
    const s = demoScenario(
      { promptTokens: 4, outputTokens: 2, useCache: true, speed: 1 },
      0,
      model.dtype,
    );
    const events = buildDemoPlan(model, s);
    const cache = new Map<string, ReturnType<typeof demoGraph>>();
    let expected = 0;
    for (const event of events) {
      const key = `${event.layer}:${event.view}:${event.sample}`;
      let graph = cache.get(key);
      if (!graph) {
        graph = demoGraph(model, s, event);
        cache.set(key, graph);
      }
      const data = graph.nodes.find((n) => n.id === event.nodeId)?.data;
      const independent = demoStepDuration(
        { ...event, durationMs: undefined },
        data,
        "guided",
      );
      expect(event.durationMs).toBe(independent);
      expected += independent;
    }
    expect(demoPlanDuration(events, "guided")).toBe(expected);
    expect(demoPlanDuration(events, "guided", 2)).toBe(expected / 2);
    expect(demoPlanDuration(events, "continuous")).toBeCloseTo(
      expected * 0.48,
    );
  });
});
