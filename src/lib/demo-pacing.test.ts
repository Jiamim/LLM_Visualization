import { expect, it } from "vitest";
import { demoBeat, demoStepDuration } from "./demo-pacing";
import { buildDemoPlan, demoScenario } from "./inference-demo";
import { buildDemoWorld } from "./demo-world";
import { REFERENCE_LAYER_HEIGHT } from "./reference-scene";
import { normalizeConfig } from "./model";
import qwen25 from "../../tests/fixtures/qwen2.5-7b.json";
const m = normalizeConfig(qwen25),
  s = demoScenario(
    { promptTokens: 4, outputTokens: 4, useCache: true, speed: 1 },
    0,
    "bfloat16",
  ),
  plan = buildDemoPlan(m, s);
it("allocates reading time to important operations with speed changes independent of progress", () => {
  const input = plan[0],
    softmax = plan.find((e) => e.nodeId === "softmax")!,
    norm = plan.find((e) => e.nodeId === "input-norm")!;
  const data = (e: typeof input) =>
    buildDemoWorld(m, s, e, false).nodes.find((n) => n.id === e.nodeId)!.data;
  expect(demoStepDuration(input, data(input), "guided")).toBe(4200);
  expect(demoStepDuration(softmax, data(softmax), "guided")).toBe(8200);
  expect(demoStepDuration(norm, data(norm), "guided")).toBe(6400);
  expect(demoStepDuration(softmax, data(softmax), "guided", 0.25)).toBe(32800);
  expect(demoStepDuration(softmax, data(softmax), "continuous")).toBeLessThan(
    demoStepDuration(softmax, data(softmax), "guided"),
  );
});
it("separates arrival, input transfer, compute scan and output reading hold", () => {
  expect(demoBeat(0)).toMatchObject({
    phase: 0,
    arrival: true,
    transfer: 0,
    computing: 0,
  });
  expect(demoBeat(0.24).transfer).toBeCloseTo(0.5);
  expect(demoBeat(0.55)).toMatchObject({
    phase: 1,
    output: false,
    hold: false,
  });
  expect(demoBeat(0.85)).toMatchObject({ phase: 2, output: true, hold: false });
  expect(demoBeat(0.96)).toMatchObject({ phase: 2, hold: true });
  expect(demoBeat(1).computing).toBe(1);
});
it("spaces distinct computation rows and parallel branches far enough for visible blocks", () => {
  const e = plan.find((e) => e.nodeId === "q-proj")!,
    graph = buildDemoWorld(m, s, e, false);
  const ys = [
      ...new Set(graph.nodes.map((n) => n.data.worldPosition![1])),
    ].sort((a, b) => a - b),
    xs = [...new Set(graph.nodes.map((n) => n.data.worldPosition![0]))].sort(
      (a, b) => a - b,
    );
  expect(ys.slice(1).every((y, i) => y - ys[i] >= 3)).toBe(true);
  expect(xs.slice(1).every((x, i) => x - xs[i] >= 4)).toBe(true);
  const second = buildDemoWorld(m, s, { ...e, layer: 1 }, false);
  expect(
    second.nodes[0].data.worldPosition![1] -
      graph.nodes[0].data.worldPosition![1],
  ).toBe(REFERENCE_LAYER_HEIGHT);
});
