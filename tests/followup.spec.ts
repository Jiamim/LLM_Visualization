import { expect, test } from "@playwright/test";
import { readFileSync } from "node:fs";
import { normalizeConfig } from "../src/lib/model";
import { buildDemoPlan, demoScenario } from "../src/lib/inference-demo";
import { traceTemplate } from "../src/lib/demo-trace";
import { selectModel } from "./model-selection";
const gemma = JSON.parse(
  readFileSync(
    new URL("../src/data/gemma4-e2b.json", import.meta.url),
    "utf8",
  ),
);
const deepseek = JSON.parse(
  readFileSync(
    new URL("../src/data/deepseek-v4.1-flash.json", import.meta.url),
    "utf8",
  ),
);
test.use({ launchOptions: { args: ["--enable-unsafe-swiftshader"] } });

test("GQA inspector switches between Query and KV head ranges without invalid slices", async ({
  page,
}) => {
  const trace = traceTemplate(normalizeConfig(gemma), {
    promptTokens: 4,
    outputTokens: 1,
    useCache: false,
    speed: 1,
  });
  trace.source = "single KV head regression";
  trace.samples = [
    {
      pass: 0,
      layer: 0,
      view: "attention",
      nodeId: "kv-project",
      direction: "output",
      tensorIndex: 0,
      shape: [1, 1, 4, 256],
      indices: [[0, 0, 0, 0]],
      values: [0.22],
    },
  ];
  await page.goto("/");
  await selectModel(page, "gemma4-e2b");
  await page.getByRole("button", { name: "示例推理", exact: true }).click();
  await page.getByRole("button", { name: "演示设置", exact: true }).click();
  await page.getByLabel("导入运行记录文件").setInputFiles({
    name: "trace.json",
    mimeType: "application/json",
    buffer: Buffer.from(JSON.stringify(trace)),
  });
  await expect(page.locator(".demo-trace-controls")).toContainText(
    trace.source,
  );
  await page.getByRole("button", { name: "返回演示", exact: true }).click();
  await page.getByLabel("演示模型层级").selectOption("0");
  await page.getByLabel("定位三维模块").selectOption("q-proj");
  await page
    .getByRole("button", { name: "张量 / 矩阵", exact: true })
    .click();
  await page.getByLabel("选择解释张量").selectOption("1");
  await page.getByLabel("解释面板注意力头").selectOption("7");
  await page.getByLabel("定位三维模块").selectOption("kv-project");
  await page.getByLabel("选择解释张量").selectOption("1");
  await expect(
    page.getByLabel("解释面板注意力头").locator("option"),
  ).toHaveCount(1);
  await expect(page.getByLabel("解释面板注意力头")).toHaveValue("0");
  await page.getByRole("button", { name: "行 0 列 0", exact: true }).click();
  await expect(page.locator(".demo-cell-readout")).toContainText("0.220000");
  await expect(page.locator(".demo-slice-description")).toContainText(
    "Nkv=0",
  );
  await page.getByLabel("矩阵起始列").fill("240");
  await page.getByLabel("定位三维模块").selectOption("scores");
  await page.getByLabel("选择解释张量").selectOption("2");
  await expect(page.getByLabel("矩阵起始列")).toHaveValue("0");
  await expect(page.locator(".demo-matrix button")).toHaveCount(16);
});

test("moving from Attention to MLP within a layer reuses the WebGL geometry", async ({
  page,
}) => {
  await page.addInitScript(() => {
    const runtime = window as typeof window & { disposedBuffers: number };
    runtime.disposedBuffers = 0;
    const original = WebGL2RenderingContext.prototype.deleteBuffer;
    WebGL2RenderingContext.prototype.deleteBuffer = function (buffer) {
      runtime.disposedBuffers++;
      return original.call(this, buffer);
    };
  });
  const model = normalizeConfig(deepseek);
  const s = demoScenario(
    { promptTokens: 4, outputTokens: 2, useCache: false, speed: 1 },
    0,
    model.dtype,
  );
  const plan = buildDemoPlan(model, s);
  const attention = plan.findIndex(
    (e) => e.layer === 2 && e.view === "attention",
  );
  const mlp = plan.findIndex((e) => e.layer === 2 && e.view === "mlp");
  await page.goto("/?demo=1");
  await page.getByLabel("演示路线").selectOption("full");
  await page.getByLabel("演示进度", { exact: true }).fill(String(attention));
  await expect(page.getByTestId("tensor-space")).toHaveAttribute(
    "data-ready",
    "true",
  );
  await page.waitForTimeout(300);
  const before = await page.evaluate(
    () =>
      (window as typeof window & { disposedBuffers: number }).disposedBuffers,
  );
  await page.getByLabel("演示进度", { exact: true }).fill(String(mlp));
  await expect(page.locator(".inference-demo")).toHaveAttribute(
    "data-step",
    String(mlp),
  );
  await page.waitForTimeout(200);
  const after = await page.evaluate(
    () =>
      (window as typeof window & { disposedBuffers: number }).disposedBuffers,
  );
  expect(after).toBe(before);
});

test("restarting at the same playback step clears a separately browsed subgraph", async ({
  page,
}) => {
  await page.goto("/?demo=1");
  await page.getByLabel("演示模型层级").selectOption("0");
  const step = await page
    .locator(".inference-demo")
    .getAttribute("data-step");
  await page.getByLabel("定位三维模块").selectOption("q-lora");
  await page
    .getByRole("button", { name: "张量 / 矩阵", exact: true })
    .click();
  await page.getByRole("button", { name: "在 2D 中查看当前组件" }).click();
  await expect(page.locator(".demo-inspector h2")).toHaveText(
    "Query Low-rank",
  );
  await page.getByRole("button", { name: "演示设置", exact: true }).click();
  await page.getByLabel("演示提示 token 数").fill("5");
  await page.getByRole("button", { name: "返回演示", exact: true }).click();
  await expect(page.locator(".inference-demo")).toHaveAttribute(
    "data-step",
    "0",
  );
  await expect(page.locator(".demo-inspector h2")).not.toHaveText(
    "Query Low-rank",
  );
  await expect(page.getByTestId("demo-shape-scenario")).toContainText("S=5");
  expect(Number(step)).toBeGreaterThan(0);
});

test("recorded prism cells distinguish signs and do not blend into unknown neighbours", async ({
  page,
}) => {
  await page.goto("/");
  const colors = await page.evaluate(async () => {
    const modulePath = "/src/rendering/reference-primitives.ts";
    const { vectorStrip, disposeReferenceGroup } = await import(
      /* @vite-ignore */ modulePath
    );
    const group = vectorStrip(3, 3, 0, [null, 1, -1]);
    let result: { start: number[]; end: number[] } | undefined;
    group.traverse(
      (object: {
        isInstancedMesh?: boolean;
        geometry: {
          getAttribute: (name: string) => { array: ArrayLike<number> };
        };
      }) => {
        if (object.isInstancedMesh)
          result = {
            start: Array.from(
              object.geometry.getAttribute("colorStart").array,
            ),
            end: Array.from(object.geometry.getAttribute("colorEnd").array),
          };
      },
    );
    disposeReferenceGroup(group);
    return result!;
  });
  expect(colors.start).toEqual(colors.end);
  const unknown = colors.start.slice(0, 3),
    positive = colors.start.slice(3, 6),
    negative = colors.start.slice(6, 9);
  expect(positive).not.toEqual(negative);
  expect(unknown).not.toEqual(positive);
  expect(positive[1]).toBeGreaterThan(positive[0]);
  expect(negative[0]).toBeGreaterThan(negative[1]);
});
