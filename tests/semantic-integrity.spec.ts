import { expect, test } from "@playwright/test";
import { readFileSync } from "node:fs";
import { openQwenModel } from "./model-selection";
import { normalizeConfig } from "../src/lib/model";
import { traceTemplate } from "../src/lib/demo-trace";
const deepseek = JSON.parse(
  readFileSync(
    new URL("../src/data/deepseek-v4.1-flash.json", import.meta.url),
    "utf8",
  ),
);

test.use({ launchOptions: { args: ["--enable-unsafe-swiftshader"] } });
test("invalid catalogue import reports the error and preserves the current model", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  const raw = JSON.parse(
    readFileSync(
      new URL("../src/data/gallery/qwen3-30b-a3b.json", import.meta.url),
      "utf8",
    ),
  );
  raw.vocab_size = Number.MAX_SAFE_INTEGER;
  await page.getByRole("button", { name: "导入模型", exact: true }).click();
  await page.getByRole("button", { name: "粘贴 JSON", exact: true }).click();
  await page.getByLabel("JSON 配置内容").fill(JSON.stringify(raw));
  await page.getByRole("button", { name: "生成模型视图" }).click();
  await expect(page.getByRole("alert")).toContainText("安全整数");
  await page.getByRole("button", { name: "关闭导入窗口" }).click();
  await expect(page.locator("h1")).toHaveText("DeepSeek V4.1 Flash");
  await expect(page.locator(".react-flow__node")).not.toHaveCount(0);
  expect(errors).toEqual([]);
});
test("selected token and component survive matrix inspection and a 2D round trip", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  const trace = traceTemplate(normalizeConfig(deepseek), {
    promptTokens: 4,
    outputTokens: 1,
    useCache: false,
    speed: 1,
  });
  trace.source = "synthetic coordinate regression";
  trace.samples = [
    {
      pass: 0,
      layer: 2,
      view: "attention",
      nodeId: "q-lora",
      direction: "output",
      tensorIndex: 0,
      shape: [1, 4, 64, 512],
      indices: [
        [0, 0, 3, 0],
        [0, 2, 3, 0],
      ],
      values: [0.11, 0.73],
    },
  ];
  // Resolve legacy records only when the adapter has a single parent instance.
  await page.goto("/?demo=1");
  await page.getByRole("button", { name: "演示设置", exact: true }).click();
  await page
    .getByLabel("导入运行记录文件")
    .setInputFiles({
      name: "trace.json",
      mimeType: "application/json",
      buffer: Buffer.from(JSON.stringify(trace)),
    });
  await expect(page.locator(".demo-trace-controls")).toContainText(
    trace.source,
  );
  await page.getByRole("button", { name: "返回演示", exact: true }).click();
  await page.getByLabel("演示模型层级").selectOption("2");
  await page.getByLabel("定位三维模块").selectOption("q-lora");
  await page
    .getByRole("button", { name: "张量 / 矩阵", exact: true })
    .click();
  await page.getByLabel("选择解释张量").selectOption("1");
  await page
    .getByRole("button", { name: "查看 token 3", exact: true })
    .click();
  await page.getByLabel("解释面板注意力头").selectOption("3");
  await page.getByRole("button", { name: "行 3 列 0", exact: true }).click();
  await expect(page.locator(".demo-cell-readout")).toContainText("0.730000");
  await expect(page.locator(".demo-slice-description")).toContainText("Q=2");
  await page.getByRole("button", { name: "在 2D 中查看当前组件" }).click();
  await expect(
    page.locator('.react-flow__node[data-id="q-lora"]'),
  ).toHaveClass(/selected/);
  await expect(page.locator(".demo-inspector h2")).toHaveText(
    "Query Low-rank",
  );
  await page
    .getByRole("button", { name: "张量 / 矩阵", exact: true })
    .click();
  await page.getByLabel("选择解释张量").selectOption("1");
  await page.getByRole("button", { name: "行 3 列 0", exact: true }).click();
  await expect(page.locator(".demo-cell-readout")).toContainText("0.730000");
  await page.getByRole("button", { name: "切换演示为 3D" }).click();
  await expect(page.locator(".demo-inspector h2")).toHaveText(
    "Query Low-rank",
  );
  await expect(page.getByTestId("tensor-space")).toHaveAttribute(
    "data-ready",
    "true",
  );
  expect(errors).toEqual([]);
});
test("the default lesson has a visible time budget and complete traversal remains available", async ({
  page,
}) => {
  await page.goto("/?demo=1");
  await expect(page.getByLabel("演示路线")).toHaveValue("overview");
  const text = await page.getByTestId("demo-route-budget").innerText();
  const minutes = Number(text.match(/预计 ([\d.]+) 分钟/)?.[1]);
  expect(minutes).toBeGreaterThan(0);
  expect(minutes).toBeLessThanOrEqual(5);
  await expect(page.getByTestId("demo-route-budget")).toContainText("省略");
  await page.getByLabel("演示路线").selectOption("full");
  await expect(page.getByTestId("demo-route-budget")).toContainText(
    "完整逐层",
  );
  expect(
    Number(await page.getByLabel("演示进度").getAttribute("max")),
  ).toBeGreaterThan(1000);
});
test("compact lesson controls stay usable on mobile", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await openQwenModel(page, "/?demo=1");
  await expect(
    page.getByRole("button", { name: "开始演示", exact: true }),
  ).toBeVisible();
  await expect(page.getByLabel("演示路线")).toBeVisible();
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth),
  ).toBe(390);
});
