import { expect, test } from "@playwright/test";
import { selectModel, openQwenModel } from "./model-selection";
test.use({ launchOptions: { args: ["--enable-unsafe-swiftshader"] } });
test.beforeEach(async ({ page }) => {
  await openQwenModel(page, "/");
});

test("start, pause and restart freeze both step progress and the generated context", async ({
  page,
}) => {
  test.setTimeout(60000);
  await page.getByRole("button", { name: "示例推理", exact: true }).click();
  const demo = page.locator(".inference-demo");
  await expect(demo).toHaveAttribute("data-status", "ready");
  await page.getByRole("button", { name: "开始演示", exact: true }).click();
  await expect
    .poll(() => demo.getAttribute("data-step"), { timeout: 12000 })
    .not.toBe("0");
  await page.getByRole("button", { name: "暂停", exact: true }).click();
  const step = await demo.getAttribute("data-step"),
    progress = await demo.getAttribute("data-progress");
  await page.waitForTimeout(1000);
  await expect(demo).toHaveAttribute("data-step", step!);
  await expect(demo).toHaveAttribute("data-progress", progress!);
  await page.getByRole("button", { name: "继续", exact: true }).click();
  await expect
    .poll(() => demo.getAttribute("data-step"), { timeout: 12000 })
    .not.toBe(step);
  await page.getByRole("button", { name: "重新开始演示" }).click();
  await expect(demo).toHaveAttribute("data-step", "0");
  await expect(demo).toHaveAttribute("data-pass", "0");
  await expect(page.locator(".demo-generated")).toHaveCount(0);
  await page.getByRole("button", { name: "暂停", exact: true }).click();
});

test("cache toggle switches from full-prefix recomputation to one-token decode and supports replay", async ({
  page,
}) => {
  await page.getByRole("button", { name: "示例推理", exact: true }).click();
  await page.getByRole("button", { name: "下一轮", exact: true }).click();
  await expect(page.getByTestId("demo-shape-scenario")).toContainText(
    "S=5 · P=0",
  );
  await expect(page.getByTestId("demo-cache-bytes")).toHaveText("0 B");
  await page.getByRole("button", { name: "演示设置", exact: true }).click();
  await page.getByLabel("启用演示 KV cache").check();
  await page.getByRole("button", { name: "返回演示", exact: true }).click();
  await expect(page.locator(".inference-demo")).toHaveAttribute(
    "data-status",
    "ready",
  );
  await page.getByRole("button", { name: "下一轮", exact: true }).click();
  await expect(page.getByTestId("demo-shape-scenario")).toContainText(
    "S=1 · P=4",
  );
  await expect(page.getByTestId("demo-cache-bytes")).toHaveText("224.00 KiB");
  await page.getByRole("button", { name: "跳到本轮输出" }).click();
  await expect(page.getByTestId("demo-cache-bytes")).toHaveText("280.00 KiB");
  await expect(page.locator(".demo-generated")).toHaveCount(1);
  await page.getByRole("button", { name: "下一轮", exact: true }).click();
  await expect(page.locator(".demo-generated")).toHaveCount(2);
  await page.locator(".demo-generated").first().click();
  await expect(page.locator(".inference-demo")).toHaveAttribute(
    "data-pass",
    "0",
  );
  await expect(page.locator(".demo-generated")).toHaveCount(0);
});

test("model-specific layer operations and state remain available after scene switches", async ({
  page,
}) => {
  await selectModel(page, "gemma4-e2b");
  await page.getByRole("button", { name: "示例推理", exact: true }).click();
  await expect(page.locator(".demo-heading")).toContainText("Gemma 4");
  await page.getByRole("button", { name: "跳过当前层" }).click();
  await expect(page.locator(".demo-layer-label")).toContainText("Layer");
  await page.getByRole("button", { name: "切换演示为 2D" }).click();
  await expect(page.locator(".tensor-node")).not.toHaveCount(0);
  const before = await page
    .locator(".inference-demo")
    .getAttribute("data-step");
  await page.getByRole("button", { name: "切换演示为 3D" }).click();
  await expect(page.getByTestId("tensor-space")).toHaveAttribute(
    "data-ready",
    "true",
  );
  await expect(page.locator(".inference-demo")).toHaveAttribute(
    "data-step",
    before!,
  );
});

test("completion stops automatically and exiting restores original workspace configuration", async ({
  page,
}) => {
  await page.getByLabel("Batch size").fill("3");
  await page.getByLabel("Sequence length").fill("64");
  await page.getByRole("button", { name: "示例推理", exact: true }).click();
  await page.getByRole("button", { name: "演示设置", exact: true }).click();
  await page.getByLabel("演示续写轮数").fill("1");
  await page.getByRole("button", { name: "返回演示", exact: true }).click();
  await page.getByRole("button", { name: "跳到本轮输出" }).click();
  await page.getByRole("button", { name: "演示下一步" }).click();
  await expect(page.locator(".inference-demo")).toHaveAttribute(
    "data-status",
    "complete",
  );
  await expect(page.locator(".demo-generated")).toHaveCount(1);
  await page.getByRole("button", { name: "返回模型视图" }).click();
  await expect(page.getByLabel("Batch size")).toHaveValue("3");
  await expect(page.getByLabel("Sequence length")).toHaveValue("64");
  await expect(page.locator("h1")).toHaveText("Qwen2.5-7B-Instruct");
});

test("mobile playback and settings are reachable without overflow", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole("button", { name: "示例推理", exact: true }).click();
  await page.getByRole("button", { name: "开始演示", exact: true }).click();
  await page.getByRole("button", { name: "暂停", exact: true }).click();
  await page.getByRole("button", { name: "演示设置", exact: true }).click();
  await expect(page.getByRole("dialog", { name: "演示设置" })).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(
    page.getByRole("button", { name: "重新开始演示" }),
  ).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(
    390,
  );
});
