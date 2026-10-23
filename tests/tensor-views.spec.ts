import { expect, test } from "@playwright/test";
import { selectModel, openQwenModel } from "./model-selection";

test.use({ launchOptions: { args: ["--enable-unsafe-swiftshader"] } });

test("2D uses the same selection, graph and decode dimensions as the structure view", async ({
  page,
}) => {
  await openQwenModel(page, "/");
  await page
    .locator(".tree-item")
    .filter({ hasText: "Self Attention" })
    .click();
  await page.getByRole("button", { name: "03 注意力计算" }).click();
  await page.getByRole("button", { name: "2D 张量", exact: true }).click();
  await expect(page.locator(".tensor-node")).toHaveCount(18);
  await expect(page.locator('[data-id="scores"] .tensor-node')).toHaveClass(
    /is-selected/,
  );
  await expect(page.locator('[data-id="scores"]')).toContainText(
    "[1, 28, 128, 128]",
  );
  await page.getByRole("button", { name: "Decode", exact: true }).click();
  await page.getByLabel("Batch size").fill("2");
  await page.getByLabel("Past KV length").fill("64");
  await expect(page.locator('[data-id="scores"]')).toContainText(
    "[2, 28, 1, 65]",
  );
  await expect(page.locator('[data-id="scores"]')).toContainText("56 个切片");
  await page.getByRole("button", { name: /切换符号维度/ }).click();
  await expect(page.locator('[data-id="scores"]')).toContainText(
    "[B, Nq, S, T]",
  );
  await page.getByRole("button", { name: "结构图", exact: true }).click();
  await expect(page.locator('[data-id="scores"]')).toHaveClass(/selected/);
  await expect(page.locator(".inspector h2")).toHaveText("Attention Scores");
  await expect(page.getByLabel("Past KV length")).toHaveValue("64");
});

test("3D draws blocks, rotates, zooms, selects and drills using the current graph", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await openQwenModel(page, "/?view=3d");
  await expect(page.getByTestId("tensor-space")).toHaveAttribute(
    "data-ready",
    "true",
  );
  await expect(page.locator(".tensor-space-host canvas")).toHaveCount(1);
  const label = page.locator('[data-spatial-id="decoder"]');
  await expect(label).toBeVisible();
  const before = await label.getAttribute("style");
  const canvas = page.locator(".tensor-space-host canvas"),
    rect = (await canvas.boundingBox())!;
  await page.mouse.move(rect.x + rect.width * 0.45, rect.y + rect.height * 0.4);
  await page.mouse.down();
  await page.mouse.move(
    rect.x + rect.width * 0.58,
    rect.y + rect.height * 0.45,
    { steps: 8 },
  );
  await page.mouse.up();
  await expect.poll(() => label.getAttribute("style")).not.toBe(before);
  await page.getByRole("button", { name: "适配三维全图" }).click();
  const embeddingBox = (await page
    .locator('[data-spatial-id="embedding"]')
    .boundingBox())!;
  await page.mouse.click(
    embeddingBox.x + embeddingBox.width / 2,
    embeddingBox.y - 35,
  );
  await expect(page.locator(".inspector h2")).toHaveText("Token Embedding");
  await page.getByLabel("定位三维模块").selectOption("embedding");
  await expect(page.locator(".inspector h2")).toHaveText("Token Embedding");
  await expect(page.locator('[data-spatial-id="embedding"]')).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  const projection = await page
    .locator('[data-spatial-id="decoder"]')
    .getAttribute("style");
  await page.getByRole("button", { name: "放大三维场景" }).click();
  await expect
    .poll(() =>
      page.locator('[data-spatial-id="decoder"]').getAttribute("style"),
    )
    .not.toBe(projection);
  await page.getByLabel("定位三维模块").selectOption("decoder");
  await page.getByRole("button", { name: "进入模块内部" }).click();
  await expect(page.locator('[data-spatial-id="attention"]')).toHaveCount(1);
  await expect(page.locator("[data-spatial-id]")).toHaveCount(7);
  await page.getByRole("button", { name: "2D 张量", exact: true }).click();
  await expect(page.locator(".tensor-node")).toHaveCount(7);
  expect(errors).toEqual([]);
});

test("switching modes releases 3D canvases and loads 3D code only when requested", async ({
  page,
}) => {
  const requests: string[] = [];
  page.on("request", (request) => requests.push(request.url()));
  await openQwenModel(page, "/");
  await expect(page.locator(".module-node")).toHaveCount(6);
  expect(
    requests.some((url) => /TensorCanvas3D|three\.js|three_module/.test(url)),
  ).toBe(false);
  for (let i = 0; i < 3; i++) {
    await page.getByRole("button", { name: "3D 空间", exact: true }).click();
    await expect(page.locator(".tensor-space-host canvas")).toHaveCount(1);
    await expect(page.locator('[data-spatial-id="decoder"]')).toBeVisible();
    await page.getByRole("button", { name: "结构图", exact: true }).click();
    await expect(page.locator(".tensor-space-host canvas")).toHaveCount(0);
    await expect(page.locator(".module-node")).toHaveCount(6);
  }
});

test("3D follows playback and preserves exact dimensions across model changes", async ({
  page,
}) => {
  await openQwenModel(page, "/?view=3d");
  await expect(page.getByTestId("tensor-space")).toHaveAttribute(
    "data-ready",
    "true",
  );
  await page.getByRole("button", { name: "数据流下一步" }).click();
  await expect(page.locator('[data-spatial-id="input"]')).toHaveAttribute(
    "data-active",
    "true",
  );
  await page.getByRole("button", { name: "数据流下一步" }).click();
  await expect(page.locator('[data-spatial-id="embedding"]')).toHaveAttribute(
    "data-active",
    "true",
  );
  for (const [model, id] of [
    ["gemma4-e2b", "ple"],
    ["deepseek-v41", "encoder"],
    ["qwen3", "decoder"],
  ]) {
    await selectModel(page, model);
    await expect(page.getByTestId("model-canvas")).toHaveAttribute(
      "data-display-mode",
      "3d",
    );
    await expect(page.locator(`[data-spatial-id="${id}"]`)).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    await expect(page.locator("[data-spatial-id][data-active]")).toHaveCount(0);
  }
});

test("2D supports tensor edge inspection and model module drilldown", async ({
  page,
}) => {
  await openQwenModel(page, "/?view=2d");
  await expect(page.locator(".tensor-node")).toHaveCount(6);
  await page.getByRole("button", { name: "适配张量全图" }).click();
  await page.locator(".react-flow__edge").first().click();
  await expect(page.locator(".inspector h2")).toHaveText("input_ids");
  await page.getByRole("button", { name: "展开 Transformer Blocks" }).click();
  await expect(page.locator(".tensor-node")).toHaveCount(7);
  await page.getByRole("button", { name: "展开 Self Attention" }).click();
  await expect(page.locator(".tensor-node")).toHaveCount(18);
  await page.getByRole("button", { name: "03 注意力计算" }).click();
  await page.getByRole("button", { name: "适配张量全图" }).click();
  await expect(page.getByTestId("model-canvas")).toHaveAttribute(
    "data-display-mode",
    "2d",
  );
});

test("touch layout keeps the canvas visible and the inspector usable", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await openQwenModel(page, "/?view=3d");
  await expect(page.getByTestId("tensor-space")).toHaveAttribute(
    "data-ready",
    "true",
  );
  await page.getByLabel("定位三维模块").selectOption("embedding");
  await expect(
    page.getByRole("dialog", { name: "模块详情", exact: true }),
  ).toBeVisible();
  await expect(page.locator(".inspector h2")).toHaveText("Token Embedding");
  await page.keyboard.press("Escape");
  await page.getByRole("button", { name: "2D 张量", exact: true }).click();
  await expect(page.locator(".tensor-node")).toHaveCount(6);
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(
    390,
  );
  expect(
    (await page.getByTestId("model-canvas").boundingBox())!.height,
  ).toBeGreaterThan(300);
});

test("WebGL failure offers a working 2D fallback without losing the model", async ({
  page,
}) => {
  await page.addInitScript(() => {
    const original = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = function (
      this: HTMLCanvasElement,
      type: string,
      ...args: unknown[]
    ) {
      if (
        type === "webgl2" ||
        type === "webgl" ||
        type === "experimental-webgl"
      )
        return null;
      return original.apply(this, [type, ...args] as Parameters<
        typeof original
      >);
    } as typeof original;
  });
  await openQwenModel(page, "/?view=3d");
  await expect(page.getByText("当前浏览器无法显示 3D 场景")).toBeVisible();
  await page
    .getByRole("button", { name: "切换到 2D 张量", exact: true })
    .click();
  await expect(page.locator(".tensor-node")).toHaveCount(6);
  await expect(page.locator("h1")).toHaveText("Qwen2.5-7B-Instruct");
  await expect(page).toHaveURL(/view=2d/);
});
