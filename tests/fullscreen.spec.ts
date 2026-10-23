import { expect, test } from "@playwright/test";
test.use({ launchOptions: { args: ["--enable-unsafe-swiftshader"] } });

test("DeepSeek opens by default with matching source and library selection", async ({
  page,
}) => {
  await page.goto("/");
  await expect(page.locator("h1")).toHaveText("DeepSeek V4.1 Flash");
  await expect(page.locator(".model-source")).toContainText(
    "deepseek-ai/DeepSeek-V4.1-Flash",
  );
  await expect(page.locator(".ced-summary")).toContainText(
    "20 Encoder + 20 Decoder",
  );
  await page
    .getByRole("button", { name: "选择或搜索模型", exact: true })
    .click();
  await expect(page.locator(".gallery-card[data-current]")).toHaveAttribute(
    "data-gallery-id",
    "deepseek-v4-1-flash",
  );
  await page.getByRole("button", { name: "关闭模型库" }).click();
  await page.getByRole("button", { name: "模型 3D 推理", exact: true }).click();
  await expect(page.locator(".demo-heading strong")).toHaveText(
    "DeepSeek V4.1 Flash",
  );
  await expect(page.getByTestId("tensor-space")).toHaveAttribute(
    "data-model-hidden",
    "5120",
  );
  await expect(page.getByTestId("tensor-space")).toHaveAttribute(
    "data-world-layers",
    "40",
  );
});

test("workspace fullscreen enlarges the canvas, keeps all display modes and restores panels", async ({
  page,
}) => {
  await page.goto("/");
  const before = (await page.getByTestId("model-canvas").boundingBox())!;
  await page.getByRole("button", { name: "全屏查看", exact: true }).click();
  await expect(page.locator(".app-shell")).toHaveAttribute(
    "data-fullscreen",
    "native",
  );
  await expect
    .poll(() =>
      page.evaluate(() =>
        document.fullscreenElement?.classList.contains("app-shell"),
      ),
    )
    .toBe(true);
  const full = (await page.getByTestId("model-canvas").boundingBox())!;
  expect(full.width).toBeGreaterThan(before.width + 200);
  expect(full.height).toBeGreaterThan(before.height + 100);
  await page.getByRole("button", { name: "切换模型面板", exact: true }).click();
  await expect(page.locator(".sidebar")).toBeVisible();
  await page.getByRole("button", { name: "切换模型面板", exact: true }).click();
  await expect(page.locator(".sidebar")).toHaveCount(0);
  await page.getByRole("button", { name: "2D 张量", exact: true }).click();
  await expect(page.locator(".tensor-node")).not.toHaveCount(0);
  await page.getByRole("button", { name: "3D 空间", exact: true }).click();
  await expect(page.getByTestId("tensor-space")).toHaveAttribute(
    "data-ready",
    "true",
  );
  await page.getByRole("button", { name: "退出全屏", exact: true }).click();
  await expect(page.locator(".app-shell")).toHaveAttribute(
    "data-fullscreen",
    "none",
  );
  await expect(page.locator(".sidebar")).toBeVisible();
  await expect(page.locator(".inspector")).toBeVisible();
  await expect(
    page.getByRole("button", { name: "全屏查看", exact: true }),
  ).toBeFocused();
  await expect(page.getByTestId("model-canvas")).toHaveAttribute(
    "data-display-mode",
    "3d",
  );
});

test("3D fullscreen preserves canvas, paused progress and model switching", async ({
  page,
}) => {
  test.setTimeout(60000);
  await page.goto("/?demo=1");
  await expect(page.getByTestId("tensor-space")).toHaveAttribute(
    "data-ready",
    "true",
  );
  await page.getByRole("button", { name: "开始演示", exact: true }).click();
  await expect
    .poll(async () =>
      Number(
        await page.locator(".inference-demo").getAttribute("data-progress"),
      ),
    )
    .toBeGreaterThan(0);
  await page.getByRole("button", { name: "暂停", exact: true }).click();
  const progress = await page
    .locator(".inference-demo")
    .getAttribute("data-progress");
  await page
    .locator(".tensor-space-host canvas")
    .evaluate((el) => el.setAttribute("data-test-retained", "yes"));
  await page.getByRole("button", { name: "全屏查看", exact: true }).click();
  await expect(page.locator(".demo-shell")).toHaveAttribute(
    "data-fullscreen",
    "native",
  );
  await expect(page.locator(".tensor-space-host canvas")).toHaveAttribute(
    "data-test-retained",
    "yes",
  );
  await expect(page.locator(".inference-demo")).toHaveAttribute(
    "data-progress",
    progress!,
  );
  // Browser-owned exit follows the same fullscreenchange path as the native Esc key.
  await page.evaluate(() => document.exitFullscreen());
  await expect(page.locator(".demo-shell")).toHaveAttribute(
    "data-fullscreen",
    "none",
  );
  await expect(page.locator(".inference-demo")).toHaveAttribute(
    "data-status",
    "paused",
  );
  await page.getByRole("button", { name: "全屏查看", exact: true }).click();
  await expect(page.locator(".demo-shell")).toHaveAttribute(
    "data-fullscreen",
    "native",
  );
  await page.getByRole("button", { name: "在 3D 演示中切换模型" }).click();
  await page.locator('[data-gallery-id="qwen3-0-6b"]').getByRole("button").click();
  await expect(page.getByTestId("tensor-space")).toHaveAttribute(
    "data-model-hidden",
    "1024",
  );
  await expect(page.locator(".demo-shell")).toHaveAttribute(
    "data-fullscreen",
    "native",
  );
  await page.getByRole("button", { name: "退出全屏", exact: true }).click();
  await expect(page.locator(".demo-shell")).toHaveAttribute(
    "data-fullscreen",
    "none",
  );
});

test("unsupported fullscreen expands the mobile view and Esc restores layout", async ({
  page,
}) => {
  await page.setViewportSize({ width: 360, height: 800 });
  await page.addInitScript(() => {
    Object.defineProperty(HTMLElement.prototype, "requestFullscreen", {
      configurable: true,
      value: undefined,
    });
  });
  await page.goto("/?view=2d");
  await page.getByRole("button", { name: "全屏查看", exact: true }).click();
  await expect(page.locator(".app-shell")).toHaveAttribute(
    "data-fullscreen",
    "expanded",
  );
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(
    360,
  );
  await page.keyboard.press("Escape");
  await expect(page.locator(".app-shell")).toHaveAttribute(
    "data-fullscreen",
    "none",
  );
  await expect(
    page.getByRole("button", { name: "全屏查看", exact: true }),
  ).toBeFocused();
  await page.goto("/?demo=1");
  await page.getByRole("button", { name: "全屏查看", exact: true }).click();
  await expect(page.locator(".demo-shell")).toHaveAttribute(
    "data-fullscreen",
    "expanded",
  );
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(
    360,
  );
  await page.getByRole("button", { name: "退出全屏", exact: true }).click();
  await expect(page.locator(".demo-shell")).toHaveAttribute(
    "data-fullscreen",
    "none",
  );
});
