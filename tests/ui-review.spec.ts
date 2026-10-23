import { openQwenModel } from "./model-selection";
import { expect, test } from "@playwright/test";

test.beforeEach(async ({ page }) => {
  await openQwenModel(page, "/");
  await page.evaluate(() => document.fonts.ready);
});

test("stage navigation keeps long graphs legible and preserves the selected node when panels change", async ({
  page,
}) => {
  await page
    .locator(".tree-item")
    .filter({ hasText: "Self Attention" })
    .click();
  await expect
    .poll(async () =>
      Number((await page.locator(".zoom-value").innerText()).replace("%", "")),
    )
    .toBeGreaterThanOrEqual(65);
  await expect(
    page.getByRole("button", { name: "01 Q / K / V 投影" }),
  ).toHaveAttribute("aria-current", "step");
  await page.getByRole("button", { name: "03 注意力计算" }).click();
  await expect(page.locator(".inspector h2")).toHaveText("Attention Scores");
  await page.getByRole("button", { name: "收起详情面板" }).click();
  await page.locator('[data-id="scores"]').click();
  await expect(page.locator(".inspector")).toBeVisible();
  await expect
    .poll(async () => {
      const node = await page.locator('[data-id="scores"]').boundingBox();
      const canvas = await page.locator(".canvas").boundingBox();
      return (
        !!node &&
        !!canvas &&
        node.x >= canvas.x &&
        node.x + node.width <= canvas.x + canvas.width
      );
    })
    .toBe(true);
  await page.getByRole("button", { name: "适配全图", exact: true }).click();
  await expect(page.locator(".readability-hint")).toBeVisible();
  await page.getByRole("button", { name: "阅读视图", exact: true }).click();
  await expect
    .poll(async () =>
      Number((await page.locator(".zoom-value").innerText()).replace("%", "")),
    )
    .toBeGreaterThanOrEqual(65);
});

test("inspector tabs work with keyboard, explain dimensions, and copy actual shapes", async ({
  page,
  context,
}) => {
  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
  const tensorTab = page.getByRole("tab", { name: /张量维度/ });
  await tensorTab.focus();
  await page.keyboard.press("ArrowRight");
  await expect(page.getByRole("tab", { name: "计算与参数" })).toHaveAttribute(
    "aria-selected",
    "true",
  );
  await expect(page.locator(".formula")).toBeVisible();
  await expect(page.locator(".parameter-summary")).toContainText("6.53B");
  await page.keyboard.press("Home");
  await expect(tensorTab).toHaveAttribute("aria-selected", "true");
  await expect(page.locator(".dimension-key")).toContainText("隐藏维度");
  await page.getByRole("button", { name: "复制张量维度" }).click();
  await expect(
    page.getByRole("button", { name: "复制张量维度" }),
  ).toContainText("已复制");
  expect(await page.evaluate(() => navigator.clipboard.readText())).toContain(
    "hidden_states: [1, 128, 3584]",
  );
});

test("focus mode enlarges the canvas and escape restores the workspace", async ({
  page,
}) => {
  const before = await page.locator(".canvas").boundingBox();
  await page.getByRole("button", { name: "更多画布工具" }).click();
  await page.getByRole("menuitem", { name: "专注模式", exact: true }).click();
  await expect(page.locator(".sidebar")).toHaveCount(0);
  await expect(page.locator(".inspector")).toHaveCount(0);
  const focused = await page.locator(".canvas").boundingBox();
  expect(focused!.width).toBeGreaterThan(before!.width + 400);
  expect(focused!.height).toBeGreaterThan(before!.height + 100);
  await page.keyboard.press("Escape");
  await expect(page.locator(".sidebar")).toBeVisible();
  await expect(page.locator(".inspector")).toBeVisible();
});

test("mobile drawers are accessible without pushing the graph below the viewport", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.locator(".panel-dialog-left")).not.toBeVisible();
  await page.getByRole("button", { name: "切换模型面板" }).click();
  await expect(
    page.getByRole("dialog", { name: "模型与输入", exact: true }),
  ).toBeVisible();
  await page.getByLabel("Batch size").fill("3");
  await page
    .locator(".tree-item")
    .filter({ hasText: "Self Attention" })
    .click();
  await expect(
    page.getByRole("dialog", { name: "模型与输入", exact: true }),
  ).not.toBeVisible();
  await expect(page.locator(".stage-navigation")).toBeVisible();
  await page.locator('[data-id="q-proj"]').click();
  await expect(
    page.getByRole("dialog", { name: "模块详情", exact: true }),
  ).toBeVisible();
  await expect(page.locator(".inspector .tensor-card").first()).toContainText(
    "3",
  );
  await page.keyboard.press("Escape");
  await expect(
    page.getByRole("dialog", { name: "模块详情", exact: true }),
  ).not.toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(
    390,
  );
  const canvas = await page.locator(".canvas").boundingBox();
  expect(canvas!.y).toBeLessThan(370);
  expect(canvas!.height).toBeGreaterThan(300);
});

test("100% control is exact and reduced motion disables animated edges", async ({
  page,
}) => {
  await page.getByRole("button", { name: /缩放到百分之百/ }).click();
  await expect(page.locator(".zoom-value")).toHaveText("100%");
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.getByRole("button", { name: "演示数据流", exact: true }).click();
  await expect(page.locator(".module-node.is-active")).toHaveCount(1);
  await expect(page.locator(".react-flow__edge.animated")).toHaveCount(0);
});
