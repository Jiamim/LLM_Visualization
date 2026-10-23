import { openQwenModel } from "./model-selection";
import { expect, test } from "@playwright/test";

test("tools menu supports keyboard navigation, restores focus, and preserves canvas state", async ({
  page,
}) => {
  await openQwenModel(page, "/");
  const trigger = page.getByRole("button", { name: "更多画布工具" });
  await trigger.click();
  await expect(
    page.getByRole("menuitem", { name: "导出当前计算图", exact: true }),
  ).toBeFocused();
  await page.keyboard.press("ArrowDown");
  await expect(
    page.getByRole("menuitem", { name: "专注模式", exact: true }),
  ).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(trigger).toBeFocused();
  await expect(page.getByRole("menu")).toHaveCount(0);
  await trigger.click();
  await page.getByRole("menuitem", { name: "专注模式", exact: true }).click();
  await expect(page.locator(".app-shell")).toHaveClass(/focus-mode/);
  await trigger.click();
  await page.keyboard.press("Escape");
  await expect(page.locator(".app-shell")).toHaveClass(/focus-mode/);
  await page.keyboard.press("Escape");
  await expect(page.locator(".app-shell")).not.toHaveClass(/focus-mode/);
});

test("mobile keeps playback, symbolic dimensions, and inspector available", async ({
  page,
}) => {
  await page.setViewportSize({ width: 360, height: 740 });
  await openQwenModel(page, "/");
  await expect(
    page.getByRole("button", { name: "数据流下一步" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "数据流下一步" }).click();
  await expect(page.locator(".module-node.is-active")).toHaveCount(1);
  await page.getByRole("button", { name: "更多画布工具" }).click();
  await page.getByRole("menuitem", { name: /切换符号维度/ }).click();
  await expect(
    page.locator('[data-id="embedding"] .node-footer code'),
  ).toHaveText("B × S × H");
  await page.getByRole("button", { name: "更多画布工具" }).click();
  await page.getByRole("menuitem", { name: "切换详情面板" }).click();
  await expect(
    page.getByRole("dialog", { name: "模块详情", exact: true }),
  ).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.locator(".canvas-topbar")).toHaveJSProperty(
    "scrollWidth",
    360,
  );
  expect(
    (await page.locator(".canvas-topbar").boundingBox())!.height,
  ).toBeLessThanOrEqual(58);
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(
    360,
  );
});

test("all views share the context bar and navigate back from tensor details", async ({
  page,
}) => {
  await openQwenModel(page, "/?view=2d");
  await page.getByRole("button", { name: "展开 Transformer Blocks" }).click();
  await page.getByRole("button", { name: "展开 Self Attention" }).click();
  await expect(page.locator(".canvas-contextbar")).toContainText(
    "Self Attention",
  );
  await expect(page.locator(".canvas-contextbar .shape-toggle")).toBeVisible();
  await page.getByRole("button", { name: "返回上一级", exact: true }).click();
  await expect(page.locator(".tensor-node")).toHaveCount(7);
  await page.getByRole("button", { name: "返回上一级", exact: true }).click();
  await expect(page.locator(".tensor-node")).toHaveCount(6);
  await page.getByRole("button", { name: "结构图", exact: true }).click();
  await expect(page.locator(".view-panel")).toHaveCount(0);
  const caption = (await page.locator(".canvas-contextbar").boundingBox())!,
    surface = (await page.locator(".graph-render-area").boundingBox())!;
  expect(caption.y + caption.height).toBeLessThanOrEqual(surface.y + 1);
});

test("model library keeps stable search controls and clears combined filters", async ({
  page,
}) => {
  await openQwenModel(page, "/");
  await page.getByRole("button", { name: /模型库/ }).click();
  const before = await page.locator(".gallery-filters").boundingBox();
  await page.getByLabel("筛选机构").selectOption("Alibaba Cloud");
  await page.getByLabel("筛选结构").selectOption("dense");
  await page.getByLabel("搜索模型库").fill("no-such-model");
  await expect(page.locator(".gallery-empty")).toBeVisible();
  expect(await page.locator(".gallery-filters").boundingBox()).toEqual(before);
  await page.getByRole("button", { name: "清除条件，查看全部" }).click();
  await expect(page.locator(".gallery-card")).toHaveCount(111);
  await expect(page.getByLabel("搜索模型库")).toBeFocused();
  await page.getByLabel("搜索模型库").fill("Gemma4 E2B");
  await page.getByRole("button", { name: "清空搜索", exact: true }).click();
  await expect(page.locator(".gallery-card")).toHaveCount(111);
  await expect(page.getByLabel("搜索模型库")).toBeFocused();
});

test("2D recenters the selected tensor after resizing to a phone", async ({
  page,
}) => {
  await openQwenModel(page, "/?view=2d");
  await page
    .locator(".tree-item")
    .filter({ hasText: "Self Attention" })
    .click();
  await page.getByRole("button", { name: "03 注意力计算" }).click();
  await expect(page.locator('[data-id="scores"] .tensor-node')).toHaveClass(
    /is-selected/,
  );
  await page.setViewportSize({ width: 390, height: 844 });
  await expect
    .poll(async () => {
      const box = await page.locator('[data-id="scores"]').boundingBox();
      const canvas = await page.locator(".tensor-render-area").boundingBox();
      return (
        !!box &&
        !!canvas &&
        box.x >= canvas.x &&
        box.x + box.width <= canvas.x + canvas.width
      );
    })
    .toBe(true);
});
