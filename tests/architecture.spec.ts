import { expect, test } from "@playwright/test";
import { selectModel, openQwenModel } from "./model-selection";
test.use({ launchOptions: { args: ["--enable-unsafe-swiftshader"] } });

test("complete architecture exposes two residual merges and only one repeated dense block", async ({
  page,
}) => {
  await openQwenModel(page, "/");
  await expect(
    page.locator('[data-id="decoder"] .node-repeat-badge'),
  ).toHaveText("× 28");
  await page.getByRole("button", { name: "展开整体架构 · 含残差" }).click();
  await expect(page.locator(".architecture-frame")).toHaveCount(1);
  await expect(page.locator(".architecture-repeat-bracket strong")).toHaveText(
    "× 28",
  );
  await expect(page.locator(".residual-node")).toHaveCount(2);
  await expect(page.locator(".architecture-frame-title")).toContainText(
    "L0–L27",
  );
  await page.locator('[data-id="arch-decoder-0/attention-residual"]').click();
  await expect(page.locator(".inspector h2")).toHaveText("Residual Add");
  await expect(page.locator(".inspector .tensor-card")).toHaveCount(3);
  await page.getByRole("tab", { name: "计算与参数" }).click();
  await expect(page.locator(".formula")).toContainText("residual + self_attn");
  await page
    .getByRole("button", { name: "展开 Self Attention", exact: true })
    .click();
  await expect(page.locator('[data-id="q-proj"]')).toHaveCount(1);
});

test("2D and 3D retain residuals, repetition counts and representative layer drilldown", async ({
  page,
}) => {
  await openQwenModel(page, "/");
  await page
    .getByRole("button", { name: "整体架构", exact: false })
    .first()
    .click();
  await page.getByRole("button", { name: "2D 张量", exact: true }).click();
  await expect(page.locator(".tensor-node.residual-node")).toHaveCount(2);
  await expect(page.locator(".architecture-repeat-bracket strong")).toHaveText(
    "× 28",
  );
  await page.getByRole("button", { name: "3D 空间", exact: true }).click();
  await expect(page.getByTestId("tensor-space")).toHaveAttribute(
    "data-ready",
    "true",
  );
  await expect(page.getByLabel("定位重复结构")).toContainText("× 28");
  await page
    .getByLabel("定位三维模块")
    .selectOption("arch-decoder-0/attention-residual");
  await expect(page.locator(".inspector h2")).toHaveText("Residual Add");
  await page.getByRole("button", { name: "数据流下一步" }).click();
  await expect(page.locator("[data-spatial-id][data-active]")).toHaveCount(1);
  await expect(page.locator('[data-spatial-id="arch-decoder-0"]')).toHaveCount(
    0,
  );
});

test("hybrid cycles and CED partitions preserve their distinct layer patterns", async ({
  page,
}) => {
  await openQwenModel(page, "/");
  await page.getByRole("button", { name: /模型库/ }).click();
  await page.getByLabel("搜索模型库").fill("Qwen3 Next");
  await page.locator(".gallery-card").getByRole("button").click();
  await page
    .getByRole("button", { name: "整体架构", exact: false })
    .first()
    .click();
  await expect(page.locator(".architecture-cycle")).toHaveCount(1);
  await expect(
    page.locator(".architecture-cycle > .architecture-repeat-bracket strong"),
  ).toHaveText("× 12");
  await expect(page.locator(".architecture-cycle")).toContainText(
    "4 层 / 周期",
  );
  await selectModel(page, "deepseek-v41");
  await page
    .getByRole("button", { name: "整体架构", exact: false })
    .first()
    .click();
  await expect(
    page
      .locator(".architecture-frame-title")
      .filter({ hasText: "Encoder" })
      .first(),
  ).toBeAttached();
  await expect(
    page
      .locator(".architecture-frame-title")
      .filter({ hasText: "Decoder" })
      .first(),
  ).toBeAttached();
  await expect(
    page
      .locator(".residual-node")
      .filter({ hasText: "mHC Attention Merge" })
      .first(),
  ).toBeAttached();
});

test("Gemma PLE has an explicit third residual and mobile can enter the whole architecture", async ({
  page,
}) => {
  await openQwenModel(page, "/");
  await selectModel(page, "gemma4-e2b");
  await page
    .locator(".tree-item")
    .filter({ hasText: "Transformer Block" })
    .click();
  await expect(page.locator(".residual-node")).toHaveCount(3);
  await expect(page.locator('[data-id="ple-add"]')).toContainText(
    "PLE Residual Add",
  );
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole("button", { name: "切换模型面板" }).click();
  await page
    .getByRole("button", { name: "整体架构", exact: false })
    .first()
    .click();
  await expect(page.locator(".architecture-description")).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth > innerWidth,
    ),
  ).toBe(false);
});
