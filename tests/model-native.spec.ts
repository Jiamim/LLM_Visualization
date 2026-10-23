import { openQwenModel } from "./model-selection";
import { test, expect } from "@playwright/test";
test.use({ launchOptions: { args: ["--enable-unsafe-swiftshader"] } });
test("same renderer follows model selection, keeps full layer branches, and resets model state", async ({
  page,
}) => {
  test.setTimeout(120000);
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await openQwenModel(page, "/");
  await page.getByRole("button", { name: "模型 3D 推理", exact: true }).click();
  const scene = page.getByTestId("tensor-space");
  await expect(scene).toHaveAttribute("data-renderer", "upstream-primitives");
  await expect(scene).toHaveAttribute("data-model-hidden", "3584");
  await page.getByLabel("演示模型层级").selectOption("0");
  for (const id of [
    "q-proj",
    "k-proj",
    "v-proj",
    "scores",
    "up-proj",
    "down-proj",
    "attention-residual",
    "mlp-residual",
  ])
    await expect(page.locator(`[data-spatial-id="${id}"]`)).toHaveCount(1);
  await page.getByLabel("定位三维模块").selectOption("q-proj");
  await expect(page.locator(".component-preview-3d canvas")).toHaveCount(1);
  await page.getByRole("button", { name: "在 3D 演示中切换模型" }).click();
  await page
    .locator('[data-gallery-id="gemma-4-e2b"]')
    .getByRole("button")
    .click();
  await expect(scene).toHaveAttribute("data-model-hidden", "1536");
  await expect(page.locator(".inference-demo")).toHaveAttribute(
    "data-status",
    "ready",
  );
  await page.getByLabel("演示模型层级").selectOption("0");
  await expect(page.locator('[data-visual-kind="residual"]')).not.toHaveCount(
    0,
  );
  await page.getByRole("button", { name: "在 3D 演示中切换模型" }).click();
  await page
    .locator('[data-gallery-id="deepseek-v4-1-flash"]')
    .getByRole("button")
    .click();
  await expect(scene).toHaveAttribute("data-model-hidden", "5120");
  await page.getByLabel("演示模型层级").selectOption("0");
  await expect(page.locator('[data-visual-kind="experts"]')).not.toHaveCount(0);
  expect(errors).toEqual([]);
});
test("mobile model chooser returns to the same native 3D experience", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await openQwenModel(page, "/?demo=1");
  await expect(
    page.getByRole("button", { name: "在 3D 演示中切换模型" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "开始演示", exact: true }).click();
  await page.getByRole("button", { name: "暂停", exact: true }).click();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(
    390,
  );
  expect(
    (await page.locator(".demo-scene").boundingBox())!.height,
  ).toBeGreaterThan(220);
});
