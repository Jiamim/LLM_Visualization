import { openQwenModel } from "./model-selection";
import { expect, test } from "@playwright/test";
test.use({ launchOptions: { args: ["--enable-unsafe-swiftshader"] } });
test("defaults to a readable pace and preserves token, camera, and fractional progress on pause", async ({
  page,
}) => {
  await openQwenModel(page, "/?demo=1");
  const demo = page.locator(".inference-demo");
  await expect(page.getByLabel("演示节奏", { exact: true })).toHaveValue(
    "guided",
  );
  await expect(demo).toHaveAttribute("data-duration", "4200");
  await page.getByRole("button", { name: "开始演示", exact: true }).click();
  await page.waitForTimeout(1200);
  await expect(demo).toHaveAttribute("data-step", "0");
  await page.getByRole("button", { name: "暂停", exact: true }).click();
  const progress = await demo.getAttribute("data-progress"),
    token = await page
      .getByTestId("tensor-space")
      .getAttribute("data-active-token");
  await page.waitForTimeout(200);
  const target = await page
    .locator(".tensor-space-host canvas")
    .getAttribute("data-camera-target");
  await page.waitForTimeout(800);
  await expect(demo).toHaveAttribute("data-progress", progress!);
  await expect(page.getByTestId("tensor-space")).toHaveAttribute(
    "data-active-token",
    token!,
  );
  await expect(page.locator(".tensor-space-host canvas")).toHaveAttribute(
    "data-camera-target",
    target!,
  );
  await page.getByLabel("演示速度").selectOption("0.25");
  await expect(demo).toHaveAttribute("data-duration", "16800");
  await expect(demo).toHaveAttribute("data-progress", progress!);
});
test("step learning pauses automatically after each operation and continues explicitly", async ({
  page,
}) => {
  await openQwenModel(page, "/?demo=1");
  await page.getByLabel("演示节奏", { exact: true }).selectOption("step");
  await page.getByLabel("演示速度").selectOption("4");
  await page.getByRole("button", { name: "开始演示", exact: true }).click();
  await expect(page.locator(".inference-demo")).toHaveAttribute(
    "data-status",
    "paused",
    { timeout: 6000 },
  );
  await expect(page.locator(".inference-demo")).toHaveAttribute(
    "data-step",
    "1",
  );
  await page.waitForTimeout(800);
  await expect(page.locator(".inference-demo")).toHaveAttribute(
    "data-step",
    "1",
  );
  await page.getByRole("button", { name: "继续", exact: true }).click();
  await expect(page.locator(".inference-demo")).toHaveAttribute(
    "data-step",
    "2",
    { timeout: 6000 },
  );
  await expect(page.locator(".inference-demo")).toHaveAttribute(
    "data-status",
    "paused",
  );
});
test("mobile rhythm controls and walkthrough keep a visible scene", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await openQwenModel(page, "/?demo=1");
  await expect(
    page.getByRole("button", { name: "开始演示", exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "开始演示", exact: true }).click();
  await page.getByRole("button", { name: "暂停", exact: true }).click();
  await expect(page.locator(".demo-scene-caption")).toContainText("已暂停");
  expect(
    (await page.locator(".demo-scene").boundingBox())!.height,
  ).toBeGreaterThan(220);
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(
    390,
  );
});
