import { expect, test } from "@playwright/test";
import { selectModel, openQwenModel } from "./model-selection";

test("cache metric opens a live formula breakdown and syncs changes back to the model", async ({
  page,
}) => {
  await openQwenModel(page, "/");
  await expect(page.getByTestId("cache-stat-value")).toContainText("7.00");
  await page
    .getByRole("button", { name: "查看 KV cache 计算", exact: true })
    .click();
  await expect(
    page.getByRole("dialog", { name: "KV cache 计算" }),
  ).toBeVisible();
  await expect(page.getByTestId("cache-kv")).toHaveText("7.00 MiB");
  await page.getByLabel("缓存批大小 B").fill("2");
  await page.getByLabel("缓存当前 token S").fill("7");
  await page.getByLabel("缓存历史 token P").fill("13");
  await expect(page.getByTestId("cache-kv")).toHaveAttribute(
    "title",
    `${2 * 28 * 2 * 4 * 20 * 128 * 2} bytes`,
  );
  await expect(page.getByTestId("cache-next")).toHaveText("112.00 KiB");
  await page.locator(".cache-row summary").first().click();
  await expect(page.locator(".cache-row-detail")).toContainText(
    "[2, 4, 20, 128]",
  );
  await expect(page.locator(".cache-formula")).toContainText("每份 × 28");
  await page.getByLabel("缓存逻辑精度").selectOption("float32");
  await expect(page.getByTestId("cache-kv")).toHaveAttribute(
    "title",
    `${2 * 28 * 2 * 4 * 20 * 128 * 4} bytes`,
  );
  await page.keyboard.press("Escape");
  await expect(
    page.getByRole("button", { name: "查看 KV cache 计算", exact: true }),
  ).toBeFocused();
  await expect(page.getByLabel("Batch size")).toHaveValue("2");
  await expect(page.getByLabel("Past KV length")).toHaveValue("13");
  await expect(page.getByLabel("计算精度", { exact: true })).toHaveValue(
    "float32",
  );
});

test("shared Gemma and DeepSeek cache reports retain their dedicated policies", async ({
  page,
}) => {
  await openQwenModel(page, "/");
  await selectModel(page, "gemma4-e2b");
  await page
    .getByRole("button", { name: "查看 KV cache 计算", exact: true })
    .click();
  await page.getByLabel("缓存当前 token S").fill("1024");
  await expect(page.getByTestId("cache-kv")).toHaveAttribute(
    "title",
    `${2 * 1 * 1 * 2 * (12 * 511 * 256 + 3 * 1024 * 512)} bytes`,
  );
  await expect(page.locator(".cache-rows")).toContainText("来源层");
  await page.getByRole("button", { name: "关闭缓存计算" }).click();
  await selectModel(page, "deepseek-v41");
  await page
    .getByRole("button", { name: "查看 KV cache 计算", exact: true })
    .click();
  await expect(page.locator(".cache-packed")).toContainText("890.00 KiB");
  await expect(page.getByTestId("cache-auxiliary")).not.toHaveText("0 B");
  await page.getByLabel("缓存历史 token P").fill("128");
  await expect(
    page.getByRole("dialog", { name: "KV cache 计算" }).getByRole("alert"),
  ).toContainText("S=1");
  await expect(page.getByLabel("缓存历史 token P")).toHaveValue("0");
});

test("library cards calculate cache for the current scenario and keep recurrent memory distinct", async ({
  page,
}) => {
  await openQwenModel(page, "/");
  await page.getByLabel("Batch size").fill("2");
  await page.getByRole("button", { name: /模型库/ }).click();
  await page.getByLabel("搜索模型库").fill("Qwen3 0.6B");
  await expect(
    page.locator('[data-gallery-id="qwen3-0-6b"] .library-cache-preview'),
  ).toContainText("28.00 MiB");
  await page.getByLabel("搜索模型库").fill("Qwen3 Next");
  await expect(page.locator(".library-cache-preview")).toContainText(
    "其他推理状态",
  );
  await page.getByLabel("搜索模型库").fill("xLSTM");
  await expect(page.locator(".library-cache-preview")).toContainText("0 B");
  await expect(page.locator(".library-cache-preview")).toContainText(
    "递归状态型",
  );
  await page.locator(".gallery-card").getByRole("button").click();
  await page
    .getByRole("button", { name: "查看 KV cache 计算", exact: true })
    .click();
  await expect(page.getByTestId("cache-kv")).toHaveText("0 B");
  await expect(page.getByTestId("cache-recurrent")).not.toHaveText("0 B");
  await expect(page.getByTestId("cache-next")).toHaveText("0 B");
});

test("mobile exposes the cache metric and detailed rows without horizontal overflow", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await openQwenModel(page, "/?view=2d");
  await expect(
    page.getByRole("button", { name: "查看 KV cache 计算", exact: true }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "查看 KV cache 计算", exact: true })
    .click();
  await page.locator(".cache-row summary").first().click();
  await expect(page.locator(".cache-row-detail")).toBeVisible();
  expect(
    await page
      .locator(".cache-dialog")
      .evaluate((el) => el.scrollWidth > el.clientWidth),
  ).toBe(false);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth > innerWidth,
    ),
  ).toBe(false);
  await page.getByRole("button", { name: "关闭缓存计算" }).click();
  await expect(page.getByTestId("model-canvas")).toHaveAttribute(
    "data-display-mode",
    "2d",
  );
});

test("export includes cache formulas, units and architecture-specific scope", async ({
  page,
}) => {
  await openQwenModel(page, "/");
  await page.getByRole("button", { name: "更多画布工具" }).click();
  const downloadEvent = page.waitForEvent("download");
  await page
    .getByRole("menuitem", { name: "导出当前计算图", exact: true })
    .click();
  const stream = await (await downloadEvent).createReadStream();
  let text = "";
  for await (const chunk of stream!) text += chunk.toString();
  const data = JSON.parse(text);
  expect(data.cache.totalBytes).toBe(7 * 1024 * 1024);
  expect(data.cache.rows[0].copies).toBe(28);
  expect(data.cache.notes.length).toBeGreaterThan(0);
});
