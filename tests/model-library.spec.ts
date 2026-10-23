import { expect, test } from "@playwright/test";
import { selectModel } from "./model-selection";
import { readFileSync } from "node:fs";

test.beforeEach(async ({ page }) => {
  await page.goto("/");
});

test("sidebar and header open the same deduplicated searchable models", async ({
  page,
}) => {
  await expect(page.getByLabel("选择示例模型")).toHaveCount(0);
  await expect(page.locator(".library-sidebar-button")).toHaveCount(0);
  await page
    .getByRole("button", { name: "选择或搜索模型", exact: true })
    .click();
  await expect(
    page.getByRole("dialog", { name: "全部模型 111" }),
  ).toBeVisible();
  await expect(page.locator(".gallery-card")).toHaveCount(111);
  await expect(page.locator(".gallery-card[data-current]")).toHaveAttribute(
    "data-gallery-id",
    "deepseek-v4-1-flash",
  );
  await expect(page.getByLabel("搜索模型库")).toBeFocused();
  await expect(page.locator(".gallery-intro")).toContainText("111 个画廊模型");
  await expect(page.locator(".gallery-intro")).not.toContainText("补充模型");
  for (const query of ["Qwen2.5", "Mistral-7B"]) {
    await page.getByLabel("搜索模型库").fill(query);
    await expect(page.locator(".gallery-card")).toHaveCount(0);
    await expect(page.locator(".gallery-empty")).toBeVisible();
  }
  for (const [query, id] of [
    ["Llama3.2 1B", "llama-3-2-1b"],
    ["Mistral Small 3.1", "mistral-small-3-1-24b"],
    ["Qwen3-0.6B", "qwen3-0-6b"],
    ["Gemma4 E2B", "gemma-4-e2b"],
  ]) {
    await page.getByLabel("搜索模型库").fill(query);
    await expect(page.locator(".gallery-card")).toHaveCount(1);
    await expect(page.locator(".gallery-card")).toHaveAttribute(
      "data-gallery-id",
      id,
    );
  }
  await page.getByRole("button", { name: "关闭模型库" }).click();
  await page.getByRole("button", { name: "模型库 111", exact: true }).click();
  await expect(page.locator(".gallery-card")).toHaveCount(111);
  await page.keyboard.press("Escape");
  await expect(
    page.getByRole("button", { name: "模型库 111", exact: true }),
  ).toBeFocused();
});

test("variant selection updates actual source, context, and current card across reopen", async ({
  page,
}) => {
  await selectModel(page, "qwen3");
  await expect(page.locator(".current-model-text strong")).toHaveText(
    "Qwen3 (0.6B) · 对话版",
  );
  await expect(page.locator(".model-source")).toContainText("Qwen/Qwen3-0.6B");
  await page.getByRole("button", { name: "选择或搜索模型" }).click();
  await page.getByLabel("搜索模型库").fill("qwen3 0.6b");
  const card = page.locator(".gallery-card");
  await expect(card).toHaveCount(1);
  await expect(card).toHaveAttribute("data-current", "true");
  await expect(card.locator("select")).toHaveValue("qwen3");
  await expect(card).toContainText("40,960 context");
  await card.locator("select").selectOption("gallery");
  await expect(card).toContainText("32,768 context");
  await card.getByRole("button").click();
  await expect(page.locator("h1")).toHaveText("Qwen3 (0.6B)");
  await expect(page.locator(".model-source")).toContainText(
    "Qwen/Qwen3-0.6B-Base",
  );
  await page.getByRole("button", { name: "选择或搜索模型" }).click();
  await page.getByLabel("搜索模型库").fill("Qwen3 0.6B");
  await expect(page.locator(".gallery-card select")).toHaveValue("gallery");
});

test("custom imports are reachable from the library and remain identified separately", async ({
  page,
}) => {
  await page.getByRole("button", { name: "选择或搜索模型" }).click();
  await page.getByRole("button", { name: "导入自定义配置" }).click();
  await expect(
    page.getByRole("dialog", { name: /全部模型/ }),
  ).not.toBeVisible();
  await expect(page.locator(".import-dialog")).toBeVisible();
  await page.getByRole("button", { name: "关闭导入窗口" }).click();
  await expect(page.locator(".current-model-text strong")).toHaveText(
    "DeepSeek V4.1 Flash",
  );
  await page.getByRole("button", { name: "选择或搜索模型" }).click();
  await page.getByRole("button", { name: "导入自定义配置" }).click();
  await page.getByRole("button", { name: "粘贴 JSON" }).click();
  await page
    .getByLabel("JSON 配置内容")
    .fill(
      readFileSync(
        new URL("../src/data/qwen3-0.6b.json", import.meta.url),
        "utf8",
      ),
    );
  await page.getByRole("button", { name: "生成模型视图" }).click();
  await expect(page.locator(".current-model-text strong")).toHaveText(
    "Qwen3ForCausalLM",
  );
  await expect(page.locator(".model-source")).toContainText(
    "自定义 · 粘贴的 config.json",
  );
  await page.getByRole("button", { name: "选择或搜索模型" }).click();
  await expect(page.locator(".gallery-card[data-current]")).toHaveCount(0);
  await expect(page.locator(".gallery-card")).toHaveCount(111);
});

test("the model selector works from the mobile configuration drawer", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole("button", { name: "切换模型面板" }).click();
  await page.getByRole("button", { name: "选择或搜索模型" }).click();
  await page.getByLabel("搜索模型库").fill("Qwen3 0.6B");
  await page.locator(".gallery-card").getByRole("button").click();
  await expect(page.locator("h1")).toHaveText("Qwen3 (0.6B)");
  await expect(
    page.getByRole("dialog", { name: "模型与输入", exact: true }),
  ).not.toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth > innerWidth,
    ),
  ).toBe(false);
});
