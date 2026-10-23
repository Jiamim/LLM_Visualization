import { expect, type Page } from "@playwright/test";
import { readFileSync } from "node:fs";
const presets: Record<string, { entry: string; variant?: string }> = {
  qwen3: { entry: "qwen3-0-6b", variant: "qwen3" },
  llama: { entry: "llama-3-2-1b" },
  "gemma4-e2b": { entry: "gemma-4-e2b" },
  "gemma4-e4b": { entry: "gemma-4-e4b" },
  "deepseek-v41": { entry: "deepseek-v4-1-flash" },
};
export async function selectModel(page: Page, preset: string) {
  const selected = presets[preset];
  if (!selected) throw new Error(`Unknown test model ${preset}`);
  await page.getByRole("button", { name: /^模型库(?: \d+)?$/ }).click();
  const dialog = page.getByRole("dialog", { name: /全部模型/ });
  await expect(dialog).toBeVisible();
  const card = dialog.locator(`[data-gallery-id="${selected.entry}"]`);
  if (selected.variant)
    await card.locator("select").selectOption(selected.variant);
  await card.getByRole("button").click();
  await expect(dialog).not.toBeVisible();
}

/** Import the dense regression fixture independently of the shipped model library. */
export async function openQwenModel(page: Page, path = "/") {
  const url = new URL(path, "http://localhost");
  const demo = url.searchParams.get("demo") === "1";
  if (demo) url.searchParams.delete("demo");
  await page.goto(`${url.pathname}${url.search}`);
  await page.getByRole("button", { name: "导入模型", exact: true }).click();
  await page.getByRole("button", { name: "本地文件" }).click();
  await page.locator(".import-dialog input[type=file]").setInputFiles({
    name: "Qwen2.5-7B-Instruct",
    mimeType: "application/json",
    buffer: readFileSync(new URL("./fixtures/qwen2.5-7b.json", import.meta.url)),
  });
  await page.getByRole("button", { name: "生成模型视图" }).click();
  await expect(page.locator(".import-dialog")).not.toBeVisible();
  if (demo) {
    await page.evaluate((route) => history.replaceState(null, "", route), path);
    await page.getByRole("button", { name: "示例推理", exact: true }).click();
  }
}
