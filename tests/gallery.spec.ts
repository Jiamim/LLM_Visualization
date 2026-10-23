import { selectModel } from "./model-selection";
import { expect, test, type Page } from "@playwright/test";
async function openLibrary(page: Page) {
  await page.getByRole("button", { name: /模型库 111/ }).click();
  await expect(page.getByRole("dialog", { name: /全部模型/ })).toBeVisible();
}
async function load(page: Page, name: string) {
  await openLibrary(page);
  await page.getByLabel("搜索模型库").fill(name);
  await page
    .locator(".gallery-card")
    .first()
    .getByRole("button", { name: /加载/ })
    .click();
  await expect(
    page.getByRole("dialog", { name: /全部模型/ }),
  ).not.toBeVisible();
}
test.beforeEach(async ({ page }) => {
  await page.goto("/");
});

test("shows all 111 deduplicated models, filters by organization and mechanism, then restores the list", async ({
  page,
}) => {
  await openLibrary(page);
  await expect(page.locator(".gallery-card")).toHaveCount(111);
  await page.getByLabel("搜索模型库").fill("Qwen3 Next");
  await expect(page.locator(".gallery-card")).toHaveCount(1);
  await expect(page.locator(".gallery-card")).toContainText("DeltaNet");
  await page.getByLabel("筛选机构").selectOption("Alibaba Cloud");
  await page.getByLabel("筛选结构").selectOption("moe");
  await expect(page.locator(".gallery-card")).toHaveCount(1);
  await page.getByLabel("筛选结构").selectOption("dense");
  await expect(page.locator(".gallery-empty")).toBeVisible();
  await page.getByLabel("筛选机构").selectOption("all");
  await page.getByLabel("筛选结构").selectOption("all");
  await page.getByLabel("搜索模型库").fill("no-such-model");
  await expect(page.locator(".gallery-empty")).toBeVisible();
  await page.getByLabel("搜索模型库").fill("");
  await page.getByLabel("筛选结构").selectOption("hybrid");
  await expect(page.locator(".gallery-card").first()).toBeVisible();
  expect(await page.locator(".gallery-card").count()).toBeLessThan(111);
  await page.getByLabel("筛选结构").selectOption("all");
  await expect(page.locator(".gallery-card")).toHaveCount(111);
});

test("loads GPT-2 and MoE models, preserving source identity in exported graph", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await load(page, "GPT-2 XL");
  await expect(page.locator("h1")).toHaveText("GPT-2 XL (1.5B)");
  await expect(page.locator('[data-id="positions"]')).toContainText(
    "Learned Position",
  );
  await load(page, "GPT-OSS (20B)");
  await page.locator(".tree-item").filter({ hasText: "前馈 / 专家" }).click();
  await expect(page.locator('[data-id="router"]')).toContainText("32 experts");
  const event = page.waitForEvent("download");
  await page.getByRole("button", { name: "更多画布工具" }).click();
  await page
    .getByRole("menuitem", { name: "导出当前计算图", exact: true })
    .click();
  const download = await event;
  const stream = await download.createReadStream();
  let text = "";
  for await (const c of stream!) text += c.toString();
  const graph = JSON.parse(text);
  expect(graph.gallery.id).toBe("gpt-oss-20b");
  expect(graph.model.model_type).toBe("gpt_oss");
  expect(graph.gallery.url).toContain("sebastianraschka");
  expect(errors).toEqual([]);
});

test("hybrid models separate Mamba states and MoE entries", async ({
  page,
}) => {
  await load(page, "Nemotron 3 Super");
  await page.locator(".tree-item").filter({ hasText: "混合算子" }).click();
  await expect(page.locator('[data-id="state-update"]')).toContainText(
    "MAMBA State Update",
  );
  await page.getByRole("button", { name: "Decode", exact: true }).click();
  await page.getByLabel("Past KV length").fill("4096");
  await page.locator(".tree-item").filter({ hasText: "前馈 / 专家" }).click();
  await expect(page.locator('[data-id="no-ffn"]')).toBeVisible();
  await page.getByLabel("选择 Transformer 层").selectOption("1");
  await expect(page.locator('[data-id="router"]')).toContainText("512 experts");
  await page.locator(".tree-item").filter({ hasText: "状态与缓存" }).click();
  await expect(page.locator('[data-id="cache-0"]')).toContainText("MAMBA");
});

test("derived sources are explicit and Gemma/DeepSeek specialized adapters remain available", async ({
  page,
}) => {
  await load(page, "BTL-3");
  await expect(page.locator(".catalog-provenance")).toContainText("来源说明");
  await page.getByRole("button", { name: "范围说明", exact: true }).click();
  await expect(page.locator(".raw-dialog")).toContainText("LoRA");
  await page.getByRole("button", { name: "关闭原始配置" }).click();
  await load(page, "Gemma 4 (E2B)");
  await expect(page.locator(".gemma-summary")).toBeVisible();
  await expect(
    page.locator(".tree-item").filter({ hasText: "Per-Layer Embeddings" }),
  ).toBeVisible();
  await load(page, "DeepSeek V4.1");
  await expect(page.locator(".ced-summary")).toContainText("20 Encoder");
  await selectModel(page, "qwen3");
  await expect(page.locator(".catalog-scope")).toHaveCount(0);
  await expect(page.locator(".catalog-provenance")).toHaveCount(0);
});

test("library is usable on mobile and closes safely during model selection", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole("button", { name: /模型库/ }).click();
  await page.getByLabel("搜索模型库").fill("Qwen3 Next");
  await page
    .locator(".gallery-card")
    .first()
    .getByRole("button", { name: /加载/ })
    .click();
  await expect(page.locator("h1")).toContainText("Qwen3 Next");
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth > innerWidth,
    ),
  ).toBe(false);
});
