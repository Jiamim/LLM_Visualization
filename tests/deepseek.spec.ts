import { selectModel } from "./model-selection";
import { expect, test } from "@playwright/test";
import { readFileSync } from "node:fs";
const raw = JSON.parse(
  readFileSync(
    new URL("../src/data/deepseek-v4.1-flash.json", import.meta.url),
    "utf8",
  ),
);
test.beforeEach(async ({ page }) => {
  await page.goto("/");
  await selectModel(page, "deepseek-v41");
});

test("imports CED, exposes encoder/decoder groups and follows exact cache owners", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await expect(
    page.getByRole("heading", { name: "DeepSeek V4.1-Flash (552B)" }),
  ).toBeVisible();
  await expect(page.locator(".ced-summary")).toContainText(
    "20 Encoder + 20 Decoder",
  );
  await expect(page.locator(".stat").first()).toContainText("763.21B");
  await page.locator(".tree-item").filter({ hasText: "CED Decoder" }).click();
  await expect(page.locator('[data-id="group-24"]')).toContainText("Reindex");
  await page.getByLabel("选择 Transformer 层").selectOption("24");
  await page
    .locator(".tree-item")
    .filter({ hasText: "CSA2 Attention" })
    .click();
  await page.getByRole("button", { name: "02 全局 KV 与索引" }).click();
  await page.locator('[data-id="global-kv"]').click();
  await expect(page.locator(".inspector h2")).toHaveText(
    "Read Shared Global KV",
  );
  await expect(page.locator(".inspector")).toContainText("Owner Layer 20");
  await page.getByRole("tab", { name: "计算与参数" }).click();
  await expect(page.locator(".inspector .config-details")).toContainText(
    "Layer 24",
  );
  await page.locator(".tree-item").filter({ hasText: "共享缓存" }).click();
  await page.getByRole("button", { name: "04 KV 来源 L20" }).click();
  await expect(page.locator('[data-id="source-20"]')).toBeVisible();
  expect(errors).toEqual([]);
});

test("bounded replay changes decoder query length while preserving global KV length", async ({
  page,
}) => {
  await page.getByLabel("Sequence length").fill("1024");
  await page.getByLabel("CED 执行路径").selectOption("bounded");
  await page.getByLabel("选择 Transformer 层").selectOption("24");
  await page
    .locator(".tree-item")
    .filter({ hasText: "CSA2 Attention" })
    .click();
  await expect(page.locator('[data-id="q-lora"] .node-footer code')).toHaveText(
    "1 × 128 × 64 × 512",
  );
  await expect(
    page.locator('[data-id="global-kv"] .node-footer code'),
  ).toHaveText("1 × 1024 × 512");
  await expect(page.locator(".memory-footer")).toContainText("890.00 KiB");
  await page.getByRole("button", { name: "Decode", exact: true }).click();
  await expect(page.locator('[data-id="q-lora"] .node-footer code')).toHaveText(
    "1 × 1 × 64 × 512",
  );
  await page.getByLabel("Sequence length").fill("2");
  await expect(page.getByRole("alert")).toContainText("S=1");
  await expect(page.locator('[data-id="q-lora"] .node-footer code')).toHaveText(
    "1 × 1 × 64 × 512",
  );
});

test("supports vision dimensions, expert routing and model-specific graph export", async ({
  page,
}) => {
  await page.getByLabel("包含图像").check();
  await expect(page.getByLabel("Sequence length")).toHaveValue("212");
  await page.locator(".tree-item").filter({ hasText: "视觉编码器" }).click();
  await expect(
    page.locator('[data-id="patch-embed"] .node-footer code'),
  ).toHaveText("1764 × 1024");
  await expect(
    page.locator('[data-id="projector"] .node-footer code'),
  ).toHaveText("196 × 5120");
  await page
    .locator(".tree-item")
    .filter({ hasText: "Mixture of Experts" })
    .click();
  await page.getByRole("button", { name: "01 路由与分发" }).click();
  await page.locator('[data-id="route"]').click();
  await expect(page.locator(".inspector h2")).toHaveText(
    "Top-K Expert Routing",
  );
  await expect(page.locator(".inspector")).toContainText("6 of 384");
  const event = page.waitForEvent("download");
  await page.getByRole("button", { name: "更多画布工具" }).click();
  await page
    .getByRole("menuitem", { name: "导出当前计算图", exact: true })
    .click();
  const download = await event;
  const stream = await download.createReadStream();
  let content = "";
  for await (const c of stream!) content += c.toString();
  const graph = JSON.parse(content);
  expect(graph.model.model_type).toBe("deepseek_v41");
  expect(graph.scenario.image.count).toBe(1);
  expect(
    graph.nodes.some((n: { id: string }) => n.id === "shared-expert"),
  ).toBe(true);
});

test("imports the actual nested JSON and restores dense views after changing models", async ({
  page,
}) => {
  await page.getByRole("button", { name: "导入模型", exact: true }).click();
  await page.getByRole("button", { name: "粘贴 JSON" }).click();
  await page.getByLabel("JSON 配置内容").fill(JSON.stringify(raw));
  await page.getByRole("button", { name: "生成模型视图" }).click();
  await expect(
    page.getByRole("heading", { name: "DeepseekV41ForCausalLM" }),
  ).toBeVisible();
  await page.locator(".tree-item").filter({ hasText: "Engram" }).click();
  await expect(page.locator('[data-id="engram-table"]')).toBeVisible();
  await selectModel(page, "qwen3");
  await expect(page.locator(".ced-summary")).toHaveCount(0);
  await page.getByRole("button", { name: "展开 Transformer Blocks" }).click();
  await page.getByRole("button", { name: "展开 Self Attention" }).click();
  await expect(page.locator('[data-id="q-proj"] .node-footer code')).toHaveText(
    "1 × 128 × 2048",
  );
});
