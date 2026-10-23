import { selectModel } from "./model-selection";
import { expect, test } from "@playwright/test";
import { readFileSync } from "node:fs";
const e2b = JSON.parse(
  readFileSync(new URL("../src/data/gemma4-e2b.json", import.meta.url), "utf8"),
);

test.beforeEach(async ({ page }) => {
  await page.goto("/");
  await selectModel(page, "gemma4-e2b");
});

test("Gemma E2B switches local/global head dimensions and shares the correct KV owner", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await expect(
    page.getByRole("heading", { name: "Gemma 4 (E2B)" }),
  ).toBeVisible();
  await expect(page.locator(".stat").first()).toContainText("5.10B");
  await page
    .locator(".tree-item")
    .filter({ hasText: "Gemma Hybrid Attention" })
    .click();
  await expect(page.locator('[data-id="q-proj"] .node-footer code')).toHaveText(
    "1 × 8 × 128 × 256",
  );
  await page.getByLabel("选择 Transformer 层").selectOption("4");
  await expect(page.locator('[data-id="q-proj"] .node-footer code')).toHaveText(
    "1 × 8 × 128 × 512",
  );
  await page.getByLabel("选择 Transformer 层").selectOption("19");
  await expect(page.locator('[data-id="shared-kv"]')).toContainText(
    "Shared from Layer 14",
  );
  await expect(page.locator('[data-id="kv-project"]')).toHaveCount(0);
  await page.getByRole("button", { name: "01 Query 与 Key/Value" }).click();
  await page.locator('[data-id="shared-kv"]').click();
  await page.getByRole("tab", { name: "计算与参数" }).click();
  await expect(page.locator(".inspector")).toContainText("不存在");
  expect(errors).toEqual([]);
});

test("PLE and double-wide MLP follow selected layer, and E4B uses its own config", async ({
  page,
}) => {
  await page
    .locator(".tree-item")
    .filter({ hasText: "Per-Layer Embeddings" })
    .click();
  await expect(
    page.locator('[data-id="ple-table"] .node-footer code'),
  ).toHaveText("1 × 128 × 35 × 256");
  await page.getByLabel("选择 Transformer 层").selectOption("15");
  await expect(page.locator('[data-id="ple-slice"]')).toContainText(
    "Slice Layer 15",
  );
  await page
    .locator(".tree-item")
    .filter({ hasText: "Gemma Gated MLP" })
    .click();
  await expect(
    page.locator('[data-id="gate-proj"] .node-footer code'),
  ).toHaveText("1 × 128 × 12288");
  await selectModel(page, "gemma4-e4b");
  await expect(page.locator(".stat").first()).toContainText("7.94B");
  await page.getByLabel("选择 Transformer 层").selectOption("24");
  await page
    .locator(".tree-item")
    .filter({ hasText: "Gemma Gated MLP" })
    .click();
  await expect(
    page.locator('[data-id="gate-proj"] .node-footer code'),
  ).toHaveText("1 × 128 × 10240");
  await page
    .locator(".tree-item")
    .filter({ hasText: "Gemma Hybrid Attention" })
    .click();
  await expect(page.locator('[data-id="shared-kv"]')).toContainText(
    "Shared from Layer 22",
  );
});

test("sliding cache retains W-1 while the attention view includes current tokens", async ({
  page,
}) => {
  await page.getByLabel("Sequence length").fill("1024");
  await page.getByLabel("选择 Transformer 层").selectOption("13");
  await page
    .locator(".tree-item")
    .filter({ hasText: "Gemma Hybrid Attention" })
    .click();
  await expect(
    page.locator('[data-id="kv-cache"] .node-footer code'),
  ).toHaveText("1 × 1 × 1024 × 256");
  await page.getByRole("button", { name: "02 缓存状态" }).click();
  await page.locator('[data-id="kv-cache"]').click();
  await expect(page.locator(".inspector .tensor-output").nth(2)).toContainText(
    "511",
  );
  await page.getByRole("button", { name: "Decode", exact: true }).click();
  await page.getByLabel("Past KV length").fill("1024");
  await page.getByLabel("Sequence length").fill("7");
  await expect(
    page.locator('[data-id="kv-cache"] .node-footer code'),
  ).toHaveText("1 × 1 × 518 × 256");
  await page.locator(".tree-item").filter({ hasText: "共享缓存" }).click();
  await expect(page.locator('[data-id="owner-13"]')).toContainText(
    "sliding_attention",
  );
  await expect(page.locator('[data-id="owner-14"]')).toContainText(
    "full_attention",
  );
});

test("image and audio controls preserve merged-token dimensions and reject invalid grids", async ({
  page,
}) => {
  await page.getByLabel("包含图像").check();
  await expect(page.getByLabel("Sequence length")).toHaveValue("282");
  await page.getByLabel("包含音频").check();
  await expect(page.getByLabel("Sequence length")).toHaveValue("316");
  await page.locator(".tree-item").filter({ hasText: "视觉编码器" }).click();
  await expect(
    page.locator('[data-id="vision-pool"] .node-footer code'),
  ).toHaveText("1 × 280 × 768");
  await page.getByLabel("Patch grid H").fill("43");
  await expect(page.getByRole("alert")).toContainText("正整数倍");
  await expect(
    page.locator('[data-id="vision-pool"] .node-footer code'),
  ).toHaveText("1 × 280 × 768");
  await page.getByLabel("Patch grid H").fill("42");
  await page.getByLabel("Sequence length").fill("400");
  await page.getByLabel("Audio feature frames").fill("129");
  await page.locator(".tree-item").filter({ hasText: "音频编码器" }).click();
  await expect(
    page.locator('[data-id="audio-subsample"] .node-footer code'),
  ).toHaveText("1 × 33 × 1024");
  await expect(
    page.locator('[data-id="audio-language"] .node-footer code'),
  ).toHaveText("1 × 33 × 1536");
  await page.getByRole("button", { name: "Decode", exact: true }).click();
  await expect(page.locator(".module-node:not(.is-inactive)")).toHaveCount(0);
  await expect(
    page.getByRole("button", { name: "演示数据流", exact: true }),
  ).toBeDisabled();
});

test("imports official nested JSON, exports PLE graph, and resets media state on model switch", async ({
  page,
}) => {
  await page.getByRole("button", { name: "导入模型", exact: true }).click();
  await page.getByRole("button", { name: "粘贴 JSON" }).click();
  await page.getByLabel("JSON 配置内容").fill(JSON.stringify(e2b));
  await page.getByRole("button", { name: "生成模型视图" }).click();
  await expect(
    page.getByRole("heading", { name: "Gemma4ForConditionalGeneration" }),
  ).toBeVisible();
  await page.getByLabel("包含音频").check();
  await page
    .locator(".tree-item")
    .filter({ hasText: "Per-Layer Embeddings" })
    .click();
  const event = page.waitForEvent("download");
  await page.getByRole("button", { name: "更多画布工具" }).click();
  await page
    .getByRole("menuitem", { name: "导出当前计算图", exact: true })
    .click();
  const download = await event,
    stream = await download.createReadStream();
  let data = "";
  for await (const chunk of stream!) data += chunk.toString();
  const graph = JSON.parse(data);
  expect(graph.model.model_type).toBe("gemma4");
  expect(graph.scenario.audio.enabled).toBe(true);
  expect(graph.nodes.some((n: { id: string }) => n.id === "ple-table")).toBe(
    true,
  );
  await selectModel(page, "qwen3");
  await expect(page.locator(".gemma-scenario")).toHaveCount(0);
  await page.getByRole("button", { name: "展开 Transformer Blocks" }).click();
  await page.getByRole("button", { name: "展开 Self Attention" }).click();
  await expect(page.locator(".react-flow__node")).toHaveCount(18);
});
