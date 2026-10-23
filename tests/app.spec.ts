import { selectModel, openQwenModel } from "./model-selection";
import { expect, test } from "@playwright/test";
import { readFileSync } from "node:fs";

const presets = ["qwen2.5-7b", "qwen3-0.6b", "llama3.2-1b", "mistral-7b"].map(
  (name) => ({
    raw: JSON.parse(
      readFileSync(
        new URL(
          name === "qwen2.5-7b" || name === "mistral-7b"
            ? `./fixtures/${name}.json`
            : `../src/data/${name}.json`,
          import.meta.url,
        ),
        "utf8",
      ),
    ),
  }),
);

test.beforeEach(async ({ page }) => {
  await page.route("https://fonts.googleapis.com/**", (route) => route.abort());
  await openQwenModel(page, "/");
});

test("explores the model, focuses modules and updates decode tensor shapes", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await expect(
    page.getByRole("heading", { name: "Qwen2.5-7B-Instruct" }),
  ).toBeVisible();
  await page.evaluate(() => document.fonts.ready);
  await expect(page.locator(".react-flow__minimap-node")).toHaveCount(0);
  await page.screenshot({ path: "test-results/overview.png", fullPage: true });
  await page.getByRole("button", { name: "展开 Transformer Blocks" }).click();
  await expect(page.locator(".react-flow__node")).toHaveCount(7);
  await page.getByRole("button", { name: "展开 Self Attention" }).click();
  await expect(page.locator(".react-flow__node")).toHaveCount(18);
  await expect(page.locator(".react-flow__minimap-node")).toHaveCount(18);
  await page.getByRole("button", { name: "Decode", exact: true }).click();
  await page.getByLabel("Batch size").fill("2");
  await page.getByLabel("Past KV length").fill("64");
  const viewportBefore = await page
    .locator(".react-flow__viewport")
    .getAttribute("style");
  await page.getByRole("button", { name: "03 注意力计算" }).click();
  await page.locator('[data-id="scores"]').click();
  await expect(page.locator(".inspector h2")).toHaveText("Attention Scores");
  await expect(page.locator(".inspector .tensor-card").last()).toContainText(
    "65",
  );
  await expect(page.locator(".inspector .tensor-card").last()).toContainText(
    "28",
  );
  await expect
    .poll(() => page.locator(".react-flow__viewport").getAttribute("style"))
    .not.toBe(viewportBefore);
  await page.getByRole("button", { name: /切换符号维度/ }).click();
  await expect(page.locator('[data-id="scores"] .node-footer code')).toHaveText(
    "B × Nq × S × T",
  );
  expect(errors).toEqual([]);
});

test("imports local and pasted configs, rejecting unsupported models without losing the graph", async ({
  page,
}) => {
  await page.getByRole("button", { name: "导入模型", exact: true }).click();
  await page.getByRole("button", { name: "粘贴 JSON" }).click();
  await page.getByLabel("JSON 配置内容").fill('{"model_type":"bert"}');
  await page.getByRole("button", { name: "生成模型视图" }).click();
  await expect(page.getByRole("alert")).toContainText("暂不支持");
  await page.getByLabel("JSON 配置内容").fill(JSON.stringify(presets[1].raw));
  await page.getByRole("button", { name: "生成模型视图" }).click();
  await expect(
    page.getByRole("heading", { name: "Qwen3ForCausalLM" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "导入模型", exact: true }).click();
  await page.getByRole("button", { name: "本地文件" }).click();
  await page.locator("input[type=file]").setInputFiles({
    name: "llama-config.json",
    mimeType: "application/json",
    buffer: Buffer.from(JSON.stringify(presets[2].raw)),
  });
  await expect(
    page.getByText("llama-config.json", { exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "生成模型视图" }).click();
  await expect(
    page.getByRole("heading", { name: "llama-config.json" }),
  ).toBeVisible();
});

test("loads a Hub configuration and shows actionable errors for gated repositories", async ({
  page,
}) => {
  await page.route("https://huggingface.co/**/config.json", (route) =>
    route.fulfill({ status: 403 }),
  );
  await page.getByRole("button", { name: "导入模型", exact: true }).click();
  await page.getByRole("button", { name: "Hugging Face", exact: true }).click();
  await page.getByRole("button", { name: "生成模型视图" }).click();
  await expect(page.getByRole("alert")).toContainText("需要访问权限");
  await page.unroute("https://huggingface.co/**/config.json");
  await page.route("https://huggingface.co/**/config.json", (route) =>
    route.fulfill({ json: presets[1].raw }),
  );
  await page.getByLabel("模型 ID").fill("Qwen/Qwen3-0.6B");
  await page.getByRole("button", { name: "生成模型视图" }).click();
  await expect(page.getByRole("heading", { name: "Qwen3-0.6B" })).toBeVisible();
  await expect(page.locator(".stat").first()).toContainText("596.05M");
});

test("switches presets, exposes original config and exports a reproducible graph", async ({
  page,
}) => {
  await selectModel(page, "qwen3");
  await expect(page.locator(".stat").first()).toContainText("596.05M");
  await page.getByRole("button", { name: "查看原始配置" }).click();
  await expect(page.locator(".raw-dialog pre")).toContainText(
    '"head_dim": 128',
  );
  await page.getByRole("button", { name: "关闭原始配置" }).click();
  const downloaded = page.waitForEvent("download");
  await page.getByRole("button", { name: "更多画布工具" }).click();
  await page
    .getByRole("menuitem", { name: "导出当前计算图", exact: true })
    .click();
  const download = await downloaded;
  expect(download.suggestedFilename()).toBe("qwen3-overview-graph.json");
  const stream = await download.createReadStream();
  let text = "";
  for await (const chunk of stream!) text += chunk.toString();
  const exported = JSON.parse(text);
  expect(exported.model.model_type).toBe("qwen3");
  expect(exported.scenario.sequence).toBe(128);
  expect(exported.nodes).toHaveLength(6);
});

test("allows edge inspection, playback, and responsive exploration", async ({
  page,
}) => {
  await page.locator(".react-flow__edge").first().click();
  await expect(page.locator(".inspector h2")).toHaveText("input_ids");
  await page.getByRole("button", { name: "演示数据流" }).click();
  await expect(page.locator(".playback-status")).toBeVisible();
  await expect(page.locator(".module-node.is-active")).toHaveCount(1);
  await page.getByRole("button", { name: "暂停", exact: true }).click();
  await page.getByRole("button", { name: "更多画布工具" }).click();
  await page.getByRole("menuitem", { name: "重置数据流", exact: true }).click();
  await expect(page.locator(".playback-status")).toHaveCount(0);
  await page.screenshot({ path: "test-results/desktop.png", fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(
    page.getByRole("button", { name: "导入模型", exact: true }),
  ).toBeVisible();
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth > window.innerWidth,
  );
  expect(overflow).toBe(false);
  await expect
    .poll(async () => {
      const node = await page.locator('[data-id="decoder"]').boundingBox();
      return !!node && node.x >= 0 && node.x + node.width <= 390;
    })
    .toBe(true);
  await page.screenshot({ path: "test-results/mobile.png", fullPage: true });
});
