import { openQwenModel } from "./model-selection";
import { expect, test } from "@playwright/test";
test.use({ launchOptions: { args: ["--enable-unsafe-swiftshader"] } });
test("route opens 3D demo and detailed explanation, matrix and math are interactive", async ({
  page,
}) => {
  await openQwenModel(page, "/?demo=1&token=4&generation=0&kvCache=0");
  await expect(page.getByTestId("tensor-space")).toHaveAttribute(
    "data-world-layers",
    "28",
  );
  await page.getByRole("button", { name: "下一轮", exact: true }).click();
  await page.getByLabel("演示模型层级").selectOption("0");
  await page.getByLabel("定位三维模块").selectOption("input-norm");
  await expect(
    page.getByRole("complementary", { name: "演示组件详解" }),
  ).toBeVisible();
  await expect(page.locator(".demo-inspector")).toContainText("为什么需要它");
  await page.getByRole("button", { name: "张量 / 矩阵", exact: true }).click();
  await page.getByRole("button", { name: "行 0 列 0", exact: true }).click();
  await expect(page.locator(".demo-cell-readout")).toContainText(
    "未加载运行数据",
  );
  await page.getByLabel("矩阵起始列").fill("16");
  await expect(
    page.getByRole("button", { name: "行 0 列 16", exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "数学示例", exact: true }).click();
  await expect(page.locator(".demo-detail-scroll")).toContainText(
    "RMSNorm 示例",
  );
  const before = await page.locator(".demo-detail-scroll code").innerText();
  await page.getByLabel("数学示例输入").fill("2");
  await expect(page.locator(".demo-detail-scroll code")).not.toHaveText(before);
  await page.getByLabel("解释面板注意力头").selectOption("1");
  await expect(page.getByLabel("解释面板注意力头")).toHaveValue("1");
  await page.getByRole("button", { name: "关闭组件详解" }).click();
  await page.getByRole("button", { name: "全景查看完整模型" }).click();
  await expect(page.getByTestId("tensor-space")).toHaveAttribute(
    "data-world-layers",
    "28",
  );
});
test("trace template imports matching data and rejects invalid coordinates", async ({
  page,
}) => {
  await openQwenModel(page, "/?demo=1");
  await page.getByRole("button", { name: "演示设置", exact: true }).click();
  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "下载记录模板" }).click();
  const stream = await (await download).createReadStream();
  let text = "";
  for await (const c of stream!) text += c.toString();
  const trace = JSON.parse(text);
  trace.source = "UI test fixture";
  trace.prompt[0] = { text: "fixture", id: 42 };
  trace.samples = [
    {
      pass: 0,
      layer: 0,
      view: "overview",
      nodeId: "input",
      direction: "output",
      tensorIndex: 0,
      shape: [1, 4],
      indices: [[0, 0]],
      values: [42],
    },
  ];
  await page.getByLabel("导入运行记录文件").setInputFiles({
    name: "trace.json",
    mimeType: "application/json",
    buffer: Buffer.from(JSON.stringify(trace)),
  });
  await expect(page.locator(".demo-trace-controls")).toContainText(
    "UI test fixture",
  );
  await page.getByRole("button", { name: "返回演示", exact: true }).click();
  await page.getByRole("button", { name: "组件详细解释" }).click();
  await expect(page.locator(".demo-token-meta")).toContainText("Token ID：42");
  await page.getByRole("button", { name: "张量 / 矩阵", exact: true }).click();
  await page.getByRole("button", { name: "行 0 列 0", exact: true }).click();
  await expect(page.locator(".demo-cell-readout")).toContainText("42.0000");
  await page.getByRole("button", { name: "演示设置", exact: true }).click();
  trace.samples[0].indices = [[0, 99]];
  await page.getByLabel("导入运行记录文件").setInputFiles({
    name: "wrong.json",
    mimeType: "application/json",
    buffer: Buffer.from(JSON.stringify(trace)),
  });
  await expect(page.getByRole("alert")).toContainText("坐标");
});
test("formula and brightness settings persist and detail panel fits mobile", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await openQwenModel(page, "/?demo=1");
  await page.getByRole("button", { name: "组件详细解释" }).click();
  await expect(page.locator(".demo-inspector")).toBeVisible();
  await page.getByRole("button", { name: "关闭组件详解" }).click();
  await page.getByRole("button", { name: "演示设置", exact: true }).click();
  await page.getByLabel("显示演示公式").uncheck();
  await page.getByLabel("演示场景亮度").fill("75");
  await page.getByRole("button", { name: "返回演示", exact: true }).click();
  await expect(page.locator(".demo-equation")).toHaveCount(0);
  await expect(page.locator(".tensor-space-host")).toHaveCSS(
    "filter",
    "brightness(0.75)",
  );
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(
    390,
  );
});
