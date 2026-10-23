# 所选模型的原站风格 3D

入口是工作台顶栏「模型 3D 推理」或「示例推理」。演示内可直接切换统一模型库中的任意模型；配置、维度、层结构与动画均随所选模型变化。发行包仅包含多模型工作台，图元副本和播放逻辑随主项目构建。

## 实现

直接复用 LLM-Visualized 的 `WeightMatrixVisualization` 和 `VectorVisualizationInstancedPrism` 及其图元依赖，保留上游出处与 ISC 声明。构造函数传入当前模型的显式维度、可视采样数和通道数。新增 `referenceVisual` 将原有架构适配器的节点转换为几何语义，`reference-primitives` 组合出模型场景。

- 权重：原站的梯形矩阵与材质，扩张 / 压缩比例随输入输出末维变化。
- 向量：原站实例化渐变 prism，按 token 沿深度排列；最多显示 24 个维度位置、16 条 token 通道。
- 注意力：Query / KV 头数从当前层与张量读取；紧凑视图最多显示 8 个头，侧栏 Head 选择器可切换；按因果关系和局部窗口展示屏蔽。
- Norm、激活、残差、缓存、递归状态、路由和专家使用对应图元；GELU / SiLU / ReLU 曲线由所选模型激活函数计算。mHC 单独标注为门控混合。
- 当前层的 Attention、MLP 与残差始终共存。动画移动向量、扫描矩阵、逐行揭示注意力，再沿连线传递输出；暂停保留进度和镜头。
- 完整模型保留可点击的紧凑层级概览，当前层使用详细图元；大量层使用原站梯形几何的实例化概览以控制 WebGL 成本。
- 侧栏嵌入同源组件的独立可旋转 3D 预览，并保留公式、输入输出维度、解释与矩阵查询。
- 演示内切换模型复用统一模型库，销毁旧 WebGL 上下文，重新建立模型、token 和缓存状态。

## 数值与复刻范围

这次迁移的是可重用图元、视觉语言和交互到多模型渲染器，不再将其他模型跳转到 GPT-2 演示。它不是原站整套 GPT-2 运行控制器的逐帧副本：模型结构不同，场景布局和阶段编排也随结构变化。

仅凭 config.json 可以知道结构与形状，不能得到权重、激活值、attention 概率或实际专家选择。未导入记录时，色带、注意力明暗、专家路线和 token 使用明确标注的教学示意。导入匹配运行记录时，3D 输出向量 / attention 坐标与矩阵面板读取相同记录；未采样坐标为灰色，不补造数值。大张量和专家数量压缩展示，详情仍保留完整维度与数量。

## 验证

新增全模型测试遍历 111 个目录条目的首尾层，检查几何尺寸、节点唯一性、连线完整性，并验证这些层中的每个播放事件都能映射回正确视图与组件。现有缓存、架构、配置与演示测试继续运行。全量单元测试 1021 项通过；最后一轮多模型与播放回归 13 项通过；工作台 3D 的 7 项测试在此次修改中亦已通过。生产构建成功，生产预览中验证了组件选择、注意力预览、播放暂停与 390px 手机布局，无页面脚本错误。

浏览器测试覆盖 Qwen → Gemma → DeepSeek 的演示内切换、持久层场景、侧栏预览、手机布局；另回归开始 / 暂停 / 重启、缓存切换、导入记录和 WebGL 回退。

截图：
- [Qwen 的 Q/K/V](qwen-attention.png)
- [注意力头与侧栏](attention-heads.png)
- [手机布局](mobile.png)
- [紧凑模型全景](full-model.png)
- [Gemma E2B](gemma-4-e2b.png)
- [DeepSeek V4.1 Flash](deepseek-v4-1-flash.png)
- [Qwen3 Next 混合结构](qwen3-next-80b-a3b.png)

源码来源：[saahithjanapati/llm-visualized](https://github.com/saahithjanapati/llm-visualized)，固定提交 `a354ca1fd303ae4aeec50722dca8f049cfd5c675`。图元副本位于 `src/vendor/llm-visualized/`；声明见 [NOTICE.md](../../src/vendor/llm-visualized/NOTICE.md)，生产构建携带 [第三方声明](../../public/THIRD-PARTY-NOTICES.txt)。
