# Tensor Atlas

使用 Hugging Face `config.json` 可视化 Transformer 架构，逐级探索模块并查看张量维度。纯前端应用，不加载模型权重，不需要 Python、GPU 或推理服务。

## 启动

需要 Node.js 20.19+ 或 22.12+。

```bash
npm ci
npm run dev
```

打开终端打印的地址，默认 `http://localhost:5173`。开发服务监听 `0.0.0.0`；在远程服务器运行时，可以转发 5173 端口到本机。

默认加载 **DeepSeek V4.1 Flash**。工作台显示工具栏和推理演示顶部都有「全屏查看」按钮；再次点击或按 Esc 退出。工作台全屏会收起摘要和侧栏，保留视图切换及播放控件，退出后恢复原先布局。不支持浏览器全屏时自动铺满当前窗口。切换全屏不会重建 3D 场景或重置播放进度。

顶栏「模型 3D 推理」打开当前所选模型的演示，并可在演示内切换模型。所有视图使用主项目的一套依赖，安装后即可启动。

```bash
npm run build    # TypeScript 检查和工作台生产构建，产物在 dist/
npm run preview  # 预览生产构建
npm test         # 配置、形状、参数量和图结构测试
npx playwright install --with-deps chromium
npm run test:e2e # 浏览器交互测试；自动启动开发服务
```

Linux 浏览器测试需要 Chromium 的系统依赖（包括 `libgbm.so.1`）。应用运行本身不依赖这些库。如果机器通过代理联网，请将 `localhost,127.0.0.1` 加入 `NO_PROXY` / `no_proxy`，避免浏览器测试的开发服务探测请求走代理。

## 提交到 Git

提交源码、测试、验证脚本、模型配置、文档、许可文件，以及根目录的 `package.json` / `package-lock.json`。`src/vendor/llm-visualized/` 中的图元、`public/fonts/`、`public/THIRD-PARTY-NOTICES.txt` 和 `public/FONT-LICENSE.txt` 是运行资源或来源声明；`docs/` 中还包含测试使用的参考 JSON，需要保留。

`.gitignore` 已排除依赖目录 `node_modules/`、产物 `dist/`、浏览器测试输出 `test-results/` / `playwright-report/`、缓存、日志、PID 文件和本地环境文件。请保留 `public/`、`docs/` 与 `src/vendor/` 中的项目资源和声明。

清理后或克隆仓库后，运行 `npm ci` 安装依赖，再运行 `npm run dev` 启动。生产部署运行 `npm run build`，托管生成的 `dist/` 目录即可。

## 功能

- 全部模型的 3D 推理：复用原站矩阵 / 向量图元，按当前模型生成 Attention、MLP、残差、专家和状态结构，并提供可旋转组件预览。
- 导入本地配置、粘贴 JSON，或填写 Hugging Face `owner/model` 和 revision 在线读取配置。单个配置上限 2 MiB。
- 模型总览 → 指定 Transformer 层 → Self Attention / Feed Forward 的逐级视图。
- 「整体架构」展开输入、输出与代表层内部，残差旁路和合并节点显式显示；相同连续层及重复周期仅展示一次，在框旁标记 × 次数。
- 「结构图 / 2D 张量 / 3D 空间」切换，共享模型、当前层、输入场景、选中模块与数据流播放状态。
- 单击模块自动放大聚焦，双击模块或点击右上角箭头展开；面包屑返回上一级。
- 画布拖动、缩放、全图适配、聚焦选中模块、可操作的小地图。
- 长图默认进入阅读视图；按计算阶段定位 Attention / MLP 分支。手动缩放后保留视角，专注模式可收起侧栏和顶部摘要。
- 连线上的张量形状、点击连线查看详情、具体数值 / 符号维度切换。
- 实时调整 batch、当前序列长度、历史 KV cache 长度及浮点精度；提供 prefill / decode 示例。
- 展示模块参数、输入输出形状、计算公式，以及权重、KV cache、logits 的逻辑内存估算。
- 每个模型提供 KV cache 主指标与可展开的计算明细；模型库卡片按当前 B / S / P / dtype 显示缓存预览，分别列出 KV、递归和辅助状态。
- 详情面板优先显示张量维度，支持维度释义、复制和「计算与参数」页签；页签支持方向键切换。
- 按拓扑顺序演示数据流，支持暂停、单步和重置。
- 顶栏「示例推理」打开独立暗色 3D 教学演示，逐层进入 Attention / MLP、残差和状态操作，再按轮追加示例 token；支持开始、暂停、继续、重新开始、进度拖动和缓存开关。
- 导出原始配置，或导出带输入场景、节点、连线和维度的 JSON 图。
- 中文界面与移动端布局；示例、计算和界面字体均可离线使用。
- 窄屏通过抽屉打开模型配置和详情，保持画布在首屏；支持减少动态效果的系统偏好。

最新的三视图 UI 评审、改动依据和截图见 [界面评审与优化](docs/ui-review/v2/REVIEW.md)，早期结构图改进见 [初次 UI review](docs/ui-review/REVIEW.md)。画布的导出、专注模式和重置入口统一位于「更多画布工具」菜单，手机还可在该菜单切换符号维度和详情面板。

## 多模型 3D 推理

顶栏「模型 3D 推理」与「示例推理」进入同一套模型演示。直接复用原站梯形权重矩阵与实例化向量，保持当前层完整结构，支持镜头跟随、向量流转、矩阵扫描、逐头注意力、残差旁路、缓存和专家分支。演示内「切换模型」打开统一模型库。详细范围、截图与测试见 [多模型原站风格 3D](docs/model-native/README.md)。

## 图元来源

3D 工作台和模型推理演示复用 [LLM-Visualized](https://github.com/saahithjanapati/llm-visualized) 的矩阵、向量图元及其辅助模块，固定提交 `a354ca1fd303ae4aeec50722dca8f049cfd5c675`。图元保存在 `src/vendor/llm-visualized/`，由本项目的模型配置、计算图和播放控制器驱动。

来源、作者和上游 ISC 声明见 [图元声明](src/vendor/llm-visualized/NOTICE.md)，生产构建同时携带 [第三方声明](public/THIRD-PARTY-NOTICES.txt)。当前发行仅包含多模型工作台；完整 GPT-2 参考应用、录制数据与预计算场景资源已移除。

## 工作台的 2D 张量与 3D 空间

画布工具栏切换显示方式，也可通过 `?view=2d` 或 `?view=3d` 直接打开对应视图。交互参考 [LLM-Visualized](https://www.llm-visualized.com/?token=4&generation=0&kvCache=0&view=2d)，使用本项目配置推导的图与维度，图元随应用打包，不需要访问参考站点。

- **2D 张量**：以矩阵平面表示各模块的首个输出（无输出时使用首个输入）。行列对应张量最后两轴，其他轴折叠为切片；保留分支、残差和缓存连线。点击模块或连线查看全部输入输出，双击或箭头进入子模块，支持缩放、全图和选中模块聚焦。
- **3D 空间**：Three.js 使用原站同源矩阵与向量图元、专用算子几何和连接路径；空间位置表示计算布局，详情显示原始张量各轴。拖动旋转、右键或双指平移、滚轮或捏合缩放。点击块体或标签查看详情，双击可展开模块；「定位模块」支持在长图中直接查找，支持间距调整、聚焦及重置视角。
- 三种视图共用现有架构适配器，包含统一模型库的 111 个条目，以及对应架构的自定义配置。场景变化后更新维度；切换显示方式不重置模型和输入。
- **网格和几何是压缩的形状示意；未导入运行记录时，颜色与注意力明暗不代表真实激活或概率。** 完整尺寸由标签和详情提供；空张量单独标明。不同模型原有的适配和估算范围继续适用。
- 3D 代码按需加载，仅在交互或状态变化时重绘；关闭视图释放 WebGL 资源。不支持 WebGL2 或上下文丢失时提供 2D 入口。参考站点的 `token` / `generation` 参数不作为本项目的真实推理输入。

图例、截图和验证说明见 [张量视图说明](docs/tensor-views/README.md)。

## KV cache 计算

点击顶部绿色「KV cache」指标或底部「缓存 / 状态」，查看总量、每序列占用、本次净增、下一 token 净增以及逐组公式、张量形状和共享来源层。弹窗可直接修改 B / S / P / dtype，并同步输入场景。所有 111 个模型条目及其配置版本均已接入；模型库预览随可见卡片加载。

标准 MHA / GQA / MQA 采用 `2 × 层数 × B × KV头数 × (P+S) × head_dim × 字节数`。Gemma 共享层不重复累计；MLA 按压缩 latent 与 rotary key 计算；混合模型区分 K/V 与递归 / 卷积状态；DeepSeek V4.1 区分共享 KV、索引 K、压缩缓冲及全局 FP4。纯递归主干显示传统 KV 为 0，并单列非零递归状态。缓存并非模型权重，MoE 专家数不直接乘到 KV 公式中。

这是遵从各适配器口径的静态容量估算，窗口、后端缓存布局、专用算子范围在明细里说明，不表示峰值显存或所有内核的实际分配。详见 [缓存计算说明](docs/cache/README.md)。

## 整体架构与残差

左侧「整体架构」或总览中的「展开整体架构 · 含残差」打开完整路径，标准 Transformer 的代表层中直接显示 Norm → Attention → Residual Add → Norm → MLP → Residual Add。橙色虚线表示残差旁路；残差是合并算子，普通加法不增加模型权重。

相同结构的连续层只画一次，分组框旁标记 `× N` 和原层范围。混合架构按原顺序折叠重复周期，例如 Qwen3 Next 的「3 个 DeltaNet + 1 个 Attention」作为 4 层周期重复 12 次。共享 KV、局部 / 全局、维度、MoE 结构或残差形式不同的层保留差异。循环模型额外标明跨轮共享权重，重复结构不默认表示权重共享。

结构图与 2D 显示重复框和括号，3D 提供重复结构定位及次数标签。点选查看代表层维度，点击内部模块展开，或使用原层选择器查看具体层。Gemma PLE 明确显示第三条残差，DeepSeek mHC 保留专用门控混合。详见 [整体架构说明](docs/architecture/README.md)。

## 示例推理演示

选择任何模型后，点击顶部「示例推理」。交互参考 [LLM-Visualized](https://www.llm-visualized.com/?token=4&generation=0&kvCache=0)：暗色 3D 场景、公式与维度、token 上下文、视角跟随、逐层前向、续写轮次和 KV cache 开关。默认 4 个教学 token、2 轮输出，缓存关闭；可设置 1–16 个输入位置、1–8 轮输出和 0.25×–4× 速度。默认「慢速讲解」每步约 4–8 秒，另有「连续浏览」和每步结束自动暂停的「逐步学习」。镜头优先保留相邻分支，计算过程分为输入流入、计算扫描、输出传递和阅读停留。详见 [节奏与视觉优化](docs/inference-demo/polish/REVIEW.md)。

默认代表层讲解展示主要计算，省略的步骤会明确标注；可切换完整逐层路线按模型实际主干层与子模块推进。暂停保留步骤内进度，继续从原处播放，重新开始清空演示状态。支持单步、跳过当前层、跳到本轮输出、下一轮、进度拖动和点击已输出 token 回看。3D / 2D 切换保留播放位置；退出后恢复原工作台配置和输入。无 LM Head 的基础模型只演示隐藏状态前向。

开启缓存时首轮 Prefill，后续使用 S=1 与历史 P；关闭时每轮从增长后的完整前缀重新开始。KV / 递归状态容量按原适配器在轮次边界估算。演示采用固定文本脚本与 token 占位，不运行模型权重或 tokenizer，不提供真实生成概率、激活值或显存轨迹，也不声称与参考站点逐帧一致。详见 [演示说明与验证](docs/inference-demo/README.md)。

点击演示中的组件可打开完整侧栏：原理、公式、全部张量、token / head、矩阵位置以及可调数学示例。设置提供运行记录模板与导入入口；匹配记录的 token ID 和采样值会按来源显示。普通 3D 视图也提供「推理与详细解释」入口。[参考站点实际运行检查及本次补齐范围](docs/inference-demo/reference-review/REVIEW.md)。


默认代表层路线预计不超过 5 分钟（标准速度、默认两轮输出），可切换完整逐层；运行记录的 `instance`、张量轴和语义约束见 [模型语义与教学路线](docs/semantics/README.md)。

## 统一模型库（111 个模型）

左侧「模型配置」的模型选择器与顶栏「模型库」打开同一个检索窗口。目录包含 [Raschka LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/) 在 2026-10-17 的全部 111 个模型条目，按需加载。

5 个快捷配置已归入对应画廊条目，不再重复显示。Qwen3 0.6B 的 Base 与对话版在同一卡片内选择，保留各自的来源与上下文配置。可搜索名称、仓库、机构及架构关键词，按机构或结构筛选，并查看当前模型标记；自定义配置可直接从模型库导入。

新目录面向语言主干结构，覆盖 MoE、MLA/DSA、DeltaNet/KDA、Mamba-2、短卷积、循环层、Gemma 大模型、DeepSeek V4 等。通用适配明确标注结构估算及专用算子范围，不能视作全部模型的逐权重或峰值显存验证。Tiny Aya、Soofi、BTL、Ember 等来源例外在界面中说明。完整覆盖和来源见 [模型库说明](docs/gallery/README.md) 与 [逐模型清单](docs/gallery/COVERAGE.md)。

## 原有专用适配范围

| `model_type`                         | 支持的标准架构                                                | 关键差异                                                    |
| ------------------------------------ | ------------------------------------------------------------- | ----------------------------------------------------------- |
| `llama`                              | LlamaModel / LlamaForCausalLM                                 | GQA、RoPE、可选 linear bias、权重共享                       |
| `qwen2`                              | Qwen2Model / Qwen2ForCausalLM                                 | 包含 Qwen2.5；Q/K/V bias、可选滑动窗口                      |
| `qwen3`                              | Qwen3Model / Qwen3ForCausalLM                                 | 显式 `head_dim`、每个 head 的 QK RMSNorm                    |
| `mistral`                            | MistralModel / MistralForCausalLM                             | GQA、滑动窗口因果 mask                                      |
| `deepseek_v41` / `deepseek_v41_text` | DeepseekV41ForCausalLM / 文本主干                             | CED、CSA2、跨层 KV/索引共享、MoE、mHC、Engram、视觉、DSpark |
| `gemma4` / `gemma4_text`             | Gemma4ForConditionalGeneration / Gemma4ForCausalLM / 基础模型 | E2B/E4B：PLE、混合 head_dim、KV 共享、视觉和音频            |

统一模型库保留了五份与画廊模型对应的快捷配置，包括 [Gemma 4 E2B](https://huggingface.co/google/gemma-4-E2B-it/blob/3e22461f65e89153144f8adb70e3b8c2cc9845a7/config.json)、[Gemma 4 E4B](https://huggingface.co/google/gemma-4-E4B-it/blob/ee0ef6023621cff504d758262d4e04895a5af4a2/config.json)，以及 [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/config.json)，以及 [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json)、[Llama-3.2-1B-Instruct](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct/blob/main/config.json)。Qwen2.5 7B 和 Mistral 7B 已移出模型库；相关配置仅作为测试夹具保存在 `tests/fixtures/`，仍可通过自定义导入使用。

`config.json` 描述模型超参数，无法单独定义任意模型的执行代码。应用通过标准架构适配器推导逻辑计算图；除专用适配和模型库登记的架构家族外，未知架构仍会显示错误。模型库中的已知 `auto_map` 只作为元数据读取，不执行其代码；多模态模型通常仅展示语言主干。缺少 `architectures` 时，默认展示相应的 `ForCausalLM`，并在配置说明中标明假设。省略关键结构字段的稀疏配置需要补齐 `hidden_size`、`intermediate_size`、`num_hidden_layers`、`num_attention_heads` 和 `vocab_size`。

## Gemma 4 E2B / E4B

选择对应示例，或导入 `gemma4` 完整嵌套配置、`gemma4_text` 文本配置。新增逐层嵌入 PLE、Hybrid Decoder、按类型共享 KV、E2B 共享层 double-wide MLP、视觉与音频编码器视图。

E2B/E4B 的完整参数分别约 **5.10B / 7.94B**，包含大型 PLE 表与媒体编码器；E 名称不是全部参数量。原始配置弹窗中可查看分项统计。局部 `head_dim=256`、全局 `head_dim=512`；E2B 后 20 层共享 L13/L14 的 KV，E4B 后 18 层共享 L22/L23 的 KV。

图像网格和音频特征帧由输入场景指定；S 已包含媒体软 token。当前是规则、无 padding 的静态示例，不执行原始媒体处理或视频时间轴分析。详见 [Gemma 适配说明](docs/gemma/README.md)。

## DeepSeek V4.1 CED

在模型库选择「DeepSeek V4.1-Flash (552B)」，或导入官方完整 `config.json`。侧栏提供 Causal Encoder、CED Decoder、共享缓存、CSA2、MoE、视觉、Engram 和 DSpark 视图。层选择器同时显示层所属分区及 SWA / Full / Reindex / Reuse 模式。

- 支持 `text_config` + `vision_config` 嵌套配置及单独 `deepseek_v41_text` 配置。不会把 `DeepseekV41ForCausalLM` 误判为普通 GQA decoder。
- 默认展示参考实现完整前向。输入场景可切换「有界重放示意」：Prefill 的 Encoder 处理全部 S，Decoder 只重放末尾窗口；全局 KV 长度保持完整上下文。此近似调度与完整前向不严格等价。
- Decode 遵照官方参考实现，要求 S=1。支持批大小、缓存长度、逻辑 dtype 和预处理后图像 patch 网格联动；S 已包含图像及分隔 token。
- 全局 KV 按唯一来源层去重，K/V 共用 latent，不乘 2；区分逻辑浮点缓存与含 scales 的紧凑 FP4 全局缓存。默认官方配置在偶数长序列下为 890 bytes/token，不含 SWA 与临时状态。
- 参数合计分别包含主干、Engram、视觉、DSpark；量化 scales、并行 padding 不计入逻辑参数。主干约 551.57B，全部逻辑权重约 763.21B。原始配置窗口中可查看分项。
- 未启用图像时，视觉路径只作示例，不进入播放。DSpark 是独立可选分支，不模拟接受率或调度循环；其 Prefill 只初始化缓存。

详细公式、固定参考版本、验证方法与边界见 [DeepSeek 适配说明](docs/deepseek/README.md)。

## 维度语义与限制

| 符号         | 含义                                         |
| ------------ | -------------------------------------------- |
| B            | Batch size                                   |
| S            | 本次输入的 token 数                          |
| P            | 历史 KV cache 的 token 数                    |
| T            | P + S                                        |
| H / I / V    | hidden size / intermediate size / vocab size |
| Nq / Nkv / D | Query heads / KV heads / head dimension      |

例如 Qwen2.5-7B，`B=2, S=7, P=13`：Q 为 `[2,28,7,128]`，缓存 K/V 各为 `[2,4,20,128]`，GQA 逻辑扩展后各为 `[2,28,20,128]`，注意力分数为 `[2,28,7,20]`。Qwen3-0.6B 的 `head_dim=128`，Q 投影输出宽度为 `16×128=2048`，并不等于其 `hidden_size=1024`。

- 这是**静态形状推导**，数据流播放不执行真实前向计算，也不展示实际激活数值、注意力热力图或性能计时。
- 使用推理模式；训练 dropout、梯度、优化器状态不在图中。
- Llama/Qwen/Mistral 图展示完整未裁剪 KV cache；Gemma E 系列区分本次可见长度和 DynamicCache 持久长度；DeepSeek V4.1 使用独立的压缩缓存规则。
- Attention 分数和 GQA 扩展是逻辑形状；FlashAttention / SDPA 等融合后端可能不分配完整中间矩阵。
- 普通 dense 模型的 LM Head 展示全部 S 个位置；DeepSeek V4.1 主干遵照参考生成代码展示最后位置 float32 logits，DSpark 单独展示草稿 block。
- 共享 Embedding / LM Head 权重在模块内分别显示，但总参数量只计一次。
- 内存估算以所选浮点 dtype 计算，**不代表峰值显存**，不含运行时工作区、量化存储、临时激活和框架开销。
- 在线导入仅访问公开 Hugging Face 配置；私有 / gated 仓库请先自行下载配置，再本地导入。不收集或保存访问 token。

## 实现结构

```text
src/
  lib/model.ts             配置校验、标准架构差异、形状与参数推导
  lib/graph.ts             分层计算图、残差和 Q/K/V 分支、模块解释
  lib/catalog.ts           模型库语言主干归一化、混合层与状态规则
  lib/catalog-graph.ts     模型库层图、MoE、递归状态及专用接口
  data/gallery-manifest.json 111 个来源条目的固定快照
  data/model-library.ts     快捷配置与画廊去重、版本选择和检索
  components/GalleryDialog.tsx 可搜索/筛选的模型库
  lib/gemma.ts             Gemma E 系列配置、参数与维度推导
  lib/gemma-graph.ts       PLE、共享 KV、视觉与音频图
  lib/huggingface.ts       在线配置读取、大小限制与错误处理
  components/ModuleNode.tsx 图节点
  components/Inspector.tsx 模块和张量详情
  components/ImportDialog.tsx 配置导入
  data/                   示例配置及来源
  App.tsx                 输入场景、导航、选中与播放状态
  styles.css              界面和响应式布局
tests/app.spec.ts         Playwright 浏览器测试
```

技术栈：React、TypeScript、Vite、React Flow。扩展架构时，先在 `model.ts` 中实现配置归一化、参数与形状规则，再在 `graph.ts` 中描述对应的拓扑和模块语义；独立测试应覆盖该架构特有的维度变化。BERT、未登记的 MoE 或多模态需要匹配的图规则，不能直接套用 dense decoder 图。DeepSeek V4.1 与 Gemma 4 E 系列分别在 `deepseek*.ts`、`gemma*.ts` 实现独立适配。

现有测试覆盖非法配置、四类真实配置、MHA / GQA / MQA、非方形 decode attention、Qwen3 QK norm、权重共享、残差拓扑、导入失败恢复、点击聚焦、导出与响应式布局。四个示例的总参数量另以 Transformers 5.12.1 在 PyTorch meta device 上实例化核对，无需分配权重存储。

## 参考与资源

结构规则参考 Hugging Face Transformers 官方实现：[Llama](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)、[Qwen2](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py)、[Qwen3](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py)、[Mistral](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py)。

界面字体是 [Noto Sans SC](https://github.com/google/fonts/tree/main/ofl/notosanssc) 的字形子集，按 SIL Open Font License 1.1 分发，许可见 `public/FONT-LICENSE.txt`。子集覆盖当前界面文本；新增中文文案可使用系统中文字体回退，或重新生成子集。
