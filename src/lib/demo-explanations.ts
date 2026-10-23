import type { ModuleData } from "./graph";
import type { ModelConfig } from "./model";
export interface Explanation {
  title: string;
  purpose: string;
  reason: string;
  steps: string[];
  caution: string;
  lab: "softmax" | "activation" | "norm" | "attention" | null;
}
export function explainDemoNode(
  m: ModelConfig,
  node: ModuleData,
): Explanation {
  const title = node.title,
    id = `${node.path} ${title}`.toLowerCase();
  const base = {
    title,
    purpose:
      node.description || node.subtitle || "执行当前架构定义的张量变换。",
    reason:
      "把当前表示变换为下一个模块所需要的输入，同时保持配置规定的轴语义。",
    steps: [] as string[],
    caution: "维度和参数来自配置；没有加载权重，色块不表示真实激活数值。",
    lab: null as Explanation["lab"],
  };
  if (node.category === "input")
    return {
      ...base,
      purpose: /layer|block|normalized|hidden/.test(id)
        ? "这是当前层或子模块接收到的 token 表示；它保留前面各层已经整合的信息。"
        : "这是文本进入模型前的离散 token 位置，后续查表将它映射成连续向量。",
      reason:
        "明确模块之间传递的是同一条残差主干的表示，还是外部输入，能帮助理解归一化、注意力与前馈如何逐步更新 token。",
      steps: [
        "确认 token 位置以及 B / S / H 等轴。",
        "从前序模块接收对应张量。",
        "传给当前模块的投影或归一化步骤。",
      ],
    };
  if (node.residual)
    return {
      ...base,
      purpose: /mhc|special/i.test(id)
        ? "将主分支与并行残差流按当前架构的门控和混合规则合并。"
        : "把子层计算结果与绕过该子层的输入逐元素相加。",
      reason:
        "保留已有表示，让子层学习需要添加的变化，避免每一步都重新构造完整信息。",
      steps: [
        "保存子层输入作为旁路。",
        "执行注意力、前馈或 PLE 信号变换。",
        "在相容的隐藏轴上合并两路结果。",
      ],
      caution: /mhc|special/i.test(id)
        ? "此模型为专用残差混合，不可简化为普通 x + f(x)。"
        : base.caution,
    };
  if (node.outputs.some((t) => t.role === "context") && /softmax/.test(id))
    return {
      ...base,
      purpose:
        "对有效分数执行 Softmax，再与 Value 加权求和并合并 heads，输出 context。",
      reason: "Context 是汇聚后的特征向量；它的数值不等于 attention 概率。",
      steps: [
        "沿键轴归一化分数。",
        "按权重汇聚 Value。",
        "合并 heads 得到 context。",
      ],
      lab: "attention",
    };
  if (node.outputs.some((t) => t.role === "probabilities"))
    return {
      ...base,
      purpose: "把每个 query 的有效 attention 分数转换为非负归一化权重。",
      reason:
        "同一行权重之和为 1，才能用来混合 Value；被 mask 的位置不能贡献结果。",
      steps: [
        "先减去行内最大值来稳定计算。",
        "对未屏蔽分数取指数。",
        "除以这一行指数总和。",
      ],
      lab: "softmax",
    };
  if (/scores|attention.core|attention score|sparse.attn/.test(id))
    return {
      ...base,
      purpose:
        "比较当前 Query 与允许访问的历史 Key，确定哪些位置能影响当前输出。",
      reason:
        "语言模型需要结合上下文；因果约束禁止读取当前 token 之后的信息。",
      steps: [
        "在当前 head 空间计算 query-key 相似度。",
        "应用缩放、位置和因果 / 窗口约束。",
        "经 softmax 或模型专用选择规则得到混合权重。",
      ],
      lab: "attention",
      caution:
        "这里的因果矩阵展示可访问位置；稀疏 Top-K、路由和真实权重须由运行数据确定。",
    };
  if (
    node.category === "norm" ||
    /rmsnorm|layernorm/.test(node.title.toLowerCase())
  )
    return {
      ...base,
      purpose: `沿特征轴执行 ${/rms/.test(id) ? "RMSNorm" : "当前归一化规则"}，调整表示的尺度。`,
      reason:
        "控制不同 token 的数值尺度，帮助后续投影保持稳定；归一化位置由模型架构决定。",
      steps: /rms/.test(id)
        ? [
            "计算特征平方的均值。",
            "用 sqrt(mean(x²) + ε) 缩放。",
            "乘可训练 scale（若该模块包含）。",
          ]
        : [
            "计算当前特征的均值与方差（LayerNorm）。",
            "中心化并按标准差缩放。",
            "应用模块定义的 scale / bias。",
          ],
      lab: "norm",
    };
  if (/activation|gelu|silu/.test(id))
    return {
      ...base,
      purpose: `逐元素应用 ${m.activation}，为前馈网络提供非线性。`,
      reason:
        "如果只串联线性矩阵，整体仍是线性变换；非线性让网络表达更丰富的模式。",
      steps: [
        "保持 batch 和 token 轴。",
        "对特征分量独立应用激活函数。",
        "门控 MLP 会继续与另一条升维分支相乘。",
      ],
      lab: "activation",
    };
  if (/cache|shared.kv|reuse key|kv.source|owner/.test(id))
    return {
      ...base,
      purpose: "保存或读取过去 token 的键值表示，或复用指定来源层的缓存。",
      reason:
        "自回归解码不必每次重新投影整个历史前缀。共享与滑动策略进一步改变持久存储量。",
      steps: [
        "区分已有历史 P 与本次输入 S。",
        "按实际 KV heads 和缓存策略更新状态。",
        "提供本次允许访问的 K/V 或压缩 latent。",
      ],
      caution:
        "局部缓存、持久缓存与本次可见 K/V 的长度可能不同；以当前模块形状和缓存说明为准。",
    };
  if (node.category === "embedding")
    return {
      ...base,
      purpose: /position/.test(id)
        ? "把位置相关的信息加入或用于变换 token 表示。"
        : "将离散 token 位置转换为连续向量，或注入当前模型所需的额外嵌入信号。",
      reason:
        "后续矩阵计算处理向量；词表查表、位置编码及 PLE 在不同模型中承担不同角色。",
      steps: [
        "根据 ID 或当前上下文读取 / 生成嵌入。",
        "应用配置规定的缩放与位置变换。",
        "将向量传给主干；多模态软 token 已计入序列长度。",
      ],
    };
  if (/query|key projection|value projection|q.proj|k.proj|v.proj/.test(id))
    return {
      ...base,
      purpose: "把同一个隐藏向量投影到 Query、Key 或 Value 子空间。",
      reason:
        "Query 表达要检索的内容，Key 用于匹配，Value 携带要聚合的信息；GQA 可以让多个 Query heads 共享较少的 KV heads。",
      steps: [
        "按权重形状执行线性投影。",
        "重排为 heads × head_dim。",
        "根据架构应用 QK norm、RoPE 或低秩表示。",
      ],
    };
  if (/router|expert|dispatch|moe/.test(id))
    return {
      ...base,
      purpose: "根据路由规则把 token 分配给专家网络，再合并专家输出。",
      reason:
        "MoE 将总参数容量与每 token 激活的专家数量分离；路由仍依赖输入和模型权重。",
      steps: [
        "计算路由分数并选择 Top-K 专家。",
        "按专家执行对应的前馈变换。",
        "按门控权重汇总，并包含配置中的共享专家分支。",
      ],
      caution:
        "不会随机指定专家选择结果；实际专家 ID、负载和路由权重需要推理记录。",
    };
  if (/state|mamba|delta|kda|convolution|conv|mlstm/.test(id))
    return {
      ...base,
      purpose: "用当前输入更新固定形状的递归或卷积状态。",
      reason:
        "用压缩的历史状态承载上下文，使部分历史存储不随序列长度线性增长。",
      steps: [
        "读取前一状态。",
        "按当前模型的门控、衰减或卷积规则更新。",
        "产生当前输出并保存下一状态。",
      ],
      caution: "递归状态不是传统逐 token K/V；内部更新按各模型适配范围展示。",
    };
  if (node.category === "mlp")
    return {
      ...base,
      purpose: "对每个 token 的特征独立执行升维、非线性或降维。",
      reason:
        "注意力负责 token 间信息交换，前馈网络进一步变换每个 token 内的特征。",
      steps: [
        "将 H 维映射到 intermediate_size。",
        "执行激活 / 门控计算。",
        "投影回 H 维，准备残差合并。",
      ],
    };
  if (node.category === "output")
    return {
      ...base,
      purpose: "将最终隐藏表示投影到词表，或输出隐藏状态。",
      reason:
        "LM Head 为下一个 token 的选择提供词表维度；没有 LM Head 的模型不会执行生成采样。",
      steps: [
        "完成最后的归一化。",
        "若含 LM Head，计算词表 logits。",
        "应用实际解码策略；当前演示只追加教学脚本 token。",
      ],
    };
  return base;
}
export function stableSoftmax(
  values: number[],
  masked: boolean[] = values.map(() => false),
): number[] {
  const valid = values.filter((_, i) => !masked[i]);
  if (!valid.length) return values.map(() => 0);
  const max = Math.max(...valid),
    exps = values.map((v, i) => (masked[i] ? 0 : Math.exp(v - max))),
    sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((x) => x / sum);
}
export function activationValue(x: number, kind: string) {
  return /silu|swish/.test(kind)
    ? x / (1 + Math.exp(-x))
    : kind === "relu2"
      ? Math.max(0, x) ** 2
      : kind === "relu"
        ? Math.max(0, x)
        : 0.5 *
          x *
          (1 +
            Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)));
}
export function normalizeExample(values: number[], rms: boolean, eps = 1e-6) {
  const mean = rms ? 0 : values.reduce((a, b) => a + b, 0) / values.length,
    variance =
      values.reduce((sum, x) => sum + (x - mean) ** 2, 0) / values.length;
  return values.map((x) => (x - mean) / Math.sqrt(variance + eps));
}
