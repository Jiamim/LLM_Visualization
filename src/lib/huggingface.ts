export const MAX_CONFIG_BYTES = 2 * 1024 * 1024;

export function configURL(repo: string, revision = "main"): string {
  const clean = repo.trim();
  if (!/^[a-zA-Z0-9][a-zA-Z0-9_.-]*\/[a-zA-Z0-9][a-zA-Z0-9_.-]*$/.test(clean)) {
    throw new Error(
      "请输入 owner/model 格式的模型 ID，例如 Qwen/Qwen2.5-7B-Instruct。",
    );
  }
  if (!revision.trim()) throw new Error("revision 不能为空。");
  return `https://huggingface.co/${clean}/resolve/${encodeURIComponent(revision.trim())}/config.json`;
}

export async function fetchConfig(
  repo: string,
  revision: string,
  signal: AbortSignal,
): Promise<string> {
  const response = await fetch(configURL(repo, revision), { signal });
  if (response.status === 401 || response.status === 403)
    throw new Error(
      "此仓库需要访问权限。请从 Hugging Face 下载 config.json 后使用本地导入。",
    );
  if (response.status === 404)
    throw new Error("未找到 config.json，请检查模型 ID 和 revision。");
  if (!response.ok)
    throw new Error(
      `Hugging Face 请求失败（HTTP ${response.status}）。可尝试下载配置后本地导入。`,
    );
  const reader = response.body?.getReader();
  if (!reader) throw new Error("服务器返回了空响应。");
  const decoder = new TextDecoder();
  let size = 0,
    text = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > MAX_CONFIG_BYTES) {
        await reader.cancel();
        throw new Error("配置文件超过 2 MiB 上限。");
      }
      text += decoder.decode(value, { stream: true });
    }
    return text + decoder.decode();
  } finally {
    reader.releaseLock();
  }
}
