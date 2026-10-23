import { useEffect, useRef, useState } from "react";
import {
  Braces,
  Check,
  FileJson,
  Globe,
  LoaderCircle,
  Upload,
  X,
} from "lucide-react";
import { parseConfig, type ModelConfig } from "../lib/model";
import { fetchConfig, MAX_CONFIG_BYTES } from "../lib/huggingface";

export default function ImportDialog({
  open,
  onClose,
  onImport,
}: {
  open: boolean;
  onClose: () => void;
  onImport: (config: ModelConfig, name: string, source: string) => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const request = useRef<AbortController | null>(null);
  const [tab, setTab] = useState<"hub" | "file" | "json">("hub");
  const [repo, setRepo] = useState("Qwen/Qwen2.5-7B-Instruct");
  const [revision, setRevision] = useState("main");
  const [json, setJson] = useState("");
  const [filename, setFilename] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    if (open) {
      dialog.current?.showModal();
      setError("");
    } else {
      dialog.current?.close();
      request.current?.abort();
    }
    return () => request.current?.abort();
  }, [open]);
  const readFile = async (file?: File) => {
    if (!file) return;
    setError("");
    if (file.size > MAX_CONFIG_BYTES) {
      setError("配置文件超过 2 MiB 上限。");
      return;
    }
    try {
      setJson(await file.text());
      setFilename(file.name);
    } catch {
      setError("无法读取文件，请重新选择。");
    }
  };
  const submit = async () => {
    setError("");
    setBusy(true);
    const controller = new AbortController();
    request.current = controller;
    const timeout = window.setTimeout(() => controller.abort("timeout"), 20000);
    try {
      const content =
        tab === "hub"
          ? await fetchConfig(repo, revision, controller.signal)
          : json;
      if (content.length > MAX_CONFIG_BYTES)
        throw new Error("配置内容超过 2 MiB 上限。");
      const config = parseConfig(content);
      if (controller.signal.aborted) return;
      onImport(
        config,
        tab === "hub"
          ? repo.trim().split("/").pop()!
          : filename || config.architecture,
        tab === "hub"
          ? `${repo.trim()} @ ${revision.trim()}`
          : tab === "json"
            ? "粘贴的 config.json"
            : filename,
      );
      onClose();
    } catch (err) {
      if (controller.signal.aborted && controller.signal.reason !== "timeout")
        return;
      setError(
        controller.signal.reason === "timeout"
          ? "请求超时。请检查网络，或下载 config.json 后本地导入。"
          : err instanceof TypeError
            ? "无法连接 Hugging Face，请检查网络或使用本地导入。"
            : err instanceof Error
              ? err.message
              : "加载失败，请检查配置。",
      );
    } finally {
      clearTimeout(timeout);
      setBusy(false);
    }
  };
  return (
    <dialog
      ref={dialog}
      className="import-dialog"
      onCancel={onClose}
      onClick={(e) => {
        if (e.target === dialog.current) onClose();
      }}
      aria-labelledby="import-title"
    >
      <div className="dialog-inner">
        <div className="dialog-header">
          <span className="dialog-icon">
            <FileJson size={23} />
          </span>
          <button
            className="icon-button"
            onClick={onClose}
            aria-label="关闭导入窗口"
          >
            <X size={18} />
          </button>
        </div>
        <h2 id="import-title">从配置，开始探索。</h2>
        <p className="dialog-intro">
          只需一个 Hugging Face config.json，无需下载模型权重。
        </p>
        <div className="import-tabs">
          {(
            [
              ["hub", Globe, "Hugging Face"],
              ["file", Upload, "本地文件"],
              ["json", Braces, "粘贴 JSON"],
            ] as const
          ).map(([id, Icon, title]) => (
            <button
              key={id}
              disabled={busy}
              className={tab === id ? "active" : ""}
              onClick={() => {
                setTab(id);
                setError("");
              }}
            >
              <Icon size={15} />
              {title}
            </button>
          ))}
        </div>
        {tab === "hub" ? (
          <div className="import-fields">
            <label>
              模型 ID
              <input
                value={repo}
                onChange={(e) => setRepo(e.target.value)}
                placeholder="owner/model-name"
                autoFocus
                disabled={busy}
              />
            </label>
            <label>
              Revision
              <input
                value={revision}
                onChange={(e) => setRevision(e.target.value)}
                placeholder="main"
                disabled={busy}
              />
            </label>
            <p>
              从公开仓库读取 config.json。需授权的模型可下载配置后本地导入。
            </p>
          </div>
        ) : tab === "file" ? (
          <>
            <button
              className="dropzone"
              disabled={busy}
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault();
                if (!busy) void readFile(e.dataTransfer.files[0]);
              }}
              onClick={() => fileInput.current?.click()}
            >
              <Upload size={28} />
              <strong>{filename || "拖放 config.json 到这里"}</strong>
              <span>或点击选择文件 · 最大 2 MiB</span>
            </button>
            <input
              ref={fileInput}
              type="file"
              accept=".json,application/json"
              hidden
              onChange={(e) => {
                void readFile(e.target.files?.[0]);
                e.target.value = "";
              }}
            />
          </>
        ) : (
          <textarea
            className="json-input"
            aria-label="JSON 配置内容"
            value={json}
            onChange={(e) => {
              setJson(e.target.value);
              setFilename("");
            }}
            spellCheck={false}
            placeholder={
              '{\n  "model_type": "llama",\n  "hidden_size": 4096,\n  ...\n}'
            }
            disabled={busy}
          />
        )}
        <div className="supported-models">
          <Check size={14} />
          <span>
            Llama · Qwen2 / 2.5 · Qwen3 · Mistral · DeepSeek V4.1 CED · Gemma 4
            E2B/E4B
          </span>
        </div>
        {error && (
          <p className="error-message" role="alert">
            {error}
          </p>
        )}
        <button
          className="primary-button import-submit"
          disabled={busy || (tab !== "hub" && !json.trim())}
          onClick={() => void submit()}
        >
          {busy ? <LoaderCircle size={16} className="spin" /> : <BoxIcon />}{" "}
          {busy ? "正在读取配置…" : "生成模型视图"}
        </button>
      </div>
    </dialog>
  );
}
function BoxIcon() {
  return <FileJson size={16} />;
}
