import { useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowUpRight,
  BookOpen,
  Boxes,
  Check,
  LoaderCircle,
  Search,
  Upload,
  X,
} from "lucide-react";
import {
  modelLibrary,
  modelLibraryCount,
  galleryModelCount,
  loadLibraryModel,
  matchesLibraryQuery,
  type ModelSelection,
  type ModelVariant,
  type LibraryEntry,
} from "../data/model-library";
import { gallerySnapshot } from "../data/gallery";
import type { ModelConfig, Scenario } from "../lib/model";
import LibraryCachePreview from "./LibraryCachePreview";

function sourceLabel(entry: ModelVariant) {
  const note = entry.sourceNote ?? "";
  return /派生|基座|转换|derivative|base architecture|reference configuration|Architecture-derived|GGUF metadata|compatible variant/.test(
    note,
  )
    ? "基座 / 转换配置"
    : /mirror/.test(note)
      ? "公开镜像"
      : "公开配置";
}
export default function GalleryDialog({
  open,
  onClose,
  onLoad,
  onImport,
  selection,
  scenario,
}: {
  open: boolean;
  onClose: () => void;
  onLoad: (
    model: ModelConfig,
    name: string,
    source: string,
    selection: ModelSelection,
  ) => void;
  onImport: () => void;
  selection: ModelSelection | null;
  scenario: Scenario;
}) {
  const [variantIds, setVariantIds] = useState<Record<string, string>>(() =>
    selection ? { [selection.entryId]: selection.variantId } : {},
  );
  const dialog = useRef<HTMLDialogElement>(null),
    searchInput = useRef<HTMLInputElement>(null),
    generation = useRef(0);
  const [query, setQuery] = useState(""),
    [company, setCompany] = useState("all"),
    [kind, setKind] = useState("all"),
    [loading, setLoading] = useState<string | null>(null),
    [error, setError] = useState("");
  useEffect(() => {
    const element = dialog.current;
    const opener =
      document.activeElement instanceof HTMLElement
        ? document.activeElement
        : null;
    if (open) {
      element?.showModal();
      searchInput.current?.focus({ preventScroll: true });
    } else {
      element?.close();
      generation.current++;
      setLoading(null);
    }
    return () => {
      generation.current++;
      element?.close();
      if (opener?.isConnected && !document.querySelector("dialog[open]"))
        opener.focus({ preventScroll: true });
    };
  }, [open]);
  const companies = useMemo(
    () => [...new Set(modelLibrary.map((m) => m.company))].sort(),
    [],
  );
  const filtered = useMemo(
    () =>
      modelLibrary.filter(
        (m) =>
          matchesLibraryQuery(m, query) &&
          (company === "all" || m.company === company) &&
          (kind === "all" ||
            (kind === "moe"
              ? /MoE|Sparse/.test(m.decoder)
              : kind === "hybrid"
                ? /hybrid|Recurrent/.test(m.decoder) ||
                  /Mamba|Delta|KDA|Lightning|LIV|mLSTM|CCA/.test(m.layerMix)
                : !/MoE|Sparse/.test(m.decoder))),
      ),
    [query, company, kind],
  );
  const select = async (entry: LibraryEntry, variant: ModelVariant) => {
    const id = ++generation.current;
    setLoading(entry.id);
    setError("");
    try {
      const chosen = { entryId: entry.id, variantId: variant.id };
      const { model, name, source } = await loadLibraryModel(chosen);
      if (id !== generation.current) return;
      onLoad(model, name, source, chosen);
      onClose();
    } catch (e) {
      if (id === generation.current)
        setError(e instanceof Error ? e.message : "加载失败");
    } finally {
      if (id === generation.current) setLoading(null);
    }
  };
  const hasFilters = !!query || company !== "all" || kind !== "all";
  const resetFilters = () => {
    setQuery("");
    setCompany("all");
    setKind("all");
    searchInput.current?.focus();
  };
  return (
    <dialog
      ref={dialog}
      className="gallery-dialog"
      aria-labelledby="gallery-title"
      onCancel={onClose}
      onClick={(e) => {
        if (e.target === dialog.current) onClose();
      }}
    >
      <div className="gallery-header">
        <span className="dialog-icon">
          <BookOpen size={24} />
        </span>
        <div>
          <span className="section-eyebrow">MODEL LIBRARY</span>
          <h2 id="gallery-title">
            全部模型 <span>{modelLibraryCount}</span>
          </h2>
        </div>
        <button
          className="icon-button"
          aria-label="关闭模型库"
          onClick={onClose}
        >
          <X size={20} />
        </button>
      </div>
      <p className="gallery-intro">
        {galleryModelCount} 个画廊模型，可按名称、机构或架构检索。
        <a
          href="https://sebastianraschka.com/llm-architecture-gallery/"
          target="_blank"
          rel="noreferrer"
        >
          原始画廊
          <ArrowUpRight size={12} />
        </a>
      </p>
      <div className="gallery-filters">
        <label className="gallery-search">
          <Search size={17} />
          <input
            ref={searchInput}
            aria-label="搜索模型库"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="搜索模型、机构或注意力类型…"
          />
          {query && (
            <button
              className="search-clear"
              aria-label="清空搜索"
              onClick={() => {
                setQuery("");
                searchInput.current?.focus();
              }}
            >
              <X size={15} />
            </button>
          )}
        </label>
        <select
          aria-label="筛选机构"
          value={company}
          onChange={(e) => setCompany(e.target.value)}
        >
          <option value="all">全部机构</option>
          {companies.map((c) => (
            <option key={c}>{c}</option>
          ))}
        </select>
        <select
          aria-label="筛选结构"
          value={kind}
          onChange={(e) => setKind(e.target.value)}
        >
          <option value="all">全部结构</option>
          <option value="moe">MoE</option>
          <option value="dense">Dense</option>
          <option value="hybrid">混合 / 递归</option>
        </select>
      </div>
      <div className="gallery-result-count">
        <span role="status" aria-live="polite">
          <strong>{filtered.length}</strong> / {modelLibraryCount} 个模型
        </span>
        {hasFilters && (
          <button className="filter-reset" onClick={resetFilters}>
            重置筛选
            <X size={12} />
          </button>
        )}
        <span>快照 {gallerySnapshot} · 按需载入配置</span>
      </div>
      {error && (
        <p role="alert" className="error-message">
          {error}
        </p>
      )}
      <div className="gallery-results">
        {filtered.length ? (
          filtered.map((entry) => {
            const variant =
              entry.variants.find((v) => v.id === variantIds[entry.id]) ??
              entry.variants[0];
            const current = selection?.entryId === entry.id;
            const sameSelection =
              current && selection?.variantId === variant.id;
            const actionLabel = sameSelection
              ? "重新加载"
              : current
                ? "切换版本"
                : "加载模型";
            return (
              <article
                key={entry.id}
                className="gallery-card"
                data-gallery-id={entry.id}
                data-current={current || undefined}
              >
                <div className="gallery-card-top">
                  <span>{entry.company}</span>
                  {current && (
                    <span className="current-model-badge">
                      <Check size={11} />
                      当前模型
                    </span>
                  )}
                  <span className="gallery-source-kind">
                    {sourceLabel(variant)}
                  </span>
                </div>
                <h3>{entry.name}</h3>
                <p className="gallery-mix">{entry.layerMix}</p>
                <p className="gallery-repo" title={variant.repo}>
                  {variant.repo}
                </p>
                <div className="gallery-tags">
                  <span>{entry.decoder}</span>
                  <span>{variant.context} context</span>
                </div>
                <LibraryCachePreview
                  selection={{ entryId: entry.id, variantId: variant.id }}
                  scenario={scenario}
                />
                {entry.variants.length > 1 && (
                  <label className="model-variant-picker">
                    配置版本
                    <select
                      aria-label={`${entry.name} 配置版本`}
                      value={variant.id}
                      disabled={loading !== null}
                      onChange={(e) =>
                        setVariantIds((previous) => ({
                          ...previous,
                          [entry.id]: e.target.value,
                        }))
                      }
                    >
                      {entry.variants.map((v) => (
                        <option value={v.id} key={v.id}>
                          {v.label}
                        </option>
                      ))}
                    </select>
                  </label>
                )}
                {variant.sourceNote && (
                  <p className="gallery-source-note" title={variant.sourceNote}>
                    {variant.sourceNote}
                  </p>
                )}
                <div className="gallery-card-actions">
                  <a href={variant.configUrl} target="_blank" rel="noreferrer">
                    配置来源
                    <ArrowUpRight size={12} />
                  </a>
                  <button
                    className="secondary-button"
                    disabled={loading !== null}
                    onClick={() => void select(entry, variant)}
                    aria-label={`${actionLabel}：${entry.name}${entry.variants.length > 1 ? ` · ${variant.label}` : ""}`}
                  >
                    {loading === entry.id ? (
                      <LoaderCircle className="spin" size={14} />
                    ) : (
                      <Boxes size={14} />
                    )}
                    {actionLabel}
                  </button>
                </div>
              </article>
            );
          })
        ) : (
          <div className="gallery-empty">
            <Search size={28} />
            <p>没有匹配的模型，请尝试其他名称或取消筛选。</p>
            <button className="secondary-button" onClick={resetFilters}>
              清除条件，查看全部
            </button>
          </div>
        )}
      </div>
      <div className="gallery-footer">
        <span>
          <Check size={13} />
          按需读取配置，无需下载权重
        </span>
        <button className="secondary-button" onClick={onImport}>
          <Upload size={14} />
          导入自定义配置
        </button>
      </div>
    </dialog>
  );
}
