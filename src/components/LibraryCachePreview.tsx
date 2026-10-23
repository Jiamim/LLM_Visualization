import { useEffect, useMemo, useRef, useState } from "react";
import { loadLibraryModel, type ModelSelection } from "../data/model-library";
import { cacheReport } from "../lib/cache";
import { formatBytes, type ModelConfig, type Scenario } from "../lib/model";

const models = new Map<string, Promise<ModelConfig>>();
export default function LibraryCachePreview({
  selection,
  scenario,
}: {
  selection: ModelSelection;
  scenario: Scenario;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);
  const [loaded, setLoaded] = useState<{
    key: string;
    model: ModelConfig;
  } | null>(null);
  const [error, setError] = useState("");
  const key = `${selection.entryId}/${selection.variantId}`;
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          setVisible(true);
          observer.disconnect();
        }
      },
      { rootMargin: "160px" },
    );
    if (ref.current) observer.observe(ref.current);
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    if (!visible) return;
    let cancelled = false;
    setError("");
    let request = models.get(key);
    if (!request) {
      request = loadLibraryModel(selection).then((result) => result.model);
      models.set(key, request);
    }
    request
      .then((model) => {
        if (!cancelled) setLoaded({ key, model });
      })
      .catch(() => {
        models.delete(key);
        if (!cancelled) setError("此配置暂未能计算");
      });
    return () => {
      cancelled = true;
    };
  }, [visible, key]);
  const report = useMemo(
    () => (loaded?.key === key ? cacheReport(loaded.model, scenario) : null),
    [loaded, key, scenario],
  );
  return (
    <div
      ref={ref}
      className="library-cache-preview"
      data-cache-ready={!!report}
    >
      {error ? (
        <span>{error}</span>
      ) : report ? (
        <>
          <div>
            <span>KV cache{report.partial ? " · 部分" : ""}</span>
            <strong>
              {report.unknownLayers.length && report.kvBytes === 0
                ? "未确定"
                : formatBytes(report.kvBytes)}
            </strong>
          </div>
          {report.recurrentBytes + report.auxiliaryBytes > 0 && (
            <div className="library-other-state">
              <span>其他推理状态</span>
              <b>
                {formatBytes(report.recurrentBytes + report.auxiliaryBytes)}
              </b>
            </div>
          )}
          <small>
            {report.kvBytes === 0 && !report.unknownLayers.length
              ? "递归状态型 · "
              : ""}
            {report.cacheDisabledByDefault ? "假设开启缓存 · " : ""}B=
            {scenario.batch} · T={report.totalTokens.toLocaleString()} ·{" "}
            {scenario.dtype}
          </small>
        </>
      ) : (
        <span className="cache-preview-pending">
          缓存估算{visible ? "计算中…" : " · 随卡片载入"}
        </span>
      )}
    </div>
  );
}
