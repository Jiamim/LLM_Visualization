import manifest from "./gallery-manifest.json";
import type { ModelConfig, RawConfig } from "../lib/model";
import { normalizeConfig } from "../lib/model";

export const galleryModels = manifest.models;
export const gallerySnapshot = manifest.snapshotDate;
export type GalleryEntry = (typeof galleryModels)[number];
const configs = import.meta.glob<RawConfig>("./gallery/*.json", {
  import: "default",
});

export async function loadGalleryModel(
  entry: GalleryEntry,
): Promise<ModelConfig> {
  const importer = configs[`./gallery/${entry.configPath}`];
  if (!importer) throw new Error("此条目的配置快照缺失。");
  const raw = await importer();
  const model = normalizeConfig(raw);
  model.gallery = {
    id: entry.id,
    name: entry.name,
    url: entry.gallery,
    configUrl: entry.configUrl ?? "",
    sourceNote: entry.sourceNote,
    snapshotDate: gallerySnapshot,
  };
  if (entry.sourceNote) model.warnings.unshift(entry.sourceNote);
  if (raw._atlas_snapshot_note)
    model.warnings.push(String(raw._atlas_snapshot_note));
  if (model.catalog && entry.layerMix)
    model.warnings.push(
      `画廊描述（仅供对照）：${entry.layerMix}；本图层数优先采用配置。`,
    );
  return model;
}
