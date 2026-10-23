import { galleryModels, loadGalleryModel, type GalleryEntry } from "./gallery";
import { presets } from "./presets";
import { normalizeConfig, type ModelConfig } from "../lib/model";

import type { ModelSelection } from "./model-library-info";
export {
  modelLibraryCount,
  galleryModelCount,
  initialSelection,
  type ModelSelection,
} from "./model-library-info";
export interface ModelVariant {
  id: string;
  label: string;
  repo: string;
  configUrl: string;
  sourceNote?: string;
  presetId?: string;
  galleryEntry?: GalleryEntry;
  context: string;
}
export interface LibraryEntry {
  id: string;
  name: string;
  company: string;
  attention: string;
  decoder: string;
  layerMix: string;
  context: string;
  modelType: string;
  aliases: string[];
  variants: ModelVariant[];
}

/** Canonical cards are gallery identities, never guessed from model family/size. */
function createLibrary(): LibraryEntry[] {
  const entries: LibraryEntry[] = galleryModels.map((entry) => ({
    id: entry.id,
    name: entry.name,
    company: entry.company,
    attention: entry.attention,
    decoder: entry.decoder,
    layerMix: entry.layerMix,
    context: entry.context,
    modelType: entry.modelType ?? "",
    aliases: [entry.id, entry.repo ?? ""],
    variants: [
      {
        id: "gallery",
        label: entry.id === "qwen3-0-6b" ? "Base" : "公开配置",
        repo: entry.repo ?? entry.name,
        configUrl: entry.configUrl ?? entry.gallery,
        sourceNote: entry.sourceNote,
        galleryEntry: entry,
        context: entry.context,
      },
    ],
  }));
  for (const preset of presets) {
    // Repository identity also catches future duplicates without adding mappings.
    const existing = entries.find(
      (entry) =>
        entry.id === preset.galleryId ||
        entry.variants.some(
          (v) => v.repo.toLowerCase() === preset.repo.toLowerCase(),
        ),
    );
    if (existing) {
      existing.aliases.push(preset.id, preset.name, preset.repo);
      const sameRepo = existing.variants.find(
        (v) => v.repo.toLowerCase() === preset.repo.toLowerCase(),
      );
      if (!sameRepo)
        existing.variants.push({
          id: preset.id,
          label: preset.variantLabel ?? preset.repo.split("/")[1],
          repo: preset.repo,
          configUrl: `https://huggingface.co/${preset.repo}/blob/main/config.json`,
          presetId: preset.id,
          context: Number(preset.raw.max_position_embeddings).toLocaleString(
            "en-US",
          ),
        });
      continue;
    }
    throw new Error(`Missing gallery identity for ${preset.id}`);
  }
  return entries;
}

export const modelLibrary = createLibrary();

export function selectionForPreset(presetId: string): ModelSelection {
  const preset = presets.find((p) => p.id === presetId);
  if (!preset) throw new Error("未知模型配置。");
  const entry = modelLibrary.find(
    (e) =>
      e.id === preset.galleryId ||
      e.variants.some(
        (v) => v.repo.toLowerCase() === preset.repo.toLowerCase(),
      ),
  );
  const variant = entry?.variants.find(
    (v) =>
      v.presetId === presetId ||
      v.repo.toLowerCase() === preset.repo.toLowerCase(),
  );
  if (!entry || !variant) throw new Error("此模型配置未登记。");
  return { entryId: entry.id, variantId: variant.id };
}

export async function loadLibraryModel(
  selection: ModelSelection,
): Promise<{ model: ModelConfig; name: string; source: string }> {
  const entry = modelLibrary.find((e) => e.id === selection.entryId),
    variant = entry?.variants.find((v) => v.id === selection.variantId);
  if (!entry || !variant) throw new Error("此模型或配置版本不存在。");
  if (variant.galleryEntry)
    return {
      model: await loadGalleryModel(variant.galleryEntry),
      name: entry.name,
      source: variant.repo,
    };
  const preset = presets.find((p) => p.id === variant.presetId);
  if (!preset) throw new Error("模型配置快照缺失。");
  return {
    model: normalizeConfig(preset.raw),
    name: `${entry.name} · ${variant.label}`,
    source: variant.repo,
  };
}

// Ignore separators and letter case, but keep decimals and complete numbers:
// 20B must not match 120B, and Qwen3 must not become Qwen3.8.
const searchTokens = (value: string) =>
  value
    .normalize("NFKC")
    .toLowerCase()
    .match(/\p{L}+|\d+(?:\.\d+)*/gu) ?? [];
export function matchesLibraryQuery(
  entry: LibraryEntry,
  query: string,
): boolean {
  if (query.normalize("NFKC").trim().toLowerCase() === entry.id) return true;
  // Slugs flatten decimal versions (3.8 -> 3-8), so only match a slug exactly.
  const fields = [
    entry.name,
    entry.company,
    entry.attention,
    entry.layerMix,
    entry.decoder,
    entry.modelType,
    ...entry.aliases.filter((alias) => alias !== entry.id),
    ...entry.variants.flatMap((v) => [v.repo, v.label]),
  ].map((value) => searchTokens(value).join(" "));
  return query
    .trim()
    .split(/\s+/)
    .map(searchTokens)
    .filter((tokens) => tokens.length)
    .every((tokens) => {
      const pattern = new RegExp(
        tokens
          .map((token) =>
            /^\d/.test(token)
              ? `(?<![\\d.])${token.replace(/\./g, "\\.")}(?![\\d.])`
              : token,
          )
          .join("\\s*"),
        "u",
      );
      return fields.some((field) => pattern.test(field));
    });
}
