import { describe, expect, it } from "vitest";
import {
  modelLibrary,
  modelLibraryCount,
  galleryModelCount,
  selectionForPreset,
  loadLibraryModel,
  matchesLibraryQuery,
} from "../data/model-library";
import { galleryModels } from "../data/gallery";
import { presets } from "../data/presets";
import { defaultPreset } from "../data/presets";
import { initialSelection } from "../data/model-library-info";

describe("unified model library", () => {
  it("opens the DeepSeek default with the matching library identity and configuration", async () => {
    expect(defaultPreset.id).toBe("deepseek-v41");
    expect(initialSelection).toEqual(selectionForPreset(defaultPreset.id));
    const loaded = await loadLibraryModel(initialSelection);
    expect(loaded.model.raw).toEqual(defaultPreset.raw);
    expect(loaded.source).toBe(defaultPreset.repo);
    expect(loaded.model.deepseek).toBeDefined();
  });
  it("keeps exactly the gallery entries without supplemental models", () => {
    expect(modelLibraryCount).toBe(111);
    expect(modelLibrary.length).toBe(modelLibraryCount);
    expect(galleryModelCount).toBe(galleryModels.length);
    expect(new Set(modelLibrary.map((e) => e.id)).size).toBe(
      modelLibrary.length,
    );
    expect(modelLibrary.map((e) => e.id)).toEqual(
      galleryModels.map((e) => e.id),
    );
    for (const e of galleryModels)
      expect(modelLibrary.filter((m) => m.id === e.id)).toHaveLength(1);
  });
  it.each(presets)(
    "preserves $id as a retrievable model/configuration without another card",
    async (preset) => {
      const selection = selectionForPreset(preset.id),
        entry = modelLibrary.find((e) => e.id === selection.entryId)!;
      expect(entry).toBeDefined();
      expect(entry.variants.some((v) => v.id === selection.variantId)).toBe(
        true,
      );
      expect(entry.id).toBe(preset.galleryId);
      const loaded = await loadLibraryModel(selection);
      expect(loaded.source).toBe(preset.repo);
      expect(loaded.model.raw).toEqual(preset.raw);
    },
  );
  it("keeps Base and chat configuration distinct inside one Qwen3 card", async () => {
    const entry = modelLibrary.find((e) => e.id === "qwen3-0-6b")!;
    expect(entry.variants).toHaveLength(2);
    const base = await loadLibraryModel({
        entryId: entry.id,
        variantId: "gallery",
      }),
      chat = await loadLibraryModel(selectionForPreset("qwen3"));
    expect(base.model.maxPosition).toBe(32768);
    expect(chat.model.maxPosition).toBe(40960);
    expect(base.source).toBe("Qwen/Qwen3-0.6B-Base");
    expect(chat.source).toBe("Qwen/Qwen3-0.6B");
    expect(chat.model.gallery).toBeUndefined();
    expect(base.model.gallery?.id).toBe(entry.id);
  });
  it.each([
    "QWEN3 0.6B",
    "Qwen3-0.6B",
    "qwen3 · 0.6b",
    "Ｑｗｅｎ３　０．６Ｂ",
    "Qwen/Qwen3-0.6B",
  ])("normalizes query %s without hiding aliases", (query) => {
    const result = modelLibrary.filter((e) => matchesLibraryQuery(e, query));
    expect(result.map((e) => e.id)).toContain("qwen3-0-6b");
    expect(result.filter((e) => e.id === "qwen3-0-6b")).toHaveLength(1);
  });
  it("does not collapse unrelated sizes or releases within one family", () => {
    expect(modelLibrary.some((e) => e.id === "mistral")).toBe(false);
    expect(modelLibrary.some((e) => e.id === "qwen25")).toBe(false);
    expect(modelLibrary.some((e) => e.id === "mistral-small-3-1-24b")).toBe(
      true,
    );
    expect(modelLibrary.filter((e) => e.id === "gemma-4-e2b")).toHaveLength(1);
    expect(modelLibrary.filter((e) => e.id === "gemma-4-e4b")).toHaveLength(1);
  });
  it.each([
    ["GPT-OSS (20B)", ["gpt-oss-20b"]],
    ["Qwen3 Next", ["qwen3-next-80b-a3b"]],
    ["Gemma4 E2B", ["gemma-4-e2b"]],
    ["Mistral-7B", []],
    ["Qwen2.5 GQA", []],
  ])("finds precise model numbers for %s", (query, ids) => {
    expect(
      modelLibrary
        .filter((e) => matchesLibraryQuery(e, query))
        .map((e) => e.id),
    ).toEqual(ids);
  });
  it("rejects missing variant IDs rather than silently loading the wrong config", async () => {
    await expect(
      loadLibraryModel({ entryId: "qwen3-0-6b", variantId: "missing" }),
    ).rejects.toThrow();
  });
});
