import { it, expect } from "vitest";
import reference from "../../docs/gallery/reference-counts.json";
import { loadGalleryModel, galleryModels } from "../data/gallery";
import { catalogParameters } from "./catalog";
it.each(Object.entries(reference))(
  "validates counted backbone weights for %s",
  async (id, ref) => {
    const model = await loadGalleryModel(
      galleryModels.find((m) => m.id === id)!,
    );
    const p = catalogParameters(model);
    expect(
      p.total,
      `first blocks: ${p.perLayer.slice(0, 4).map((p) => p.total)} vs ${ref.layers.slice(0, 4)}`,
    ).toBe(ref.total);
  },
);
