import { defaultPreset } from "./presets";

// Checked against gallery-manifest.json in model-library.test.ts. This small
// summary keeps the workspace from eagerly importing the complete catalog.
export const galleryModelCount = 111;
export const modelLibraryCount = galleryModelCount;
export interface ModelSelection {
  entryId: string;
  variantId: string;
}
export const initialSelection: ModelSelection = {
  entryId: defaultPreset.galleryId,
  variantId: "gallery",
};
