import qwen3 from "./qwen3-0.6b.json";
import llama from "./llama3.2-1b.json";
import deepseek from "./deepseek-v4.1-flash.json";
import gemmaE2B from "./gemma4-e2b.json";
import gemmaE4B from "./gemma4-e4b.json";
import type { RawConfig } from "../lib/model";

export interface Preset {
  id: string;
  name: string;
  repo: string;
  raw: RawConfig;
  /** Explicit identity: variants of a gallery model share one library card. */
  galleryId: string;
  variantLabel?: string;
}
export const presets: Preset[] = [
  {
    id: "qwen3",
    name: "Qwen3 · 0.6B",
    repo: "Qwen/Qwen3-0.6B",
    raw: qwen3,
    galleryId: "qwen3-0-6b",
    variantLabel: "对话版",
  },
  {
    id: "llama",
    name: "Llama 3.2 · 1B",
    repo: "unsloth/Llama-3.2-1B-Instruct",
    raw: llama,
    galleryId: "llama-3-2-1b",
  },
  {
    id: "deepseek-v41",
    name: "DeepSeek · V4.1 Flash",
    repo: "deepseek-ai/DeepSeek-V4.1-Flash",
    raw: deepseek,
    galleryId: "deepseek-v4-1-flash",
  },
  {
    id: "gemma4-e2b",
    name: "Gemma 4 · E2B",
    repo: "google/gemma-4-E2B-it",
    raw: gemmaE2B,
    galleryId: "gemma-4-e2b",
  },
  {
    id: "gemma4-e4b",
    name: "Gemma 4 · E4B",
    repo: "google/gemma-4-E4B-it",
    raw: gemmaE4B,
    galleryId: "gemma-4-e4b",
  },
];

export const defaultPreset = presets.find(
  (preset) => preset.id === "deepseek-v41",
)!;
