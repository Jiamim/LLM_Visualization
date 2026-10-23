import type { ModelGraph, ModuleNode } from "./graph";
import type { Tensor } from "./model";
import { categories } from "./presentation";

export type DisplayMode = "graph" | "2d" | "3d";
export interface TensorVolume {
  rows: number;
  columns: number;
  slices: number;
  rowSymbol: string;
  columnSymbol: string;
  sliceSymbol: string;
  empty: boolean;
  width: number;
  height: number;
  depth: number;
  gridRows: number;
  gridColumns: number;
}

// Geometry is a bounded illustration; labels always retain the original shape.
// A rank > 3 tensor becomes a stack of its last-two-axis planes.
export function tensorVolume(tensor?: Tensor): TensorVolume {
  const shape = tensor?.shape ?? [];
  const symbols = tensor?.symbols ?? [];
  const columns = shape.at(-1) ?? 1;
  const rows = shape.length > 1 ? shape.at(-2)! : 1;
  const slices = shape
    .slice(0, -2)
    .reduce((product, value) => product * value, 1);
  const extent = (n: number, min: number, max: number) =>
    min + Math.min(1, Math.log2(Math.max(1, n)) / 14) * (max - min);
  return {
    rows,
    columns,
    slices,
    rowSymbol: shape.length > 1 ? (symbols.at(-2) ?? "rows") : "1",
    columnSymbol: symbols.at(-1) ?? "1",
    sliceSymbol: symbols.slice(0, -2).join(" × ") || "1",
    empty: shape.some((n) => n === 0),
    width: extent(columns, 2.4, 5),
    height: extent(rows, 0.6, 2.3),
    depth: extent(slices, 0.24, 1.7),
    gridRows: Math.min(10, rows),
    gridColumns: Math.min(18, columns),
  };
}

export const TENSOR_NODE_WIDTH = 332;
export const TENSOR_NODE_HEIGHT = 256;
export interface SpatialBlock {
  id: string;
  node: ModuleNode;
  tensor?: Tensor;
  volume: TensorVolume;
  color: string;
  position: [number, number, number];
}

/** Keep graph positions and edges: attention branches and cache reuse stay intact. */
export function spatialBlocks(graph: ModelGraph): SpatialBlock[] {
  if (!graph.nodes.length) return [];
  const xs = graph.nodes.map((n) => n.position.x);
  const ys = graph.nodes.map((n) => n.position.y);
  const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
  const cy = (Math.min(...ys) + Math.max(...ys)) / 2;
  return graph.nodes
    .filter((node) => !node.data.frame)
    .map((node) => {
      const tensor = node.data.outputs[0] ?? node.data.inputs[0];
      return {
        id: node.id,
        node,
        tensor,
        volume: tensorVolume(tensor),
        color: node.data.residual
          ? "#aa6b35"
          : categories[node.data.category].color,
        position: node.data.worldPosition ?? [
          (node.position.x - cx) / 58,
          -(node.position.y - cy) / 46,
          0,
        ],
      };
    });
}

export function tensorLabel(
  tensor: Tensor | undefined,
  symbolic: boolean,
): string {
  return tensor
    ? `[${(symbolic ? tensor.symbols : tensor.shape).join(", ")}]`
    : "无张量输出";
}

export function displayModeFromURL(): DisplayMode {
  const value = new URLSearchParams(window.location.search).get("view");
  return value === "2d" || value === "3d" ? value : "graph";
}
