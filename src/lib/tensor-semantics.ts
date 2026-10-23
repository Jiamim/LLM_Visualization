import type { AxisRole, Tensor, TensorRole } from "./model";
import type { TraceSample } from "./demo-trace";

const axisRoles: Record<string, AxisRole> = {
  B: "batch",
  S: "query",
  Q: "query",
  T: "key",
  Klen: "key",
  Tcache: "key",
  P: "key",
  Pcache: "key",
  W: "key",
  C: "key",
  R: "key",
  Nq: "head",
  Hq: "head",
  Nstate: "head",
  Nkv: "kv-head",
  Hkv: "kv-head",
};
const tensorRoles: Record<string, TensorRole> = {
  query: "query",
  q: "query",
  key: "key",
  current_key: "key",
  value: "value",
  current_value: "value",
  attention_scores: "scores",
  scores: "scores",
  attention_weights: "probabilities",
  context: "context",
  key_cache: "state",
  value_cache: "state",
  compressed_KV: "state",
  main_kv: "state",
  window_kv: "state",
  index_key: "state",
};
/** Adapter symbols are translated once; consumers use typed axes and tensor roles. */
export function annotateTensor(tensor: Tensor): Tensor {
  return {
    ...tensor,
    axes: tensor.axes ?? tensor.symbols.map((s) => axisRoles[s] ?? "feature"),
    role: tensor.role ?? tensorRoles[tensor.name] ?? "activation",
  };
}
export function tensorAxes(tensor: Tensor): AxisRole[] {
  return annotateTensor(tensor).axes!;
}
export interface TensorSlice {
  token: number;
  head: number;
  batch?: number;
}
/** Exact tensor-local coordinates: no interpolation, clamping, or filling missing data. */
export function tensorCoordinates(
  tensor: Tensor,
  slice: TensorSlice,
  plane?: { row: number; column: number },
): number[] {
  const axes = tensorAxes(tensor);
  return tensor.shape.map((_size, i) => {
    if (plane && i === tensor.shape.length - 1) return plane.column;
    if (plane && i === tensor.shape.length - 2) return plane.row;
    if (axes[i] === "query" || axes[i] === "key") return slice.token;
    if (axes[i] === "head" || axes[i] === "kv-head") return slice.head;
    if (axes[i] === "batch") return slice.batch ?? 0;
    return 0;
  });
}
const coordinateIndices = new WeakMap<number[][], Map<string, number>>();

export function traceValue(
  sample: TraceSample | undefined,
  tensor: Tensor,
  indices: number[],
): number | null {
  if (
    !sample ||
    indices.length !== tensor.shape.length ||
    indices.some(
      (x, i) => !Number.isInteger(x) || x < 0 || x >= tensor.shape[i],
    )
  )
    return null;
  // Parsed records are immutable. Scene remapping creates new sample objects,
  // but preserves the coordinate array, so every view shares the same index.
  let index = coordinateIndices.get(sample.indices);
  if (!index) {
    index = new Map(sample.indices.map((coords, i) => [coords.join(","), i]));
    coordinateIndices.set(sample.indices, index);
  }
  const i = index.get(indices.join(","));
  return i === undefined ? null : sample.values[i];
}
/** Width of a projected feature vector, independent of a later head reshape. */
export function featureWidth(tensor: Tensor): number {
  const axes = tensorAxes(tensor);
  return tensor.shape.reduce(
    (n, size, i) =>
      ["batch", "query", "key"].includes(axes[i]) ? n : n * size,
    1,
  );
}

/** A KV tensor exposes KV heads, even when the model has more Query heads. */
export function tensorHeadCount(
  tensor: Tensor | undefined,
): number | undefined {
  if (!tensor) return undefined;
  const axis = tensorAxes(tensor).findIndex(
    (role) => role === "head" || role === "kv-head",
  );
  return axis < 0 ? undefined : tensor.shape[axis];
}
