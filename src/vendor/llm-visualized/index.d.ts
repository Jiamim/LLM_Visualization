import * as THREE from "three";
export class WeightMatrixVisualization {
  constructor(data?: number[] | null, position?: THREE.Vector3, width?: number, height?: number, depth?: number, topWidthFactor?: number, cornerRadius?: number, numberOfSlits?: number, slitWidth?: number, slitDepthFactor?: number, slitBottomWidthFactor?: number, slitTopWidthFactor?: number | null, useInstancedSlices?: boolean);
  group: THREE.Group;
  mesh: THREE.Mesh;
}
export class VectorVisualizationInstancedPrism {
  constructor(data?: number[], position?: THREE.Vector3, subsections?: number, instances?: number);
  group: THREE.Group;
  mesh: THREE.InstancedMesh<THREE.BufferGeometry, THREE.MeshBasicMaterial>;
  updateKeyColorsFromData(data: number[], samples?: number, options?: Record<string, unknown> | null): void;
  setInstanceGradientColors(index: number, start: THREE.Color, end: THREE.Color, update?: boolean): void;
  dispose(): void;
}
