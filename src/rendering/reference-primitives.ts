import * as THREE from "three";
import {
  WeightMatrixVisualization,
  VectorVisualizationInstancedPrism,
} from "../vendor/llm-visualized/index.js";
import {
  recordedValueColor,
  UNKNOWN_VALUE_COLOR,
} from "../lib/visual-values";
import { demoBeat } from "../lib/demo-pacing";
import {
  recordedLane,
  sampleCoordinates,
  TOKEN_COLORS,
  type ReferenceVisual,
} from "../lib/reference-scene";
import type { TraceSample } from "../lib/demo-trace";
import { activationValue } from "../lib/demo-explanations";

export interface ReferenceObject {
  group: THREE.Group;
  hit: THREE.Mesh<THREE.BoxGeometry, THREE.MeshLambertMaterial>;
  outline: THREE.LineSegments;
  animate: (
    progress: number,
    active: boolean,
    token: number,
    reduced: boolean,
  ) => void;
  select: (selected: boolean) => void;
}
const basic = (color: string, opacity = 1) =>
  new THREE.MeshBasicMaterial({
    color,
    transparent: opacity < 1,
    opacity,
    depthWrite: opacity >= 1,
  });
function line(points: THREE.Vector3[], color: string, opacity = 0.45) {
  return new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(points),
    new THREE.LineBasicMaterial({ color, transparent: true, opacity }),
  );
}
export function textSprite(
  text: string,
  color = "#d4d8e0",
  height = 0.45,
): THREE.Sprite {
  const canvas = document.createElement("canvas");
  canvas.width = 512;
  canvas.height = 80;
  const ctx = canvas.getContext("2d")!;
  ctx.clearRect(0, 0, 512, 80);
  ctx.font = '500 34px "Atlas Sans", monospace';
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillStyle = color;
  ctx.fillText(text, 256, 40, 500);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(
    new THREE.SpriteMaterial({
      map: texture,
      transparent: true,
      depthTest: false,
    }),
  );
  sprite.scale.set(height * 6.4, height, 1);
  return sprite;
}
export function disposeReferenceGroup(group: THREE.Group) {
  const geometries = new Set<THREE.BufferGeometry>(),
    materials = new Set<THREE.Material>(),
    textures = new Set<THREE.Texture>();
  group.traverse((o) => {
    const obj = o as THREE.Mesh;
    if (obj.geometry) geometries.add(obj.geometry);
    if (obj.material)
      for (const m of Array.isArray(obj.material)
        ? obj.material
        : [obj.material]) {
        materials.add(m);
        const map = (m as THREE.MeshBasicMaterial).map;
        if (map) textures.add(map);
      }
  });
  for (const g of geometries) g.dispose();
  for (const m of materials) m.dispose();
  for (const t of textures) t.dispose();
  group.clear();
}
/** Uses the upstream instanced prism shader with explicit target-model dimensions. */
export function vectorStrip(
  width: number,
  dimension: number,
  lane: number,
  values?: (number | null)[],
  color?: string,
) {
  const coordinates = sampleCoordinates(dimension, 24),
    count = Math.max(1, coordinates.length);
  // Deterministic teaching palette is visual scaffolding, never presented as activation data.
  const data = coordinates.map(
    (_, i) => Math.sin((i + 1) * 1.13 + lane * 2.4) * 0.8,
  );
  const v = new VectorVisualizationInstancedPrism(
    data,
    new THREE.Vector3(),
    count,
    count,
  );
  v.group.scale.set(width / (count * 9.6), 0.022, 0.026);
  const hue = new THREE.Color(
    color ?? TOKEN_COLORS[lane % TOKEN_COLORS.length],
  ).getHSL({ h: 0, s: 0, l: 0 }).h;
  v.updateKeyColorsFromData(data, count, {
    type: "hueRange",
    baseHue: hue,
    hueSpread: 0.07,
    saturation: 0.67,
    minLightness: 0.08,
    maxLightness: 0.42,
    valueMin: -1,
    valueMax: 1,
  });
  // Instanced color is multiplied after the upstream gradient shader. Keep it
  // neutral so constructor colors do not tint the new model's lane palette.
  for (let i = 0; i < count; i++)
    v.mesh.setColorAt(i, new THREE.Color(1, 1, 1));
  if (v.mesh.instanceColor) v.mesh.instanceColor.needsUpdate = true;
  for (let i = 0; i < count; i++) {
    const colorAt = (index: number) => {
      if (!values)
        return new THREE.Color().setHSL(
          hue,
          0.65,
          0.08 + (data[index] + 1) * 0.17,
        );
      const c = new THREE.Color(recordedValueColor(values[index]));
      return c;
    };
    v.setInstanceGradientColors(
      i,
      colorAt(i),
      colorAt(values ? i : Math.min(count - 1, i + 1)),
    );
  }
  v.mesh.frustumCulled = true;
  v.mesh.computeBoundingSphere();
  v.mesh.material.transparent = true;
  v.mesh.material.depthWrite = false;
  v.mesh.material.toneMapped = false;
  return v.group;
}
function matrix(
  width: number,
  height: number,
  depth: number,
  ratio: number,
  color: string,
) {
  const m = new WeightMatrixVisualization(
    null,
    new THREE.Vector3(),
    width,
    height,
    depth,
    ratio,
    0.06,
    0,
    0.1,
    1,
    0.9,
    null,
    false,
  );
  m.group.traverse((o) => {
    if (o instanceof THREE.Mesh) {
      o.geometry = o.geometry.clone();
      for (const mat of Array.isArray(o.material)
        ? o.material
        : [o.material]) {
        const physical = mat as THREE.MeshPhysicalMaterial;
        physical.envMapIntensity = 0.5;
        physical.emissive.set(color);
        physical.emissiveIntensity = 0.12;
        const uniforms = physical.userData.sciFiUniforms;
        if (uniforms) {
          uniforms.uAccentColor.value.set(color);
          uniforms.uSecondaryColor.value.set("#111923");
          uniforms.uEdgeColor.value.set(color);
        }
      }
    }
  });
  // Slice markings align with depth/token lanes as on the upstream matrices.
  for (let i = 1; i < Math.min(16, Math.ceil(depth / 0.6)); i++) {
    const z = -depth / 2 + (i * depth) / Math.ceil(depth / 0.6);
    m.group.add(
      line(
        [
          new THREE.Vector3(-width / 2, -height / 2, z),
          new THREE.Vector3((-width * ratio) / 2, height / 2, z),
          new THREE.Vector3((width * ratio) / 2, height / 2, z),
          new THREE.Vector3(width / 2, -height / 2, z),
        ],
        color,
        0.27,
      ),
    );
  }
  return m.group;
}

/** Shared upstream silhouette for the instanced distant-layer overview. */
export function overviewMatrixGeometry() {
  const m = new WeightMatrixVisualization(
    null,
    new THREE.Vector3(),
    1,
    0.55,
    1,
    0.7,
    0.035,
    0,
    0.1,
    1,
    0.9,
    null,
    false,
  );
  const geometry = m.mesh.geometry.clone();
  geometry.computeBoundingBox();
  const size = geometry.boundingBox!.getSize(new THREE.Vector3());
  geometry.scale(1 / size.x, 1 / size.y, 1 / size.z);
  geometry.center();
  m.group.traverse((o) => {
    if (o instanceof THREE.Mesh)
      for (const mat of Array.isArray(o.material) ? o.material : [o.material])
        mat.dispose();
  });
  return geometry;
}

export function createReferenceObject(
  spec: ReferenceVisual,
  id: string,
  tokenLabels: string[] = [],
  samples: TraceSample[] = [],
  head = 0,
): ReferenceObject {
  const group = new THREE.Group();
  group.userData.id = id;
  group.userData.visualKind = spec.kind;
  const { width: w, height: h, depth: d } = spec;
  const geometry = new THREE.BoxGeometry(w, h + 1.7, d + 0.6);
  const hit = new THREE.Mesh(
    geometry,
    new THREE.MeshLambertMaterial({
      transparent: true,
      opacity: 0,
      depthWrite: false,
      colorWrite: false,
    }),
  );
  hit.userData.id = id;
  group.add(hit);
  const outline = new THREE.LineSegments(
    new THREE.EdgesGeometry(geometry),
    new THREE.LineBasicMaterial({
      color: spec.color,
      transparent: true,
      opacity: 0,
    }),
  );
  group.add(outline);
  const core = new THREE.Group();
  group.add(core);
  const incoming: THREE.Group[] = [],
    outgoing: THREE.Group[] = [],
    attentionTiles: {
      mesh: THREE.InstancedMesh;
      base: THREE.Color[];
      rows: number;
      cols: number;
    }[] = [];
  const sample = samples.find(
    (s) => s.nodeId === id && s.direction === "output" && s.tensorIndex === 0,
  );
  const inputSample = samples.find(
    (s) => s.nodeId === id && s.direction === "input" && s.tensorIndex === 0,
  );
  const widthRatio = Math.max(
    0.38,
    Math.min(1.9, spec.outputWidth / spec.inputWidth),
  );
  const matrixWidth = widthRatio > 1 ? w / widthRatio : w;
  if (spec.kind === "matrix")
    core.add(matrix(matrixWidth, h, d, widthRatio, spec.color));
  if (spec.kind === "block") {
    for (let i = 0; i < 4; i++) {
      const m = matrix(w * 0.75, 0.22, d, 1, spec.color);
      m.position.y = (i - 1.5) * 0.42;
      core.add(m);
    }
  }
  if (spec.kind === "norm") {
    const ring = new THREE.Mesh(
      new THREE.TorusGeometry(w * 0.32, 0.14, 10, 48),
      new THREE.MeshStandardMaterial({
        color: "#d7dee8",
        metalness: 0.4,
        roughness: 0.4,
      }),
    );
    ring.rotation.x = Math.PI / 2;
    core.add(ring);
    const gamma = textSprite(
      spec.rms ? "γ · x / RMS(x)" : "γ · x̂ + β",
      "#abbacb",
      0.42,
    );
    gamma.position.set(0, 0, d / 2 + 0.5);
    core.add(gamma);
  }
  if (spec.kind === "residual") {
    const ring = new THREE.Mesh(
      new THREE.TorusGeometry(0.47, 0.055, 8, 36),
      basic("#d4c7a2"),
    );
    ring.position.z = d / 2;
    core.add(ring);
    if (spec.residualMix) {
      const label = textSprite("gated residual mix", "#e9d5a0", 0.35);
      label.position.y = 0.9;
      core.add(label);
    }
    if (!spec.residualMix)
      core.add(
        line(
          [
            new THREE.Vector3(-0.27, 0, d / 2),
            new THREE.Vector3(0.27, 0, d / 2),
          ],
          "#e9d5a0",
          1,
        ),
      );
    if (!spec.residualMix)
      core.add(
        line(
          [
            new THREE.Vector3(0, -0.27, d / 2),
            new THREE.Vector3(0, 0.27, d / 2),
          ],
          "#e9d5a0",
          1,
        ),
      );
    const path = new THREE.CatmullRomCurve3([
      new THREE.Vector3(-w * 0.42, -2, 0),
      new THREE.Vector3(-w * 0.7, -1, 0),
      new THREE.Vector3(-w * 0.7, 0, 0),
      new THREE.Vector3(0, 0, 0),
    ]);
    core.add(
      new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(path.getPoints(30)),
        new THREE.LineBasicMaterial({
          color: "#c9ad68",
          transparent: true,
          opacity: 0.8,
        }),
      ),
    );
  }
  if (spec.kind === "attention") {
    const firstHead = Math.max(0, Math.min(head, spec.heads - 1));
    const count = Math.min(spec.heads - firstHead, 8),
      rows = Math.min(spec.queryLength, 8),
      cols = Math.max(1, Math.min(spec.keyLength, 12)),
      tileW = Math.min(1.65, 5.8 / count);
    for (let headIndex = 0; headIndex < count; headIndex++) {
      const tiles = new THREE.InstancedMesh(
        new THREE.BoxGeometry(
          (tileW / Math.max(1, cols)) * 0.85,
          0.035,
          0.55 / Math.max(1, rows),
        ),
        new THREE.MeshBasicMaterial(),
        Math.max(1, rows * cols),
      );
      const dummy = new THREE.Object3D();
      const base: THREE.Color[] = [];
      let index = 0;
      for (let r = 0; r < rows; r++)
        for (let c = 0; c < cols; c++) {
          dummy.position.set(
            (headIndex - (count - 1) / 2) * (tileW + 0.12) +
              ((c - (cols - 1) / 2) * tileW) / cols,
            0,
            ((r - (rows - 1) / 2) * 0.65) / Math.max(1, rows),
          );
          dummy.updateMatrix();
          tiles.setMatrixAt(index, dummy.matrix);
          const key = sampleCoordinates(spec.keyLength, 12)[c];
          const position = spec.keyOffset + key;
          const masked =
            spec.visibility === "causal" &&
            (position > spec.past + r ||
              (spec.window !== null &&
                position <= spec.past + r - spec.window));
          const recorded =
            sample && spec.attentionGrid
              ? recordedLane(sample, spec.tensor, r, firstHead + headIndex, [
                  key,
                ])[0]
              : undefined;
          const unknown =
            recorded === null ||
            (spec.recordedMode && recorded === undefined) ||
            spec.visibility === "unknown" ||
            !spec.attentionGrid;
          const color = new THREE.Color(
            masked
              ? "#18202c"
              : unknown
                ? UNKNOWN_VALUE_COLOR
                : recorded !== undefined
                  ? recordedValueColor(recorded)
                  : TOKEN_COLORS[r % 6],
          ).multiplyScalar(
            masked || unknown || recorded !== undefined
              ? 1
              : 0.3 + ((c + r + headIndex) % 5) * 0.14,
          );
          base.push(color);
          tiles.setColorAt(index++, color);
        }
      tiles.position.y = 0.15;
      core.add(tiles);
      attentionTiles.push({ mesh: tiles, base, rows, cols });
      for (const [offset, color] of [
        [-0.28, "#7db4ff"],
        [0, "#e1a4cf"],
        [0.28, "#77dcb6"],
      ] as const) {
        const qkv = vectorStrip(
          tileW * 0.72,
          spec.outputWidth,
          headIndex,
          spec.recordedMode
            ? Array(sampleCoordinates(spec.outputWidth).length).fill(null)
            : undefined,
          color,
        );
        qkv.position.set(
          (headIndex - (count - 1) / 2) * (tileW + 0.12),
          -0.6,
          offset,
        );
        qkv.scale.y *= 0.5;
        core.add(qkv);
      }
    }
    const label = textSprite(
      `${spec.heads} Q · ${spec.kvHeads} KV${spec.heads > 8 ? ` · ${firstHead + 1}–${firstHead + count}` : ""}${spec.visibility === "unknown" ? " · candidates unknown" : !spec.attentionGrid ? " · context (no probabilities)" : ""}`,
      "#9caabd",
      0.35,
    );
    label.position.set(0, 0.7, 0);
    core.add(label);
  }
  if (spec.kind === "activation") {
    const points = Array.from({ length: 50 }, (_, i) => {
      const x = (i / 49 - 0.5) * 5;
      return new THREE.Vector3(
        x,
        activationValue(x, spec.activation) * 0.5,
        0,
      );
    });
    core.add(line(points, "#d9a5f0", 1));
    core.add(
      line(
        [new THREE.Vector3(-2.7, 0, 0), new THREE.Vector3(2.7, 0, 0)],
        "#647083",
      ),
    );
  }
  if (spec.kind === "cache" || spec.kind === "state") {
    for (let i = 0; i < 3; i++) {
      const m = matrix(
        w * 0.76,
        0.25,
        d,
        1,
        spec.kind === "cache" ? "#70bbbd" : "#d0ac8b",
      );
      m.position.y = (i - 1) * 0.44;
      core.add(m);
    }
    const label = textSprite(
      spec.kind === "cache" ? spec.cacheLabel : "recurrent state",
      "#83cbbb",
      0.38,
    );
    label.position.set(0, 0.95, 0);
    core.add(label);
  }
  if (spec.kind === "router" || spec.kind === "experts") {
    const n = Math.min(8, spec.experts),
      cellWidth = Math.min(1.2, w / n - 0.15);
    for (let i = 0; i < n; i++) {
      const m = matrix(
        cellWidth,
        0.75,
        Math.min(d, 2),
        0.85,
        i < spec.topK ? "#cfb0f4" : "#677082",
      );
      m.position.x = (i - (n - 1) / 2) * (cellWidth + 0.2);
      core.add(m);
      core.add(
        line(
          [
            new THREE.Vector3(0, -1.1, 0),
            new THREE.Vector3(m.position.x, -0.4, 0),
          ],
          i < spec.topK ? "#c695f4" : "#303944",
          i < spec.topK ? 0.9 : 0.4,
        ),
      );
    }
    const label = textSprite(
      `${spec.experts} experts · top ${spec.topK}`,
      "#c4a5e1",
      0.35,
    );
    label.position.y = 1;
    core.add(label);
  }
  if (spec.kind === "token") {
    for (let lane = 0; lane < spec.lanes; lane++) {
      const chip = new THREE.Mesh(
        new THREE.BoxGeometry(Math.min(w, 2.3), 0.76, 0.15),
        new THREE.MeshStandardMaterial({
          color: TOKEN_COLORS[lane % 6],
          roughness: 0.8,
          metalness: 0.1,
        }),
      );
      chip.position.set(0, 0, (lane - (spec.lanes - 1) / 2) * 0.85);
      core.add(chip);
      const label = textSprite(
        tokenLabels[lane] ?? `t${lane + 1}`,
        "#ffffff",
        0.5,
      );
      label.position.copy(chip.position).add(new THREE.Vector3(0, 0, 0.12));
      core.add(label);
    }
  } else {
    for (let lane = 0; lane < spec.lanes; lane++) {
      const z = (lane - (spec.lanes - 1) / 2) * 0.85;
      const inputDimension =
        spec.inputTensor?.shape.at(-1) ?? spec.inputWidth;
      const inputValues = inputSample
        ? recordedLane(
            inputSample,
            spec.inputTensor,
            lane,
            head,
            sampleCoordinates(inputDimension),
          )
        : spec.recordedMode
          ? Array(sampleCoordinates(inputDimension).length).fill(null)
          : undefined;
      const source = vectorStrip(
        Math.min(w, 5.8),
        inputDimension,
        lane,
        inputValues,
      );
      const values = sample
        ? recordedLane(
            sample,
            spec.tensor,
            lane,
            head,
            sampleCoordinates(spec.tensor?.shape.at(-1) ?? spec.outputWidth),
          )
        : spec.recordedMode
          ? Array(sampleCoordinates(spec.outputWidth).length).fill(null)
          : undefined;
      const target = vectorStrip(
        Math.min(w, 5.8),
        spec.tensor?.shape.at(-1) ?? spec.outputWidth,
        lane,
        values,
      );
      source.position.set(0, -h / 2 - 0.6, z);
      target.position.set(0, h / 2 + 0.4, z);
      group.add(source, target);
      incoming.push(source);
      outgoing.push(target);
    }
  }
  if (spec.kind === "vector") core.visible = false;
  const scanner = new THREE.Mesh(
    new THREE.PlaneGeometry(0.06, d + 0.4),
    basic("#e8f1ff", 0.75),
  );
  scanner.rotation.x = -Math.PI / 2;
  scanner.position.y = h / 2 + 0.03;
  scanner.visible = false;
  group.add(scanner);
  // Probability-like bars are not fabricated: output vectors use the same teaching palette.
  const animate = (
    p: number,
    active: boolean,
    token: number,
    reduced: boolean,
  ) => {
    const beat = demoBeat(p);
    const i = Math.max(0, Math.min(spec.lanes - 1, token));
    incoming.forEach((v, j) => {
      v.position.y =
        -h / 2 -
        0.6 +
        (active && j === i && !reduced && !beat.output
          ? beat.transfer * 0.48
          : 0);
      v.traverse((o) => {
        if (o instanceof THREE.Mesh)
          o.material.opacity = active ? 0.95 : 0.42;
      });
    });
    outgoing.forEach((v, j) => {
      v.visible = !active || p >= 0.33;
      v.scale.y =
        0.022 *
        (active && j === i && !reduced
          ? 0.008 + 0.992 * (beat.output ? 1 : beat.computing)
          : 1);
    });
    scanner.visible = active && p >= 0.33 && p < 0.76 && !reduced;
    scanner.position.x = (beat.computing - 0.5) * w;
    for (const { mesh, base, cols } of attentionTiles) {
      for (let n = 0; n < base.length; n++) {
        const row = Math.floor(n / cols),
          shown =
            !active ||
            reduced ||
            beat.output ||
            row < i ||
            (row === i && (n % cols) / cols <= beat.computing);
        mesh.setColorAt(n, base[n].clone().multiplyScalar(shown ? 1 : 0.12));
      }
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    }
  };
  const select = (selected: boolean) => {
    (outline.material as THREE.LineBasicMaterial).opacity = selected
      ? 0.16
      : 0;
    core.traverse((o) => {
      if (
        o instanceof THREE.Mesh &&
        o.material instanceof THREE.MeshStandardMaterial
      )
        o.material.emissiveIntensity = selected ? 0.3 : 0.1;
    });
  };
  return { group, hit, outline, animate, select };
}
