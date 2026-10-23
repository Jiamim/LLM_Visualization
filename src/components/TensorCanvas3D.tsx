import { useEffect, useMemo, useRef, useState } from "react";
import { Box, Focus, Minus, Plus, RotateCcw, Scan } from "lucide-react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import {
  spatialBlocks,
  tensorLabel,
  type SpatialBlock,
} from "../lib/tensor-layout";
import type { TensorCanvasProps } from "./TensorCanvas2D";
import {
  REFERENCE_LAYER_HEIGHT,
  referenceVisual,
} from "../lib/reference-scene";
import {
  createReferenceObject,
  overviewMatrixGeometry,
  disposeReferenceGroup,
  vectorStrip,
  type ReferenceObject,
} from "../rendering/reference-primitives";
import { demoBeat } from "../lib/demo-pacing";

interface BlockObject {
  group: THREE.Group;
  mesh: THREE.Mesh<THREE.BoxGeometry, THREE.MeshLambertMaterial>;
  outline: THREE.LineSegments;
  block: SpatialBlock;
  tokenBands?: THREE.Mesh[];
  scan?: THREE.Mesh;
  visual?: ReferenceObject;
}
interface SceneState {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  controls: OrbitControls;
  content: THREE.Group;
  objects: Map<string, BlockObject>;
  draw: () => void;
  fit: (wholeModel?: boolean) => void;
  focus: (id: string, following?: boolean) => void;
  stopMotion: () => void;
  flows: {
    source: string;
    target: string;
    curve: THREE.QuadraticBezierCurve3;
    marker: THREE.Object3D;
  }[];
}
const disposeGroup = disposeReferenceGroup;

export default function TensorCanvas3D(props: TensorCanvasProps) {
  const {
    graph,
    selectedId,
    activeId,
    symbolic,
    showShapes,
    sceneKey,
    followPlayback,
    onSelect,
    onDrill,
    onFallback,
    immersive = false,
    animationProgress,
    worldLayers,
    tokenCount = 1,
    activeToken = 0,
    brightness = 100,
  } = props;
  const blocks = useMemo(
    () =>
      spatialBlocks(graph).map((block) => {
        const visual = referenceVisual(
          block.node,
          props.model,
          props.scenario,
        );
        return {
          ...block,
          color: visual.color,
          volume: {
            ...block.volume,
            width: visual.width,
            height: visual.height + 1.7,
            depth: visual.depth + 0.6,
          },
        };
      }),
    [graph, props.model, props.scenario],
  );
  const [spacing, setSpacing] = useState(1);
  const [failed, setFailed] = useState(false);
  const [ready, setReady] = useState(false);
  const host = useRef<HTMLDivElement>(null);
  const labels = useRef(new Map<string, HTMLButtonElement>());
  const runtime = useRef<SceneState | null>(null);
  const current = useRef(props);
  current.current = props;
  const previousScene = useRef("");
  const previousSelection = useRef(selectedId);
  const selected = activeId ?? selectedId;
  const selectedBlock = blocks.find((block) => block.id === selected);
  const world = useRef<THREE.Group | null>(null);
  const followedGroup = useRef<string | null>(null);

  useEffect(() => {
    const container = host.current!;
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    } catch {
      setFailed(true);
      return;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.7));
    renderer.setClearColor("#050608", 1);
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.25;
    renderer.domElement.setAttribute("aria-label", "可旋转的三维张量场景");
    renderer.domElement.setAttribute("role", "img");
    container.prepend(renderer.domElement);
    const scene = new THREE.Scene();

    scene.add(new THREE.AmbientLight(0xffffff, 2.0));
    const light = new THREE.DirectionalLight(0xffffff, 2.7);
    light.position.set(8, 16, 25);
    scene.add(light);
    const worldExtent = (worldLayers?.length ?? 0) * REFERENCE_LAYER_HEIGHT;
    const camera = new THREE.PerspectiveCamera(
      42,
      1,
      0.05,
      Math.max(10000, worldExtent * 12),
    );
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = false;
    controls.minDistance = 2;
    controls.maxDistance = Math.max(6000, worldExtent * 10);
    controls.screenSpacePanning = true;
    const content = new THREE.Group();
    scene.add(content);
    const worldGroup = new THREE.Group();
    scene.add(worldGroup);
    world.current = worldGroup;
    if (worldLayers) {
      // One instance per real layer part; overview remains cheap for hundreds of layers.
      const parts = worldLayers.flatMap((layer) =>
        (layer.parts ?? []).flatMap((part, i) =>
          Array.from(
            {
              length:
                part.visual?.kind === "attention"
                  ? Math.min(8, part.visual.heads)
                  : 1,
            },
            (_, head) => ({ layer, part, i, head }),
          ),
        ),
      );
      const instance = new THREE.InstancedMesh(
        overviewMatrixGeometry(),
        new THREE.MeshStandardMaterial({
          color: "#728098",
          transparent: true,
          opacity: 0.28,
          roughness: 0.85,
          metalness: 0.15,
        }),
        Math.max(1, parts.length),
      );
      const dummy = new THREE.Object3D();
      parts.forEach(({ layer, part, i, head }, index) => {
        const count =
          part.visual?.kind === "attention"
            ? Math.min(8, part.visual.heads)
            : 1;
        dummy.position.set(
          count > 1 ? (head - (count - 1) / 2) * 2.1 : 0,
          layer.index * REFERENCE_LAYER_HEIGHT +
            (i * REFERENCE_LAYER_HEIGHT) /
              Math.max(1, layer.parts?.length ?? 1),
          -2,
        );
        dummy.scale.set(
          part.residual ? 5 : count > 1 ? 1.6 : 9,
          part.residual ? 0.2 : 0.55,
          Math.max(2, Math.min(tokenCount, 8) * 0.85),
        );
        dummy.updateMatrix();
        instance.setMatrixAt(index, dummy.matrix);
        instance.setColorAt(index, new THREE.Color(part.color));
      });
      instance.userData.layerInstances = parts.map((p) => p.layer.index);
      instance.userData.baseHeights = parts.map((p) =>
        p.part.residual ? 0.2 : 0.55,
      );
      instance.computeBoundingSphere();
      worldGroup.add(instance);
      const centerline = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(-6, -20, -2),
          new THREE.Vector3(
            -6,
            worldLayers.length * REFERENCE_LAYER_HEIGHT,
            -2,
          ),
        ]),
        new THREE.LineBasicMaterial({
          color: "#778498",
          transparent: true,
          opacity: 0.32,
        }),
      );
      worldGroup.add(centerline);
    }
    const objects = new Map<string, BlockObject>();
    let cameraTween: {
      from: THREE.Vector3;
      to: THREE.Vector3;
      start: THREE.Vector3;
      target: THREE.Vector3;
      at: number;
      duration: number;
    } | null = null;
    let frame = 0,
      disposed = false;
    const projected = new THREE.Vector3();
    const render = () => {
      frame = 0;
      if (disposed) return;
      renderer.domElement.dataset.cameraTarget = controls.target
        .toArray()
        .map((v) => v.toFixed(2))
        .join(",");
      if (cameraTween) {
        const p = Math.min(
            1,
            (performance.now() - cameraTween.at) / cameraTween.duration,
          ),
          eased = p * p * (3 - 2 * p);
        camera.position.lerpVectors(cameraTween.from, cameraTween.to, eased);
        controls.target.lerpVectors(
          cameraTween.start,
          cameraTween.target,
          eased,
        );
        controls.update();
        if (p === 1) cameraTween = null;
        else draw();
      }
      renderer.render(scene, camera);
      const width = container.clientWidth,
        height = container.clientHeight;
      const selected = current.current.activeId ?? current.current.selectedId;
      const placed: {
        left: number;
        top: number;
        right: number;
        bottom: number;
      }[] = [];
      const selectedPosition = objects.get(selected ?? "")?.group.position;
      const ordered = [...objects.values()].sort(
        (a, b) =>
          Number(b.block.id === selected) - Number(a.block.id === selected),
      );
      for (const object of ordered) {
        const label = labels.current.get(object.block.id);
        if (!label) continue;
        if (!content.visible) {
          label.style.display = "none";
          continue;
        }
        projected.copy(object.group.position);
        projected.y -= object.block.volume.height / 2 + 0.38;
        projected.z += object.block.volume.depth / 2;
        projected.project(camera);
        const x = ((projected.x + 1) * width) / 2,
          y = ((1 - projected.y) * height) / 2;
        label.style.display = "block";
        const labelWidth = label.offsetWidth,
          labelHeight = label.offsetHeight;
        const left = Math.max(
          10,
          Math.min(x - labelWidth / 2, width - labelWidth - 10),
        );
        const top = Math.max(58, Math.min(y, height - labelHeight - 62));
        const bounds = {
          left,
          top,
          right: left + labelWidth,
          bottom: top + labelHeight,
        };
        const visible =
          projected.z > -1 &&
          projected.z < 1 &&
          x > 25 &&
          x < width - 25 &&
          y > 10 &&
          y < height - 42 &&
          (!immersive ||
            object.block.id === selected ||
            !selectedPosition ||
            object.group.position.distanceTo(selectedPosition) < 24) &&
          (object.block.id === selected ||
            !placed.some(
              (p) =>
                bounds.left < p.right + 8 &&
                bounds.right + 8 > p.left &&
                bounds.top < p.bottom + 6 &&
                bounds.bottom + 6 > p.top,
            ));
        label.style.display = visible ? "block" : "none";
        label.style.transform = `translate(${left}px,${top}px)`;
        label.style.zIndex = object.block.id === selected ? "3" : "2";
        if (visible) placed.push(bounds);
      }
    };
    const draw = () => {
      if (!disposed && !frame) frame = requestAnimationFrame(render);
    };
    const orient = new THREE.Vector3(
      immersive ? 0.45 : 0.35,
      immersive ? 0.32 : 0.28,
      1,
    ).normalize();
    const lookAt = (center: THREE.Vector3, distance: number) => {
      controls.target.copy(center);
      camera.position.copy(center).addScaledVector(orient, distance);
      controls.update();
      draw();
    };
    const overview = (enabled: boolean) => {
      content.visible = !enabled;
      worldGroup.scale.set(
        enabled ? 4 : 1,
        enabled ? 0.028 : 1,
        enabled ? 3 : 1,
      );
      worldGroup.traverse((o) => {
        if (o instanceof THREE.InstancedMesh && o.userData.baseHeights) {
          const m = new THREE.Matrix4(),
            p = new THREE.Vector3(),
            q = new THREE.Quaternion(),
            s = new THREE.Vector3();
          for (let i = 0; i < o.count; i++) {
            o.getMatrixAt(i, m);
            m.decompose(p, q, s);
            s.y = enabled
              ? o.userData.baseHeights[i] * 20
              : o.userData.layerInstances[i] === current.current.worldLayer
                ? 0.0001
                : o.userData.baseHeights[i];
            m.compose(p, q, s);
            o.setMatrixAt(i, m);
          }
          o.instanceMatrix.needsUpdate = true;
          (o.material as THREE.MeshStandardMaterial).opacity = enabled
            ? 0.75
            : 0.28;
        }
      });
    };
    const fit = (wholeModel = false) => {
      followedGroup.current = null;
      cameraTween = null;
      overview(wholeModel && !!worldLayers);
      if (scene.fog instanceof THREE.FogExp2)
        scene.fog.density = wholeModel ? 0 : 0.004;
      const bounds = new THREE.Box3().setFromObject(
        wholeModel && worldLayers ? worldGroup : content,
      );
      if (bounds.isEmpty()) return;
      const size = bounds.getSize(new THREE.Vector3()),
        center = bounds.getCenter(new THREE.Vector3());
      const halfFov = THREE.MathUtils.degToRad(camera.fov / 2);
      const distance =
        (Math.max(size.y, size.x / Math.max(camera.aspect, 0.1), size.z) /
          2 /
          Math.tan(halfFov)) *
          1.6 +
        size.z;
      lookAt(center, Math.max(distance, 12));
    };
    const focus = (id: string, following = false) => {
      const object = objects.get(id);
      if (!object) return;
      overview(false);
      let center = object.group.position.clone();
      const size = new THREE.Box3()
        .setFromObject(object.group)
        .getSize(new THREE.Vector3());
      let distance = Math.max(
        12,
        (Math.max(size.x / Math.max(camera.aspect, 0.5), size.y, size.z) /
          (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)))) *
          1.5 +
          size.z,
      );
      if (immersive) {
        if (scene.fog instanceof THREE.FogExp2) scene.fog.density = 0.004;
        const neighbors = [...objects.values()]
          .filter(
            (o) =>
              o.block.id === id ||
              (Math.abs(o.group.position.y - object.group.position.y) <
                0.05 &&
                o.block.node.data.category ===
                  object.block.node.data.category) ||
              current.current.graph.edges.some(
                (e) =>
                  (e.source === id && e.target === o.block.id) ||
                  (e.target === id && e.source === o.block.id),
              ),
          )
          .filter(
            (o) => o.group.position.distanceTo(object.group.position) < 24,
          );
        const bounds = new THREE.Box3();
        for (const o of neighbors) bounds.expandByObject(o.group);
        const size = bounds.getSize(new THREE.Vector3());
        center = bounds.getCenter(new THREE.Vector3());
        const siblings = neighbors.filter(
          (o) =>
            Math.abs(o.group.position.y - object.group.position.y) < 0.05,
        );
        center.lerp(object.group.position, siblings.length > 1 ? 0 : 0.4);
        distance = Math.max(
          23,
          (Math.max(size.y * 1.25, size.x / Math.max(camera.aspect, 0.55)) /
            (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)))) *
            1.3,
        );
        if (following) {
          const projected = object.group.position.clone().project(camera);
          const sameGroup =
            followedGroup.current === current.current.cameraGroup;
          const framed =
            Math.abs(projected.x) < 0.54 &&
            Math.abs(projected.y) < 0.48 &&
            projected.z > -1 &&
            projected.z < 1;
          const nearEnough =
            camera.position.distanceTo(object.group.position) <
            distance * 1.8;
          if (sameGroup && framed && nearEnough) {
            draw();
            return;
          }
          followedGroup.current = current.current.cameraGroup ?? null;
        }
      }
      if (
        immersive &&
        !current.current.reducedMotion &&
        current.current.playing
      ) {
        cameraTween = {
          from: camera.position.clone(),
          to: center.clone().addScaledVector(orient, distance),
          start: controls.target.clone(),
          target: center,
          at: performance.now(),
          duration: Math.min(
            1250,
            Math.max(700, (current.current.stepDuration ?? 5000) * 0.17),
          ),
        };
        draw();
      } else lookAt(center, distance);
    };
    runtime.current = {
      renderer,
      scene,
      camera,
      controls,
      content,
      objects,
      draw,
      fit,
      focus,
      stopMotion: () => {
        cameraTween = null;
      },
      flows: [],
    };
    controls.addEventListener("change", draw);
    const resize = () => {
      const width = container.clientWidth,
        height = container.clientHeight;
      if (!width || !height) return;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
      draw();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(container);
    resize();
    let start = { x: 0, y: 0 };
    const pointerDown = (event: PointerEvent) => {
      cameraTween = null;
      start = { x: event.clientX, y: event.clientY };
    };
    const pick = (event: MouseEvent) => {
      const rect = renderer.domElement.getBoundingClientRect();
      const point = new THREE.Vector2(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        (-(event.clientY - rect.top) / rect.height) * 2 + 1,
      );
      const ray = new THREE.Raycaster();
      ray.setFromCamera(point, camera);
      const hits = ray.intersectObjects(
        content.visible ? [...objects.values()].map((o) => o.mesh) : [],
      );
      if (hits[0]) return hits[0].object.userData.id as string;
      const layerHit = ray.intersectObjects(worldGroup.children)[0];
      if (layerHit?.instanceId !== undefined)
        current.current.onWorldLayer?.(
          layerHit.object.userData.layerInstances?.[layerHit.instanceId],
        );
      return undefined;
    };
    const pointerUp = (event: PointerEvent) => {
      if (
        event.button !== 0 ||
        Math.hypot(event.clientX - start.x, event.clientY - start.y) > 5
      )
        return;
      const id = pick(event);
      if (id) current.current.onSelect(id);
    };
    const doubleClick = (event: MouseEvent) => {
      const id = pick(event);
      if (id) current.current.onDrill(id);
    };
    const contextLost = (event: Event) => {
      event.preventDefault();
      setFailed(true);
    };
    renderer.domElement.addEventListener("pointerdown", pointerDown);
    renderer.domElement.addEventListener("pointerup", pointerUp);
    renderer.domElement.addEventListener("dblclick", doubleClick);
    renderer.domElement.addEventListener("webglcontextlost", contextLost);
    setReady(true);
    return () => {
      previousScene.current = "";
      disposed = true;
      cancelAnimationFrame(frame);
      observer.disconnect();
      renderer.domElement.removeEventListener("pointerdown", pointerDown);
      renderer.domElement.removeEventListener("pointerup", pointerUp);
      renderer.domElement.removeEventListener("dblclick", doubleClick);
      renderer.domElement.removeEventListener(
        "webglcontextlost",
        contextLost,
      );
      controls.removeEventListener("change", draw);
      controls.dispose();
      disposeGroup(content);
      disposeGroup(worldGroup);
      followedGroup.current = null;
      world.current = null;
      renderer.dispose();
      renderer.forceContextLoss();
      renderer.domElement.remove();
      runtime.current = null;
    };
  }, []);

  useEffect(() => {
    const group = world.current;
    if (!group) return;
    group.traverse((o) => {
      if (o instanceof THREE.InstancedMesh && o.userData.layerInstances) {
        const matrix = new THREE.Matrix4();
        for (let i = 0; i < o.count; i++) {
          o.getMatrixAt(i, matrix);
          const position = new THREE.Vector3(),
            rotation = new THREE.Quaternion(),
            scale = new THREE.Vector3();
          matrix.decompose(position, rotation, scale);
          scale.y =
            o.userData.layerInstances[i] === props.worldLayer
              ? 0.0001
              : o.userData.baseHeights[i];
          matrix.compose(position, rotation, scale);
          o.setMatrixAt(i, matrix);
        }
        o.instanceMatrix.needsUpdate = true;
      }
    });
    runtime.current?.draw();
  }, [props.worldLayer, ready]);

  useEffect(() => {
    const state = runtime.current;
    if (!state) return;
    disposeGroup(state.content);
    state.objects.clear();
    state.flows = [];
    for (const block of blocks) {
      const group = new THREE.Group();
      group.position.set(
        block.position[0],
        block.position[1] * spacing,
        block.position[2],
      );
      const visual = createReferenceObject(
        {
          ...referenceVisual(
            block.node,
            current.current.model,
            current.current.scenario,
          ),
          recordedMode: current.current.recordedMode,
        },
        block.id,
        current.current.tokenLabels,
        current.current.traceSamples,
        current.current.headIndex,
      );
      group.add(visual.group);
      state.content.add(group);
      state.objects.set(block.id, {
        group,
        mesh: visual.hit,
        outline: visual.outline,
        block,
        visual,
      });
    }

    for (const edge of graph.edges) {
      const source = state.objects.get(edge.source),
        target = state.objects.get(edge.target);
      if (!source || !target) continue;
      const a = source.group.position.clone(),
        b = target.group.position.clone();
      const direction = Math.sign(b.y - a.y) || 1;
      a.y += (direction * source.block.volume.height) / 2;
      b.y -= (direction * target.block.volume.height) / 2;
      const mid = a.clone().lerp(b, 0.5);
      mid.z -= edge.data?.residual ? 1.4 : 0.5;
      if (edge.data?.residual) mid.x -= 11;
      const curve = new THREE.QuadraticBezierCurve3(a, mid, b);
      const line = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(curve.getPoints(20)),
        new THREE.LineBasicMaterial({
          color: edge.data?.residual ? "#bd6f35" : "#9baabe",
          transparent: true,
          opacity: 0.6,
        }),
      );
      state.content.add(line);
      const arrow = new THREE.Mesh(
        new THREE.ConeGeometry(0.075, 0.25, 6),
        new THREE.MeshBasicMaterial({ color: "#8c9db4" }),
      );
      arrow.position.copy(curve.getPoint(0.88));
      arrow.quaternion.setFromUnitVectors(
        new THREE.Vector3(0, 1, 0),
        curve.getTangent(0.88).normalize(),
      );
      state.content.add(arrow);
      if (immersive) {
        const marker = vectorStrip(
          2.0,
          source.block.tensor?.shape.at(-1) ?? 16,
          0,
          undefined,
          edge.data?.residual ? "#f5b472" : "#7fd9bd",
        );
        marker.visible = false;
        state.content.add(marker);
        state.flows.push({
          source: edge.source,
          target: edge.target,
          curve,
          marker,
        });
      }
    }
    if (previousScene.current !== sceneKey) {
      if (!immersive) state.fit();
      else if (!previousScene.current) {
        const id = current.current.activeId ?? current.current.selectedId;
        if (id) state.focus(id);
        else state.fit();
      }
      previousScene.current = sceneKey;
    }
    state.draw();
  }, [
    blocks,
    graph.edges,
    spacing,
    sceneKey,
    ready,
    tokenCount,
    props.traceSamples,
    props.headIndex,
  ]);
  useEffect(() => {
    const state = runtime.current;
    if (!state) return;
    for (const [id, object] of state.objects)
      for (const [index, band] of (object.tokenBands ?? []).entries())
        (band.material as THREE.MeshBasicMaterial).opacity =
          id === activeId
            ? index === Math.max(0, activeToken) % 16
              ? 0.75
              : 0.16
            : 0.08;
    state.draw();
  }, [activeToken, activeId, ready, blocks]);

  useEffect(() => {
    const state = runtime.current;
    if (!state) return;
    for (const [id, object] of state.objects) {
      const active = id === selected;
      object.visual?.select(active);
      object.visual?.animate(
        current.current.animationProgress ?? 0,
        id === activeId && !object.block.node.data.summary,
        current.current.activeToken ?? 0,
        current.current.reducedMotion,
      );
    }
    if (activeId && followPlayback) state.focus(activeId, true);
    state.draw();
  }, [
    selected,
    blocks,
    spacing,
    activeId,
    followPlayback,
    ready,
    symbolic,
    showShapes,
  ]);
  useEffect(() => {
    const state = runtime.current;
    if (!state || animationProgress === undefined) return;
    const beat = demoBeat(animationProgress);
    if (!current.current.playing) state.controls.update();
    for (const flow of state.flows) {
      flow.marker.visible =
        (beat.output ? flow.source === activeId : flow.target === activeId) &&
        (beat.output
          ? animationProgress < 0.92
          : animationProgress >= 0.15 && animationProgress < 0.33) &&
        !current.current.reducedMotion;
      if (flow.marker.visible)
        flow.marker.position.copy(flow.curve.getPoint(beat.transfer));
    }
    for (const [id, object] of state.objects) {
      object.visual?.animate(
        animationProgress,
        id === activeId && !object.block.node.data.summary,
        current.current.activeToken ?? 0,
        current.current.reducedMotion,
      );
    }
    state.draw();
  }, [animationProgress, activeId, ready]);
  useEffect(() => {
    if (!props.playing) runtime.current?.stopMotion();
  }, [props.playing]);

  useEffect(() => {
    if (
      selectedId &&
      previousSelection.current !== selectedId &&
      (!immersive || !activeId)
    )
      runtime.current?.focus(selectedId);
    previousSelection.current = selectedId;
  }, [selectedId]);

  const zoom = (factor: number) => {
    const state = runtime.current;
    if (!state) return;
    const offset = state.camera.position
      .clone()
      .sub(state.controls.target)
      .multiplyScalar(factor);
    offset.clampLength(
      state.controls.minDistance,
      state.controls.maxDistance,
    );
    state.camera.position.copy(state.controls.target).add(offset);
    state.controls.update();
    state.draw();
  };
  return (
    <div
      className={`tensor-space reference-space ${immersive ? "immersive-space" : ""}`}
      data-testid="tensor-space"
      data-ready={ready && !failed}
      data-world-layers={worldLayers?.length}
      data-active-token={activeToken}
      data-renderer="upstream-primitives"
      data-model-hidden={props.model?.hidden}
      data-model-heads={props.model?.heads}
    >
      <div
        ref={host}
        className="tensor-space-host"
        style={
          immersive ? { filter: `brightness(${brightness}%)` } : undefined
        }
        data-testid="tensor-space-renderer"
        onKeyDown={(e) => {
          if (
            e.target instanceof HTMLButtonElement ||
            e.target instanceof HTMLSelectElement ||
            e.target instanceof HTMLInputElement
          )
            return;
          if (e.key === "+" || e.key === "=") {
            e.preventDefault();
            zoom(0.8);
          }
          if (e.key === "-") {
            e.preventDefault();
            zoom(1.25);
          }
          if (e.key.toLowerCase() === "f") {
            e.preventDefault();
            if (selected) runtime.current?.focus(selected);
          }
        }}
        tabIndex={0}
        aria-label="三维张量画布，拖动旋转，滚轮缩放；加减号缩放，F 聚焦选中模块"
      >
        {!failed && (
          <div className="spatial-labels">
            {blocks.map((block) => (
              <button
                key={block.id}
                ref={(element) => {
                  if (element) labels.current.set(block.id, element);
                  else labels.current.delete(block.id);
                }}
                className={`spatial-label ${block.id === selected ? "selected" : ""} ${block.node.data.inactive ? "inactive" : ""}`}
                data-spatial-id={block.id}
                data-visual-kind={
                  referenceVisual(block.node, props.model, props.scenario)
                    .kind
                }
                data-active={block.id === activeId || undefined}
                aria-label={`选择 ${block.node.data.title}`}
                aria-pressed={block.id === selected}
                onClick={() => onSelect(block.id)}
                onDoubleClick={() => onDrill(block.id)}
              >
                <i style={{ background: block.color }} />
                <strong>
                  {block.node.data.residual ? "⊕ " : ""}
                  {block.node.data.title}
                </strong>
                {block.node.data.repeat && (
                  <b className="spatial-repeat-badge">
                    × {block.node.data.repeat}
                  </b>
                )}
                {showShapes && (
                  <code>{tensorLabel(block.tensor, symbolic)}</code>
                )}
              </button>
            ))}
          </div>
        )}
      </div>
      {failed ? (
        <div className="spatial-fallback" role="status">
          <Box size={30} />
          <strong>当前浏览器无法显示 3D 场景</strong>
          <p>切换到 2D 张量视图，继续查看同一模型和维度。</p>
          <button className="primary-button" onClick={onFallback}>
            切换到 2D 张量
          </button>
        </div>
      ) : (
        <>
          <div className="spatial-toolbar">
            <label>
              定位模块
              <select
                aria-label="定位三维模块"
                value={
                  graph.nodes.some((n) => n.id === selected)
                    ? (selected ?? "")
                    : ""
                }
                onChange={(e) => {
                  onSelect(e.target.value);
                  runtime.current?.focus(e.target.value);
                }}
              >
                <option value="" disabled>
                  选择模块
                </option>
                {graph.nodes
                  .filter((node) => !node.data.frame)
                  .map((node) => (
                    <option key={node.id} value={node.id}>
                      {node.data.title}
                    </option>
                  ))}
              </select>
            </label>
            <label className="spatial-spacing">
              模块间距
              <input
                type="range"
                aria-label="三维模块间距"
                min="0.8"
                max="2"
                step="0.1"
                value={spacing}
                onChange={(e) => setSpacing(Number(e.target.value))}
              />
            </label>
          </div>
          <div className="spatial-controls">
            {worldLayers && (
              <button
                aria-label="全景查看完整模型"
                title="模型全景"
                onClick={() => runtime.current?.fit(true)}
              >
                <Box size={16} />
              </button>
            )}
            <button aria-label="缩小三维场景" onClick={() => zoom(1.25)}>
              <Minus size={17} />
            </button>
            <button aria-label="放大三维场景" onClick={() => zoom(0.8)}>
              <Plus size={17} />
            </button>
            <span />
            <button
              aria-label="适配三维全图"
              onClick={() => runtime.current?.fit()}
            >
              <Scan size={17} />
            </button>
            <button
              aria-label="聚焦三维模块"
              disabled={!selected}
              onClick={() => selected && runtime.current?.focus(selected)}
            >
              <Focus size={17} />
            </button>
            <button
              aria-label="重置三维视角"
              onClick={() => runtime.current?.fit()}
            >
              <RotateCcw size={16} />
            </button>
          </div>
          {graph.sections && (
            <div className="spatial-repeat-list">
              <select
                aria-label="定位重复结构"
                value={
                  graph.sections.find((section) =>
                    section.nodes.includes(selected ?? ""),
                  )?.anchor ??
                  graph.sections[0]?.anchor ??
                  ""
                }
                onChange={(e) => {
                  const section = graph.sections!.find(
                    (s) => s.anchor === e.target.value,
                  );
                  if (section) {
                    onSelect(section.anchor);
                    runtime.current?.focus(section.anchor);
                  }
                }}
              >
                <option value="" disabled>
                  重复结构 · {graph.sections.length} 组
                </option>
                {graph.sections.map((s) => (
                  <option key={s.anchor} value={s.anchor}>
                    {s.title}
                  </option>
                ))}
              </select>
            </div>
          )}
          {selectedBlock?.tensor && showShapes && (
            <div className="spatial-axis-card" aria-label="三维张量轴映射">
              <span>
                当前张量 · {selectedBlock.tensor.name}
                {selectedBlock.node.data.outputs.length > 1
                  ? " · 首个输出"
                  : ""}
              </span>
              <div>
                <b>末轴</b>
                <span>{selectedBlock.volume.columnSymbol}</span>
                <code>{selectedBlock.volume.columns.toLocaleString()}</code>
              </div>
              <div>
                <b>行</b>
                <span>{selectedBlock.volume.rowSymbol}</span>
                <code>{selectedBlock.volume.rows.toLocaleString()}</code>
              </div>
              <div>
                <b>切片</b>
                <span>{selectedBlock.volume.sliceSymbol}</span>
                <code>
                  {selectedBlock.volume.slices.toLocaleString()} 切片
                </code>
              </div>
              <small>空间：横向算子 · 纵向层级 · 深度 token</small>
            </div>
          )}
          <span className="spatial-gesture-hint">
            拖动旋转 · 右键 / 双指平移 · 滚轮 / 捏合缩放
          </span>
        </>
      )}
    </div>
  );
}
