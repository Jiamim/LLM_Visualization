import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import type { ModuleNode } from "../lib/graph";
import type { ModelConfig, Scenario } from "../lib/model";
import type { TraceSample } from "../lib/demo-trace";
import { referenceVisual } from "../lib/reference-scene";
import {
  createReferenceObject,
  disposeReferenceGroup,
} from "../rendering/reference-primitives";

/** Inspector uses the same object factory as the scene, including imported samples. */
export default function ComponentPreview3D({
  node,
  model,
  scenario,
  tokenLabels,
  samples,
  head,
  recordedMode,
}: {
  node: ModuleNode;
  model: ModelConfig;
  scenario: Scenario;
  tokenLabels: string[];
  samples: TraceSample[];
  head: number;
  recordedMode?: boolean;
}) {
  const host = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = host.current!;
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true });
    } catch {
      return;
    }
    renderer.setClearColor("#050608");
    renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.25;
    el.append(renderer.domElement);
    const scene = new THREE.Scene();
    scene.add(new THREE.AmbientLight(0xffffff, 2));
    const light = new THREE.DirectionalLight(0xffffff, 3);
    light.position.set(3, 6, 8);
    scene.add(light);
    const visual = createReferenceObject(
      { ...referenceVisual(node, model, scenario), recordedMode },
      node.id,
      tokenLabels,
      samples,
      head,
    );
    scene.add(visual.group);
    visual.animate(0.9, false, 0, true);
    const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 1000);
    const control = new OrbitControls(camera, renderer.domElement);
    control.enablePan = false;
    const render = () => renderer.render(scene, camera);
    const resize = () => {
      const w = el.clientWidth,
        h = el.clientHeight;
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      const box = new THREE.Box3().setFromObject(visual.group);
      const size = box.getSize(new THREE.Vector3());
      const center = box.getCenter(new THREE.Vector3());
      control.target.copy(center);
      camera.position
        .copy(center)
        .add(
          new THREE.Vector3(0.45, 0.35, 1)
            .normalize()
            .multiplyScalar(
              Math.max(size.x / camera.aspect, size.y, size.z) * 2.2,
            ),
        );
      control.update();
      render();
    };
    control.addEventListener("change", render);
    const observer = new ResizeObserver(resize);
    observer.observe(el);
    resize();
    return () => {
      observer.disconnect();
      control.dispose();
      disposeReferenceGroup(visual.group);
      renderer.dispose();
      renderer.forceContextLoss();
      renderer.domElement.remove();
    };
  }, [
    node.id,
    node.data.layerIndex,
    model,
    scenario.sequence,
    scenario.past,
    head,
    samples,
    recordedMode,
  ]);
  return (
    <div
      className="component-preview-3d"
      ref={host}
      aria-label="当前组件三维预览，拖动旋转"
    />
  );
}
