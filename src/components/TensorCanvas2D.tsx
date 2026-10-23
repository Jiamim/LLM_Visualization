import { useEffect, useMemo, useRef, useState } from "react";
import {
  Background,
  BackgroundVariant,
  Panel,
  ReactFlow,
  ReactFlowProvider,
  useReactFlow,
  type Edge,
} from "@xyflow/react";
import { Focus, Minus, Plus, Scan } from "lucide-react";
import TensorNode from "./TensorNode";
import type { ModelGraph, ModuleNode } from "../lib/graph";
import { TENSOR_NODE_WIDTH, TENSOR_NODE_HEIGHT } from "../lib/tensor-layout";
import type { ModelConfig, Scenario } from "../lib/model";
import type { ReferenceVisual } from "../lib/reference-scene";
import type { TraceSample } from "../lib/demo-trace";

export interface TensorCanvasProps {
  model?: ModelConfig;
  scenario?: Scenario;
  tokenLabels?: string[];
  traceSamples?: TraceSample[];
  recordedMode?: boolean;
  headIndex?: number;
  graph: ModelGraph;
  selectedId: string | null;
  activeId: string | null;
  selectedEdgeId: string | null;
  symbolic: boolean;
  showShapes: boolean;
  playing: boolean;
  followPlayback: boolean;
  reducedMotion: boolean;
  sceneKey: string;
  onSelect: (id: string) => void;
  onSelectEdge: (id: string) => void;
  onDrill: (id: string) => void;
  onFallback?: () => void;
  animationProgress?: number;
  immersive?: boolean;
  worldLayers?: {
    index: number;
    label: string;
    color: string;
    parts?: {
      id: string;
      title: string;
      color: string;
      residual: boolean;
      visual?: ReferenceVisual;
    }[];
  }[];
  worldLayer?: number;
  onWorldLayer?: (index: number) => void;
  tokenCount?: number;
  activeToken?: number;
  brightness?: number;
  cameraGroup?: string;
  stepDuration?: number;
}
const nodeTypes = { module: TensorNode };
function Canvas(props: TensorCanvasProps) {
  const {
    graph,
    selectedId,
    activeId,
    selectedEdgeId,
    symbolic,
    showShapes,
    playing,
    followPlayback,
    reducedMotion,
    sceneKey,
    onSelect,
    onSelectEdge,
    onDrill,
  } = props;
  const { fitView, zoomIn, zoomOut, setCenter, getZoom } =
    useReactFlow<ModuleNode>();
  const [zoom, setZoom] = useState(100);
  const container = useRef<HTMLDivElement>(null);
  const resizeFocus = useRef<() => void>(() => {});
  const nodes = useMemo(
    () =>
      graph.nodes.map((node) => ({
        ...node,
        width: node.data.frame
          ? node.data.frame.width * 1.2
          : TENSOR_NODE_WIDTH,
        height: node.data.frame
          ? node.data.frame.height * 2
          : TENSOR_NODE_HEIGHT,
        style: node.data.frame
          ? {
              width: node.data.frame.width * 1.2,
              height: node.data.frame.height * 2,
            }
          : node.style,
        position: { x: node.position.x * 1.2, y: node.position.y * 2 },
        selected: node.id === (activeId ?? selectedId),
        data: {
          ...node.data,
          frame: node.data.frame
            ? {
                ...node.data.frame,
                width: node.data.frame.width * 1.2,
                height: node.data.frame.height * 2,
              }
            : undefined,
          symbolic,
          active: node.id === activeId,
          onDrill: () => onDrill(node.id),
        },
      })),
    [graph.nodes, selectedId, activeId, symbolic, onDrill],
  );
  const edges = useMemo(
    () =>
      graph.edges.map((edge) => ({
        ...edge,
        label: showShapes ? edge.label : undefined,
        selected: edge.id === selectedEdgeId,
        animated: !reducedMotion && playing && edge.source === activeId,
      })),
    [
      graph.edges,
      showShapes,
      selectedEdgeId,
      reducedMotion,
      playing,
      activeId,
    ],
  );
  const focus = (id: string | null) => {
    const node = nodes.find((n) => n.id === id);
    if (node)
      void setCenter(
        node.position.x + TENSOR_NODE_WIDTH / 2,
        node.position.y + TENSOR_NODE_HEIGHT / 2,
        { zoom: Math.max(getZoom(), 0.9), duration: reducedMotion ? 0 : 260 },
      );
  };
  resizeFocus.current = () => focus(activeId ?? selectedId);
  useEffect(() => {
    let previous = "",
      timer: number | undefined;
    const observer = new ResizeObserver((entries) => {
      const box = entries[0].contentRect,
        size = `${box.width}-${box.height}`;
      if (previous && previous !== size) {
        clearTimeout(timer);
        timer = window.setTimeout(() => resizeFocus.current(), 80);
      }
      previous = size;
    });
    if (container.current) observer.observe(container.current);
    return () => {
      observer.disconnect();
      clearTimeout(timer);
    };
  }, []);
  useEffect(() => {
    const timer = window.setTimeout(() => {
      if (selectedId) focus(selectedId);
      else void fitView({ padding: 0.22, maxZoom: 0.95, duration: 0 });
    }, 50);
    return () => clearTimeout(timer);
  }, [sceneKey, fitView]);
  useEffect(() => {
    if (activeId && followPlayback) focus(activeId);
  }, [activeId, followPlayback]);
  useEffect(() => {
    if (selectedId) focus(selectedId);
  }, [selectedId]);
  return (
    <div ref={container} className="tensor-plane-canvas">
      <ReactFlow<ModuleNode, Edge>
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.22, maxZoom: 0.95 }}
        onNodeClick={(_, node) => onSelect(node.id)}
        onNodeDoubleClick={(_, node) => onDrill(node.id)}
        onEdgeClick={(_, edge) => onSelectEdge(edge.id)}
        onMove={(_, viewport) => setZoom(Math.round(viewport.zoom * 100))}
        nodesDraggable={false}
        nodesConnectable={false}
        deleteKeyCode={null}
        zoomOnDoubleClick={false}
        minZoom={0.06}
        maxZoom={2}
        onKeyDown={(e) => {
          const id = (e.target as HTMLElement).dataset.id;
          if (id && (e.key === "Enter" || e.key === " ")) {
            e.preventDefault();
            onSelect(id);
          }
        }}
      >
        <Background
          variant={BackgroundVariant.Dots}
          color="#d9dfeb"
          gap={20}
        />
        <Panel
          position="bottom-left"
          className="viewport-controls tensor-viewport-controls"
        >
          <div className="zoom-controls">
            <button
              className="icon-button"
              aria-label="缩小张量图"
              onClick={() => void zoomOut()}
            >
              <Minus size={17} />
            </button>
            <span className="zoom-value">{zoom}%</span>
            <button
              className="icon-button"
              aria-label="放大张量图"
              onClick={() => void zoomIn()}
            >
              <Plus size={17} />
            </button>
            <button
              className="icon-button"
              aria-label="适配张量全图"
              onClick={() =>
                void fitView({
                  padding: 0.22,
                  maxZoom: 0.95,
                  duration: reducedMotion ? 0 : 260,
                })
              }
            >
              <Scan size={17} />
            </button>
            <button
              className="icon-button"
              aria-label="聚焦张量模块"
              onClick={() => focus(activeId ?? selectedId)}
            >
              <Focus size={17} />
            </button>
          </div>
        </Panel>
      </ReactFlow>
    </div>
  );
}
export default function TensorCanvas2D(props: TensorCanvasProps) {
  return (
    <ReactFlowProvider>
      <Canvas {...props} />
    </ReactFlowProvider>
  );
}
