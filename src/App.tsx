import {
  lazy,
  Suspense,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  Background,
  BackgroundVariant,
  MiniMap,
  Panel,
  ReactFlow,
  useReactFlow,
  type Edge,
} from "@xyflow/react";
import {
  ArrowDown,
  ArrowLeft,
  ArrowUpRight,
  Boxes,
  Braces,
  Check,
  ChevronDown,
  ChevronRight,
  CircleHelp,
  Code2,
  Download,
  Database,
  FileJson,
  Focus,
  PanelLeft,
  PanelRight,
  Scan,
  Minus,
  BookOpen,
  Box,
  Grid2X2,
  GitBranch,
  Layers3,
  Network,
  Pause,
  Play,
  Plus,
  Settings2,
  SkipForward,
  Sparkles,
  Upload,
  X,
} from "lucide-react";
import ModuleNode from "./components/ModuleNode";
import CanvasTools from "./components/CanvasTools";
import Inspector from "./components/Inspector";
import SidePanel, { useMediaQuery } from "./components/SidePanel";
import {
  NODE_WIDTH,
  NODE_HEIGHT,
  categories,
  modelStages,
  type Stage,
} from "./lib/presentation";
import ImportDialog from "./components/ImportDialog";
const GalleryDialog = lazy(() => import("./components/GalleryDialog"));
const InferenceDemo = lazy(() => import("./components/InferenceDemo"));
const CacheDialog = lazy(() => import("./components/CacheDialog"));
import { cacheReport } from "./lib/cache";
const TensorCanvas2D = lazy(() => import("./components/TensorCanvas2D"));
const TensorCanvas3D = lazy(() => import("./components/TensorCanvas3D"));
import { displayModeFromURL, type DisplayMode } from "./lib/tensor-layout";
import {
  buildGraph,
  type ModuleNode as ModuleNodeType,
  type View,
} from "./lib/graph";
import {
  compactNumber,
  formatBytes,
  formatNumber,
  memoryEstimates,
  normalizeConfig,
  parameterCounts,
  validateScenario,
  type DType,
  type ModelConfig,
  type Scenario,
  type Tensor,
} from "./lib/model";
import {
  deepseekMemory,
  deepseekParameters,
  layerProfile,
  validateDeepseekScenario,
} from "./lib/deepseek";
import {
  gemmaLayer,
  gemmaParameters,
  gemmaMemory,
  gemmaVisionShapes,
  gemmaAudioShapes,
  validateGemmaScenario,
} from "./lib/gemma";
import { defaultPreset } from "./data/presets";
import { useFullscreen } from "./lib/use-fullscreen";
import FullscreenButton from "./components/FullscreenButton";
import {
  initialSelection,
  modelLibraryCount,
  type ModelSelection,
} from "./data/model-library-info";

const nodeTypes = { module: ModuleNode };
const defaultViewTitles: Record<View, string> = {
  architecture: "整体架构",
  auxiliary: "架构补充",
  ple: "Per-Layer Embeddings",
  audio: "音频编码器",
  encoder: "Causal Encoder",
  decoder: "CED Decoder",
  cache: "共享缓存",
  vision: "视觉编码器",
  engram: "Engram",
  dspark: "DSpark",
  overview: "模型总览",
  layer: "Transformer Block",
  attention: "Self Attention",
  mlp: "Feed Forward",
};
const initialModel = normalizeConfig(defaultPreset.raw);
const nodeColors = Object.fromEntries(
  Object.entries(categories).map(([id, value]) => [id, value.color]),
);

function downloadJSON(value: unknown, filename: string) {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(value, null, 2)], { type: "application/json" }),
  );
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 500);
}

function InputField({
  label,
  symbol,
  value,
  min,
  onChange,
}: {
  label: string;
  symbol: string;
  value: number;
  min: number;
  onChange: (value: number) => void;
}) {
  const [draft, setDraft] = useState(String(value));
  useEffect(() => setDraft(String(value)), [value]);
  const valid =
    /^\d+$/.test(draft) &&
    Number.isSafeInteger(Number(draft)) &&
    Number(draft) >= min;
  return (
    <label className="scenario-field">
      <span>
        {label}
        <code>{symbol}</code>
      </span>
      <input
        type="number"
        min={min}
        step="1"
        value={draft}
        aria-invalid={!valid}
        onChange={(event) => {
          const next = event.target.value;
          setDraft(next);
          if (
            /^\d+$/.test(next) &&
            Number.isSafeInteger(Number(next)) &&
            Number(next) >= min
          )
            onChange(Number(next));
        }}
        onBlur={() => setDraft(String(value))}
      />
    </label>
  );
}

export default function App() {
  const fullscreen = useFullscreen<HTMLDivElement>();
  const [model, setModel] = useState(initialModel);
  const [modelName, setModelName] = useState("DeepSeek V4.1 Flash");
  const [source, setSource] = useState(defaultPreset.repo);
  const [activeModelSelection, setActiveModelSelection] =
    useState<ModelSelection | null>(initialSelection);
  const [scenario, setScenario] = useState<Scenario>({
    batch: 1,
    sequence: 128,
    past: 0,
    dtype: initialModel.dtype,
    cedExecution: "reference",
  });
  const [scenarioError, setScenarioError] = useState("");
  const [view, setView] = useState<View>("overview");
  const [displayMode, setDisplayMode] =
    useState<DisplayMode>(displayModeFromURL);
  const [layerIndex, setLayerIndex] = useState(0);
  const [selectedId, setSelectedId] = useState<string | null>("encoder");
  const [selectedEdgeId, setSelectedEdgeId] = useState<string | null>(null);
  const [symbolic, setSymbolic] = useState(false);
  const [showShapes, setShowShapes] = useState(true);
  const [importOpen, setImportOpen] = useState(false);
  const [galleryOpen, setGalleryOpen] = useState(false);
  const [cacheOpen, setCacheOpen] = useState(false);
  const [demoOpen, setDemoOpen] = useState(
    () => new URLSearchParams(window.location.search).get("demo") === "1",
  );
  const [playing, setPlaying] = useState(false);
  const [step, setStep] = useState(-1);
  const [zoom, setZoom] = useState(100);
  const [showHelp, setShowHelp] = useState(false);
  const smallScreen = useMediaQuery("(max-width: 760px)");
  const compactScreen = useMediaQuery("(max-width: 1199px)");
  const reducedMotion = useMediaQuery("(prefers-reduced-motion: reduce)");
  const motion = reducedMotion ? 0 : 280;
  const [sidebarOpen, setSidebarOpen] = useState(
    () => window.innerWidth > 760,
  );
  const [inspectorOpen, setInspectorOpen] = useState(
    () => window.innerWidth >= 1200,
  );
  const [focusMode, setFocusMode] = useState(false);
  const [viewMode, setViewMode] = useState<"read" | "fit">("read");
  const [stageIndex, setStageIndex] = useState(0);
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [followPlayback, setFollowPlayback] = useState(true);
  const [notice, setNotice] = useState("");
  const fullscreenPanels = useRef<{
    sidebar: boolean;
    inspector: boolean;
  } | null>(null);
  useEffect(() => {
    if (fullscreen.active && !fullscreenPanels.current) {
      fullscreenPanels.current = {
        sidebar: sidebarOpen,
        inspector: inspectorOpen,
      };
      setSidebarOpen(false);
      setInspectorOpen(false);
    } else if (!fullscreen.active && fullscreenPanels.current) {
      setSidebarOpen(fullscreenPanels.current.sidebar);
      setInspectorOpen(fullscreenPanels.current.inspector);
      fullscreenPanels.current = null;
    }
  }, [fullscreen.active, sidebarOpen, inspectorOpen]);
  useEffect(() => setSidebarOpen(!smallScreen), [smallScreen]);
  useEffect(() => setInspectorOpen(!compactScreen), [compactScreen]);
  useEffect(() => {
    if (!notice) return;
    const timer = window.setTimeout(() => setNotice(""), 2600);
    return () => clearTimeout(timer);
  }, [notice]);
  const rawDialog = useRef<HTMLDialogElement>(null);
  const canvasRef = useRef<HTMLDivElement>(null);
  const focusTarget = useRef<string | null>(null);
  const manualViewport = useRef(false);
  const refitTimer = useRef<number | undefined>(undefined);
  const { fitView, setCenter, getZoom, zoomIn, zoomOut, zoomTo } =
    useReactFlow<ModuleNodeType>();
  const graph = useMemo(
    () => buildGraph(model, scenario, view, layerIndex, symbolic),
    [model, scenario, view, layerIndex, symbolic],
  );
  const viewStages = useMemo(
    () =>
      (graph.sections ?? modelStages(model, view))
        .map((stage) => {
          const visibleIds = stage.nodes.filter((id) =>
            graph.nodes.some((n) => n.id === id),
          );
          return {
            ...stage,
            nodes: visibleIds,
            anchor: visibleIds.includes(stage.anchor)
              ? stage.anchor
              : visibleIds[0],
          };
        })
        .filter((stage) => stage.nodes.length > 0),
    [model, view, graph],
  );
  const viewTitles = model.deepseek
    ? {
        ...defaultViewTitles,
        layer: "CED Transformer Layer",
        attention: "CSA2 Attention",
        mlp: "Mixture of Experts",
      }
    : model.catalog
      ? {
          ...defaultViewTitles,
          decoder: "语言主干层",
          attention: "混合算子",
          mlp: "前馈 / 专家",
          cache: "状态与缓存",
        }
      : model.gemma
        ? {
            ...defaultViewTitles,
            decoder: "Hybrid Decoder",
            attention: "Gemma Hybrid Attention",
            mlp: "Gemma Gated MLP",
          }
        : defaultViewTitles;
  const dsParams = useMemo(
    () => (model.deepseek ? deepseekParameters(model) : null),
    [model],
  );
  const dsMemory = useMemo(
    () => (model.deepseek ? deepseekMemory(model, scenario) : null),
    [model, scenario],
  );
  const gmParams = useMemo(
    () => (model.gemma ? gemmaParameters(model) : null),
    [model],
  );
  const gmMemory = useMemo(
    () => (model.gemma ? gemmaMemory(model, scenario) : null),
    [model, scenario],
  );
  const counts = useMemo(() => parameterCounts(model), [model]);
  const memory = useMemo(
    () => memoryEstimates(model, scenario),
    [model, scenario],
  );
  const cache = useMemo(
    () => cacheReport(model, scenario),
    [model, scenario],
  );
  const playbackNodes = useMemo(
    () => graph.nodes.filter((node) => !node.data.inactive),
    [graph],
  );
  const activeId =
    step >= 0
      ? playbackNodes[Math.min(step, playbackNodes.length - 1)]?.id
      : null;

  const navigate = useCallback(
    (next: View, targetLayer?: number) => {
      manualViewport.current = false;
      focusTarget.current = null;
      if (targetLayer !== undefined) setLayerIndex(targetLayer);
      else if (
        model.deepseek &&
        next === "engram" &&
        !model.deepseek.engramLayers.includes(layerIndex)
      )
        setLayerIndex(model.deepseek.engramLayers[0] ?? 0);
      setView(next);
      const plannedStage = modelStages(model, next)[0];
      const actualGraph = buildGraph(
        model,
        scenario,
        next,
        targetLayer ?? layerIndex,
      );
      const validAnchor = plannedStage?.nodes.find((id) =>
        actualGraph.nodes.some((node) => node.id === id),
      );
      setSelectedId(
        next === "overview"
          ? model.deepseek
            ? "encoder"
            : model.gemma
              ? "ple"
              : "decoder"
          : (actualGraph.sections?.[0]?.anchor ??
              validAnchor ??
              actualGraph.nodes[0]?.id ??
              null),
      );
      setSelectedEdgeId(null);
      setPlaying(false);
      setStep(-1);
      setStageIndex(0);
      setViewMode(
        next === "overview" &&
          !model.deepseek &&
          !model.gemma &&
          !model.catalog
          ? "fit"
          : "read",
      );
      if (window.innerWidth <= 760) setSidebarOpen(false);
      if (window.innerWidth < 1200) setInspectorOpen(false);
    },
    [model, layerIndex, scenario],
  );
  const replaceModel = (
    config: ModelConfig,
    name: string,
    sourceName: string,
    selection: ModelSelection | null = null,
  ) => {
    const nextScenario: Scenario = {
      ...scenario,
      dtype: config.dtype,
      sequence: config.deepseek && scenario.past > 0 ? 1 : scenario.sequence,
      cedExecution: "reference",
      image: undefined,
      audio: undefined,
    };
    validateScenario(nextScenario);
    const nextGraph = buildGraph(config, nextScenario, "overview", 0);
    parameterCounts(config);
    cacheReport(config, nextScenario);
    setModel(config);
    setModelName(name);
    setSource(sourceName);
    setScenario(nextScenario);
    setScenarioError("");
    setLayerIndex(0);
    setActiveModelSelection(selection);
    manualViewport.current = false;
    focusTarget.current = null;
    setView("overview");
    setPlaying(false);
    setStep(-1);
    setStageIndex(0);
    setSelectedEdgeId(null);
    if (smallScreen) setSidebarOpen(false);
    if (compactScreen) setInspectorOpen(false);
    setViewMode(
      config.deepseek || config.gemma || config.catalog ? "read" : "fit",
    );
    setSelectedId(
      nextGraph.nodes.find(
        (n) =>
          n.id ===
          (config.deepseek ? "encoder" : config.gemma ? "ple" : "decoder"),
      )?.id ??
        nextGraph.nodes[0]?.id ??
        null,
    );
  };
  const updateScenario = (update: Partial<Scenario>) => {
    const next = { ...scenario, ...update };
    try {
      validateScenario(next);
      if (model.deepseek) validateDeepseekScenario(model, next);
      if (model.gemma) validateGemmaScenario(model, next);
      setScenario(next);
      setScenarioError("");
      setPlaying(false);
      setStep(-1);
    } catch (err) {
      setScenarioError(err instanceof Error ? err.message : "输入无效。");
    }
  };
  const resetView = useCallback(() => {
    if (demoOpen || displayMode !== "graph") return;
    if (manualViewport.current) return;
    const focused = graph.nodes.find(
      (node) => node.id === focusTarget.current,
    );
    if (focused) {
      void setCenter(
        focused.position.x + NODE_WIDTH / 2,
        focused.position.y + NODE_HEIGHT / 2,
        { zoom: smallScreen ? 0.95 : 1.05, duration: motion },
      );
      return;
    }
    if (viewMode === "fit") {
      void fitView({
        padding: { top: "54px", bottom: "58px", left: "30px", right: "30px" },
        duration: motion,
        minZoom: 0.15,
        maxZoom: 1,
      });
    } else {
      const stage = viewStages[stageIndex];
      if (stage && !smallScreen) {
        void fitView({
          nodes: stage.nodes
            .filter((id) => graph.nodes.some((n) => n.id === id))
            .map((id) => ({ id })),
          padding: {
            top: "62px",
            bottom: "78px",
            left: "24px",
            right: "24px",
          },
          minZoom: view === "architecture" ? 0.35 : 0.65,
          maxZoom: 1,
          duration: motion,
        });
        return;
      }
      const anchor =
        graph.nodes.find(
          (node) => node.id === (viewStages[stageIndex]?.anchor ?? "decoder"),
        ) ?? graph.nodes[0];
      void setCenter(
        anchor.position.x + NODE_WIDTH / 2,
        anchor.position.y + NODE_HEIGHT / 2,
        { zoom: smallScreen ? 0.95 : 1, duration: motion },
      );
    }
  }, [
    demoOpen,
    displayMode,
    viewMode,
    view,
    stageIndex,
    model,
    fitView,
    setCenter,
    motion,
    smallScreen,
    layerIndex,
  ]);
  useEffect(() => {
    let previousSize = "";
    const refit = (entries?: ResizeObserverEntry[]) => {
      if (entries?.[0]) {
        const { width, height } = entries[0].contentRect;
        const size = `${Math.round(width)}-${Math.round(height)}`;
        if (previousSize && size !== previousSize)
          manualViewport.current = false;
        previousSize = size;
      }
      clearTimeout(refitTimer.current);
      refitTimer.current = window.setTimeout(resetView, 80);
    };
    const observer = new ResizeObserver(refit);
    if (canvasRef.current) observer.observe(canvasRef.current);
    refit();
    return () => {
      clearTimeout(refitTimer.current);
      observer.disconnect();
    };
  }, [resetView]);
  useEffect(() => {
    if (displayMode !== "graph" || step < 0 || !followPlayback) return;
    const node = playbackNodes[Math.min(step, playbackNodes.length - 1)];
    if (node)
      void setCenter(
        node.position.x + NODE_WIDTH / 2,
        node.position.y + NODE_HEIGHT / 2,
        { zoom: Math.max(getZoom(), 0.9), duration: motion },
      );
  }, [step, followPlayback, setCenter, getZoom, motion, displayMode]);
  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(
      () => setStep((previous) => previous + 1),
      950,
    );
    return () => clearInterval(timer);
  }, [playing]);
  useEffect(() => {
    if (step >= playbackNodes.length) {
      setPlaying(false);
      setStep(playbackNodes.length ? playbackNodes.length - 1 : -1);
    }
  }, [step, playbackNodes.length]);

  const nodes = useMemo(
    () =>
      graph.nodes.map((node) => ({
        ...node,
        selected: node.id === (activeId ?? selectedId),
        data: {
          ...node.data,
          active: node.id === activeId,
          onDrill: node.data.drill
            ? () => navigate(node.data.drill!, node.data.layerIndex)
            : undefined,
        },
      })),
    [graph.nodes, selectedId, activeId, navigate],
  );
  const edges = useMemo(
    () =>
      graph.edges.map((edge) => ({
        ...edge,
        label: showShapes ? edge.label : undefined,
        animated: !reducedMotion && playing && edge.source === activeId,
        selected: edge.id === selectedEdgeId,
        style: {
          ...edge.style,
          ...(hoveredId &&
          (edge.source === hoveredId || edge.target === hoveredId)
            ? { stroke: "#7856c6", strokeWidth: 2.5 }
            : {}),
          ...(edge.id === selectedEdgeId ||
          (playing && edge.source === activeId)
            ? { stroke: "#7c65ce", strokeWidth: 2.5 }
            : {}),
        },
      })),
    [
      graph.edges,
      showShapes,
      activeId,
      playing,
      selectedEdgeId,
      hoveredId,
      reducedMotion,
    ],
  );
  const inspected =
    graph.nodes.find((node) => node.id === (activeId ?? selectedId))?.data ??
    null;
  const selectedEdge = graph.edges.find((edge) => edge.id === selectedEdgeId)
    ?.data?.tensor as Tensor | undefined;
  const focusNode = (node: ModuleNodeType) => {
    if (displayMode !== "graph") return;
    manualViewport.current = false;
    focusTarget.current = node.id;
    setViewMode("read");
    void setCenter(
      node.position.x + NODE_WIDTH / 2,
      node.position.y + NODE_HEIGHT / 2,
      {
        zoom: Math.max(getZoom(), smallScreen ? 0.95 : 1.05),
        duration: motion,
      },
    );
  };
  const selectNode = (_: React.MouseEvent, node: ModuleNodeType) => {
    setPlaying(false);
    setStep(-1);
    setSelectedId(node.id);
    setSelectedEdgeId(null);
    setInspectorOpen(true);
    focusNode(node);
  };
  const focusSelected = () => {
    const node = nodes.find((n) => n.id === (activeId ?? selectedId));
    if (node) focusNode(node);
    else void fitView({ padding: 0.2, duration: motion });
  };
  const selectEdge = (_: React.MouseEvent, edge: Edge) => {
    setPlaying(false);
    setStep(-1);
    setSelectedId(null);
    setSelectedEdgeId(edge.id);
    setInspectorOpen(true);
  };
  const selectStage = (stage: Stage, index: number) => {
    manualViewport.current = false;
    focusTarget.current = null;
    setStageIndex(index);
    setViewMode("read");
    setPlaying(false);
    setStep(-1);
    setSelectedId(stage.anchor);
    setSelectedEdgeId(null);
    if (displayMode !== "graph") return;
    if (smallScreen) {
      const node = graph.nodes.find((n) => n.id === stage.anchor);
      if (node) focusNode(node);
    } else {
      void fitView({
        nodes: stage.nodes
          .filter((id) => graph.nodes.some((n) => n.id === id))
          .map((id) => ({ id })),
        padding: { top: "62px", bottom: "78px", left: "24px", right: "24px" },
        minZoom: view === "architecture" ? 0.35 : 0.65,
        maxZoom: 1,
        duration: motion,
      });
    }
  };
  const isBeyondContext =
    model.maxPosition !== null &&
    scenario.sequence + scenario.past > model.maxPosition;
  const selectTensorNode = useCallback((id: string) => {
    setSelectedId(id);
    setSelectedEdgeId(null);
    setPlaying(false);
    setStep(-1);
    setInspectorOpen(true);
  }, []);
  const selectTensorEdge = useCallback((id: string) => {
    setSelectedId(null);
    setSelectedEdgeId(id);
    setPlaying(false);
    setStep(-1);
    setInspectorOpen(true);
  }, []);
  const drillTensorNode = useCallback(
    (id: string) => {
      const node = graph.nodes.find((n) => n.id === id);
      if (node?.data.drill) navigate(node.data.drill, node.data.layerIndex);
    },
    [graph.nodes, navigate],
  );
  const changeDisplayMode = (next: DisplayMode) => {
    manualViewport.current = false;
    setDisplayMode(next);
    const url = new URL(window.location.href);
    if (next === "graph") url.searchParams.delete("view");
    else url.searchParams.set("view", next);
    window.history.replaceState(null, "", url);
  };
  const tensorCanvasProps = {
    graph,
    selectedId,
    activeId,
    selectedEdgeId,
    symbolic,
    showShapes,
    playing,
    followPlayback,
    reducedMotion,
    sceneKey: `${source}-${view}-${layerIndex}`,
    onSelect: selectTensorNode,
    onSelectEdge: selectTensorEdge,
    onDrill: drillTensorNode,
  };
  const exportGraph = () => {
    downloadJSON(
      {
        version: 1,
        source,
        model: model.raw,
        scenario,
        view,
        displayMode,
        layerIndex,
        gallery: model.gallery,
        inference: model.catalog
          ? "Language backbone structural visualization. Component estimates may be partial; specialized operations and source substitutions are recorded in model warnings."
          : model.gemma
            ? "Gemma 4 E-series static graph. Per-layer embeddings, typed KV sharing, DynamicCache sliding retention W-1. All S logits; optional image/audio encoder branches."
            : model.deepseek
              ? "DeepSeek V4.1 CED static logical graph. Unique global KV owners; local SWA; optional bounded replay approximation. Last-position float32 logits. No runtime routing simulation."
              : "Static logical shapes; not a runtime trace. Full uncropped KV cache. All sequence logits. Inference mode.",
        parameters: counts,
        memoryEstimates: memory,
        cache,
        nodes: graph.nodes,
        edges: graph.edges,
        architectureSections: graph.sections,
      },
      `${model.family}-${view}-graph.json`,
    );
    setNotice("计算图已导出");
  };

  if (demoOpen)
    return (
      <div
        ref={fullscreen.ref}
        className={`demo-shell ${fullscreen.active ? "view-fullscreen" : ""}`}
        data-fullscreen={fullscreen.mode}
      >
        <Suspense
          fallback={<div className="demo-loading">正在准备示例推理…</div>}
        >
          <InferenceDemo
            key={modelName + source}
            model={model}
            name={modelName}
            dtype={scenario.dtype}
            fullscreen={fullscreen}
            onChooseModel={() => setGalleryOpen(true)}
            onClose={() => {
              setDemoOpen(false);
              const url = new URL(location.href);
              ["demo", "token", "generation", "kvCache"].forEach((key) =>
                url.searchParams.delete(key),
              );
              history.replaceState(null, "", url);
            }}
          />
          {galleryOpen && (
            <GalleryDialog
              open={galleryOpen}
              onClose={() => setGalleryOpen(false)}
              onLoad={replaceModel}
              selection={activeModelSelection}
              scenario={scenario}
              onImport={() => {
                setGalleryOpen(false);
                setDemoOpen(false);
                setImportOpen(true);
              }}
            />
          )}
        </Suspense>
      </div>
    );

  return (
    <div
      ref={fullscreen.ref}
      data-fullscreen={fullscreen.mode}
      className={`app-shell ${focusMode ? "focus-mode" : ""} ${fullscreen.active ? "view-fullscreen" : ""}`}
      onKeyDown={(event) => {
        if (event.key === "Escape" && focusMode) setFocusMode(false);
        const target = event.target as HTMLElement;
        const nodeId = target.dataset.id;
        if (
          (event.key === "Enter" || event.key === " ") &&
          target.classList.contains("react-flow__node") &&
          displayMode === "graph" &&
          nodeId
        ) {
          const node = nodes.find((n) => n.id === nodeId);
          if (node) {
            event.preventDefault();
            setSelectedId(node.id);
            setSelectedEdgeId(null);
            setStep(-1);
            setPlaying(false);
            setInspectorOpen(true);
            focusNode(node);
          }
        }
      }}
    >
      <header className="app-header">
        <a
          className="brand"
          href="#"
          onClick={(event) => {
            event.preventDefault();
            navigate("overview");
          }}
        >
          <span className="brand-mark">
            <Boxes size={25} strokeWidth={1.5} />
          </span>
          <span>
            Tensor<span className="brand-light">Atlas</span>
          </span>
          <span className="beta-badge">LAB</span>
        </a>
        <div className="header-center">
          <span className="header-divider" />
          模型架构工作台
          <span className="workspace-badge">
            <span className="status-dot" />
            本地计算
          </span>
        </div>
        <div className="header-actions">
          <button
            className="secondary-button original-entry-button"
            onClick={() => {
              setPlaying(false);
              setDemoOpen(true);
            }}
            title="使用原站同源图元演示当前模型"
            aria-label="模型 3D 推理"
          >
            <Box size={15} />
            <span>模型 3D 推理</span>
          </button>
          <button
            className="secondary-button demo-entry-button"
            onClick={() => {
              setPlaying(false);
              setDemoOpen(true);
            }}
          >
            <Play size={15} />
            示例推理
          </button>
          <button
            className="secondary-button library-button"
            onClick={() => setGalleryOpen(true)}
          >
            <BookOpen size={15} />
            模型库 <span>{modelLibraryCount}</span>
          </button>
          <button
            className="text-button help-toggle"
            onClick={() => setShowHelp(!showHelp)}
          >
            <CircleHelp size={15} /> 使用指南
          </button>
          <button
            className="primary-button"
            onClick={() => setImportOpen(true)}
          >
            <Plus size={16} /> 导入模型
          </button>
        </div>
      </header>
      {showHelp && (
        <div className="help-banner">
          <CircleHelp size={17} />
          <p>
            <strong>从总览到每一次矩阵乘法。</strong>{" "}
            点选模块自动聚焦；点击模块右上角箭头或双击展开。拖动画布、滚轮缩放；点击连线查看张量。左侧调整输入，顶部播放数据流。
          </p>
          <button
            className="icon-button"
            onClick={() => setShowHelp(false)}
            aria-label="关闭指南"
          >
            <X size={16} />
          </button>
        </div>
      )}
      <div
        className={`workspace ${sidebarOpen && !smallScreen && !focusMode ? "with-sidebar" : ""} ${inspectorOpen && !compactScreen && !focusMode ? "with-inspector" : ""}`}
      >
        <SidePanel
          side="left"
          open={sidebarOpen && !focusMode}
          modal={smallScreen}
          onClose={() => setSidebarOpen(false)}
        >
          <aside className="sidebar" aria-label="模型与输入">
            <div className="sidebar-title">
              <span>
                <Settings2 size={16} />
                模型与输入
              </span>
              <button
                className="icon-button"
                aria-label="收起模型面板"
                onClick={() => setSidebarOpen(false)}
              >
                <PanelLeft size={17} />
              </button>
            </div>
            <div className="sidebar-section model-section">
              <div className="section-label">
                模型配置 <span>MODEL</span>
              </div>
              <button
                className="model-select unified-model-select"
                aria-label="选择或搜索模型"
                aria-haspopup="dialog"
                onClick={() => setGalleryOpen(true)}
              >
                <span className="model-family-icon">
                  <Boxes size={20} />
                </span>
                <span className="current-model-text">
                  <strong title={modelName}>{modelName}</strong>
                  <small>检索全部 {modelLibraryCount} 个模型</small>
                </span>
                <ChevronDown size={14} />
              </button>
              <div className="model-source" title={source}>
                <span className="status-dot" />
                {activeModelSelection ? source : `自定义 · ${source}`}
              </div>
              <button
                className="upload-button"
                onClick={() => setImportOpen(true)}
              >
                <Upload size={14} />
                导入 config.json
                <ArrowUpRight size={13} />
              </button>
            </div>
            <div className="sidebar-section structure-section">
              <div className="section-label">
                结构导航 <span>EXPLORER</span>
              </div>
              <button
                className={`tree-item ${view === "overview" ? "active" : ""}`}
                onClick={() => navigate("overview")}
              >
                <Boxes size={16} />
                <span>模型总览</span>
                <ChevronRight size={13} />
              </button>
              <button
                className={`tree-item architecture-nav ${view === "architecture" ? "active" : ""}`}
                onClick={() => navigate("architecture")}
              >
                <Network size={15} />
                <span>整体架构</span>
                <span className="tree-count">× N</span>
              </button>
              {model.deepseek && (
                <div className="ced-navigation">
                  {(
                    [
                      "encoder",
                      "decoder",
                      "cache",
                      ...(model.deepseek.vision ? ["vision"] : []),
                      ...(model.deepseek.engramLayers.length
                        ? ["engram"]
                        : []),
                      ...(model.deepseek.draftLayers ? ["dspark"] : []),
                    ] as View[]
                  ).map((item) => (
                    <button
                      key={item}
                      className={`tree-item ${view === item ? "active" : ""}`}
                      onClick={() => navigate(item)}
                    >
                      <Layers3 size={14} />
                      <span>{viewTitles[item]}</span>
                      {(item === "encoder" || item === "decoder") && (
                        <span className="tree-count">{model.layers / 2}</span>
                      )}
                    </button>
                  ))}
                </div>
              )}
              {model.gemma && (
                <div className="ced-navigation">
                  {(
                    [
                      "decoder",
                      "cache",
                      "ple",
                      ...(model.gemma.vision ? ["vision"] : []),
                      ...(model.gemma.audio ? ["audio"] : []),
                    ] as View[]
                  ).map((item) => (
                    <button
                      key={item}
                      className={`tree-item ${view === item ? "active" : ""}`}
                      onClick={() => navigate(item)}
                    >
                      <Layers3 size={14} />
                      <span>{viewTitles[item]}</span>
                    </button>
                  ))}
                </div>
              )}
              {model.catalog && (
                <div className="ced-navigation">
                  {(["decoder", "cache", "auxiliary"] as View[]).map(
                    (item) => (
                      <button
                        key={item}
                        className={`tree-item ${view === item ? "active" : ""}`}
                        onClick={() => navigate(item)}
                      >
                        <Layers3 size={14} />
                        <span>{viewTitles[item]}</span>
                      </button>
                    ),
                  )}
                </div>
              )}
              <div className="tree-branch">
                <button
                  className={`tree-item ${view === "layer" ? "active" : ""}`}
                  onClick={() => navigate("layer")}
                >
                  <Layers3 size={15} />
                  <span>Transformer Block</span>
                  <span className="tree-count">{model.layers}</span>
                </button>
                <div className="tree-branch">
                  <button
                    className={`tree-item ${view === "attention" ? "active" : ""}`}
                    onClick={() => navigate("attention")}
                  >
                    <Network size={14} />
                    <span>
                      {model.catalog
                        ? "混合算子"
                        : model.deepseek
                          ? "CSA2 Attention"
                          : model.gemma
                            ? "Gemma Hybrid Attention"
                            : "Self Attention"}
                    </span>
                  </button>
                  <button
                    className={`tree-item ${view === "mlp" ? "active" : ""}`}
                    onClick={() => navigate("mlp")}
                  >
                    <Sparkles size={14} />
                    <span>
                      {model.catalog
                        ? "前馈 / 专家"
                        : model.deepseek
                          ? "Mixture of Experts"
                          : model.gemma
                            ? "Gemma Gated MLP"
                            : "Feed Forward"}
                    </span>
                  </button>
                </div>
              </div>
              <label className="layer-picker">
                查看层
                <select
                  aria-label="选择 Transformer 层"
                  value={layerIndex}
                  onChange={(e) => {
                    manualViewport.current = false;
                    focusTarget.current = null;
                    setStageIndex(0);
                    setLayerIndex(Number(e.target.value));
                    if (["overview", "encoder", "decoder"].includes(view))
                      navigate("layer");
                  }}
                >
                  {Array.from({ length: model.layers }, (_, i) => (
                    <option key={i} value={i}>
                      {model.catalog
                        ? `L${i} · ${model.catalog.profiles[i].label}`
                        : model.deepseek
                          ? `${i < model.deepseek.encoderLayers ? "E" : "D"} · Layer ${i} · ${layerProfile(model, i).mode}`
                          : model.gemma
                            ? `L${i} · ${gemmaLayer(model, i).sliding ? "Local" : "Global"}${gemmaLayer(model, i).shared ? " · shared" : ""}`
                            : `Layer ${i}`}{" "}
                      {i === 0
                        ? "· first"
                        : i === model.layers - 1
                          ? "· last"
                          : ""}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <div className="sidebar-section scenario-section">
              <div className="section-label">
                输入场景 <Settings2 size={14} />
              </div>
              <div className="scenario-mode">
                <button
                  className={scenario.past === 0 ? "active" : ""}
                  onClick={() =>
                    updateScenario({
                      sequence: model.gemma
                        ? Math.max(
                            128,
                            (scenario.image?.count
                              ? gemmaVisionShapes(model, scenario).tokens + 2
                              : 0) +
                              (scenario.audio?.enabled
                                ? gemmaAudioShapes(model, scenario).time2 + 2
                                : 0),
                          )
                        : scenario.image?.count
                          ? Math.max(scenario.sequence, 212)
                          : 128,
                      past: 0,
                    })
                  }
                >
                  Prefill
                </button>
                <button
                  className={scenario.past > 0 ? "active" : ""}
                  onClick={() => updateScenario({ sequence: 1, past: 128 })}
                >
                  Decode
                </button>
              </div>
              <InputField
                label="Batch size"
                symbol="B"
                min={1}
                value={scenario.batch}
                onChange={(batch) => updateScenario({ batch })}
              />
              <InputField
                label="Sequence length"
                symbol="S"
                min={1}
                value={scenario.sequence}
                onChange={(sequence) => updateScenario({ sequence })}
              />
              <InputField
                label="Past KV length"
                symbol="P"
                min={0}
                value={scenario.past}
                onChange={(past) => updateScenario({ past })}
              />
              <label className="scenario-field">
                <span>
                  计算精度<code>dtype</code>
                </span>
                <select
                  aria-label="计算精度"
                  value={scenario.dtype}
                  onChange={(e) =>
                    updateScenario({ dtype: e.target.value as DType })
                  }
                >
                  <option value="bfloat16">bfloat16</option>
                  <option value="float16">float16</option>
                  <option value="float32">float32</option>
                </select>
              </label>
              {model.deepseek && (
                <div className="ced-scenario">
                  <label className="scenario-field">
                    <span>CED 路径</span>
                    <select
                      aria-label="CED 执行路径"
                      value={scenario.cedExecution ?? "reference"}
                      onChange={(e) =>
                        updateScenario({
                          cedExecution: e.target.value as
                            | "reference"
                            | "bounded",
                        })
                      }
                    >
                      <option value="reference">完整参考前向</option>
                      <option value="bounded">有界重放示意</option>
                    </select>
                  </label>
                  {model.deepseek.vision && (
                    <>
                      <label className="scenario-field">
                        <span>包含图像</span>
                        <input
                          type="checkbox"
                          aria-label="包含图像"
                          checked={Boolean(scenario.image?.count)}
                          onChange={(e) =>
                            updateScenario({
                              image: e.target.checked
                                ? { count: 1, gridH: 42, gridW: 42 }
                                : undefined,
                              ...(e.target.checked && scenario.past === 0
                                ? {
                                    sequence: Math.max(
                                      scenario.sequence,
                                      212,
                                    ),
                                  }
                                : {}),
                            })
                          }
                        />
                      </label>
                      {scenario.image?.count && (
                        <>
                          <InputField
                            label="Patch grid H"
                            symbol="Gh"
                            min={1}
                            value={scenario.image.gridH}
                            onChange={(gridH) =>
                              updateScenario({
                                image: { ...scenario.image!, gridH },
                              })
                            }
                          />
                          <InputField
                            label="Patch grid W"
                            symbol="Gw"
                            min={1}
                            value={scenario.image.gridW}
                            onChange={(gridW) =>
                              updateScenario({
                                image: { ...scenario.image!, gridW },
                              })
                            }
                          />
                          <p className="ced-small-note">
                            每样本一张图，网格为预处理结果；S 包含图像位置。
                          </p>
                        </>
                      )}
                    </>
                  )}
                  <p className="ced-small-note">
                    {scenario.cedExecution === "bounded"
                      ? "有界重放为近似调度示意；Decoder 只重放末尾窗口，全局 KV 仍覆盖完整上下文。"
                      : "完整前向与官方参考代码一致。Decode 必须 S=1。"}
                  </p>
                </div>
              )}
              {model.gemma && (
                <div className="gemma-scenario ced-scenario">
                  {model.gemma.vision && (
                    <>
                      <label className="scenario-field">
                        <span>包含图像</span>
                        <input
                          type="checkbox"
                          aria-label="包含图像"
                          checked={Boolean(scenario.image?.count)}
                          onChange={(event) => {
                            const image = event.target.checked
                              ? { count: 1, gridH: 42, gridW: 60 }
                              : undefined;
                            const next = { ...scenario, image };
                            updateScenario({
                              image,
                              ...(scenario.past === 0
                                ? {
                                    sequence: Math.max(
                                      scenario.sequence,
                                      (image
                                        ? gemmaVisionShapes(model, next)
                                            .tokens + 2
                                        : 0) +
                                        (scenario.audio?.enabled
                                          ? gemmaAudioShapes(model, next)
                                              .time2 + 2
                                          : 0),
                                    ),
                                  }
                                : {}),
                            });
                          }}
                        />
                      </label>
                      {scenario.image?.count && (
                        <>
                          <InputField
                            label="Patch grid H"
                            symbol="Gh"
                            min={1}
                            value={scenario.image.gridH}
                            onChange={(gridH) =>
                              updateScenario({
                                image: { ...scenario.image!, gridH },
                              })
                            }
                          />
                          <InputField
                            label="Patch grid W"
                            symbol="Gw"
                            min={1}
                            value={scenario.image.gridW}
                            onChange={(gridW) =>
                              updateScenario({
                                image: { ...scenario.image!, gridW },
                              })
                            }
                          />
                        </>
                      )}
                    </>
                  )}
                  {model.gemma.audio && (
                    <>
                      <label className="scenario-field">
                        <span>包含音频</span>
                        <input
                          type="checkbox"
                          aria-label="包含音频"
                          checked={Boolean(scenario.audio?.enabled)}
                          onChange={(event) => {
                            const audio = event.target.checked
                              ? { enabled: true, frames: 128 }
                              : undefined;
                            const next = { ...scenario, audio };
                            updateScenario({
                              audio,
                              ...(scenario.past === 0
                                ? {
                                    sequence: Math.max(
                                      scenario.sequence,
                                      (scenario.image?.count
                                        ? gemmaVisionShapes(model, next)
                                            .tokens + 2
                                        : 0) +
                                        (audio
                                          ? gemmaAudioShapes(model, next)
                                              .time2 + 2
                                          : 0),
                                    ),
                                  }
                                : {}),
                            });
                          }}
                        />
                      </label>
                      {scenario.audio?.enabled && (
                        <InputField
                          label="Audio feature frames"
                          symbol="F"
                          min={1}
                          value={scenario.audio.frames}
                          onChange={(frames) =>
                            updateScenario({
                              audio: { enabled: true, frames },
                            })
                          }
                        />
                      )}
                    </>
                  )}
                  <p className="ced-small-note">
                    每样本一张图/一段音频；输入为预处理后的网格和特征帧。S
                    已包含软 token；Decode 不重复执行媒体编码器。
                  </p>
                </div>
              )}
              {scenarioError && (
                <p className="error-message" role="alert">
                  {scenarioError}
                </p>
              )}
              {isBeyondContext && (
                <p className="warning-message" role="status">
                  P + S 超过配置的 max_position_embeddings (
                  {formatNumber(model.maxPosition!)})，仅展示理论形状。
                </p>
              )}
              <div className="input-preview">
                <span>input_ids</span>
                <code>
                  [{scenario.batch}, {scenario.sequence}]
                </code>
                <ArrowDown size={13} />
              </div>
            </div>
            <div className="sidebar-bottom">
              <span className="local-indicator">
                <span className="status-dot" />
                浏览器本地计算
              </span>
              <p>无需 GPU，无需加载权重。</p>
              <button
                className="text-button"
                onClick={() => rawDialog.current?.showModal()}
              >
                <Code2 size={14} />
                查看原始配置
                <ArrowUpRight size={12} />
              </button>
            </div>
          </aside>
        </SidePanel>
        <main className="main-panel">
          <div className="model-header">
            <div>
              <div className="eyebrow">
                <span>MODEL ARCHITECTURE</span>
                <span className="model-type-badge">
                  {model.catalog?.type ?? model.family}
                </span>
              </div>
              <h1>{modelName}</h1>
              <p>
                <span className="status-dot" />
                配置已就绪<span className="model-sub-divider">/</span>
                {model.architecture}
              </p>
            </div>
            <button
              className="secondary-button config-button"
              onClick={() => rawDialog.current?.showModal()}
            >
              <FileJson size={15} />
              config.json
            </button>
          </div>
          {model.deepseek && (
            <div className="ced-summary">
              <span>
                <b>CED</b>
                {model.deepseek.encoderLayers} Encoder +{" "}
                {model.layers - model.deepseek.encoderLayers} Decoder
              </span>
              <span>
                {model.deepseek.experts} experts · Top-
                {model.deepseek.topExperts}
              </span>
              <button onClick={() => navigate("cache")}>
                KV 共享关系
                <ArrowUpRight size={12} />
              </button>
            </div>
          )}
          {model.gemma && (
            <div className="gemma-summary ced-summary">
              <span>
                <b>PLE</b>
                {model.gemma.pleDim} dim / layer
              </span>
              <span>{model.gemma.sharedLayers} 层共享 KV</span>
              <button onClick={() => navigate("cache")}>
                KV 来源与统计
                <ArrowUpRight size={12} />
              </button>
            </div>
          )}
          {model.gallery && (
            <div className="catalog-provenance">
              <a href={model.gallery.url} target="_blank" rel="noreferrer">
                画廊条目
                <ArrowUpRight size={11} />
              </a>
              <a
                href={model.gallery.configUrl}
                target="_blank"
                rel="noreferrer"
              >
                配置来源
                <ArrowUpRight size={11} />
              </a>
              <span>
                {model.gallery.sourceNote ? "含来源说明" : "公开配置快照"}
              </span>
            </div>
          )}
          {model.catalog && (
            <div className="catalog-scope">
              <span>语言主干结构视图</span>
              <p>
                {model.catalog.special
                  ? "含专用算子的接口级展示；未知内部参数/缓存不视作零。"
                  : "参数与缓存为逻辑结构估算；未执行模型推理。"}
              </p>
              <button onClick={() => rawDialog.current?.showModal()}>
                范围说明
              </button>
            </div>
          )}
          <div className="stats-strip">
            <div className="stat">
              <span>
                {model.catalog
                  ? "主干权重估算"
                  : model.deepseek || model.gemma
                    ? "全部逻辑参数"
                    : "模型参数"}{" "}
                <span className="stat-hint">推导</span>
              </span>
              <strong>
                {model.catalog ? "≈ " : ""}
                {compactNumber(counts.total)}
                <small>params</small>
              </strong>
            </div>
            <div className="stat">
              <span>Transformer 层</span>
              <strong>
                {model.layers}
                <small>layers</small>
              </strong>
            </div>
            <div className="stat dimension-stat">
              <span>隐藏维度</span>
              <strong>
                {formatNumber(model.hidden)}
                <small>dim</small>
              </strong>
            </div>
            <div className="stat">
              <span>注意力头</span>
              <strong>
                {model.heads}
                <small>
                  {model.deepseek ? "Q / shared KV" : `/ ${model.kvHeads} KV`}
                </small>
              </strong>
            </div>
            <div
              className="stat memory-stat"
              title="浮点等效权重，非混合量化文件大小；不包含激活、缓存或运行时开销"
            >
              <span>
                {model.catalog
                  ? "主干浮点估算"
                  : model.deepseek || model.gemma
                    ? "浮点等效权重"
                    : "权重内存"}{" "}
                <span className="stat-hint">估算</span>
              </span>
              <strong>
                {formatBytes(memory.weights).split(" ")[0]}
                <small>{formatBytes(memory.weights).split(" ")[1]}</small>
              </strong>
            </div>
            <button
              className="stat cache-stat"
              aria-label="查看 KV cache 计算"
              aria-haspopup="dialog"
              onClick={() => setCacheOpen(true)}
            >
              <span>
                KV cache{" "}
                <span className="stat-hint">
                  {cache.partial ? "部分" : "估算"}
                </span>
              </span>
              <strong data-testid="cache-stat-value">
                {cache.unknownLayers.length && cache.kvBytes === 0
                  ? "未确定"
                  : formatBytes(cache.kvBytes).split(" ")[0]}
                <small>
                  {cache.unknownLayers.length && cache.kvBytes === 0
                    ? ""
                    : formatBytes(cache.kvBytes).split(" ")[1]}
                </small>
              </strong>
              <Database size={12} />
            </button>
          </div>
          <div className="canvas-topbar">
            <div className="navigation-tools">
              <button
                className="icon-button"
                aria-label="切换模型面板"
                title="模型与输入"
                aria-pressed={sidebarOpen && !focusMode}
                onClick={() => {
                  if (focusMode) setFocusMode(false);
                  setSidebarOpen(!sidebarOpen || focusMode);
                }}
              >
                <PanelLeft size={18} />
              </button>
            </div>
            <div
              className="display-mode-switch"
              role="group"
              aria-label="可视化方式"
            >
              {(
                [
                  { id: "graph", label: "结构图", Icon: Network },
                  { id: "2d", label: "2D 张量", Icon: Grid2X2 },
                  { id: "3d", label: "3D 空间", Icon: Box },
                ] as const
              ).map(({ id, label, Icon }) => (
                <button
                  key={id}
                  aria-pressed={displayMode === id}
                  aria-label={label}
                  onClick={() => changeDisplayMode(id)}
                >
                  <Icon size={14} />
                  <span>{label}</span>
                </button>
              ))}
            </div>
            <div className="canvas-actions">
              <FullscreenButton
                fullscreen={fullscreen}
                className="icon-button fullscreen-button"
              />
              <button
                className={`symbol-mode ${symbolic ? "active" : ""}`}
                title="切换符号维度"
                aria-label={`切换符号维度，当前${symbolic ? "符号" : "数值"}`}
                aria-pressed={symbolic}
                onClick={() => setSymbolic(!symbolic)}
              >
                <Braces size={16} />
                <span className="symbol-mode-label">
                  {symbolic ? "符号" : "数值"}
                </span>
              </button>
              <button
                className="icon-button"
                title="切换详情面板"
                aria-label="切换详情面板"
                aria-pressed={inspectorOpen && !focusMode}
                onClick={() => {
                  if (focusMode) setFocusMode(false);
                  setInspectorOpen(!inspectorOpen || focusMode);
                }}
              >
                <PanelRight size={18} />
              </button>
              <span className="toolbar-separator" />
              <button
                className={`play-button ${playing ? "playing" : ""}`}
                disabled={playbackNodes.length === 0}
                onClick={() => {
                  if (playing) setPlaying(false);
                  else {
                    if (step >= playbackNodes.length - 1 || step < 0)
                      setStep(0);
                    setPlaying(true);
                    setSelectedEdgeId(null);
                  }
                }}
              >
                {playing ? <Pause size={13} /> : <Play size={13} />}
                {playing ? "暂停" : "演示数据流"}
              </button>
              <button
                className="icon-button step-button"
                title="下一步"
                aria-label="数据流下一步"
                disabled={playbackNodes.length === 0}
                onClick={() => {
                  setPlaying(false);
                  setSelectedEdgeId(null);
                  setStep((step + 1) % playbackNodes.length);
                }}
              >
                <SkipForward size={15} />
              </button>
              <CanvasTools
                focusMode={focusMode}
                compact={smallScreen}
                symbolic={symbolic}
                onSymbol={() => setSymbolic(!symbolic)}
                onInspector={() => {
                  if (focusMode) setFocusMode(false);
                  setInspectorOpen(!inspectorOpen || focusMode);
                }}
                onExport={exportGraph}
                onFocus={() => setFocusMode(!focusMode)}
                onReset={() => {
                  setPlaying(false);
                  setStep(-1);
                }}
              />
            </div>
          </div>
          {viewStages.length > 0 && (
            <nav className="stage-navigation" aria-label="计算阶段">
              {viewStages.map((stage, index) => (
                <button
                  key={stage.title}
                  aria-current={
                    index === stageIndex && viewMode === "read"
                      ? "step"
                      : undefined
                  }
                  title={stage.description}
                  onClick={() => selectStage(stage, index)}
                >
                  <span>{String(index + 1).padStart(2, "0")}</span>
                  {stage.title}
                </button>
              ))}
            </nav>
          )}
          {view === "architecture" && (
            <p className="architecture-description">
              整体路径 · 相同的连续层 / 周期只画一次，右侧 ×
              标记重复次数。残差为加法或架构专用混合；层权重不默认共享。
              {model.catalog && model.catalog.loops > 1
                ? ` 主干按 ${model.catalog.loops} 轮执行，跨轮共享权重。`
                : ""}
            </p>
          )}
          <div
            ref={canvasRef}
            className={`canvas display-${displayMode}`}
            data-testid="model-canvas"
            data-display-mode={displayMode}
          >
            <div className="canvas-contextbar">
              <div className="canvas-location">
                {view === "overview" && (
                  <button
                    className="architecture-open-button"
                    onClick={() => navigate("architecture")}
                  >
                    展开整体架构 · 含残差
                    <ArrowUpRight size={12} />
                  </button>
                )}
                {view !== "overview" && (
                  <button
                    className="icon-button"
                    title="返回上一级"
                    aria-label="返回上一级"
                    onClick={() =>
                      navigate(
                        view === "architecture"
                          ? "overview"
                          : model.catalog
                            ? ["decoder", "cache", "auxiliary"].includes(view)
                              ? "overview"
                              : view === "layer"
                                ? "decoder"
                                : "layer"
                            : model.gemma
                              ? [
                                  "decoder",
                                  "cache",
                                  "ple",
                                  "vision",
                                  "audio",
                                ].includes(view)
                                ? "overview"
                                : view === "layer"
                                  ? "decoder"
                                  : "layer"
                              : model.deepseek
                                ? [
                                    "encoder",
                                    "decoder",
                                    "cache",
                                    "vision",
                                    "engram",
                                    "dspark",
                                  ].includes(view)
                                  ? "overview"
                                  : view === "layer"
                                    ? layerIndex <
                                      model.deepseek.encoderLayers
                                      ? "encoder"
                                      : "decoder"
                                    : "layer"
                                : view === "layer"
                                  ? "overview"
                                  : "layer",
                      )
                    }
                  >
                    <ArrowLeft size={14} />
                  </button>
                )}
                <div className="breadcrumbs">
                  <button onClick={() => navigate("overview")}>
                    <Boxes size={14} />
                    <span>模型</span>
                  </button>
                  {["layer", "attention", "mlp"].includes(view) && (
                    <>
                      <ChevronRight size={12} />
                      <button onClick={() => navigate("layer")}>
                        {model.deepseek
                          ? `${layerIndex < model.deepseek.encoderLayers ? "Encoder" : "Decoder"} L${layerIndex}`
                          : `Layer ${layerIndex}`}
                      </button>
                    </>
                  )}
                  {view !== "overview" && view !== "layer" && (
                    <>
                      <ChevronRight size={12} />
                      <span>{viewTitles[view]}</span>
                    </>
                  )}
                </div>
                <span className="module-count">
                  {graph.nodes.filter((node) => !node.data.frame).length}{" "}
                  个模块
                </span>
              </div>
              <div className="canvas-context-actions">
                {displayMode === "3d" && (
                  <button
                    className="architecture-open-button"
                    onClick={() => {
                      setPlaying(false);
                      setDemoOpen(true);
                    }}
                  >
                    推理与详细解释
                    <Play size={12} />
                  </button>
                )}
                <span className="view-purpose">
                  {displayMode === "graph"
                    ? "结构与连接"
                    : displayMode === "2d"
                      ? "矩阵与切片"
                      : "块体与深度"}
                </span>
                <button
                  className="shape-toggle"
                  aria-label="维度标注"
                  aria-pressed={showShapes}
                  onClick={() => setShowShapes(!showShapes)}
                >
                  <span
                    className={`toggle-track ${showShapes ? "on" : ""}`}
                  />
                  维度标注
                </button>
              </div>
            </div>
            {displayMode === "graph" ? (
              <div className="graph-render-area">
                <ReactFlow<ModuleNodeType>
                  nodes={nodes}
                  edges={edges}
                  nodeTypes={nodeTypes}
                  onNodeClick={selectNode}
                  onNodeDoubleClick={(_, node) => {
                    if (node.data.drill)
                      navigate(node.data.drill, node.data.layerIndex);
                  }}
                  onEdgeClick={selectEdge}
                  onNodeMouseEnter={(_, node) => setHoveredId(node.id)}
                  onNodeMouseLeave={() => setHoveredId(null)}
                  onPaneClick={() => {
                    setSelectedId(null);
                    setSelectedEdgeId(null);
                    setPlaying(false);
                    setStep(-1);
                  }}
                  onMoveStart={(event) => {
                    if (event) {
                      manualViewport.current = true;
                      clearTimeout(refitTimer.current);
                    }
                  }}
                  onMove={(_, viewport) =>
                    setZoom(Math.round(viewport.zoom * 100))
                  }
                  deleteKeyCode={null}
                  nodesDraggable={false}
                  nodesConnectable={false}
                  elementsSelectable
                  zoomOnDoubleClick={false}
                  minZoom={0.15}
                  maxZoom={2.5}
                  proOptions={{ hideAttribution: false }}
                >
                  <Background
                    color="#d9dfeb"
                    gap={20}
                    size={1}
                    variant={BackgroundVariant.Dots}
                  />
                  <Panel position="bottom-left" className="viewport-controls">
                    <div className="zoom-controls">
                      <button
                        className="icon-button"
                        aria-label="缩小画布"
                        title="缩小"
                        onClick={() => {
                          manualViewport.current = true;
                          clearTimeout(refitTimer.current);
                          void zoomOut({ duration: motion });
                        }}
                      >
                        <Minus size={17} />
                      </button>
                      <button
                        className="zoom-value"
                        aria-label={`${zoom}%，缩放到百分之百`}
                        title="100%"
                        onClick={() => {
                          manualViewport.current = true;
                          clearTimeout(refitTimer.current);
                          void zoomTo(1, { duration: motion });
                        }}
                      >
                        {zoom}%
                      </button>
                      <button
                        className="icon-button"
                        aria-label="放大画布"
                        title="放大"
                        onClick={() => {
                          manualViewport.current = true;
                          clearTimeout(refitTimer.current);
                          void zoomIn({ duration: motion });
                        }}
                      >
                        <Plus size={17} />
                      </button>
                      <span />
                      <button
                        className="icon-button"
                        title="适配全图"
                        aria-label="适配全图"
                        onClick={() => {
                          focusTarget.current = null;
                          manualViewport.current = false;
                          setViewMode("fit");
                          void fitView({
                            padding: {
                              top: "54px",
                              bottom: "58px",
                              left: "30px",
                              right: "30px",
                            },
                            duration: motion,
                          });
                        }}
                      >
                        <Scan size={17} />
                      </button>
                    </div>
                    <div className="reading-controls">
                      <button
                        aria-label="阅读视图"
                        aria-pressed={viewMode === "read"}
                        onClick={() => {
                          focusTarget.current = null;
                          manualViewport.current = false;
                          setViewMode("read");
                          if (viewMode === "read") resetView();
                        }}
                      >
                        <BookOpen size={14} />
                        <span>阅读视图</span>
                      </button>
                      <button
                        className="icon-button"
                        title="聚焦选中模块"
                        aria-label="聚焦选中模块"
                        onClick={focusSelected}
                      >
                        <Focus size={17} />
                      </button>
                    </div>
                  </Panel>
                  {(view !== "overview" || viewMode === "read") && (
                    <MiniMap
                      onNodeClick={(_, node) => {
                        setSelectedId(node.id);
                        setSelectedEdgeId(null);
                        const target = nodes.find((n) => n.id === node.id);
                        if (target) focusNode(target);
                      }}
                      style={{ width: 138, height: 90 }}
                      nodeColor={(node) =>
                        nodeColors[node.data.category as string] ?? "#a4adbc"
                      }
                      maskColor="rgba(241,244,250,0.75)"
                      zoomable
                      pannable
                      position="bottom-right"
                    />
                  )}
                  {step >= 0 && (
                    <Panel
                      position="bottom-center"
                      className="playback-status"
                    >
                      <span className="status-dot" />
                      <span>
                        {step + 1} / {playbackNodes.length}
                      </span>
                      <strong>
                        {
                          playbackNodes[
                            Math.min(step, playbackNodes.length - 1)
                          ]?.data.title
                        }
                      </strong>
                      <button
                        className="follow-button"
                        aria-pressed={followPlayback}
                        onClick={() => setFollowPlayback(!followPlayback)}
                      >
                        <Focus size={13} />
                        跟随
                      </button>
                    </Panel>
                  )}
                </ReactFlow>
                {zoom < 65 && (
                  <div className="readability-hint">
                    <BookOpen size={14} />
                    <span>当前为全图预览，切换阅读视图查看细节</span>
                    <button
                      onClick={() => {
                        focusTarget.current = null;
                        manualViewport.current = false;
                        setViewMode("read");
                        if (viewMode === "read") resetView();
                      }}
                    >
                      放大阅读
                      <ArrowUpRight size={14} />
                    </button>
                  </div>
                )}
              </div>
            ) : (
              <>
                <div className="tensor-render-area">
                  <Suspense
                    fallback={
                      <div className="tensor-loading" role="status">
                        正在加载
                        {displayMode === "3d" ? "三维场景" : "张量视图"}…
                      </div>
                    }
                  >
                    {displayMode === "2d" ? (
                      <TensorCanvas2D {...tensorCanvasProps} />
                    ) : (
                      <TensorCanvas3D
                        {...tensorCanvasProps}
                        model={model}
                        scenario={scenario}
                        onFallback={() => changeDisplayMode("2d")}
                      />
                    )}
                  </Suspense>
                </div>
                <div className="tensor-view-note">
                  <span>形状示意 · 网格与块体压缩显示，标签保留完整维度</span>
                  {step >= 0 && (
                    <span className="tensor-playback" role="status">
                      {step + 1} / {playbackNodes.length} · {inspected?.title}
                      <button
                        aria-pressed={followPlayback}
                        onClick={() => setFollowPlayback(!followPlayback)}
                      >
                        跟随
                      </button>
                    </span>
                  )}
                </div>
              </>
            )}
          </div>
          <div className="canvas-footer">
            <div className="legend">
              {[
                ["embedding", "Embedding"],
                ["attention", "Attention"],
                ["mlp", "MLP"],
                ["norm", "Norm"],
                ["residual", "Residual"],
              ].map(([id, label]) => (
                <span key={id}>
                  <i
                    style={{
                      background:
                        id === "residual" ? "#bd6f35" : nodeColors[id],
                    }}
                  />
                  {label}
                </span>
              ))}
            </div>
            <span className="graph-method">
              <GitBranch size={12} />
              静态结构推导
            </span>
          </div>
          <div className="memory-footer">
            <button
              className="cache-footer-button"
              aria-label="查看缓存与状态明细"
              onClick={() => setCacheOpen(true)}
            >
              <Database size={12} />
              <span>{cache.partial ? "已知缓存 / 状态" : "缓存 / 状态"}</span>
              <b>{formatBytes(cache.totalBytes)}</b>
              <ChevronRight size={12} />
            </button>
            {gmMemory && (
              <span title="本次所有层的 PLE 逻辑激活，不是嵌入表权重">
                PLE <b>{formatBytes(gmMemory.pleActivations)}</b>
              </span>
            )}
            {dsMemory && (
              <span title="全局 main KV + index K，含 scales；不含 SWA / 临时状态">
                全局 FP4 <b>{formatBytes(dsMemory.globalPacked)}</b>
              </span>
            )}
            <span>
              输出 logits <b>{formatBytes(memory.logits)}</b>
            </span>
            <span title="不包含临时激活、运行时工作区与框架开销">
              逻辑大小 · {scenario.dtype}
            </span>
          </div>
        </main>
        <SidePanel
          side="right"
          open={inspectorOpen && !focusMode}
          modal={compactScreen}
          onClose={() => setInspectorOpen(false)}
        >
          <Inspector
            data={inspected}
            selectedEdge={selectedEdge}
            onDrill={(next) => navigate(next, inspected?.layerIndex)}
            onClose={() => setInspectorOpen(false)}
            selectionKey={`${view}-${layerIndex}-${selectedEdgeId ?? activeId ?? selectedId}`}
          />
        </SidePanel>
      </div>
      <footer className="app-footer">
        <span>
          <Check size={12} />
          {model.architecture}
        </span>
        <span>
          {model.warnings.length > 0 ? (
            <button
              className="footer-warning"
              onClick={() => rawDialog.current?.showModal()}
            >
              查看 {model.warnings.length} 条配置说明
            </button>
          ) : (
            "配置已解析"
          )}
          <span className="footer-divider">·</span>不执行真实前向计算
        </span>
        <span className="footer-brand">TENSOR ATLAS</span>
      </footer>
      {notice && (
        <div className="toast" role="status">
          <Check size={17} />
          {notice}
        </div>
      )}
      {galleryOpen && (
        <Suspense
          fallback={
            <div className="toast" role="status">
              正在加载模型库…
            </div>
          }
        >
          <GalleryDialog
            open={galleryOpen}
            onClose={() => setGalleryOpen(false)}
            onLoad={replaceModel}
            selection={activeModelSelection}
            scenario={scenario}
            onImport={() => {
              setGalleryOpen(false);
              setImportOpen(true);
            }}
          />
        </Suspense>
      )}
      {cacheOpen && (
        <Suspense
          fallback={
            <div className="toast" role="status">
              正在打开缓存计算…
            </div>
          }
        >
          <CacheDialog
            model={model}
            name={modelName}
            scenario={scenario}
            report={cache}
            onChange={updateScenario}
            error={scenarioError}
            onClose={() => setCacheOpen(false)}
          />
        </Suspense>
      )}
      <ImportDialog
        open={importOpen}
        onClose={() => setImportOpen(false)}
        onImport={replaceModel}
      />
      <dialog
        ref={rawDialog}
        className="raw-dialog"
        aria-labelledby="raw-title"
        onClick={(e) => {
          if (e.target === rawDialog.current) rawDialog.current?.close();
        }}
      >
        <div className="raw-dialog-header">
          <div>
            <span className="section-eyebrow">MODEL CONFIGURATION</span>
            <h2 id="raw-title">config.json</h2>
          </div>
          <button
            className="icon-button"
            aria-label="关闭原始配置"
            onClick={() => rawDialog.current?.close()}
          >
            <X size={19} />
          </button>
        </div>
        <div className="raw-dialog-content">
          <p className="raw-source">{source}</p>
          {model.warnings.map((warning, i) => (
            <p className="warning-message" key={i}>
              {warning}
            </p>
          ))}
          <p className="config-method-note">
            {model.catalog
              ? "画廊主干结构适配：层级与逻辑接口来自所选配置，通用结构估算不是完整权重验证。专用算子/媒体/辅助头的范围见上方说明。"
              : model.gemma
                ? "Gemma 4 E 系列专用适配器：两类 KV 共享、PLE 和局部/全局 head_dim 按配置推导。总参数含媒体编码器；本次持久 KV 与 attention 可见长度分别计算。"
                : model.deepseek
                  ? "DeepSeek V4.1 专用适配器。全局 KV 按来源层去重；主 LM Head 使用最后位置 float32 logits。浮点逻辑内存不等于量化文件大小或峰值显存。"
                  : `使用标准 ${model.family} 架构适配器推导，展示完整未裁剪缓存与全部位置 logits。`}
          </p>
          {dsParams && (
            <dl className="config-details">
              {[
                ["CED 主干", dsParams.backbone],
                ["Engram", dsParams.engram],
                ["视觉路径", dsParams.vision],
                ["DSpark 独立权重", dsParams.draft],
                ["合计", dsParams.total],
              ].map(([label, value]) => (
                <div key={label}>
                  <dt>{label}</dt>
                  <dd>{formatNumber(value as number)}</dd>
                </div>
              ))}
            </dl>
          )}
          {gmParams && (
            <dl className="config-details gemma-parameter-breakdown">
              {[
                ["文本参数（含 PLE）", gmParams.text],
                ["其中主 embedding", gmParams.embedding],
                ["其中 PLE 表", gmParams.pleTable],
                ["其中 PLE 输入投影", gmParams.pleProjection],
                ["视觉（含投影）", gmParams.vision],
                ["音频（含投影）", gmParams.audio],
                ["合计", gmParams.total],
              ].map(([label, value]) => (
                <div key={label}>
                  <dt>{label}</dt>
                  <dd>{formatNumber(value as number)}</dd>
                </div>
              ))}
            </dl>
          )}
          <pre>{JSON.stringify(model.raw, null, 2)}</pre>
        </div>
        <div className="raw-dialog-footer">
          <button
            className="secondary-button"
            onClick={() => downloadJSON(model.raw, "config.json")}
          >
            <Download size={15} />
            下载配置
          </button>
          <button
            className="primary-button"
            onClick={() => {
              rawDialog.current?.close();
              setImportOpen(true);
            }}
          >
            <Upload size={15} />
            导入其他配置
          </button>
        </div>
      </dialog>
    </div>
  );
}
