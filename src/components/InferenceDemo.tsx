import {
  lazy,
  Suspense,
  useEffect,
  useMemo,
  useReducer,
  useRef,
  useState,
} from "react";
import {
  ArrowLeft,
  Box,
  Braces,
  Check,
  ChevronRight,
  Focus,
  Grid2X2,
  Pause,
  Play,
  RotateCcw,
  Settings2,
  SkipForward,
  PanelRight,
  X,
} from "lucide-react";
import {
  buildDemoPlan,
  demoCache,
  demoGraph,
  demoHasHead,
  demoReducer,
  demoScenario,
  demoSettingsValid,
  exampleContinuation,
  examplePrompt,
  initialDemoState,
  readDemoRoute,
  type DemoSettings,
  type DemoEvent,
} from "../lib/inference-demo";
import {
  compactNumber,
  formatBytes,
  type ModelConfig,
  type Scenario,
} from "../lib/model";
import { useMediaQuery } from "./SidePanel";
import FullscreenButton from "./FullscreenButton";
import type { FullscreenControl } from "../lib/use-fullscreen";
import DemoInspector from "./DemoInspector";
import {
  parseDemoTrace,
  traceTemplate,
  type DemoTrace,
} from "../lib/demo-trace";
import {
  buildDemoWorld,
  demoSceneNodeId,
  demoWorldLayers,
  demoPhaseSteps,
  selectedTokenPosition,
} from "../lib/demo-world";
import {
  demoBeat,
  demoPhaseStarts,
  demoStepDuration,
  demoPlanDuration,
  paceLabels,
  stageLabel,
  type DemoPace,
} from "../lib/demo-pacing";
import { eventForNode, sceneTraceSamples } from "../lib/demo-selection";
import { teachingPlan, routeLabels, type DemoRoute } from "../lib/demo-route";
const Space = lazy(() => import("./TensorCanvas3D"));
const Plane = lazy(() => import("./TensorCanvas2D"));

export default function InferenceDemo({
  model,
  name,
  dtype,
  onClose,
  onChooseModel,
  fullscreen,
}: {
  model: ModelConfig;
  name: string;
  dtype: Scenario["dtype"];
  onClose: () => void;
  onChooseModel?: () => void;
  fullscreen: FullscreenControl;
}) {
  const route = useMemo(() => readDemoRoute(window.location.search), []);
  const [settings, setSettings] = useState<DemoSettings>({
    promptTokens: Math.min(
      route.promptTokens,
      model.maxPosition ?? route.promptTokens,
    ),
    outputTokens: Math.min(
      Math.max(2, route.pass + 1),
      Math.max(1, (model.maxPosition ?? 32) - route.promptTokens + 1),
    ),
    useCache: route.useCache,
    speed: 1,
  });
  const [state, dispatch] = useReducer(demoReducer, {
    ...initialDemoState,
    pass: demoHasHead(model)
      ? Math.min(route.pass, settings.outputTokens - 1)
      : 0,
    generated: demoHasHead(model)
      ? Math.min(route.pass, settings.outputTokens - 1)
      : 0,
  });
  const [follow, setFollow] = useState(true),
    [mode, setMode] = useState<"3d" | "2d">("3d"),
    [symbolic, setSymbolic] = useState(false),
    [settingsOpen, setSettingsOpen] = useState(false),
    [inspectId, setInspectId] = useState<string | null>(null);
  const [detailOpen, setDetailOpen] = useState(false),
    [tokenIndex, setTokenIndex] = useState(0),
    [head, setHead] = useState(0),
    [brightness, setBrightness] = useState(100),
    [showEquations, setShowEquations] = useState(true),
    [devMode, setDevMode] = useState(false);
  const layers = useMemo(() => demoWorldLayers(model), [model]);
  const [trace, setTrace] = useState<DemoTrace | null>(null),
    [traceError, setTraceError] = useState("");
  const [pace, setPace] = useState<DemoPace>("guided");
  const traceInput = useRef<HTMLInputElement>(null);
  const reducedMotion = useMediaQuery("(prefers-reduced-motion: reduce)");
  const hasHead = demoHasHead(model),
    passes = hasHead ? settings.outputTokens : 1;
  const scenario = useMemo(
    () => demoScenario(settings, state.pass, dtype),
    [settings.promptTokens, settings.useCache, state.pass, dtype],
  );
  const [lessonRoute, setLessonRoute] = useState<DemoRoute>("overview");
  const fullPlan = useMemo(
    () => buildDemoPlan(model, scenario),
    [model, scenario],
  );
  const plan = useMemo(
    () =>
      lessonRoute === "full"
        ? fullPlan
        : teachingPlan(model, fullPlan, state.pass),
    [lessonRoute, fullPlan, model, state.pass],
  );
  const budgetPlans = useMemo(() => {
    return [0, 1].map((pass) => {
      const s = demoScenario(settings, pass, dtype);
      const all = buildDemoPlan(model, s);
      return lessonRoute === "full" ? all : teachingPlan(model, all, pass);
    });
  }, [model, settings.promptTokens, settings.useCache, dtype, lessonRoute]);
  const estimatedMs = useMemo(
    () =>
      demoPlanDuration(budgetPlans[0], pace, settings.speed) +
      (passes > 1
        ? (passes - 1) *
          demoPlanDuration(budgetPlans[1], pace, settings.speed)
        : 0),
    [budgetPlans, pace, settings.speed, passes],
  );
  const event = plan[Math.min(state.index, plan.length - 1)];
  const [browseEvent, setBrowseEvent] = useState<DemoEvent | null>(null);
  const graphEvent = browseEvent ?? event;
  const graph = useMemo(
    () =>
      mode === "3d"
        ? buildDemoWorld(model, scenario, graphEvent, symbolic)
        : demoGraph(model, scenario, graphEvent, symbolic),
    [
      model,
      scenario,
      mode === "3d" ? graphEvent.view === "overview" : graphEvent.view,
      graphEvent.layer,
      graphEvent.sample,
      mode === "2d" ? graphEvent.instance : undefined,
      symbolic,
      mode,
    ],
  );
  const sceneNodeId =
    mode === "3d" ? demoSceneNodeId(graph, event) : event.nodeId;
  const sceneSamples = useMemo(
    () =>
      trace
        ? sceneTraceSamples(
            graph,
            trace.samples,
            state.pass,
            event.loop,
            graphEvent.layer,
          )
        : undefined,
    [trace, state.pass, event.loop, graphEvent.layer, graph],
  );
  const node =
    graph.nodes.find((n) => n.id === (inspectId ?? sceneNodeId)) ??
    graph.nodes.find((n) => n.id === sceneNodeId);
  const inspectorSamples = useMemo(
    () => sceneSamples?.filter((s) => s.nodeId === node?.id),
    [sceneSamples, node?.id],
  );
  const changeMode = (next: "2d" | "3d") => {
    if (inspectId && node) {
      const target = eventForNode(node, graphEvent);
      setBrowseEvent(target);
      const targetGraph =
        next === "3d"
          ? buildDemoWorld(model, scenario, target, symbolic)
          : demoGraph(model, scenario, target, symbolic);
      setInspectId(
        next === "3d" ? demoSceneNodeId(targetGraph, target) : target.nodeId,
      );
    }
    setMode(next);
  };
  const playingNode = graph.nodes.find((n) => n.id === sceneNodeId);
  const stepDuration = demoStepDuration(
    event,
    playingNode?.data,
    pace,
    settings.speed,
  );
  const beat = demoBeat(state.progress);
  const valid = demoSettingsValid(model, settings);
  const committed = !!event.sample || state.status === "complete";
  const cache = useMemo(
    () => demoCache(model, scenario, settings.useCache, committed),
    [model, scenario, settings.useCache, committed],
  );
  const scopeNote =
    model.deepseek || model.gemma || model.catalog
      ? "沿用当前架构的逻辑适配；媒体、草稿和未展开专用算子不在此文本演示中。"
      : "按当前架构的静态形状演示，不执行真实前向。";
  const progressRef = useRef(0);
  useEffect(() => {
    const url = new URL(window.location.href);
    if (url.searchParams.get("demo") === "1") {
      url.searchParams.set("token", String(settings.promptTokens));
      url.searchParams.set("generation", String(state.pass));
      url.searchParams.set("kvCache", settings.useCache ? "1" : "0");
      history.replaceState(null, "", url);
    }
  }, [settings.promptTokens, settings.useCache, state.pass]);
  progressRef.current = state.progress;
  const settingsRef = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    if (state.status !== "running") return;
    let last = performance.now();
    const timer = window.setInterval(() => {
      const now = performance.now(),
        next = progressRef.current + Math.min(now - last, 160) / stepDuration;
      last = now;
      if (next >= 1) {
        progressRef.current = 0;
        dispatch({ type: "advance", steps: plan.length, passes });
        if (pace === "step") dispatch({ type: "pause" });
      } else {
        progressRef.current = next;
        dispatch({ type: "progress", value: next });
      }
    }, 50);
    return () => clearInterval(timer);
  }, [
    state.status,
    state.index,
    state.pass,
    state.run,
    stepDuration,
    pace,
    plan.length,
    passes,
  ]);
  useEffect(() => {
    setInspectId(null);
    setBrowseEvent(null);
    setTokenIndex(-1);
  }, [event.nodeId, state.pass, event.segment, state.run]);
  useEffect(() => {
    const pause = () => {
      if (document.hidden) dispatch({ type: "pause" });
    };
    document.addEventListener("visibilitychange", pause);
    return () => document.removeEventListener("visibilitychange", pause);
  }, []);
  useEffect(() => {
    const d = settingsRef.current;
    if (settingsOpen) {
      dispatch({ type: "pause" });
      d?.showModal();
    } else d?.close();
  }, [settingsOpen]);
  const changeSettings = (update: Partial<DemoSettings>) => {
    if (
      update.promptTokens !== undefined &&
      (!Number.isInteger(update.promptTokens) ||
        update.promptTokens < 1 ||
        update.promptTokens > 16)
    )
      return;
    if (
      update.outputTokens !== undefined &&
      (!Number.isInteger(update.outputTokens) ||
        update.outputTokens < 1 ||
        update.outputTokens > 8)
    )
      return;
    setInspectId(null);
    setBrowseEvent(null);
    setTrace(null);
    setTraceError("");
    setSettings((s) => ({ ...s, ...update }));
    dispatch({ type: "reset" });
  };
  const pauseSelect = (id: string) => {
    dispatch({ type: "pause" });
    setInspectId(id);
    setDetailOpen(true);
  };
  const step = () => {
    dispatch({ type: "pause" });
    dispatch({ type: "advance", steps: plan.length, passes });
  };
  const restart = () => {
    if (!valid) return;
    setInspectId(null);
    setBrowseEvent(null);
    setTokenIndex(-1);
    dispatch({ type: "restart" });
  };
  const skipLayer = () => {
    const next = plan.findIndex(
      (e, i) =>
        i > state.index &&
        (e.layer !== event.layer || e.loop !== event.loop || e.sample),
    );
    dispatch({ type: "seek", index: next < 0 ? plan.length - 1 : next });
  };
  const phase =
    state.pass === 0
      ? "Prefill"
      : settings.useCache
        ? "Decode"
        : "全前缀重算";
  const activeId = inspectId ? null : sceneNodeId;
  const phases = demoPhaseSteps(graphEvent, node);
  const omittedBefore =
    lessonRoute === "overview"
      ? fullPlan.indexOf(event) -
        (state.index > 0
          ? fullPlan.indexOf(plan[Math.min(state.index - 1, plan.length - 1)])
          : -1) -
        1
      : 0;
  const currentToken =
    state.status === "running" || tokenIndex < 0
      ? selectedTokenPosition(scenario, beat.computing)
      : Math.min(tokenIndex, settings.promptTokens + state.generated - 1);
  const tokens = [
    ...Array.from(
      { length: settings.promptTokens },
      (_, i) => trace?.prompt[i].text ?? examplePrompt[i] ?? `t${i + 1}`,
    ),
    ...(trace
      ? trace.generated.map((t) => t.text)
      : exampleContinuation
    ).slice(0, hasHead ? state.generated : 0),
  ];
  const seekLayer = (index: number) => {
    const next = fullPlan.findIndex(
      (e) =>
        e.view !== "overview" && e.layer === index && e.loop === event.loop,
    );
    if (next >= 0) {
      setLessonRoute("full");
      dispatch({ type: "seek", index: next });
      setInspectId(null);
    }
  };
  const componentStep = (offset: number) => {
    const nodes = graph.nodes.filter(
        (n) => !n.data.frame && !n.data.inactive,
      ),
      index = nodes.findIndex((n) => n.id === (inspectId ?? sceneNodeId));
    pauseSelect(nodes[(index + offset + nodes.length) % nodes.length].id);
  };
  const canvasProps = {
    model,
    scenario,
    tokenLabels: tokens.slice(scenario.past),
    headIndex: head,
    traceSamples: sceneSamples,
    recordedMode: !!trace,
    graph,
    selectedId: inspectId ?? sceneNodeId,
    activeId,
    selectedEdgeId: null,
    symbolic,
    showShapes: true,
    playing: state.status === "running",
    followPlayback: follow,
    reducedMotion,
    sceneKey: `demo:${state.run}:${state.pass}:${event.loop}:${event.layer}:${event.view === "overview" ? "overview" : "layer"}:${event.sample ? "sample" : ""}`,
    onSelect: pauseSelect,
    onSelectEdge: () => dispatch({ type: "pause" }),
    onDrill: pauseSelect,
    animationProgress: state.progress,
    immersive: true,
    worldLayers: mode === "3d" ? layers : undefined,
    worldLayer: graphEvent.view === "overview" ? undefined : graphEvent.layer,
    onWorldLayer: seekLayer,
    tokenCount: scenario.sequence,
    activeToken: currentToken - scenario.past,
    brightness,
    cameraGroup: `${state.run}:${state.pass}:${event.loop}:${event.layer}:${event.view}:${!!event.sample}`,
    stepDuration,
  };
  const shownTokens = hasHead ? state.generated : 0;
  return (
    <section
      className={`inference-demo demo-reference-layout model-native-demo ${detailOpen ? "with-demo-details" : ""}`}
      aria-label="示例推理演示"
      data-status={state.status}
      data-pass={state.pass}
      data-step={state.index}
      data-progress={state.progress.toFixed(3)}
      data-duration={stepDuration}
      data-pace={pace}
      onKeyDown={(e) => {
        const el = e.target as HTMLElement;
        if (/INPUT|SELECT|TEXTAREA|BUTTON/.test(el.tagName) || settingsOpen)
          return;
        if (e.code === "Space") {
          e.preventDefault();
          if (!valid) return;
          if (state.status !== "running") {
            setInspectId(null);
            setBrowseEvent(null);
          }
          dispatch({ type: state.status === "running" ? "pause" : "start" });
        }
        if (e.key.toLowerCase() === "r") {
          e.preventDefault();
          restart();
        }
      }}
    >
      <header className="demo-toolbar">
        <button
          className="demo-icon"
          aria-label="返回模型视图"
          title="返回模型视图"
          onClick={onClose}
        >
          <ArrowLeft size={17} />
        </button>
        <div className="demo-heading">
          <span>MODEL INFERENCE · 3D</span>
          <strong>{name}</strong>
        </div>
        {onChooseModel && (
          <button
            className="demo-model-switch"
            onClick={() => {
              dispatch({ type: "pause" });
              onChooseModel();
            }}
            aria-label="在 3D 演示中切换模型"
          >
            切换模型
          </button>
        )}
        <div className="demo-controls">
          <FullscreenButton
            fullscreen={fullscreen}
            className="demo-icon fullscreen-button"
          />
          <button
            className="demo-start"
            disabled={!valid}
            onClick={() => {
              setInspectId(null);
              setBrowseEvent(null);
              setTokenIndex(-1);
              dispatch({
                type: state.status === "running" ? "pause" : "start",
              });
            }}
          >
            {state.status === "running" ? (
              <Pause size={14} />
            ) : (
              <Play size={14} />
            )}
            {state.status === "running"
              ? "暂停"
              : state.status === "paused"
                ? "继续"
                : state.status === "complete"
                  ? "再次播放"
                  : "开始"}
          </button>
          <button
            className="demo-icon"
            disabled={!valid}
            aria-label="重新开始演示"
            title="重新开始"
            onClick={restart}
          >
            <RotateCcw size={16} />
          </button>
          <button
            className="demo-icon"
            disabled={state.status === "complete" || !valid}
            aria-label="演示下一步"
            title="下一步"
            onClick={step}
          >
            <SkipForward size={17} />
          </button>
          <button
            className="demo-follow"
            aria-label="跟随演示视角"
            aria-pressed={follow}
            onClick={() => setFollow(!follow)}
          >
            <Focus size={15} />
            <span>跟随</span>
          </button>
          <button
            className="demo-icon"
            aria-label={mode === "3d" ? "切换演示为 2D" : "切换演示为 3D"}
            title={mode === "3d" ? "2D View" : "3D View"}
            onClick={() => changeMode(mode === "3d" ? "2d" : "3d")}
          >
            {mode === "3d" ? <Grid2X2 size={16} /> : <Box size={16} />}
          </button>
          <button
            className="demo-icon"
            aria-label="演示设置"
            onClick={() => setSettingsOpen(true)}
          >
            <Settings2 size={17} />
          </button>
          <button
            className="demo-icon"
            aria-label="组件详细解释"
            aria-pressed={detailOpen}
            onClick={() => {
              dispatch({ type: "pause" });
              setDetailOpen(!detailOpen);
            }}
          >
            <PanelRight size={16} />
          </button>
        </div>
      </header>
      <div className="demo-token-bar">
        <span className="demo-prompt-label">示例 token</span>
        <div className="demo-tokens">
          {Array.from({ length: settings.promptTokens }, (_, i) => (
            <button
              key={`p${i}`}
              aria-label={`解释输入 token ${i + 1}`}
              onClick={() => {
                setTokenIndex(i);
                setDetailOpen(true);
                dispatch({ type: "pause" });
              }}
              className={
                state.pass === 0 || !settings.useCache
                  ? "processing"
                  : "cached"
              }
            >
              {trace?.prompt[i].text ?? examplePrompt[i] ?? ` t${i + 1}`}
            </button>
          ))}
          {Array.from({ length: shownTokens }, (_, i) => (
            <button
              key={`g${i}`}
              className={`demo-generated ${i === state.pass - 1 ? "processing" : ""}`}
              title={`查看第 ${i + 1} 轮`}
              onClick={() => dispatch({ type: "replay", pass: i })}
            >
              {trace?.generated[i].text ?? exampleContinuation[i]}
            </button>
          ))}
          {state.status === "running" && (
            <span className="demo-cursor" aria-hidden="true" />
          )}
        </div>
        <span className="demo-script-note" title={scopeNote}>
          {trace
            ? `导入运行记录 · ${trace.source}`
            : "固定教学脚本 · 非模型真实输出"}
        </span>
      </div>
      <div className="demo-body">
        <aside className="demo-hud">
          <div className="demo-phase">
            <span>{phase}</span>
            <small>
              第 {state.pass + 1} / {passes} 轮
            </small>
          </div>
          <div className="demo-layer-label">
            {event.sample
              ? "输出与续写"
              : event.view === "overview"
                ? "输入 / 输出"
                : `${model.deepseek ? (event.layer < model.deepseek.encoderLayers ? "Encoder" : "Decoder") : "Layer"} ${event.layer + 1} / ${model.layers}`}
            {event.loop > 0 ? ` · 第 ${event.loop + 1} 轮主干循环` : ""}
          </div>
          <span className="demo-narration-eyebrow">
            {stageLabel(graphEvent)}
          </span>
          {omittedBefore > 0 && (
            <p className="demo-skipped-note">
              代表路线已略过 {omittedBefore} 个计算步骤；可切换完整逐层查看。
            </p>
          )}
          <h2>{node?.data.title ?? event.title}</h2>
          <div className="demo-time-budget">
            <span>{inspectId ? "已暂停 · 自由查看" : paceLabels[pace]}</span>
            <span>
              {inspectId
                ? "可打开右侧详解"
                : `本步 ${(stepDuration / 1000).toFixed(1)} 秒`}
            </span>
          </div>
          {node &&
            (node.data.params > 0 ||
              node.data.parameterStatus === "unknown") && (
              <span className="demo-module-params">
                当前模块权重：
                {node.data.parameterStatus === "unknown"
                  ? "未细分"
                  : compactNumber(node.data.params)}
                {model.catalog ? "（结构估算）" : ""}
              </span>
            )}
          <div className="demo-operation-progress">
            <i style={{ width: `${state.progress * 100}%` }} />
          </div>
          <div className="demo-phase-steps">
            {phases.map((p, i) => (
              <button
                key={p}
                aria-current={beat.phase === i ? "step" : undefined}
                onClick={() => {
                  dispatch({ type: "seek", index: state.index });
                  dispatch({
                    type: "set-progress",
                    value: demoPhaseStarts[i],
                  });
                }}
              >
                <span>{i + 1}</span>
                {p}
              </button>
            ))}
          </div>
          {showEquations && (
            <code className="demo-equation">
              {node?.data.formula ||
                node?.data.subtitle ||
                "按模型配置推进当前模块"}
            </code>
          )}
          <p className="demo-explanation">{node?.data.description}</p>
          <button
            className="demo-read-more"
            onClick={() => {
              setDetailOpen(true);
              dispatch({ type: "pause" });
            }}
          >
            查看完整原理与矩阵 <ChevronRight size={13} />
          </button>
          <div className="demo-dimensions">
            <span>
              输入 → 输出{" "}
              <button
                aria-label="演示符号维度"
                aria-pressed={symbolic}
                onClick={() => setSymbolic(!symbolic)}
              >
                <Braces size={14} />
              </button>
            </span>
            {[
              ...(node?.data.inputs ?? []).slice(0, 1),
              ...(node?.data.outputs ?? []).slice(0, 1),
            ].map((t, i) => (
              <div key={`${i}:${t.name}`}>
                <small>{t.name}</small>
                <code>[{(symbolic ? t.symbols : t.shape).join(", ")}]</code>
              </div>
            ))}
          </div>
          <div className="demo-cache">
            <span>
              {settings.useCache ? "KV cache 已启用" : "KV cache 已关闭"}
            </span>
            <strong data-testid="demo-cache-bytes">
              {formatBytes(cache.totalBytes)}
            </strong>
            <small>
              {cache.partial ? "已知状态小计 · " : ""}
              {settings.useCache
                ? `轮次边界已保留 ${cache.tokens} 个 token`
                : "跨轮不保留，每轮从完整前缀重算"}
            </small>
            <details className="demo-cache-details">
              <summary>查看缓存构成</summary>
              <dl>
                <dt>KV</dt>
                <dd>{formatBytes(cache.kvBytes)}</dd>
                <dt>递归 / 辅助状态</dt>
                <dd>{formatBytes(cache.otherBytes)}</dd>
                <dt>本轮完成时理论容量</dt>
                <dd>{formatBytes(cache.targetBytes)}</dd>
              </dl>
            </details>
          </div>
          <p className="demo-visual-provenance">
            矩阵 / 向量图元源自{" "}
            <a
              href="https://github.com/saahithjanapati/llm-visualized"
              target="_blank"
              rel="noreferrer"
            >
              LLM-Visualized · ISC
            </a>
            。
            {trace
              ? "已载入采样值，未采样坐标保持灰色。"
              : "向量色带、注意力明暗与专家路线为教学示意。"}
          </p>
        </aside>
        <div className={`demo-scene demo-scene-${mode}`}>
          {state.status !== "ready" && (
            <div className="demo-scene-caption" aria-live="off">
              <span className="demo-scene-step">
                {String(beat.phase + 1).padStart(2, "0")} / 03
              </span>
              <div>
                <strong>{phases[beat.phase]}</strong>
                <small>
                  {beat.arrival
                    ? "镜头就位，先看输入"
                    : beat.hold
                      ? "结果已就绪，准备进入下一步"
                      : `Token ${currentToken + 1} · ${playingNode?.data.outputs[0]?.symbols.join(" × ") ?? "模型接口"}`}
                </small>
              </div>
              <span className="demo-scene-phase">
                {state.status === "paused"
                  ? "已暂停"
                  : beat.hold
                    ? "停留阅读"
                    : "演示中"}
              </span>
            </div>
          )}
          <label className="demo-world-navigation">
            模型层级
            <select
              aria-label="演示模型层级"
              value={event.view === "overview" ? "overview" : event.layer}
              onChange={(e) => {
                if (e.target.value === "overview") {
                  dispatch({ type: "seek", index: 0 });
                } else seekLayer(Number(e.target.value));
              }}
            >
              <option value="overview">输入 / 输出</option>
              {layers.map((l) => (
                <option key={l.index} value={l.index}>
                  {l.label}
                </option>
              ))}
            </select>
          </label>
          <Suspense
            fallback={<div className="demo-loading">正在准备模型演示…</div>}
          >
            {mode === "3d" ? (
              <Space {...canvasProps} onFallback={() => changeMode("2d")} />
            ) : (
              <Plane {...canvasProps} />
            )}
          </Suspense>
          {state.status === "ready" && (
            <div className="demo-intro">
              <Play size={24} />
              <h2>跟随一个 token，读懂一次推理</h2>
              <p>
                {settings.promptTokens} 个示例 token → {model.layers} 层主干 →{" "}
                {hasHead ? `${settings.outputTokens} 轮续写` : "隐藏状态输出"}
              </p>
              <p
                className="demo-route-budget"
                data-testid="demo-route-budget"
              >
                {routeLabels[lessonRoute]} · 预计{" "}
                {(estimatedMs / 60000).toFixed(1)} 分钟
                {lessonRoute === "overview" &&
                  " · 展示代表层与异构边界，其余计算步骤省略"}
              </p>
              <div className="demo-pace-choices" aria-label="选择演示节奏">
                {(["guided", "continuous", "step"] as const).map((p) => (
                  <button
                    key={p}
                    aria-pressed={pace === p}
                    onClick={() => setPace(p)}
                  >
                    <strong>{paceLabels[p]}</strong>
                    <span>
                      {p === "guided"
                        ? "每步 4–8 秒，留出阅读时间"
                        : p === "continuous"
                          ? "熟悉流程后更快浏览"
                          : "每步结束后自动暂停"}
                    </span>
                  </button>
                ))}
              </div>
              <button
                className="demo-start"
                disabled={!valid}
                onClick={() => dispatch({ type: "start" })}
              >
                开始演示
              </button>
              <small>
                原站同源图元 · 结构与维度来自当前模型 · 颜色与 token
                为教学示意
              </small>
            </div>
          )}
          {state.status === "complete" && (
            <div className="demo-complete" role="status">
              <Check size={18} />
              <span>
                {hasHead
                  ? "示例演示完成"
                  : "隐藏状态前向演示完成（无 LM Head）"}
              </span>
              <button onClick={restart}>重新开始</button>
            </div>
          )}
        </div>
        {detailOpen && node && (
          <DemoInspector
            key={`${graphEvent.view}:${graphEvent.layer}`}
            node={node}
            model={model}
            scenario={scenario}
            tokenLabels={tokens}
            tokenIndex={currentToken}
            onToken={(i) => {
              dispatch({ type: "pause" });
              setTokenIndex(i);
            }}
            head={head}
            onHead={setHead}
            onPrevious={() => componentStep(-1)}
            onNext={() => componentStep(1)}
            onClose={() => setDetailOpen(false)}
            onMatrix={() => changeMode("2d")}
            devMode={devMode}
            traceTokens={
              trace
                ? [
                    ...trace.prompt,
                    ...trace.generated.slice(0, state.generated),
                  ]
                : undefined
            }
            traceSource={trace?.source}
            samples={inspectorSamples}
          />
        )}
      </div>
      <footer className="demo-timeline">
        <span className="demo-mobile-disclaimer">
          {trace ? "正在回放导入记录" : "教学脚本 · 非模型真实输出"}
        </span>
        <div className="demo-timeline-top">
          <span>
            {state.status === "ready"
              ? "准备就绪"
              : state.status === "running"
                ? "演示中"
                : state.status === "paused"
                  ? "已暂停"
                  : "已完成"}{" "}
            · {routeLabels[lessonRoute]} · 步骤{" "}
            {Math.min(state.index + 1, plan.length)} / {plan.length}· 预计{" "}
            {(estimatedMs / 60000).toFixed(1)} 分钟
          </span>
          <span data-testid="demo-shape-scenario">
            B=1 · S={scenario.sequence} · P={scenario.past} · {dtype}
          </span>
        </div>
        <input
          type="range"
          aria-label="演示进度"
          min="0"
          max={plan.length - 1}
          value={state.index}
          onChange={(e) =>
            dispatch({ type: "seek", index: Number(e.target.value) })
          }
        />
        <div className="demo-timeline-actions">
          <label>
            路线
            <select
              aria-label="演示路线"
              value={lessonRoute}
              disabled={!!trace}
              onChange={(e) => {
                setLessonRoute(e.target.value as DemoRoute);
                setInspectId(null);
                setBrowseEvent(null);
                dispatch({ type: "reset" });
              }}
            >
              <option value="overview">代表层讲解</option>
              <option value="full">完整逐层</option>
            </select>
          </label>
          <label className="demo-pace-select">
            节奏
            <select
              aria-label="演示节奏"
              value={pace}
              onChange={(e) => setPace(e.target.value as DemoPace)}
            >
              {Object.entries(paceLabels).map(([id, label]) => (
                <option key={id} value={id}>
                  {label}
                </option>
              ))}
            </select>
          </label>
          <button disabled={state.status === "complete"} onClick={skipLayer}>
            跳过当前层
            <ChevronRight size={12} />
          </button>
          <button
            disabled={
              state.status === "complete" || event.view !== "attention"
            }
            onClick={() => {
              const next = plan.findIndex(
                (e, i) => i > state.index && e.view !== "attention",
              );
              dispatch({
                type: "seek",
                index: next < 0 ? plan.length - 1 : next,
              });
            }}
          >
            跳过注意力
          </button>
          <button
            disabled={state.status === "complete"}
            onClick={() => dispatch({ type: "seek", index: plan.length - 1 })}
          >
            跳到本轮输出
          </button>
          <button
            disabled={state.status === "complete"}
            onClick={() =>
              dispatch({ type: "next-pass", passes, steps: plan.length })
            }
          >
            下一轮
          </button>
          <button
            disabled={state.pass === passes - 1}
            onClick={() => dispatch({ type: "replay", pass: passes - 1 })}
          >
            最后一轮
          </button>
          <label>
            速度
            <select
              aria-label="演示速度"
              value={settings.speed}
              onChange={(e) =>
                setSettings((s) => ({ ...s, speed: Number(e.target.value) }))
              }
            >
              <option value={0.25}>0.25×</option>
              <option value={0.5}>0.5×</option>
              <option value={0.75}>0.75×</option>
              <option value={1}>1×</option>
              <option value={2}>2×</option>
              <option value={4}>4×</option>
            </select>
          </label>
        </div>
      </footer>
      <dialog
        ref={settingsRef}
        className="demo-settings"
        aria-labelledby="demo-settings-title"
        onCancel={() => setSettingsOpen(false)}
      >
        <header>
          <h2 id="demo-settings-title">演示设置</h2>
          <button
            className="demo-icon"
            aria-label="关闭演示设置"
            onClick={() => setSettingsOpen(false)}
          >
            <X size={19} />
          </button>
        </header>
        <div className="demo-trace-controls">
          <strong>运行数据</strong>
          <p>
            {trace
              ? `当前来源：${trace.source}`
              : "可导入与当前配置匹配的 token 与张量采样记录，数据仅在浏览器本地读取。"}
          </p>
          <input
            ref={traceInput}
            hidden
            type="file"
            accept="application/json,.json"
            aria-label="导入运行记录文件"
            onChange={async (e) => {
              const file = e.target.files?.[0];
              if (!file) return;
              try {
                if (file.size > 8 * 1024 * 1024)
                  throw new Error("文件超过 8 MiB 上限。");
                const parsed = parseDemoTrace(await file.text(), model);
                setInspectId(null);
                setBrowseEvent(null);
                setTrace(parsed);
                setLessonRoute("full");
                setSettings((s) => ({ ...s, ...parsed.settings }));
                dispatch({ type: "reset" });
                setTraceError("");
              } catch (error) {
                setTraceError(
                  error instanceof Error ? error.message : "无法读取记录",
                );
              }
              e.target.value = "";
            }}
          />
          <button onClick={() => traceInput.current?.click()}>
            导入运行记录
          </button>
          <button
            onClick={() => {
              const url = URL.createObjectURL(
                new Blob(
                  [JSON.stringify(traceTemplate(model, settings), null, 2)],
                  { type: "application/json" },
                ),
              );
              const a = document.createElement("a");
              a.href = url;
              a.download = "inference-trace-template.json";
              a.click();
              window.setTimeout(() => URL.revokeObjectURL(url), 500);
            }}
          >
            下载记录模板
          </button>
          {trace && (
            <button
              onClick={() => {
                setInspectId(null);
                setBrowseEvent(null);
                setTrace(null);
                dispatch({ type: "reset" });
              }}
            >
              移除记录
            </button>
          )}
          {traceError && <p role="alert">{traceError}</p>}
        </div>
        <label>
          提示 token 数
          <input
            aria-label="演示提示 token 数"
            type="number"
            min="1"
            max="16"
            value={settings.promptTokens}
            onChange={(e) =>
              changeSettings({ promptTokens: Number(e.target.value) })
            }
          />
        </label>
        <label>
          续写轮数
          <input
            aria-label="演示续写轮数"
            type="number"
            min="1"
            max="8"
            disabled={!hasHead}
            value={settings.outputTokens}
            onChange={(e) =>
              changeSettings({ outputTokens: Number(e.target.value) })
            }
          />
        </label>
        <label className="demo-cache-toggle">
          启用 KV cache
          <input
            aria-label="启用演示 KV cache"
            type="checkbox"
            checked={settings.useCache}
            onChange={(e) => changeSettings({ useCache: e.target.checked })}
          />
        </label>
        <label>
          显示公式
          <input
            type="checkbox"
            aria-label="显示演示公式"
            checked={showEquations}
            onChange={(e) => setShowEquations(e.target.checked)}
          />
        </label>
        <label>
          开发数据模式
          <input
            type="checkbox"
            aria-label="显示开发数据"
            checked={devMode}
            onChange={(e) => setDevMode(e.target.checked)}
          />
        </label>
        <label>
          场景亮度 {brightness}%
          <input
            type="range"
            aria-label="演示场景亮度"
            min="45"
            max="150"
            value={brightness}
            onChange={(e) => setBrightness(Number(e.target.value))}
          />
        </label>
        <p>
          修改提示长度、轮数或缓存模式会重置演示。开启缓存后续轮次
          S=1；关闭后每轮重算全部前缀。递归模型沿用自身的状态规则。
        </p>
        {!valid && (
          <p role="alert">
            提示数需为 1–16、轮数需为 1–8，且总长度不能超过模型上下文。
          </p>
        )}
        <p>
          示例 token 不使用模型
          tokenizer；不提供真实概率或激活值。批大小固定为
          1，媒体与草稿分支不在此文本演示中。
        </p>
        <button
          className="demo-start"
          disabled={!valid}
          onClick={() => setSettingsOpen(false)}
        >
          返回演示
        </button>
      </dialog>
    </section>
  );
}
