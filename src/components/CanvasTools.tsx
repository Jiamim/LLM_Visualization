import { useEffect, useRef, useState } from "react";
import {
  Braces,
  Download,
  Ellipsis,
  Maximize2,
  Minimize2,
  PanelRight,
  RotateCcw,
} from "lucide-react";

export default function CanvasTools({
  focusMode,
  onExport,
  onFocus,
  onReset,
  compact,
  symbolic,
  onSymbol,
  onInspector,
}: {
  focusMode: boolean;
  onExport: () => void;
  onFocus: () => void;
  onReset: () => void;
  compact: boolean;
  symbolic: boolean;
  onSymbol: () => void;
  onInspector: () => void;
}) {
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null),
    trigger = useRef<HTMLButtonElement>(null),
    menu = useRef<HTMLDivElement>(null);
  const close = () => {
    setOpen(false);
    trigger.current?.focus();
  };
  useEffect(() => {
    if (!open) return;
    menu.current?.querySelector<HTMLButtonElement>("button")?.focus();
    const outside = (event: PointerEvent) => {
      if (!root.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("pointerdown", outside);
    return () => document.removeEventListener("pointerdown", outside);
  }, [open]);
  const actions = [
    ...(compact
      ? [
          {
            label: `切换符号维度，当前${symbolic ? "符号" : "数值"}`,
            detail: symbolic ? "当前符号" : "当前数值",
            display: "切换符号维度",
            Icon: Braces,
            run: onSymbol,
          },
          {
            label: "切换详情面板",
            detail: "模块与维度",
            Icon: PanelRight,
            run: onInspector,
          },
        ]
      : []),
    { label: "导出当前计算图", detail: "JSON", Icon: Download, run: onExport },
    {
      label: focusMode ? "退出专注模式" : "专注模式",
      detail: "Esc 退出",
      Icon: focusMode ? Minimize2 : Maximize2,
      run: onFocus,
    },
    { label: "重置数据流", detail: "回到开始", Icon: RotateCcw, run: onReset },
  ];
  return (
    <div
      ref={root}
      className="canvas-tools"
      onBlur={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget)) setOpen(false);
      }}
    >
      <button
        ref={trigger}
        className="icon-button"
        aria-label="更多画布工具"
        aria-haspopup="menu"
        aria-expanded={open}
        title="更多工具"
        onClick={() => setOpen(!open)}
      >
        <Ellipsis size={18} />
      </button>
      {open && (
        <div
          ref={menu}
          className="canvas-tools-menu"
          role="menu"
          aria-label="画布工具"
          onKeyDown={(event) => {
            if (event.key === "Escape") {
              event.stopPropagation();
              event.preventDefault();
              close();
              return;
            }
            const buttons = Array.from(
              menu.current?.querySelectorAll<HTMLButtonElement>("button") ?? [],
            );
            const index = buttons.indexOf(
              document.activeElement as HTMLButtonElement,
            );
            if (["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)) {
              event.preventDefault();
              buttons[
                event.key === "Home"
                  ? 0
                  : event.key === "End"
                    ? buttons.length - 1
                    : (index +
                        (event.key === "ArrowDown" ? 1 : -1) +
                        buttons.length) %
                      buttons.length
              ]?.focus();
            }
          }}
        >
          <span className="tools-menu-caption">画布工具</span>
          {actions.map((action) => {
            const { label, detail, Icon, run } = action;
            return (
              <button
                key={label}
                role="menuitem"
                aria-label={label}
                onClick={() => {
                  close();
                  run();
                }}
              >
                <Icon size={16} />
                <span>{"display" in action ? action.display : label}</span>
                <small>{detail}</small>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
