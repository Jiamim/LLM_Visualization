import { Maximize, Minimize } from "lucide-react";
import type { FullscreenControl } from "../lib/use-fullscreen";

export default function FullscreenButton({
  fullscreen,
  className,
}: {
  fullscreen: FullscreenControl;
  className: string;
}) {
  const label = fullscreen.active ? "退出全屏" : "全屏查看";
  return (
    <button
      type="button"
      className={className}
      aria-label={label}
      title={fullscreen.active ? "退出全屏 · Esc" : label}
      aria-pressed={fullscreen.active}
      disabled={fullscreen.pending}
      onClick={() => void fullscreen.toggle()}
    >
      {fullscreen.active ? <Minimize size={17} /> : <Maximize size={17} />}
    </button>
  );
}
