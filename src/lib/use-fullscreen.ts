import { useCallback, useEffect, useRef, useState } from "react";

export interface FullscreenControl {
  active: boolean;
  pending: boolean;
  mode: "none" | "native" | "expanded";
  toggle: () => Promise<void>;
}

/** Native fullscreen when available; viewport expansion on unsupported browsers. */
export function useFullscreen<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [mode, setMode] = useState<FullscreenControl["mode"]>("none");
  const [pending, setPending] = useState(false);
  const opener = useRef<HTMLElement | null>(null);
  const restoreFocus = useCallback(() => {
    requestAnimationFrame(() => {
      if (opener.current?.isConnected)
        opener.current.focus({ preventScroll: true });
    });
  }, []);

  useEffect(() => {
    const sync = () => {
      if (document.fullscreenElement === ref.current && ref.current)
        setMode("native");
      else
        setMode((previous) => {
          if (previous === "native") {
            restoreFocus();
            return "none";
          }
          return previous;
        });
    };
    document.addEventListener("fullscreenchange", sync);
    return () => document.removeEventListener("fullscreenchange", sync);
  }, [restoreFocus]);

  useEffect(() => {
    if (mode === "none") return;
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const escape = (event: KeyboardEvent) => {
      if (event.key === "Escape" && mode === "expanded") {
        event.preventDefault();
        event.stopPropagation();
        setMode("none");
        restoreFocus();
      }
    };
    document.addEventListener("keydown", escape, true);
    return () => {
      document.body.style.overflow = previous;
      document.removeEventListener("keydown", escape, true);
    };
  }, [mode, restoreFocus]);

  const toggle = async () => {
    if (pending || !ref.current) return;
    if (mode === "expanded") {
      setMode("none");
      restoreFocus();
      return;
    }
    setPending(true);
    try {
      if (document.fullscreenElement === ref.current) {
        try {
          await document.exitFullscreen();
          setMode("none");
          restoreFocus();
        } catch {
          // Keep the exit control available if the browser rejects the request.
          if (!document.fullscreenElement) setMode("none");
        }
      } else {
        opener.current =
          document.activeElement instanceof HTMLElement
            ? document.activeElement
            : null;
        try {
          if (!ref.current.requestFullscreen)
            throw new Error("Fullscreen unavailable");
          await ref.current.requestFullscreen();
          setMode("native");
        } catch {
          setMode("expanded");
        }
      }
    } finally {
      setPending(false);
    }
  };
  return { ref, mode, active: mode !== "none", pending, toggle };
}
