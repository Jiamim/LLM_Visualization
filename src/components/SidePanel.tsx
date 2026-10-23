import { useEffect, useRef, useState, type ReactNode } from "react";

export function useMediaQuery(query: string) {
  const [matches, setMatches] = useState(
    () => window.matchMedia(query).matches,
  );
  useEffect(() => {
    const media = window.matchMedia(query);
    const update = () => setMatches(media.matches);
    media.addEventListener("change", update);
    update();
    return () => media.removeEventListener("change", update);
  }, [query]);
  return matches;
}

export default function SidePanel({
  open,
  modal,
  side,
  onClose,
  children,
}: {
  open: boolean;
  modal: boolean;
  side: "left" | "right";
  onClose: () => void;
  children: ReactNode;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    if (!modal) return;
    if (open) ref.current?.showModal();
    else ref.current?.close();
  }, [open, modal]);
  if (!modal) return open ? children : null;
  return (
    <dialog
      ref={ref}
      className={`panel-dialog panel-dialog-${side}`}
      aria-label={side === "left" ? "模型与输入" : "模块详情"}
      onCancel={onClose}
      onClick={(event) => {
        if (event.target === ref.current) onClose();
      }}
    >
      {children}
    </dialog>
  );
}
