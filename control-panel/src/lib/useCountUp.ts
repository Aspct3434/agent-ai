import { useEffect, useRef, useState } from "react";

/**
 * Animate a number from 0 up to `target` once on mount / when target changes.
 * Respects prefers-reduced-motion (snaps to the value).
 */
export function useCountUp(target: number, durationMs = 600): number {
  const [value, setValue] = useState(0);
  const frame = useRef<number | undefined>(undefined);

  useEffect(() => {
    const reduced =
      typeof window !== "undefined" &&
      Boolean(window.matchMedia?.("(prefers-reduced-motion: reduce)").matches);
    const start = performance.now();
    const tick = (now: number) => {
      if (reduced) {
        setValue(target);
        return;
      }
      const t = Math.min(1, (now - start) / durationMs);
      const eased = 1 - Math.pow(1 - t, 3); // ease-out cubic
      setValue(Math.round(target * eased));
      if (t < 1) frame.current = requestAnimationFrame(tick);
    };
    frame.current = requestAnimationFrame(tick);
    return () => {
      if (frame.current !== undefined) cancelAnimationFrame(frame.current);
    };
  }, [target, durationMs]);

  return value;
}
