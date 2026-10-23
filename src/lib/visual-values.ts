export const UNKNOWN_VALUE_COLOR = "#3e4650";

/** Recorded magnitudes use separate hues for signs; zero remains neutral. */
export function recordedValueColor(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value))
    return UNKNOWN_VALUE_COLOR;
  if (value === 0) return "#a4afbb";
  const lightness = 30 + 35 * Math.min(1, Math.abs(value) / 2);
  return `hsl(${value > 0 ? 155 : 340}, 55%, ${lightness}%)`;
}
