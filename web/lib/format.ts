export function formatFitness(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "—";
  return value.toFixed(4);
}

export function formatMetric(value: number): string {
  if (!Number.isFinite(value)) return "—";
  const abs = Math.abs(value);
  if (abs >= 100) return value.toFixed(1);
  if (abs >= 10) return value.toFixed(2);
  if (abs >= 1) return value.toFixed(3);
  return value.toFixed(4);
}

export function formatSeconds(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "—";
  if (value < 10) return `${value.toFixed(2)}s`;
  if (value < 100) return `${value.toFixed(1)}s`;
  return `${Math.round(value)}s`;
}

export function formatUsd(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "—";
  return `$${value.toFixed(value === 0 ? 2 : 4)}`;
}

export function shortHash(value: string | null | undefined, size = 10): string {
  if (!value) return "—";
  if (value.length <= size * 2) return value;
  return `${value.slice(0, size)}…${value.slice(-4)}`;
}

export function familyOf(metadata: { family?: string | null } | undefined): string {
  const family = metadata?.family;
  return family && family.length > 0 ? family : "unspecified";
}
