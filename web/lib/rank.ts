export type Rank = "best" | "worst" | "mid" | "none";

/**
 * Marks the best value in a column (all ties are best) and, when there are at least
 * three distinct values, the worst. null / non-finite values are not ranked.
 */
export function rankColumn(values: (number | null | undefined)[], better: "lower" | "higher"): Rank[] {
  const finite = values.filter((value): value is number => value != null && Number.isFinite(value));
  if (finite.length === 0) return values.map(() => "none");
  const best = better === "lower" ? Math.min(...finite) : Math.max(...finite);
  const worst = better === "lower" ? Math.max(...finite) : Math.min(...finite);
  const distinct = new Set(finite.map((value) => value.toFixed(6))).size;
  const eps = 1e-9;
  return values.map((value) => {
    if (value == null || !Number.isFinite(value)) return "none";
    if (Math.abs(value - best) <= eps) return "best";
    if (distinct >= 3 && Math.abs(value - worst) <= eps) return "worst";
    return "mid";
  });
}

/** Shared cell styles: green highlight + ✓ for best, muted red for worst. */
export const RANK_CLASS: Record<Rank, string> = {
  best: "rounded-lg bg-teal/15 px-2 py-1 font-semibold text-teal",
  worst: "rounded-lg px-2 py-1 text-danger/80",
  mid: "px-2 py-1 text-ivory",
  none: "px-2 py-1 text-ivory",
};

/** Accent for the GeneAI row in any comparison table. */
export const GENEAI_ROW = "bg-teal/[0.06] shadow-[inset_3px_0_0_var(--teal)]";
