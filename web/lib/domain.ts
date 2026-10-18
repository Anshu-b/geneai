import type { ShoppingItem } from "./types";

const METRIC_LABELS: Record<string, string> = {
  mean_completion: "Mean completion",
  sla_miss_rate: "SLA miss rate",
  hard_tardiness: "Hard-deadline tardiness",
  hard_tardiness_ratio_vs_round_robin: "Tardiness vs round robin",
  completion_ratio_vs_round_robin: "Completion vs round robin",
  energy: "Energy",
  energy_ratio_vs_round_robin: "Energy vs round robin",
  violations: "Capacity violations",
  workloads: "Workloads",
  gpu_foreign_work: "GPU foreign work",
  highmem_foreign_work: "High-memory foreign work",
  availability_confidence: "Availability confidence",
  budget_efficiency: "Budget efficiency",
  color_compatibility: "Color compatibility",
  occasion_fit: "Occasion fit",
  purchasable: "Purchasable",
  shoppers: "Shoppers",
  spend: "Spend",
  style_match: "Style match",
  subjective_rubric: "Subjective rubric flag",
};

export function metricLabel(key: string): string {
  return METRIC_LABELS[key] ?? key.replaceAll("_", " ");
}

export function fitnessHelp(domain: string): string {
  if (domain === "cloud_scheduling") {
    return "Higher is better. Cloud fitness is negative: the evaluator penalizes completion time, SLA misses, hard-deadline tardiness, and energy. Capacity violations are infeasible.";
  }
  if (domain === "shopping") {
    return "Higher is better. Shopping fitness is a subjective rubric averaged across shoppers on that split. It is not comparable to cloud milliseconds.";
  }
  return "Higher is better under the fitness recorded in this export.";
}

export function outfitItems(payload: Record<string, unknown>): ShoppingItem[] {
  const items = payload.items;
  if (!Array.isArray(items)) return [];
  return items.filter((item): item is ShoppingItem => Boolean(item) && typeof item === "object");
}

export function isFixtureProduct(item: ShoppingItem): boolean {
  if (item.fixture === true) return true;
  return (item.product_id ?? "").startsWith("fx_");
}

const SWATCH: Record<string, string> = {
  black: "#191919",
  bone: "#e6dcc8",
  sage: "#8ea184",
  navy: "#243056",
  neon: "#d6f25c",
};

export function colorSwatch(colorId: string | undefined): string | null {
  if (!colorId) return null;
  return SWATCH[colorId] ?? null;
}
