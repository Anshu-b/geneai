// Types and pure helpers for the Liquid showcase (geneai-liquid-v1).
// Every number shown comes from this file's input JSON, which the Python evaluator wrote.

export interface LiquidLot {
  lot_id: string;
  symbol: string;
  market_value_cents: number;
  basis_cents: number;
  holding_period: string;
  sellable: boolean;
}

export interface LiquidScenario {
  scenario_id: string;
  purchase_cents: number;
  horizon_days: number;
  checking: { balance_cents: number; minimum_cents: number };
  savings: { balance_cents: number; minimum_cents: number; apy: number };
  lots: LiquidLot[];
  credit: { approved: boolean; maximum_draw_cents: number; apr: number; fee_cents: number };
  assumptions: { long_term_gain_rate: number; short_term_gain_rate: number; loss_tax_benefit_mode: string; trade_fee_cents: number };
  constraints: { allow_borrowing: boolean; max_modeled_borrow_cents: number; protect: string[] };
}

export interface StressResult {
  rate_shock: { label: string; passed: boolean; extra_interest_cents: number };
  market_drop: { label: string; passed: boolean; portfolio_after_drop_cents: number; loan_to_value_after_drop: number | null };
  emergency: { label: string; passed: boolean; cash_above_reserve_cents: number; shortfall_cents: number };
  post_purchase: { checking_cents: number; savings_cents: number };
}

export interface ScenarioEval {
  scenario_id: string;
  feasible: boolean;
  constraint_violations: string[];
  allocation_cents: Record<string, number>;
  costs_cents: { estimated_tax: number; foregone_savings_interest: number; borrowing_interest: number; fees: number };
  total_cost_cents: number;
  realized_gain_cents: number;
  realized_loss_cents: number;
  stress: StressResult;
  risk_breakdown_cents: Record<string, number>;
  risk_penalty_cents: number;
  stress_passes: number;
  fitness: number | null;
}

export interface SplitEval {
  feasible: boolean;
  fitness: number | null;
  metrics: Record<string, number>;
}

export interface LiquidCandidate {
  key: string;
  label: string;
  provider: string;
  valid: boolean;
  invalid_reason?: string | null;
  policy?: Record<string, unknown>;
  summary?: string;
  base?: ScenarioEval;
  splits?: Record<"train" | "validation" | "test", SplitEval>;
  oneshot_meta?: Record<string, unknown>;
}

export interface LiquidGeneration {
  generation: number;
  born: number;
  by_provider: Record<string, number>;
  best_fitness: number | null;
  best_so_far: number | null;
}

export interface LiquidFile {
  schema: "geneai-liquid-v1";
  mode: string;
  run: {
    id: string;
    mode: string;
    recorded_at?: string;
    stop_reason?: string;
    generations_completed?: number;
    population_size?: number;
    wall_seconds?: number;
    model_calls?: number;
    evaluations?: number;
    champion_id: string;
    evaluator_version: string;
  };
  scenario: LiquidScenario;
  variants: Record<string, { scenario_id: string; note: string }[]>;
  model: Record<string, string | number>;
  candidates: LiquidCandidate[];
  generations: LiquidGeneration[];
  champion_lineage: { id: string; generation: number; operator: string; provider: string; parent_ids: string[]; hypothesis: string; train_fitness: number | null; baseline_name?: string | null }[];
  provenance: {
    genomes_by_provider: Record<string, number>;
    behavior_origin: { id: string; provider: string; operator: string; generation: number; provenance?: string | null };
    champion_hypothesis?: string;
    providers: { name: string; model: string | null; mode: string }[];
  };
}

export function parseLiquid(raw: unknown): LiquidFile {
  if (!raw || typeof raw !== "object") throw new Error("Liquid file is not an object.");
  const obj = raw as Record<string, unknown>;
  if (obj.schema !== "geneai-liquid-v1") throw new Error("Liquid file has the wrong schema.");
  for (const key of ["run", "scenario", "candidates", "generations", "provenance"]) {
    if (!(key in obj)) throw new Error(`Liquid file is missing ${key}.`);
  }
  if (!Array.isArray(obj.candidates)) throw new Error("candidates must be an array.");
  return raw as LiquidFile;
}

export function usd(cents: number, digits = 2): string {
  const sign = cents < 0 ? "-" : "";
  return `${sign}$${(Math.abs(cents) / 100).toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits })}`;
}

export const SOURCE_LABEL: Record<string, string> = {
  checking: "Checking",
  savings: "Savings",
  "nvda-1": "NVDA",
  "aapl-1": "AAPL",
  "msft-1": "MSFT",
  "etf-1": "ETF",
  credit: "Credit line",
};

export const SOURCE_COLOR: Record<string, string> = {
  checking: "var(--amber)",
  savings: "var(--teal)",
  "aapl-1": "var(--violet)",
  "msft-1": "var(--edge-rose)",
  "etf-1": "var(--prov-veri)",
  "nvda-1": "var(--faint)",
  credit: "var(--danger)",
};

export type Better = "lower" | "higher";

/** Indices of the best value in a column. Ties all win. Nulls never win. */
export function bestIndices(values: (number | null | undefined)[], better: Better): Set<number> {
  const real = values.map((v, i) => [v, i] as const).filter(([v]) => typeof v === "number" && Number.isFinite(v)) as [number, number][];
  if (real.length === 0) return new Set();
  const target = better === "lower" ? Math.min(...real.map(([v]) => v)) : Math.max(...real.map(([v]) => v));
  return new Set(real.filter(([v]) => Math.abs(v - target) < 1e-9).map(([, i]) => i));
}

export function totalAssetsCents(s: LiquidScenario): number {
  return s.checking.balance_cents + s.savings.balance_cents + s.lots.reduce((sum, lot) => sum + lot.market_value_cents, 0);
}

/** One honest sentence comparing GeneAI with the best other method on held-out test fitness. */
export function takeaway(file: LiquidFile): string {
  const geneai = file.candidates.find((c) => c.key === "geneai");
  const others = file.candidates.filter((c) => c.key !== "geneai" && c.valid && c.splits?.test.fitness != null);
  const g = geneai?.splits?.test.fitness;
  if (g == null || others.length === 0) return "Held-out comparison not recorded.";
  const best = others.reduce((a, b) => ((a.splits!.test.fitness as number) >= (b.splits!.test.fitness as number) ? a : b));
  const b = best.splits!.test.fitness as number;
  const gap = Math.abs(g - b).toFixed(2);
  const gv = geneai?.splits?.validation.fitness;
  const ov = others.map((c) => c.splits?.validation.fitness).filter((v): v is number => typeof v === "number");
  const valNote = gv != null && ov.length ? ` On validation, where the champion was chosen, it scores ${gv.toFixed(2)} vs a best alternative of ${Math.max(...ov).toFixed(2)}.` : "";
  const twin = others.find((c) => c.key !== "one_shot" && (["train", "validation", "test"] as const).every((sp) => c.splits?.[sp].fitness === geneai?.splits?.[sp].fitness));
  const twinNote = twin ? ` Its scores equal the ${twin.label} baseline on every split, so on these scenarios the champion behaves like that rule.` : "";
  return headline() + valNote + twinNote;
  function headline(): string {
  if (g! > b + 1e-9) return `On the held-out test scenarios GeneAI's policy scores ${g!.toFixed(2)}, ${gap} better than the best alternative (${best.label}, ${b.toFixed(2)}).`;
  if (Math.abs(g! - b) <= 1e-9) return `On the held-out test scenarios GeneAI's policy ties the best alternative (${best.label}) at ${g!.toFixed(2)}.`;
  return `On the held-out test scenarios GeneAI's policy scores ${g!.toFixed(2)}, ${gap} worse than ${best.label} (${b.toFixed(2)}). We show that as recorded.`;
  }
}
