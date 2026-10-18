export interface ShowcaseMachine {
  id: string;
  kind: string;
  cpu?: number;
  mem_gb?: number;
  gpu?: number;
}

export interface ShowcaseJob {
  id: string;
  class: string;
  cpu?: number;
  mem_gb?: number;
  gpu?: number;
  arrival_min: number;
  duration_min: number;
  deadline_min: number;
  hard_deadline?: boolean;
}

export interface ScheduleBar {
  job_id: string;
  machine_id: string;
  start_min: number;
  end_min: number;
  wait_min: number;
  deadline_min: number;
  missed_deadline: boolean;
}

export interface PolicySummary {
  avg_completion_min: number;
  p95_wait_min: number;
  sla_misses: number;
  hard_deadline_misses: number;
  energy_kwh: number;
  makespan_min: number;
  capacity_violations: number;
  fitness?: number | null;
}

export interface CloudAggregate extends PolicySummary {
  n_cases: number;
}

export interface ShoppingAggregate {
  score_mean?: number | null;
  feasible_count: number;
  n_shoppers: number;
  total_usd_mean: number;
  budget_usd_mean: number;
}

export interface BehaviorOrigin {
  /** Engine (geneai_core/showcase.py) writes `id`; older files used `genome_id`. */
  id?: string | null;
  genome_id?: string | null;
  provider: string;
  operator?: string;
  generation?: number;
}

export interface CloudPolicy {
  key: string;
  label: string;
  provider?: string;
  genome_id?: string | null;
  description?: string;
  rules_plain?: string[];
  feasible: boolean;
  infeasible_reason?: string | null;
  schedule: ScheduleBar[];
  summary?: PolicySummary | null;
  aggregate?: CloudAggregate | null;
  behavior_origin?: BehaviorOrigin | null;
}

export interface CloudShowcase {
  units?: { time?: string; energy?: string };
  machines: ShowcaseMachine[];
  jobs: ShowcaseJob[];
  policies: CloudPolicy[];
}

export interface ShopperProfile {
  id: string;
  style?: string;
  budget_usd: number;
  apparel_size?: string;
  shoe_size?: number | string;
  occasions?: string[];
  likes?: string[];
  dislikes?: string[];
  notes?: string;
}

export interface OutfitItem {
  id: string;
  name: string;
  role?: string;
  price_usd: number;
  url?: string;
  image_url?: string;
  source?: string;
}

export interface OutfitCard {
  key: string;
  label: string;
  provider?: string;
  genome_id?: string | null;
  feasible: boolean;
  infeasible_reason?: string | null;
  items: OutfitItem[];
  total_usd: number;
  score?: number | null;
  why?: string[];
  aggregate?: ShoppingAggregate | null;
  behavior_origin?: BehaviorOrigin | null;
}

export interface ShoppingShowcase {
  shopper: ShopperProfile;
  outfits: OutfitCard[];
}

export interface ShowcaseFile {
  schema: "geneai-showcase-v1";
  domain: string;
  run_id: string;
  export_integrity_sha256?: string;
  split?: string;
  case_id: string;
  case_note: string;
  cloud?: CloudShowcase;
  shopping?: ShoppingShowcase;
}

export type ShowcaseLoad =
  | { status: "ready"; file: ShowcaseFile }
  | { status: "missing" }
  | { status: "invalid"; message: string };

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" ? (value as Record<string, unknown>) : null;
}

export function parseShowcase(raw: unknown): ShowcaseFile {
  const data = asRecord(raw);
  if (!data) throw new Error("Showcase file is empty.");
  if (data.schema !== "geneai-showcase-v1") throw new Error("Showcase schema is not geneai-showcase-v1.");
  if (typeof data.domain !== "string" || typeof data.case_id !== "string" || typeof data.case_note !== "string") {
    throw new Error("Showcase file is missing domain, case id, or note.");
  }
  if (data.domain === "cloud_scheduling") {
    const cloud = asRecord(data.cloud);
    if (!cloud || !Array.isArray(cloud.policies) || !Array.isArray(cloud.machines)) {
      throw new Error("Cloud showcase is missing machines or policies.");
    }
  }
  if (data.domain === "shopping") {
    const shopping = asRecord(data.shopping);
    const shopper = shopping ? asRecord(shopping.shopper) : null;
    if (!shopping || !shopper || !Array.isArray(shopping.outfits)) {
      throw new Error("Shopping showcase is missing the shopper or outfits.");
    }
    for (const outfit of shopping.outfits) {
      const card = asRecord(outfit);
      if (card) card.aggregate = normalizeShoppingAggregate(card.aggregate);
    }
  }
  return raw as ShowcaseFile;
}

/** The engine builder writes score / n_feasible / mean_total_usd / mean_budget_usd. Accept both shapes. */
function normalizeShoppingAggregate(value: unknown): ShoppingAggregate | null {
  const row = asRecord(value);
  if (!row) return null;
  const pick = (...keys: string[]) => {
    for (const key of keys) if (typeof row[key] === "number") return row[key] as number;
    return null;
  };
  const n = pick("n_shoppers");
  if (n == null) return null;
  return {
    n_shoppers: n,
    feasible_count: pick("feasible_count", "n_feasible") ?? 0,
    score_mean: pick("score_mean", "score"),
    total_usd_mean: pick("total_usd_mean", "mean_total_usd") ?? 0,
    budget_usd_mean: pick("budget_usd_mean", "mean_budget_usd") ?? 0,
  };
}

export function isSampleShowcase(file: ShowcaseFile): boolean {
  return `${file.case_id} ${file.case_note}`.toLowerCase().includes("sample");
}

const BASELINE_KEY = /round_robin|shortest_job|least_loaded|deadline_weighted|neutral_outfit|best_baseline|handwritten|(^|[^a-z])baseline([^a-z]|$)/;

/** Hand-written schedulers and outfit templates. Agentic arms stay visible. */
export function isManualBaseline(key: string, provider?: string | null): boolean {
  return BASELINE_KEY.test(`${key} ${provider ?? ""}`.toLowerCase());
}

export function keyScoreLines(file: ShowcaseFile): string[] {
  if (file.domain === "cloud_scheduling" && file.cloud) return cloudScoreLines(file.cloud.policies.filter((policy) => !isManualBaseline(policy.key, policy.provider)));
  if (file.domain === "shopping" && file.shopping) {
    return shoppingScoreLines(
      file.shopping.outfits.filter((outfit) => !isManualBaseline(outfit.key, outfit.provider)),
      file.shopping.shopper.budget_usd,
    );
  }
  return [];
}

function cloudScoreLines(policies: CloudPolicy[]): string[] {
  const pair = agenticPair(policies);
  if (!pair) return [];
  const [gene, other] = pair;
  const lines: string[] = [];
  if (gene.summary && other.summary) {
    lines.push(`Missed deadlines: ${gene.summary.sla_misses} vs ${other.summary.sla_misses}. ${gene.label}, then ${other.label}.`);
    lines.push(`Average finish: ${compact(gene.summary.avg_completion_min)} min vs ${compact(other.summary.avg_completion_min)} min.`);
  }
  if (gene.aggregate && other.aggregate && gene.aggregate.n_cases > 0) {
    lines.push(`Across ${gene.aggregate.n_cases} held-out cases, missed deadlines: ${gene.aggregate.sla_misses} vs ${other.aggregate.sla_misses}.`);
  }
  return lines.slice(0, 3);
}

function shoppingScoreLines(outfits: OutfitCard[], budget: number): string[] {
  const pair = agenticPair(outfits);
  if (!pair) return [];
  const [gene, other] = pair;
  const lines = [`Spend on this shopper: $${compact(gene.total_usd, 0)} vs $${compact(other.total_usd, 0)}, against a $${compact(budget, 0)} budget.`];
  if (gene.aggregate && other.aggregate && gene.aggregate.n_shoppers > 0) {
    lines.push(`Feasible for ${gene.aggregate.feasible_count} of ${gene.aggregate.n_shoppers} held-out shoppers vs ${other.aggregate.feasible_count} of ${other.aggregate.n_shoppers}.`);
    if (gene.aggregate.score_mean != null && other.aggregate.score_mean != null) {
      lines.push(`Mean score: ${compact(gene.aggregate.score_mean, 2)} vs ${compact(other.aggregate.score_mean, 2)}. Higher fits the shopper better.`);
    }
  }
  return lines.slice(0, 3);
}

function agenticPair<T extends { key: string }>(items: T[]): [T, T] | null {
  if (items.length < 2) return null;
  const gene = items.find((item) => item.key === "champion" || item.key === "geneai") ?? items[0];
  const other = items.find((item) => item.key !== gene.key && /gpt|one_shot|openai/i.test(item.key)) ?? items.find((item) => item.key !== gene.key);
  if (!other) return null;
  return [gene, other];
}

export function interpretShowcaseResponse(status: number, body: unknown): ShowcaseLoad {
  if (status === 404) return { status: "missing" };
  if (status < 200 || status >= 300) return { status: "invalid", message: `Showcase request failed (${status}).` };
  try {
    return { status: "ready", file: parseShowcase(body) };
  } catch (error) {
    return { status: "invalid", message: error instanceof Error ? error.message : "Showcase file could not be read." };
  }
}

function compact(value: number, digits = 1): string {
  if (!Number.isFinite(value)) return "—";
  const fixed = value.toFixed(digits);
  return fixed.replace(/\.0+$/, "").replace(/(\.\d*?)0+$/, "$1");
}

export function providerLabel(provider: string | undefined): string {
  const value = (provider || "").toLowerCase();
  if (value === "creao") return "CREAO";
  if (value === "veri") return "Veri";
  if (value === "openai") return "OpenAI";
  return provider || "unknown";
}

/** Credit the first genome with this behavior when a later copy carries a different provider. */
export function originCredit(provider: string | undefined, origin: BehaviorOrigin | null | undefined): string | null {
  if (!origin?.provider) return null;
  if ((provider || "").toLowerCase() === origin.provider.toLowerCase()) return null;
  const genomeId = origin.id ?? origin.genome_id;
  const id = genomeId ? ` ${genomeId}` : "";
  const operator = origin.operator || "variation";
  const generation = origin.generation ?? 0;
  return `Found by ${providerLabel(origin.provider)} ${operator}${id} in gen ${generation}.`;
}

export function caseMissCount(policy: CloudPolicy): number {
  if (policy.summary && Number.isFinite(policy.summary.sla_misses)) return policy.summary.sla_misses;
  return policy.schedule.filter((bar) => bar.missed_deadline).length;
}

export function exampleMissLine(policies: CloudPolicy[]): string {
  if (policies.length === 0) return "This file has no policies on the example case.";
  const bits = policies.map((policy) => {
    if (!policy.feasible) return `${policy.label} is infeasible`;
    return `${policy.label} missed ${caseMissCount(policy)}`;
  });
  return `On this case, ${bits.join(", ")}.`;
}

export function cloudVerdict(policies: CloudPolicy[]): string {
  const withSplit = policies.filter((policy) => policy.aggregate && policy.aggregate.n_cases > 0);
  if (withSplit.length > 0) {
    const count = withSplit[0].aggregate!.n_cases;
    const champion = withSplit.find((policy) => policy.key === "champion") ?? withSplit[0];
    const aggregate = champion.aggregate!;
    const others = withSplit
      .filter((policy) => policy.key !== champion.key)
      .map((policy) => {
        const row = policy.aggregate!;
        return `${policy.label} missed ${row.sla_misses} in total (mean finish ${compact(row.avg_completion_min)} min, ${compact(row.energy_kwh, 2)} kWh)`;
      });
    return `Across all ${count} held-out test cases, ${champion.label} missed ${aggregate.sla_misses} deadlines in total, with a mean finish of ${compact(aggregate.avg_completion_min)} min and ${compact(aggregate.energy_kwh, 2)} kWh. ${others.join("; ")}.`;
  }
  const champion = policies.find((policy) => policy.key === "champion");
  if (!champion) return "This file has no champion policy to compare.";
  if (!champion.feasible || !champion.summary) {
    return `${champion.label} is marked infeasible${champion.infeasible_reason ? `: ${champion.infeasible_reason}` : "."}`;
  }
  return exampleMissLine(policies);
}

export function shoppingSplitLine(outfits: OutfitCard[]): string | null {
  const withSplit = outfits.filter((outfit) => outfit.aggregate && outfit.aggregate.n_shoppers > 0);
  if (withSplit.length === 0) return null;
  const count = withSplit[0].aggregate!.n_shoppers;
  const bits = withSplit.map((outfit) => {
    const row = outfit.aggregate!;
    const score = row.score_mean == null ? "" : `, mean score ${compact(row.score_mean, 2)}`;
    return `${outfit.label} was feasible for ${row.feasible_count} of ${row.n_shoppers}, mean spend $${compact(row.total_usd_mean, 0)} of $${compact(row.budget_usd_mean, 0)}${score}`;
  });
  return `Across all ${count} held-out shoppers, ${bits.join("; ")}.`;
}

export function budgetLine(total: number, budget: number): string {
  const delta = Math.round((budget - total) * 100) / 100;
  const money = (value: number) => `$${compact(value, 2)}`;
  if (delta >= 0) return `${money(total)} of ${money(budget)}, under by ${money(delta)}.`;
  return `${money(total)} of ${money(budget)}, over by ${money(Math.abs(delta))}.`;
}
