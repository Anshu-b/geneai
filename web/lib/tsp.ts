export const TSP_KEYS = ["held_karp", "classic_ga", "llm_one_shot", "geneai"] as const;

export type TspMethodKey = (typeof TSP_KEYS)[number];

export interface TspCity {
  id: string;
  x: number;
  y: number;
}

export interface TspPoint {
  gen?: number;
  iter?: number;
  best_length: number;
  elapsed_ms: number;
}

export interface TspMethod {
  key: TspMethodKey;
  label: string;
  provider: string;
  tour: string[];
  /** null when the method returned no tour (for example an empty model reply). */
  length: number | null;
  gap_pct: number | null;
  time_ms: number;
  llm_calls: number;
  valid: boolean;
  invalid_reason: string | null;
  convergence: TspPoint[];
  /** GeneAI only: genomes added during the run, by source, e.g. {"llm:veri": 0, "llm:creao": 0, "algorithmic": 2}. */
  provenance_counts?: Record<string, number> | null;
  /** GeneAI only: the genome that first produced the best tour. */
  behavior_origin?: { id?: string; provider?: string; operator?: string; provenance?: string; generation?: number } | null;
}

export interface TspOptimum {
  tour: string[];
  length: number;
  time_ms: number;
}

export interface TspHeldoutSummary {
  mean_gap_pct: number | null;
  optimal_hits: number;
  mean_time_ms: number;
  valid_count: number;
}

export interface TspFile {
  schema: "geneai-tsp-v1";
  cities: TspCity[];
  optimum: TspOptimum;
  methods: TspMethod[];
  heldout: {
    n: number;
    instances: unknown[];
    summary: Record<TspMethodKey, TspHeldoutSummary>;
  };
  notes: string;
}

export type TspLoad =
  | { status: "missing" }
  | { status: "invalid"; message: string }
  | { status: "ready"; file: TspFile };

const KEY_SET = new Set<string>(TSP_KEYS);

function asRecord(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  return value as Record<string, unknown>;
}

function num(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) throw new Error(`${label} must be a number.`);
  return value;
}

function tourOf(value: unknown, label: string): string[] {
  if (!Array.isArray(value) || value.length < 2) throw new Error(`${label} is missing a tour.`);
  return value.map((item) => String(item));
}

function parsePoint(value: unknown, label: string): TspPoint {
  const point = asRecord(value);
  if (!point) throw new Error(`${label} has a broken convergence point.`);
  const parsed: TspPoint = {
    best_length: num(point.best_length, `${label} best length`),
    elapsed_ms: num(point.elapsed_ms, `${label} elapsed`),
  };
  if (typeof point.gen === "number") parsed.gen = point.gen;
  if (typeof point.iter === "number") parsed.iter = point.iter;
  return parsed;
}

function parseMethod(value: unknown): TspMethod {
  const row = asRecord(value);
  if (!row || typeof row.key !== "string" || !KEY_SET.has(row.key)) {
    throw new Error("A method key must be held_karp, classic_ga, llm_one_shot, or geneai.");
  }
  if (typeof row.label !== "string" || typeof row.provider !== "string") {
    throw new Error(`Method ${row.key} is missing a label or provider.`);
  }
  if (typeof row.valid !== "boolean") throw new Error(`Method ${row.key} is missing valid.`);
  const convergence = Array.isArray(row.convergence) ? row.convergence.map((point, index) => parsePoint(point, `${row.key} point ${index + 1}`)) : [];
  return {
    key: row.key as TspMethodKey,
    label: row.label,
    provider: row.provider,
    // The engine writes tour: null and length: null for an arm that returned no tour.
    tour: row.tour == null && row.valid === false ? [] : tourOf(row.tour, row.key),
    length: row.length == null && row.valid === false ? null : num(row.length, `${row.key} length`),
    gap_pct: row.gap_pct == null ? null : num(row.gap_pct, `${row.key} gap`),
    time_ms: num(row.time_ms, `${row.key} time`),
    llm_calls: num(row.llm_calls, `${row.key} calls`),
    valid: row.valid,
    invalid_reason: typeof row.invalid_reason === "string" ? row.invalid_reason : null,
    convergence,
    provenance_counts: parseCounts(row.provenance_counts),
    behavior_origin: asRecord(row.behavior_origin) as TspMethod["behavior_origin"],
  };
}

function parseCounts(value: unknown): Record<string, number> | null {
  const row = asRecord(value);
  if (!row) return null;
  const out: Record<string, number> = {};
  for (const [key, count] of Object.entries(row)) if (typeof count === "number" && Number.isFinite(count)) out[key] = count;
  return Object.keys(out).length > 0 ? out : null;
}

function parseSummary(value: unknown, key: string): TspHeldoutSummary {
  const row = asRecord(value);
  if (!row) throw new Error(`Held-out summary is missing ${key}.`);
  return {
    mean_gap_pct: row.mean_gap_pct == null ? null : num(row.mean_gap_pct, `${key} mean gap`),
    optimal_hits: num(row.optimal_hits, `${key} optimal hits`),
    mean_time_ms: num(row.mean_time_ms, `${key} mean time`),
    valid_count: num(row.valid_count, `${key} valid count`),
  };
}

export function parseTsp(raw: unknown): TspFile {
  const data = asRecord(raw);
  if (!data || data.schema !== "geneai-tsp-v1") throw new Error("TSP file schema must be geneai-tsp-v1.");
  if (!Array.isArray(data.cities) || data.cities.length < 3) throw new Error("TSP file is missing cities.");
  const cities = data.cities.map((item, index) => {
    const city = asRecord(item);
    if (!city) throw new Error(`City ${index + 1} is not an object.`);
    return { id: String(city.id), x: num(city.x, `city ${city.id} x`), y: num(city.y, `city ${city.id} y`) };
  });
  const ids = new Set(cities.map((city) => city.id));
  if (!Array.isArray(data.methods)) throw new Error("TSP file is missing methods.");
  if (data.methods.some((item) => asRecord(item)?.key === "heuristic")) {
    throw new Error("TSP file must not include a heuristic method.");
  }
  const methods = data.methods.map(parseMethod);
  for (const key of TSP_KEYS) {
    if (!methods.some((method) => method.key === key)) throw new Error(`TSP file is missing ${key}.`);
  }
  for (const method of methods) {
    for (const stop of method.tour) {
      if (!ids.has(stop)) throw new Error(`${method.key} names an unknown city ${stop}.`);
    }
  }
  const held = methods.find((method) => method.key === "held_karp")!;
  if (held.length == null) throw new Error("Held-Karp must return a tour.");
  const optimum: TspOptimum = { tour: held.tour.slice(), length: held.length, time_ms: held.time_ms };
  const heldout = asRecord(data.heldout);
  if (!heldout || typeof heldout.n !== "number") throw new Error("TSP file is missing the held-out count.");
  const summaryRaw = asRecord(heldout.summary);
  if (!summaryRaw) throw new Error("TSP file is missing the held-out summary.");
  const summary = Object.fromEntries(TSP_KEYS.map((key) => [key, parseSummary(summaryRaw[key], key)])) as Record<TspMethodKey, TspHeldoutSummary>;
  return {
    schema: "geneai-tsp-v1",
    cities,
    optimum,
    methods: TSP_KEYS.map((key) => methods.find((method) => method.key === key)!),
    heldout: {
      n: heldout.n,
      instances: Array.isArray(heldout.instances) ? heldout.instances : [],
      summary,
    },
    notes: typeof data.notes === "string" ? data.notes : "",
  };
}

export function isSampleTsp(file: TspFile): boolean {
  return file.notes.toLowerCase().includes("sample");
}

export function interpretTspResponse(status: number, body: unknown): TspLoad {
  if (status === 404) return { status: "missing" };
  if (status < 200 || status >= 300) return { status: "invalid", message: `Benchmark request failed (${status}).` };
  try {
    return { status: "ready", file: parseTsp(body) };
  } catch (error) {
    return { status: "invalid", message: error instanceof Error ? error.message : "Benchmark file could not be read." };
  }
}

export function hitsOptimum(method: TspMethod): boolean {
  return method.valid && method.gap_pct != null && method.gap_pct <= 0.05;
}

export function bestAt(method: TspMethod, elapsed: number): number | null {
  let best: number | null = null;
  for (const point of method.convergence) {
    if (point.elapsed_ms <= elapsed + 1e-6) best = point.best_length;
  }
  return best;
}

export function maxRaceMs(file: TspFile): number {
  const times = file.methods.flatMap((method) => [method.time_ms, ...method.convergence.map((point) => point.elapsed_ms)]);
  return Math.max(1, ...times);
}

export function timeFraction(elapsed: number, max: number, log: boolean): number {
  if (max <= 0) return 0;
  const clamped = Math.min(Math.max(elapsed, 0), max);
  if (!log) return clamped / max;
  return Math.log10(1 + clamped) / Math.log10(1 + max);
}

export function elapsedFromFraction(fraction: number, max: number, log: boolean): number {
  const portion = Math.min(Math.max(fraction, 0), 1);
  if (!log) return portion * max;
  return 10 ** (portion * Math.log10(1 + max)) - 1;
}

export function formatDuration(ms: number): string {
  if (!Number.isFinite(ms)) return "—";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  const seconds = ms / 1000;
  return `${seconds.toFixed(seconds < 10 ? 1 : 0)} s`;
}

export function formatLength(value: number | null): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return value.toFixed(2);
}

export function formatGap(value: number | null): string {
  if (value == null || !Number.isFinite(value)) return "—";
  if (Math.abs(value) < 0.05) return "0%";
  return `${value.toFixed(1)}%`;
}

export function tspVerdict(file: TspFile): string {
  const by = Object.fromEntries(file.methods.map((method) => [method.key, method])) as Record<TspMethodKey, TspMethod>;
  const parts = [
    `Held-Karp proves a tour of ${formatLength(by.held_karp.length)} in ${formatDuration(by.held_karp.time_ms)}.`,
  ];
  if (hitsOptimum(by.geneai)) parts.push("GeneAI matches that tour.");
  else if (by.geneai.valid) parts.push(`GeneAI finishes ${formatGap(by.geneai.gap_pct)} above it.`);
  else parts.push("GeneAI does not return a tour.");
  if (by.classic_ga.valid) parts.push(`The classic genetic algorithm finishes ${formatGap(by.classic_ga.gap_pct)} above it.`);
  else parts.push("The classic genetic algorithm does not return a tour.");
  if (!by.llm_one_shot.valid) parts.push(by.llm_one_shot.invalid_reason || "GPT-5.5 does not return a tour.");
  else if (hitsOptimum(by.llm_one_shot)) parts.push("GPT-5.5 matches that tour.");
  else parts.push(`GPT-5.5 finishes ${formatGap(by.llm_one_shot.gap_pct)} above it.`);
  return parts.join(" ");
}

export function heldoutLine(file: TspFile): string {
  const n = file.heldout.n;
  const name: Record<TspMethodKey, string> = {
    held_karp: "Held-Karp",
    classic_ga: "The classic genetic algorithm",
    llm_one_shot: "GPT-5.5",
    geneai: "GeneAI",
  };
  const bits = TSP_KEYS.map((key) => {
    const row = file.heldout.summary[key];
    return `${name[key]} hits ${row.optimal_hits} of ${n} (mean gap ${formatGap(row.mean_gap_pct)}, mean ${formatDuration(row.mean_time_ms)})`;
  });
  return `Across ${n} held-out maps, ${bits.join("; ")}.`;
}

export function provenanceLine(file: TspFile): string | null {
  const gene = file.methods.find((method) => method.key === "geneai");
  const counts = gene?.provenance_counts;
  if (!counts) return null;
  const name = (key: string) => (key === "llm:veri" ? "Veri" : key === "llm:creao" ? "CREAO" : key === "algorithmic" ? "algorithmic operators (OX, ERX, swap, 2-opt)" : key);
  const bits = Object.entries(counts).map(([key, count]) => `${count} from ${name(key)}`);
  const origin = gene?.behavior_origin;
  let found = "";
  if (origin && (origin.provenance || origin.provider)) {
    const source = origin.provenance === "algorithmic" ? "an algorithmic" : `a ${name(origin.provenance || origin.provider || "")}`;
    found = ` The best tour first appeared as ${source} ${origin.operator || "child"}${origin.id ? ` (${origin.id})` : ""} in generation ${origin.generation ?? 0}.`;
  }
  return `Genomes GeneAI added on this map, by source: ${bits.join(", ")}.${found}`;
}

export const TSP_COLOR: Record<TspMethodKey, string> = {
  held_karp: "var(--amber)",
  classic_ga: "var(--violet)",
  llm_one_shot: "var(--danger)",
  geneai: "var(--teal)",
};
