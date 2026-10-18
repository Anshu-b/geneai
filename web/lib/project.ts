import { familyOf } from "./format";
import type { Evaluation, GeneEvent, Genome, RunExport } from "./types";

export type BeatPhase =
  | "elites"
  | "births"
  | "prunes"
  | "settled"
  | "validation"
  | "frozen"
  | "tested"
  | "complete";

export interface BeatStats {
  births?: number;
  prunes?: number;
  elites?: number;
  living?: number;
  rejected?: number;
  bestFitness?: number | null;
  bestGenomeId?: string | null;
  finalists?: number;
}

export interface Beat {
  id: string;
  sequence: number;
  generation: number;
  phase: BeatPhase;
  title: string;
  focusGeneration: number | null;
  stats: BeatStats;
}

export type NodeStatus = "incoming" | "kept" | "elite" | "pruned" | "champion";

export interface GenomeView {
  genome: Genome;
  status: NodeStatus;
  laterPruned: boolean;
  feasible: boolean | null;
  trainFitness: number | null;
  train: Evaluation | null;
  validation: Evaluation | null;
  test: Evaluation | null;
  violations: string[];
}

export interface BudgetSnapshot {
  cancelled: boolean | null;
  elapsed_seconds: number | null;
  evaluations: number | null;
  exhausted_reason: string | null;
  model_calls: number | null;
  spend_usd: number | null;
}

export interface FitnessPoint {
  generation: number;
  fitness: number;
  genomeId: string;
}

export interface Projection {
  sequence: number;
  views: GenomeView[];
  byId: Map<string, GenomeView>;
  populationIds: string[];
  eliteIds: string[];
  prunedIds: string[];
  championFrozen: boolean;
  testVisible: boolean;
  validationVisible: boolean;
  championId: string | null;
  best: FitnessPoint | null;
  budget: BudgetSnapshot | null;
  series: FitnessPoint[];
  settledGeneration: number | null;
  frozenSequence: number | null;
}

export interface ProviderCall {
  provider: string | null;
  providerMode: string | null;
  providerRole: string | null;
  providerRunId: string | null;
  model: string | null;
  deploymentId: string | null;
  latencyMs: number | null;
  costUsd: number | null;
  operation: string | null;
  callType: string;
}

export interface LayoutNode {
  id: string;
  x: number;
  y: number;
  r: number;
}

export interface LayoutLane {
  family: string;
  y: number;
  height: number;
  maxStack: number;
}

export interface LayoutModel {
  nodes: Map<string, LayoutNode>;
  lanes: LayoutLane[];
  generations: number[];
  width: number;
  height: number;
  xFor: (generation: number) => number;
  minFitness: number | null;
  maxFitness: number | null;
}

const X0 = 220;
const X_STEP = 118;
const HEADER = 52;
const STACK = 18;
const LANE_PAD = 18;

function num(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function str(value: unknown): string | null {
  return typeof value === "string" && value.length > 0 ? value : null;
}

function strings(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is string => typeof item === "string");
}

function splitOf(payload: Record<string, unknown>): string | null {
  return str(payload.evaluation_split) || str(payload.split);
}

export function sortedEvents(events: GeneEvent[]): GeneEvent[] {
  return [...events].sort((a, b) => a.sequence - b.sequence);
}

export function buildBeats(events: GeneEvent[]): Beat[] {
  const ordered = sortedEvents(events);
  const starts = ordered.filter(
    (event) => event.type === "generation.started" && num(event.payload.generation) != null,
  );
  const beats: Beat[] = [];

  for (const start of starts) {
    const generation = num(start.payload.generation)!;
    const nextStart = starts.find((event) => event.sequence > start.sequence);
    const completed = ordered.find(
      (event) =>
        event.type === "generation.completed" &&
        event.payload.generation === generation &&
        event.sequence >= start.sequence &&
        (!nextStart || event.sequence < nextStart.sequence),
    );
    let end = completed?.sequence ?? start.sequence;
    const budget = ordered.find(
      (event) =>
        event.type === "budget.updated" &&
        event.sequence > end &&
        (!nextStart || event.sequence < nextStart.sequence),
    );
    if (budget) end = budget.sequence;
    const window = ordered.filter((event) => event.sequence >= start.sequence && event.sequence <= end);
    const elites = window.filter((event) => event.type === "selection.elite");
    const prunes = window.filter((event) => event.type === "selection.pruned");
    const firstPrune = prunes[0]?.sequence ?? Number.POSITIVE_INFINITY;
    const births = window.filter((event) => {
      if (event.sequence >= firstPrune) return false;
      return (
        event.type === "genome.seeded" ||
        event.type === "genome.mutated" ||
        event.type === "genome.crossed" ||
        event.type === "selection.immigrant"
      );
    });
    const rejected = window.filter((event) => event.type === "genome.rejected").length;
    const birthIds = new Set(births.map((event) => str(event.payload.genome_id)).filter(Boolean));

    if (elites.length > 0) {
      const last = elites[elites.length - 1];
      beats.push({
        id: `g${generation}-elites`,
        sequence: last.sequence,
        generation,
        phase: "elites",
        title: `Generation ${generation} · elites carried`,
        focusGeneration: generation,
        stats: { elites: strings(last.payload.elite_ids).length },
      });
    }

    if (births.length > 0) {
      const evals = window.filter(
        (event) =>
          event.type === "genome.evaluated" &&
          event.sequence < firstPrune &&
          (splitOf(event.payload) === "train" || splitOf(event.payload) == null),
      );
      const sequence = evals.length > 0 ? evals[evals.length - 1].sequence : births[births.length - 1].sequence;
      beats.push({
        id: `g${generation}-births`,
        sequence,
        generation,
        phase: "births",
        title: `Generation ${generation} · births`,
        focusGeneration: generation,
        stats: { births: birthIds.size, rejected },
      });
    }

    if (prunes.length > 0) {
      const last = prunes[prunes.length - 1];
      beats.push({
        id: `g${generation}-prunes`,
        sequence: last.sequence,
        generation,
        phase: "prunes",
        title: `Generation ${generation} · pruned`,
        focusGeneration: generation,
        stats: { prunes: prunes.length },
      });
    }

    if (completed) {
      beats.push({
        id: `g${generation}-settled`,
        sequence: end,
        generation,
        phase: "settled",
        title: `Generation ${generation} settled`,
        focusGeneration: generation,
        stats: {
          living: strings(completed.payload.population_ids).length,
          prunes: strings(completed.payload.pruned_ids).length,
          elites: strings(completed.payload.elite_ids).length,
          bestFitness: num(completed.payload.best_fitness),
          bestGenomeId: str(completed.payload.best_genome_id),
          rejected,
        },
      });
    }
  }

  const lastEnd = beats.length > 0 ? beats[beats.length - 1].sequence : 0;
  const tail = ordered.filter((event) => event.sequence > lastEnd);
  const validation = tail.filter(
    (event) => event.type === "genome.evaluated" && splitOf(event.payload) === "validation",
  );
  if (validation.length > 0) {
    const last = validation[validation.length - 1];
    beats.push({
      id: "validation",
      sequence: last.sequence,
      generation: num(last.payload.generation) ?? 0,
      phase: "validation",
      title: "Validation finalists",
      focusGeneration: null,
      stats: { finalists: validation.length },
    });
  }
  const frozen = tail.find((event) => event.type === "champion.frozen");
  if (frozen) {
    beats.push({
      id: "frozen",
      sequence: frozen.sequence,
      generation: num(frozen.payload.generation) ?? 0,
      phase: "frozen",
      title: "Champion frozen",
      focusGeneration: num(frozen.payload.generation),
      stats: {
        bestFitness: num(frozen.payload.fitness),
        bestGenomeId: str(frozen.payload.genome_id),
      },
    });
  }
  const tested = tail.find((event) => event.type === "champion.tested");
  const testEval = tail.find(
    (event) => event.type === "genome.evaluated" && splitOf(event.payload) === "test",
  );
  if (tested || testEval) {
    const sequence = Math.max(tested?.sequence ?? 0, testEval?.sequence ?? 0);
    const source = tested ?? testEval!;
    beats.push({
      id: "tested",
      sequence,
      generation: num(source.payload.generation) ?? 0,
      phase: "tested",
      title: "Champion tested",
      focusGeneration: num(source.payload.generation),
      stats: {
        bestFitness: num(source.payload.fitness),
        bestGenomeId: str(source.payload.genome_id),
      },
    });
  }
  const done = tail.find((event) => event.type === "run.completed");
  if (done) {
    beats.push({
      id: "complete",
      sequence: done.sequence,
      generation: num(done.payload.generations_completed) ?? beats[beats.length - 1]?.generation ?? 0,
      phase: "complete",
      title: "Run complete",
      focusGeneration: null,
      stats: { bestGenomeId: str(done.payload.champion_id) },
    });
  }
  return beats;
}

function readBudget(payload: Record<string, unknown>): BudgetSnapshot {
  return {
    cancelled: typeof payload.cancelled === "boolean" ? payload.cancelled : null,
    elapsed_seconds: num(payload.elapsed_seconds),
    evaluations: num(payload.evaluations),
    exhausted_reason: str(payload.exhausted_reason),
    model_calls: num(payload.model_calls),
    spend_usd: num(payload.spend_usd),
  };
}

function evaluationFromRecord(evaluation: Evaluation): Evaluation {
  return evaluation;
}

export function projectRun(exp: RunExport, sequence: number): Projection {
  const events = sortedEvents(exp.events).filter((event) => event.sequence <= sequence);
  const born = new Set<string>();
  const prunedAt = new Map<string, number>();
  let populationIds: string[] = [];
  let populationSeq = -1;
  let carriedElites: string[] = [];
  let carriedSeq = -1;
  let settledElites: string[] = [];
  let settledSeq = -1;
  let championFrozen = false;
  let frozenSequence: number | null = null;
  let budget: BudgetSnapshot | null = null;
  let settledGeneration: number | null = null;
  const series: FitnessPoint[] = [];
  const seenSplits = new Set<string>();

  for (const event of events) {
    const payload = event.payload;
    const genomeId = str(payload.genome_id);
    if (
      genomeId &&
      (event.type === "genome.seeded" ||
        event.type === "genome.mutated" ||
        event.type === "genome.crossed" ||
        event.type === "selection.immigrant" ||
        event.type === "genome.evaluated")
    ) {
      born.add(genomeId);
    }
    if (event.type === "selection.pruned" && genomeId) prunedAt.set(genomeId, event.sequence);
    if (event.type === "selection.elite") {
      carriedElites = strings(payload.elite_ids);
      carriedSeq = event.sequence;
    }
    if (event.type === "generation.completed") {
      populationIds = strings(payload.population_ids);
      populationSeq = event.sequence;
      settledElites = strings(payload.elite_ids);
      settledSeq = event.sequence;
      settledGeneration = num(payload.generation);
      const fitness = num(payload.best_fitness);
      const bestId = str(payload.best_genome_id);
      const generation = num(payload.generation);
      if (fitness != null && bestId && generation != null) {
        series.push({ generation, fitness, genomeId: bestId });
      }
    }
    if (event.type === "budget.updated") budget = readBudget(payload);
    if (event.type === "champion.frozen") {
      championFrozen = true;
      frozenSequence = event.sequence;
    }
    if ((event.type === "genome.evaluated" || event.type === "champion.tested") && genomeId) {
      const split = splitOf(payload);
      if (split) seenSplits.add(`${genomeId}:${split}`);
    }
  }

  const records = new Map<string, Evaluation>();
  for (const evaluation of exp.evaluations) {
    records.set(`${evaluation.genome_id}:${evaluation.data_split}`, evaluationFromRecord(evaluation));
  }

  const population = new Set(populationIds);
  const useSettledElites = settledSeq >= carriedSeq && populationSeq >= 0;
  const eliteIds = (useSettledElites ? settledElites : carriedElites).filter((id) => {
    const pruned = prunedAt.get(id);
    return pruned == null || pruned > (useSettledElites ? settledSeq : carriedSeq);
  });
  const eliteSet = new Set(eliteIds);
  const championId = exp.champion_id;
  const testVisible =
    championFrozen && championId != null && seenSplits.has(`${championId}:test`);
  const validationVisible = [...seenSplits].some((key) => key.endsWith(":validation"));

  const views: GenomeView[] = [];
  for (const genome of exp.genomes) {
    if (!born.has(genome.id)) continue;
    const train = seenSplits.has(`${genome.id}:train`) ? records.get(`${genome.id}:train`) ?? null : null;
    const validation = seenSplits.has(`${genome.id}:validation`)
      ? records.get(`${genome.id}:validation`) ?? null
      : null;
    const test =
      testVisible && genome.id === championId && seenSplits.has(`${genome.id}:test`)
        ? records.get(`${genome.id}:test`) ?? null
        : null;
    const pruned = prunedAt.has(genome.id);
    const isChampion = championFrozen && genome.id === championId;
    let status: NodeStatus;
    if (isChampion) status = "champion";
    else if (pruned) status = "pruned";
    else if (eliteSet.has(genome.id)) status = "elite";
    else if (population.has(genome.id)) status = "kept";
    else status = "incoming";
    const primary = train ?? validation;
    views.push({
      genome,
      status,
      laterPruned: Boolean(isChampion && pruned),
      feasible: primary ? primary.feasible : null,
      trainFitness: train?.fitness ?? null,
      train,
      validation,
      test,
      violations: primary?.constraint_violations ?? [],
    });
  }

  views.sort((a, b) => a.genome.generation - b.genome.generation || a.genome.id.localeCompare(b.genome.id));
  const byId = new Map(views.map((view) => [view.genome.id, view]));
  const best = series.length > 0 ? series[series.length - 1] : null;

  return {
    sequence,
    views,
    byId,
    populationIds,
    eliteIds,
    prunedIds: [...prunedAt.keys()],
    championFrozen,
    testVisible,
    validationVisible,
    championId,
    best,
    budget,
    series,
    settledGeneration,
    frozenSequence,
  };
}

export function ancestryIds(championId: string | null, genomes: Map<string, Genome>): Set<string> {
  const set = new Set<string>();
  const stack = championId ? [championId] : [];
  while (stack.length > 0) {
    const id = stack.pop()!;
    if (set.has(id)) continue;
    set.add(id);
    const genome = genomes.get(id);
    if (!genome) continue;
    for (const parent of genome.parent_ids) stack.push(parent);
  }
  return set;
}

export function genomeMap(exp: RunExport): Map<string, Genome> {
  return new Map(exp.genomes.map((genome) => [genome.id, genome]));
}

export function isHandwrittenBaseline(genome: Genome): boolean {
  return (
    genome.metadata.seeded_baseline === true ||
    genome.provider === "baseline" ||
    genome.metadata.provider_role === "baseline"
  );
}

export function isCeilingReference(genome: Genome): boolean {
  const blob = `${genome.metadata.role ?? ""} ${genome.metadata.baseline_name ?? ""} ${genome.provider}`.toLowerCase();
  return blob.includes("ceiling");
}

export function operatorLabel(genome: Genome): string {
  const role = genome.metadata.provider_role || genome.metadata.role;
  if (role === "immigrant") return "immigrant";
  if (genome.operator === "mutation") return "mutate";
  if (genome.operator === "crossover") return "crossover";
  if (genome.operator === "repair") return "repair";
  if (genome.operator === "elite") return "elite";
  if (genome.operator === "seed") return "seed";
  return genome.operator || "unknown";
}

export function providerBucket(provider: string | null | undefined): "veri" | "creao" | "baseline" | "mock" | "random" | "other" {
  const value = (provider || "").toLowerCase();
  if (value.includes("veri")) return "veri";
  if (value.includes("creao")) return "creao";
  if (value === "baseline") return "baseline";
  if (value === "random") return "random";
  if (value.includes("mock")) return "mock";
  return "other";
}

export const PROVIDER_COLOR: Record<ReturnType<typeof providerBucket>, string> = {
  veri: "var(--prov-veri)",
  creao: "var(--prov-creao)",
  baseline: "var(--prov-baseline)",
  mock: "var(--prov-mock)",
  random: "var(--prov-random)",
  other: "var(--prov-other)",
};

function radiusFor(fitness: number | null, feasible: boolean | null, min: number | null, max: number | null): number {
  if (feasible === false) return 4.2;
  if (fitness == null || min == null || max == null) return 4.6;
  if (max === min) return 8;
  const t = (fitness - min) / (max - min);
  return 4.4 + Math.min(1, Math.max(0, t)) * 8.2;
}

export function layoutRun(exp: RunExport): LayoutModel {
  const trains = exp.evaluations.filter((evaluation) => evaluation.data_split === "train" && evaluation.fitness != null);
  const fitnesses = trains.map((evaluation) => evaluation.fitness as number);
  const minFitness = fitnesses.length ? Math.min(...fitnesses) : null;
  const maxFitness = fitnesses.length ? Math.max(...fitnesses) : null;
  const trainById = new Map<string, Evaluation>();
  for (const evaluation of trains) trainById.set(evaluation.genome_id, evaluation);

  const generations = [...new Set(exp.genomes.map((genome) => genome.generation))].sort((a, b) => a - b);
  const byFamily = new Map<string, Genome[]>();
  for (const genome of exp.genomes) {
    const family = familyOf(genome.metadata);
    const list = byFamily.get(family) ?? [];
    list.push(genome);
    byFamily.set(family, list);
  }

  const families = [...byFamily.entries()].sort((a, b) => {
    const aFirst = Math.min(...a[1].map((genome) => genome.generation));
    const bFirst = Math.min(...b[1].map((genome) => genome.generation));
    if (aFirst !== bFirst) return aFirst - bFirst;
    const aBest = Math.max(...a[1].map((genome) => trainById.get(genome.id)?.fitness ?? Number.NEGATIVE_INFINITY));
    const bBest = Math.max(...b[1].map((genome) => trainById.get(genome.id)?.fitness ?? Number.NEGATIVE_INFINITY));
    if (aBest !== bBest) return bBest - aBest;
    return a[0].localeCompare(b[0]);
  });

  const lanes: LayoutLane[] = [];
  let y = HEADER;
  const nodes = new Map<string, LayoutNode>();
  for (const [family, members] of families) {
    const perGen = new Map<number, Genome[]>();
    for (const genome of members) {
      const list = perGen.get(genome.generation) ?? [];
      list.push(genome);
      perGen.set(genome.generation, list);
    }
    let maxStack = 1;
    for (const group of perGen.values()) maxStack = Math.max(maxStack, group.length);
    const height = Math.max(54, maxStack * STACK + LANE_PAD);
    lanes.push({ family, y, height, maxStack });
    for (const group of perGen.values()) {
      group.sort((a, b) => a.id.localeCompare(b.id));
      group.forEach((genome, index) => {
        const train = trainById.get(genome.id);
        nodes.set(genome.id, {
          id: genome.id,
          x: X0 + genome.generation * X_STEP,
          y: y + ((index + 1) / (group.length + 1)) * height,
          r: radiusFor(train?.fitness ?? null, train?.feasible ?? null, minFitness, maxFitness),
        });
      });
    }
    y += height;
  }

  return {
    nodes,
    lanes,
    generations,
    width: X0 + Math.max(generations.length, 1) * X_STEP + 48,
    height: y + 28,
    xFor: (generation: number) => X0 + generation * X_STEP,
    minFitness,
    maxFitness,
  };
}

export function providerCallFor(genome: Genome, events: GeneEvent[], sequence: number): ProviderCall | null {
  const runId = genome.metadata.provider_run_id;
  if (!runId) return null;
  let found: GeneEvent | null = null;
  for (const event of events) {
    if (event.sequence > sequence) continue;
    if (event.type !== "model.completed" && event.type !== "agent.completed") continue;
    if (event.payload.provider_run_id === runId) found = event;
  }
  if (!found) return null;
  const payload = found.payload;
  const provenance = found.provenance;
  return {
    provider: str(payload.provider) || str(provenance?.model_provider) || str(provenance?.workflow_provider),
    providerMode: str(payload.provider_mode) || str(provenance?.provider_mode),
    providerRole: str(payload.provider_role),
    providerRunId: str(payload.provider_run_id),
    model: str(payload.model) || str(provenance?.model),
    deploymentId: str(payload.deployment_id) || str(provenance?.deployment_id),
    latencyMs: num(payload.latency_ms) ?? num(provenance?.latency_ms),
    costUsd: num(payload.cost_usd),
    operation: str(payload.operation),
    callType: found.type,
  };
}

export function stableJitter(id: string): number {
  let hash = 2166136261;
  for (let i = 0; i < id.length; i += 1) {
    hash ^= id.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return ((hash >>> 0) % 1000) / 1000;
}
