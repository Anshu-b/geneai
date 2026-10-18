import { EXPORT_KEYS, type GeneEvent, type Genome, type RunExport } from "./types";

export function parseExport(input: unknown): RunExport {
  if (!input || typeof input !== "object" || Array.isArray(input)) {
    throw new Error("Export is not a JSON object.");
  }
  const obj = input as Record<string, unknown>;
  for (const key of EXPORT_KEYS) {
    if (!(key in obj)) throw new Error(`Export is missing ${key}.`);
  }
  if (obj.schema_version !== "1.0") {
    throw new Error(`Unsupported schema_version ${String(obj.schema_version)}. This lab reads 1.0.`);
  }
  const run = obj.run;
  if (!run || typeof run !== "object") throw new Error("Export run is missing.");
  const runObj = run as Record<string, unknown>;
  if (typeof runObj.id !== "string" || typeof runObj.domain !== "string") {
    throw new Error("Export run needs an id and a domain.");
  }
  if (typeof runObj.mode !== "string") throw new Error("Export run.mode is missing.");
  if (!Array.isArray(obj.genomes)) throw new Error("genomes must be an array.");
  if (!Array.isArray(obj.evaluations)) throw new Error("evaluations must be an array.");
  if (!Array.isArray(obj.events)) throw new Error("events must be an array.");
  if (!Array.isArray(obj.providers)) throw new Error("providers must be an array.");
  const events = obj.events as GeneEvent[];
  for (const event of events) {
    if (!event || typeof event !== "object") throw new Error("An event is not an object.");
    if (typeof event.sequence !== "number" || typeof event.type !== "string") {
      throw new Error("Every event needs a numeric sequence and a type.");
    }
  }
  const genomes = obj.genomes as Genome[];
  const ids = new Set<string>();
  for (const genome of genomes) {
    if (!genome || typeof genome.id !== "string") throw new Error("A genome is missing an id.");
    if (typeof genome.generation !== "number") throw new Error(`Genome ${genome.id} has no generation.`);
    if (!Array.isArray(genome.parent_ids)) throw new Error(`Genome ${genome.id} has no parent_ids.`);
    if (!genome.metadata || typeof genome.metadata !== "object") {
      genome.metadata = {};
    }
    if (ids.has(genome.id)) throw new Error(`Duplicate genome id ${genome.id}.`);
    ids.add(genome.id);
  }
  return obj as unknown as RunExport;
}
