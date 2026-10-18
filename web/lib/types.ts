/**
 * Contract derived from engine exports at git 6bbf16d (schema 1.0).
 * Fields the fixtures do not carry stay optional. Nothing here is a sample score.
 */

export interface RunMeta {
  id: string;
  domain: string;
  mode: string;
  seed: number;
  git_sha?: string;
  status?: string;
  stop_reason?: string | null;
}

export interface ObjectiveSpec {
  domain: string;
  prompt: string;
  constraints: Record<string, unknown>;
  evaluation_version: string;
}

export interface EvolutionConfig {
  seed: number;
  population_size: number;
  generations: number;
  elite_count: number;
  mutation_probability: number;
  crossover_probability: number;
  random_immigrant_probability: number;
  max_model_calls: number;
  max_evaluations: number;
  max_wall_seconds: number;
  max_usd_estimate: number;
  max_concurrent_model_calls: number;
  finalist_count?: number;
  finalist_family_cap?: number;
  strategist_calls_per_generation?: number;
}

export interface Datasets {
  train_hash?: string;
  validation_hash?: string;
  test_hash?: string;
}

export interface ProviderRecord {
  name: string;
  mode?: string | null;
  model?: string | null;
  deployment_id?: string | null;
  app_id?: string | null;
}

export interface GenomeMetadata {
  family?: string | null;
  species?: string | null;
  provider_mode?: string | null;
  provider_role?: string | null;
  provider_run_id?: string | null;
  model?: string | null;
  deployment_id?: string | null;
  thread_id?: string | null;
  baseline_name?: string | null;
  seeded_baseline?: boolean;
  origin_operator?: string | null;
  role?: string | null;
}

export interface Genome {
  id: string;
  run_id: string;
  generation: number;
  parent_ids: string[];
  operator: string;
  payload: Record<string, unknown>;
  hypothesis: string;
  provider: string;
  created_at: string;
  content_hash: string;
  metadata: GenomeMetadata;
}

export interface Evaluation {
  genome_id: string;
  feasible: boolean;
  fitness: number | null;
  metrics: Record<string, number>;
  constraint_violations: string[];
  feedback: string;
  evaluator_version: string;
  data_split: string;
  elapsed_ms: number;
}

export interface Provenance {
  mode?: string | null;
  model_provider?: string | null;
  workflow_provider?: string | null;
  provider_run_id?: string | null;
  provider_mode?: string | null;
  evaluator_version?: string | null;
  latency_ms?: number | null;
  model?: string | null;
  deployment_id?: string | null;
  prompt_version?: string | null;
  started_at?: string | null;
  completed_at?: string | null;
  thread_id?: string | null;
}

export interface GeneEvent {
  schema_version?: string;
  event_id: string;
  run_id: string;
  sequence: number;
  timestamp: string;
  type: string;
  payload: Record<string, unknown>;
  provenance?: Provenance;
}

export interface OneShotBaseline {
  arm?: string;
  calls?: number;
  comparison_label?: string;
  compute_matched?: boolean;
  cost_usd?: number | null;
  feasible?: boolean | null;
  fitness?: number | null;
  reason?: string | null;
  status?: string;
  metrics?: Record<string, number> | null;
  model?: string | null;
  provider?: string | null;
  wall_seconds?: number | null;
  split?: string | null;
  [key: string]: unknown;
}

export interface RunExport {
  schema_version: string;
  run: RunMeta;
  objective: ObjectiveSpec;
  config: EvolutionConfig;
  datasets: Datasets;
  providers: ProviderRecord[];
  genomes: Genome[];
  evaluations: Evaluation[];
  events: GeneEvent[];
  champion_id: string | null;
  one_shot_baseline: OneShotBaseline | null;
  recorded_at?: string;
  integrity_sha256?: string;
}

export interface ShoppingItem {
  product_id?: string;
  color_id?: string;
  size?: string;
  role?: string;
  title?: string;
  price?: number;
  currency?: string;
  product_url?: string;
  image_url?: string;
  availability_checked_at?: string;
  fixture?: boolean;
}

export const EXPORT_KEYS = [
  "schema_version",
  "run",
  "objective",
  "config",
  "datasets",
  "providers",
  "genomes",
  "evaluations",
  "events",
  "champion_id",
  "one_shot_baseline",
  "integrity_sha256",
] as const;
