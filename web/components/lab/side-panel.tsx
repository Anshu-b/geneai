"use client";

import { useState } from "react";
import { colorSwatch, fitnessHelp, isFixtureProduct, metricLabel, outfitItems } from "@/lib/domain";
import { formatFitness, formatMetric, formatSeconds, formatUsd, shortHash } from "@/lib/format";
import {
  isCeilingReference,
  isHandwrittenBaseline,
  operatorLabel,
  providerCallFor,
  type GenomeView,
  type Projection,
} from "@/lib/project";
import type { Evaluation, Genome, RunExport } from "@/lib/types";

type Tab = "provenance" | "results" | "domain";

export function SidePanel({
  exp,
  projection,
  selected,
  onSelect,
}: {
  exp: RunExport;
  projection: Projection;
  selected: GenomeView | null;
  onSelect: (id: string) => void;
}) {
  const [tab, setTab] = useState<Tab>("provenance");
  const domainLabel = exp.run.domain === "shopping" ? "Outfit" : exp.run.domain === "cloud_scheduling" ? "Workload" : "Domain";
  const tabs: { id: Tab; label: string }[] = [
    { id: "provenance", label: "Provenance" },
    { id: "results", label: "Results" },
    { id: "domain", label: domainLabel },
  ];
  return (
    <aside className="border-line xl:border-l">
      <div className="flex gap-1 border-b border-line px-3 py-2" role="tablist" aria-label="Genome details">
        {tabs.map((item) => (
          <button
            key={item.id}
            type="button"
            role="tab"
            aria-selected={tab === item.id}
            className={`rounded-full px-3 py-1 text-sm ${tab === item.id ? "bg-ivory/10 text-ivory" : "text-dim"}`}
            onClick={() => setTab(item.id)}
          >
            {item.label}
          </button>
        ))}
      </div>
      <div className="ghost-scroll max-h-[80vh] overflow-auto p-4 xl:max-h-[calc(100vh-8.5rem)]">
        {tab === "provenance" ? (
          <Provenance exp={exp} projection={projection} selected={selected} onSelect={onSelect} />
        ) : null}
        {tab === "results" ? <Results exp={exp} projection={projection} onSelect={onSelect} /> : null}
        {tab === "domain" ? <DomainView exp={exp} projection={projection} selected={selected} /> : null}
      </div>
    </aside>
  );
}

function Provenance({
  exp,
  projection,
  selected,
  onSelect,
}: {
  exp: RunExport;
  projection: Projection;
  selected: GenomeView | null;
  onSelect: (id: string) => void;
}) {
  if (!selected) {
    return (
      <div data-testid="provenance-panel">
        <p className="kicker">Provenance</p>
        <p className="mt-3 font-serif text-2xl">Select a genome.</p>
        <p className="mt-2 text-sm leading-relaxed text-dim">
          The payload, parents, operator, and provider call are taken from the export. Test scores stay hidden until the champion is frozen.
        </p>
      </div>
    );
  }
  const genome = selected.genome;
  const call = providerCallFor(genome, exp.events, projection.sequence);
  const born = projection.byId.has(genome.id);
  return (
    <div data-testid="provenance-panel" className="space-y-5">
      {!born ? <p className="text-sm text-amber">This genome has not been born at this point in the replay.</p> : null}
      <div>
        <p className="kicker">{selected.status}{selected.laterPruned ? " · later pruned" : ""}</p>
        <h2 className="mt-1 font-serif text-3xl">{genome.id}</h2>
        <p className="mt-2 text-sm leading-relaxed text-dim">{genome.hypothesis || "No hypothesis text was recorded."}</p>
      </div>
      <dl className="grid grid-cols-2 gap-x-3 gap-y-3 text-sm">
        <Fact label="Generation" value={String(genome.generation)} />
        <Fact label="Family" value={genome.metadata.family || "—"} />
        <Fact label="Species" value={genome.metadata.species || "—"} />
        <Fact label="Operator" value={`${genome.operator} · ${operatorLabel(genome)}`} />
        <Fact label="Provider" value={genome.provider} />
        <Fact label="Role" value={genome.metadata.provider_role || genome.metadata.role || "—"} />
        <Fact label="Provider mode" value={genome.metadata.provider_mode || "—"} />
        <Fact label="Model" value={call?.model || genome.metadata.model || "—"} />
      </dl>
      <div>
        <p className="kicker">Provider call</p>
        {isHandwrittenBaseline(genome) ? (
          <p className="mt-2 text-sm text-dim">Hand-written baseline. No model call was made for this genome.</p>
        ) : call ? (
          <dl className="mt-2 space-y-1 text-sm">
            <div className="flex justify-between gap-3"><dt className="text-dim">Run id</dt><dd className="font-mono text-xs">{call.providerRunId}</dd></div>
            <div className="flex justify-between gap-3"><dt className="text-dim">Latency</dt><dd>{call.latencyMs == null ? "not recorded" : `${call.latencyMs} ms`}</dd></div>
            <div className="flex justify-between gap-3"><dt className="text-dim">Cost</dt><dd>{formatUsd(call.costUsd)}</dd></div>
            <div className="flex justify-between gap-3"><dt className="text-dim">Event</dt><dd className="font-mono text-xs">{call.callType}</dd></div>
            {call.deploymentId ? <div className="flex justify-between gap-3"><dt className="text-dim">Deployment</dt><dd className="font-mono text-xs">{call.deploymentId}</dd></div> : null}
          </dl>
        ) : (
          <p className="mt-2 text-sm text-dim">No completed provider call with this provider_run_id is in the log yet.</p>
        )}
        <p className="mt-2 break-all font-mono text-[10px] text-faint">hash {shortHash(genome.content_hash, 12)}</p>
      </div>
      <div>
        <p className="kicker">Parents</p>
        {genome.parent_ids.length === 0 ? (
          <p className="mt-2 text-sm text-dim">No parents. {operatorLabel(genome) === "immigrant" ? "Recorded as an immigrant." : "Recorded as a seed."}</p>
        ) : (
          <ul className="mt-2 space-y-1">
            {genome.parent_ids.map((id) => {
              const parent = exp.genomes.find((item) => item.id === id);
              return (
                <li key={id}>
                  <button type="button" className="text-sm text-teal underline-offset-2 hover:underline" onClick={() => onSelect(id)}>
                    {id}
                  </button>
                  <span className="text-sm text-dim"> · {parent ? `${parent.operator}, ${parent.metadata.family ?? "unspecified"}` : "missing from export"}</span>
                </li>
              );
            })}
          </ul>
        )}
      </div>
      <Scores view={selected} frozen={projection.championFrozen} testVisible={projection.testVisible} />
      <div>
        <p className="kicker">Payload</p>
        <pre className="mt-2 max-h-72 overflow-auto rounded border border-line bg-black/30 p-3 font-mono text-[11px] leading-relaxed text-ivory/90">
          {JSON.stringify(genome.payload, null, 2)}
        </pre>
      </div>
    </div>
  );
}

function Scores({ view, frozen, testVisible }: { view: GenomeView; frozen: boolean; testVisible: boolean }) {
  return (
    <div>
      <p className="kicker">Scores</p>
      <div className="mt-2 space-y-2">
        <ScoreRow label="Train" evaluation={view.train} empty="Not evaluated on train yet." />
        <ScoreRow label="Validation" evaluation={view.validation} empty="Not scored on validation. Only finalists are." />
        <ScoreRow
          label="Test"
          evaluation={view.test}
          empty={
            !frozen
              ? "Hidden until champion.frozen. The held-out split is not part of selection."
              : testVisible
                ? "This genome was not the frozen champion, so it has no held-out test."
                : "The champion is frozen. The test evaluation has not been reached in the replay."
          }
        />
      </div>
    </div>
  );
}

function ScoreRow({ label, evaluation, empty }: { label: string; evaluation: Evaluation | null; empty: string }) {
  if (!evaluation) return <p className="text-sm text-dim"><span className="text-ivory">{label}.</span> {empty}</p>;
  return (
    <div className="rounded border border-line px-3 py-2">
      <div className="flex items-baseline justify-between gap-3">
        <p className="text-sm text-ivory">{label}</p>
        <p className="font-mono text-sm">{formatFitness(evaluation.fitness)}</p>
      </div>
      <p className="text-xs text-dim">{evaluation.feasible ? "Feasible" : "Infeasible"} · evaluator {evaluation.elapsed_ms} ms</p>
      {evaluation.constraint_violations.length > 0 ? (
        <p className="mt-1 text-xs text-danger">{evaluation.constraint_violations.join(" ")}</p>
      ) : null}
      {evaluation.feedback ? <p className="mt-1 text-xs leading-relaxed text-dim">{evaluation.feedback}</p> : null}
    </div>
  );
}

function Results({
  exp,
  projection,
  onSelect,
}: {
  exp: RunExport;
  projection: Projection;
  onSelect: (id: string) => void;
}) {
  const champion = projection.championId ? projection.byId.get(projection.championId) ?? null : null;
  const baselines = exp.genomes.filter(isHandwrittenBaseline);
  const ceilings = exp.genomes.filter(isCeilingReference);
  const oneShot = exp.one_shot_baseline;
  const finalists = projection.views.filter((view) => view.validation);
  const comparison = oneShot?.comparison_label || "single attempt vs evolutionary search";
  return (
    <div data-testid="results-panel" className="space-y-5">
      <div>
        <p className="kicker">Comparison</p>
        <h2 className="mt-1 font-serif text-2xl leading-tight">{comparison}</h2>
        <p className="mt-2 text-sm leading-relaxed text-dim">{fitnessHelp(exp.run.domain)}</p>
        {oneShot?.compute_matched === false ? (
          <p className="mt-2 text-sm text-amber">Not compute-matched. GeneAI evaluates more candidates than the single attempt.</p>
        ) : null}
      </div>
      <ChampionCard champion={champion} frozen={projection.championFrozen} onSelect={onSelect} />
      <section>
        <p className="kicker">Validation finalists</p>
        {finalists.length === 0 ? (
          <p className="mt-2 text-sm text-dim">Finalists have not been scored on validation at this point.</p>
        ) : (
          <ul className="mt-2 divide-y divide-line border-y border-line">
            {finalists
              .slice()
              .sort((a, b) => (b.validation?.fitness ?? -Infinity) - (a.validation?.fitness ?? -Infinity))
              .map((view) => (
                <li key={view.genome.id} className="flex items-center justify-between gap-3 py-2 text-sm">
                  <button type="button" className="text-teal" onClick={() => onSelect(view.genome.id)}>
                    {view.genome.id}
                  </button>
                  <span className="text-dim">{view.genome.id === exp.champion_id && projection.championFrozen ? "frozen" : view.genome.metadata.family}</span>
                  <span className="font-mono">{formatFitness(view.validation?.fitness)}</span>
                </li>
              ))}
          </ul>
        )}
      </section>
      <section>
        <p className="kicker">Hand-written baselines · not GeneAI</p>
        {baselines.length === 0 ? (
          <p className="mt-2 text-sm text-dim">This export has no hand-written baselines.</p>
        ) : (
          <ul className="mt-2 space-y-2">
            {baselines.map((genome) => (
              <BaselineRow key={genome.id} genome={genome} projection={projection} onSelect={onSelect} />
            ))}
          </ul>
        )}
        <p className="mt-2 text-xs text-faint">Train split only, unless a test evaluation was recorded. These rows are reference heuristics, not evolutionary children.</p>
      </section>
      <section className="rounded border border-line p-3">
        <p className="kicker">One-shot arm</p>
        <p className="mt-2 text-sm text-ivory">{oneShot?.status === "skipped" ? "Not run" : oneShot?.status || "Absent"}</p>
        <p className="mt-1 font-mono text-xs text-dim">
          fitness {formatFitness(typeof oneShot?.fitness === "number" ? oneShot.fitness : null)} · calls {oneShot?.calls ?? "—"} · cost {formatUsd(oneShot?.cost_usd)}
        </p>
        {oneShot?.reason ? <p className="mt-2 whitespace-pre-wrap text-xs leading-relaxed text-dim">{oneShot.reason}</p> : null}
        {!oneShot ? <p className="mt-2 text-sm text-dim">This export has no one-shot record, so there is no delta to show.</p> : null}
        {oneShot && oneShot.fitness == null ? <p className="mt-2 text-xs text-faint">No one-shot score was recorded, so there is no quality delta.</p> : null}
      </section>
      <section>
        <p className="kicker">Ceiling reference · not GeneAI</p>
        {ceilings.length === 0 ? (
          <p className="mt-2 text-sm text-dim">No ceiling-reference row is in this export.</p>
        ) : (
          <ul className="mt-2 space-y-2">
            {ceilings.map((genome) => (
              <BaselineRow key={genome.id} genome={genome} projection={projection} onSelect={onSelect} />
            ))}
          </ul>
        )}
      </section>
      <section>
        <p className="kicker">Best train in the living population</p>
        {projection.best ? (
          <button type="button" className="mt-2 block text-left" onClick={() => onSelect(projection.best!.genomeId)}>
            <span className="font-mono text-sm text-teal">{projection.best.genomeId}</span>
            <span className="mt-1 block text-sm text-dim">
              Generation {projection.best.generation} train fitness {formatFitness(projection.best.fitness)}. This is not the held-out champion unless the ids match.
            </span>
          </button>
        ) : (
          <p className="mt-2 text-sm text-dim">No generation has settled yet.</p>
        )}
      </section>
      <Spark series={projection.series} />
      <BudgetLine projection={projection} />
    </div>
  );
}

function ChampionCard({
  champion,
  frozen,
  onSelect,
}: {
  champion: GenomeView | null;
  frozen: boolean;
  onSelect: (id: string) => void;
}) {
  if (!frozen || !champion) {
    return (
      <section className="rounded border border-line p-3">
        <p className="kicker">Champion</p>
        <p className="mt-2 text-sm text-dim">Not frozen yet. Selection up to here uses train scores only.</p>
      </section>
    );
  }
  return (
    <section className="rounded border border-amber/40 p-3">
      <p className="kicker text-amber">Champion · frozen on validation</p>
      <button type="button" className="mt-1 font-serif text-3xl text-ivory" onClick={() => onSelect(champion.genome.id)}>
        {champion.genome.id}
      </button>
      <p className="mt-2 text-sm text-dim">
        Validation {formatFitness(champion.validation?.fitness)} · test {champion.test ? formatFitness(champion.test.fitness) : "not opened yet"}
      </p>
      {champion.laterPruned ? (
        <p className="mt-2 text-sm leading-relaxed text-dim">
          This genome was later pruned from the living population. The held-out test still belongs to the frozen champion, not to whoever led train fitness afterward.
        </p>
      ) : null}
    </section>
  );
}

function BaselineRow({
  genome,
  projection,
  onSelect,
}: {
  genome: Genome;
  projection: Projection;
  onSelect: (id: string) => void;
}) {
  const view = projection.byId.get(genome.id);
  return (
    <li className="flex items-center justify-between gap-2 text-sm">
      <button type="button" className="text-left text-teal" onClick={() => onSelect(genome.id)}>
        {genome.metadata.baseline_name || genome.id}
        <span className="block font-mono text-[10px] text-faint">{genome.id}</span>
      </button>
      <span className="text-right font-mono text-xs">
        train {formatFitness(view?.train?.fitness)}
        <span className="block text-faint">test {view?.test ? formatFitness(view.test.fitness) : "not evaluated"}</span>
      </span>
    </li>
  );
}

function Spark({ series }: { series: Projection["series"] }) {
  if (series.length < 2) return <p className="text-sm text-dim">Not enough settled generations to plot best train fitness.</p>;
  const width = 320;
  const height = 96;
  const values = series.map((point) => point.fitness);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const points = series
    .map((point, index) => {
      const x = 8 + (index / (series.length - 1)) * (width - 16);
      const t = max === min ? 0.5 : (point.fitness - min) / (max - min);
      const y = height - 12 - t * (height - 24);
      return `${x},${y}`;
    })
    .join(" ");
  return (
    <section>
      <p className="kicker">Best train fitness by generation</p>
      <svg viewBox={`0 0 ${width} ${height}`} className="mt-2 w-full" role="img" aria-label="Best train fitness by generation">
        <polyline fill="none" stroke="#7ee0c6" strokeWidth="1.6" points={points} />
      </svg>
      <p className="font-mono text-[10px] text-faint">
        {formatFitness(min)} to {formatFitness(max)}. The line is the recorded best, including flats.
      </p>
    </section>
  );
}

function BudgetLine({ projection }: { projection: Projection }) {
  const budget = projection.budget;
  if (!budget) return null;
  return (
    <p className="font-mono text-[11px] leading-relaxed text-faint">
      Engine elapsed {formatSeconds(budget.elapsed_seconds)} · evaluations {budget.evaluations ?? "—"} · model calls {budget.model_calls ?? "—"} · spend {formatUsd(budget.spend_usd)}. This is budget.updated, not the replay duration.
    </p>
  );
}

function DomainView({
  exp,
  projection,
  selected,
}: {
  exp: RunExport;
  projection: Projection;
  selected: GenomeView | null;
}) {
  const view = selected ?? (projection.championFrozen && projection.championId ? projection.byId.get(projection.championId) ?? null : null);
  if (exp.run.domain === "shopping") return <Outfit exp={exp} view={view} />;
  if (exp.run.domain === "cloud_scheduling") return <Workload view={view} />;
  return <p className="text-sm text-dim">No domain view is defined for {exp.run.domain}. The lineage and scores above are still from the export.</p>;
}

function Workload({ view }: { view: GenomeView | null }) {
  const evaluation = view?.test ?? view?.validation ?? view?.train ?? null;
  return (
    <div>
      <p className="kicker">Simulator metrics</p>
      <p className="mt-2 text-sm leading-relaxed text-dim">
        These are evaluator outputs for the selected genome. The lab does not invent a live machine diagram.
      </p>
      {!view || !evaluation ? (
        <p className="mt-3 text-sm text-dim">Select a scored genome to read its metrics.</p>
      ) : (
        <MetricList evaluation={evaluation} />
      )}
    </div>
  );
}

function Outfit({ exp, view }: { exp: RunExport; view: GenomeView | null }) {
  if (!view) return <p className="text-sm text-dim">Select an outfit genome.</p>;
  const items = outfitItems(view.genome.payload);
  const currency = typeof exp.objective.constraints.currency === "string" ? exp.objective.constraints.currency : "";
  const spend = view.train?.metrics.spend ?? view.validation?.metrics.spend;
  return (
    <div>
      <p className="kicker">Catalog pieces</p>
      <p className="mt-2 text-sm leading-relaxed text-dim">
        {typeof view.genome.payload.styling_notes === "string" ? view.genome.payload.styling_notes : "No styling note was recorded."}
      </p>
      {spend != null ? (
        <p className="mt-2 font-mono text-xs text-dim">
          Outfit spend on the scored split: {currency} {formatMetric(spend)}. Per-item prices were not in the export.
        </p>
      ) : null}
      <ul className="mt-3 space-y-3">
        {items.map((item, index) => {
          const fixture = isFixtureProduct(item);
          const swatch = colorSwatch(item.color_id);
          const showLink = Boolean(item.product_url) && !fixture;
          return (
            <li key={`${item.product_id}-${index}`} className="rounded border border-line p-3">
              <div className="flex gap-3">
                {item.image_url && !fixture ? (
                  // Remote catalog shots are optional and may fail. The id remains the record.
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={item.image_url} alt={item.title || item.product_id || "Catalog item"} className="h-16 w-16 rounded object-cover" />
                ) : (
                  <span
                    className="grid h-16 w-16 place-items-center rounded border border-line text-[10px] text-faint"
                    style={swatch ? { background: swatch } : undefined}
                    aria-hidden
                  >
                    {!swatch ? "no image" : ""}
                  </span>
                )}
                <div className="min-w-0">
                  <p className="font-mono text-sm">{item.product_id || "unnamed"}</p>
                  <p className="text-xs text-dim">
                    {item.role || "role unknown"} · {item.color_id || "color unknown"} · {item.size || "size unknown"}
                  </p>
                  {fixture ? (
                    <p className="mt-1 inline-block rounded border border-amber/40 px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-widest text-amber" data-testid="fixture-badge">
                      fixture
                    </p>
                  ) : null}
                  {showLink ? (
                    <a className="mt-1 block text-sm text-teal underline-offset-2 hover:underline" href={item.product_url} target="_blank" rel="noreferrer">
                      {item.title || "Product page"}
                    </a>
                  ) : (
                    <p className="mt-1 text-xs text-faint">{fixture ? "No store link. Fixture rows are not purchasable." : "No product URL was recorded."}</p>
                  )}
                </div>
              </div>
            </li>
          );
        })}
      </ul>
      {items.length === 0 ? <p className="mt-3 text-sm text-dim">This genome has no outfit items.</p> : null}
    </div>
  );
}

function MetricList({ evaluation }: { evaluation: Evaluation }) {
  const entries = Object.entries(evaluation.metrics);
  return (
    <div className="mt-3">
      <p className="text-xs text-dim">
        {evaluation.data_split} · {evaluation.feasible ? "feasible" : "infeasible"} · fitness {formatFitness(evaluation.fitness)}
      </p>
      <dl className="mt-2 divide-y divide-line border-y border-line">
        {entries.map(([key, value]) => (
          <div key={key} className="flex items-baseline justify-between gap-3 py-1.5 text-sm">
            <dt className="text-dim">{metricLabel(key)}</dt>
            <dd className="font-mono">{formatMetric(value)}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

function Fact({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="kicker">{label}</dt>
      <dd className="mt-1 break-words text-ivory">{value}</dd>
    </div>
  );
}
