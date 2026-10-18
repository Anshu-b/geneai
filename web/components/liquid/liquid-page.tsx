"use client";

import { useEffect, useMemo, useState } from "react";
import { Header } from "@/components/header";
import { LineageSvg } from "@/components/lab/lineage-svg";
import { parseExport } from "@/lib/parse";
import { ancestryIds, genomeMap, layoutRun, projectRun, type Projection } from "@/lib/project";
import type { RunExport } from "@/lib/types";
import { RankLegend } from "@/components/rank-legend";
import { GENEAI_ROW, RANK_CLASS, rankColumn } from "@/lib/rank";
import {
  parseLiquid,
  SOURCE_COLOR,
  SOURCE_LABEL,
  takeaway,
  totalAssetsCents,
  usd,
  type LiquidCandidate,
  type LiquidFile,
  type ScenarioEval,
} from "@/lib/liquid";

const PROVIDER_LABEL: Record<string, string> = {
  veri: "Veri (Qwen2.5-7B)",
  creao: "CREAO agent",
  baseline: "Hand-written baseline",
  algorithmic: "Algorithmic operator",
  random: "Random immigrant",
};

export function LiquidPage({ src, runSrc }: { src: string; runSrc: string }) {
  const [file, setFile] = useState<LiquidFile | null>(null);
  const [exp, setExp] = useState<RunExport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [reduced, setReduced] = useState(false);

  useEffect(() => {
    const media = window.matchMedia("(prefers-reduced-motion: reduce)");
    setReduced(media.matches);
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const response = await fetch(src);
        if (!response.ok) throw new Error(`Could not fetch the Liquid run (${response.status}).`);
        const parsed = parseLiquid(await response.json());
        if (!cancelled) setFile(parsed);
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : "Could not read the Liquid run.");
      }
      try {
        const response = await fetch(runSrc);
        if (response.ok) {
          const parsed = parseExport(await response.json());
          if (!cancelled) setExp(parsed);
        }
      } catch {
        // The lineage graph is optional. The page says so below.
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [src, runSrc]);

  return (
    <div>
      <Header />
      <main className="mx-auto max-w-[1600px] px-4 py-8 md:px-6" data-testid="liquid-page">
        <div className="flex flex-wrap items-center gap-3">
          <p className="kicker">Liquid</p>
          <span className="rounded-full border border-amber/50 px-3 py-1 font-mono text-[10px] uppercase tracking-[0.16em] text-amber" data-testid="liquid-mode">
            Saved run replay
          </span>
        </div>
        <h1 className="mt-3 font-serif text-5xl tracking-tight text-ivory md:text-6xl">
          LIQUID <span className="text-dim">—</span> Find the cheapest dollar you own.
        </h1>
        <div className="mt-6 max-w-3xl rounded-2xl border border-line bg-raise px-5 py-4" data-testid="liquid-request">
          <p className="kicker">The request</p>
          <p className="mt-2 text-lg leading-relaxed text-ivory">
            “I need $6,000 for a laptop and a monitor setup. Keep at least $5,000 in savings and never sell my NVDA. I could sell stock (all of it is up, some short-term), draw a 7% line against stock I pledge, or use a 0% card that jumps to 24% after 60 days. Over 90 days, which mix costs me the least without leaving me exposed?”
          </p>
        </div>
        <p className="mt-3 max-w-3xl text-sm text-faint" data-testid="liquid-disclaimer">
          Fictional data. Simplified tax model. Not financial advice. No accounts are connected and nothing is bought, sold, or borrowed.
        </p>

        {error ? <p className="mt-6 text-sm text-danger" role="alert">{error}</p> : null}
        {!file && !error ? <p className="mt-6 text-sm text-dim" role="status">Reading the saved run…</p> : null}
        {file ? <Body file={file} /> : null}
        {file ? <Evolution file={file} exp={exp} reduced={reduced} /> : null}
        {file ? <Assumptions file={file} /> : null}
      </main>
    </div>
  );
}

function Body({ file }: { file: LiquidFile }) {
  const s = file.scenario;
  const geneai = file.candidates.find((c) => c.key === "geneai");
  const oneShot = file.candidates.find((c) => c.key === "one_shot");
  return (
    <>
      <section className="mt-12" data-testid="liquid-portfolio">
        <h2 className="font-serif text-3xl tracking-tight text-ivory">The fictional portfolio</h2>
        <div className="mt-5 grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          <Card title="Checking" value={usd(s.checking.balance_cents, 0)} note="Earns nothing." />
          <Card title="Savings" value={usd(s.savings.balance_cents, 0)} note={`${(s.savings.apy * 100).toFixed(1)}% APY · ${usd(s.savings.minimum_cents, 0)} emergency floor`} />
          <div className="rounded-2xl border border-dashed border-danger/50 p-4 md:col-span-2 xl:col-span-1" data-testid="liquid-credit">
            <p className="kicker">Credit (liabilities, not assets)</p>
            <div className="mt-3 grid gap-3 sm:grid-cols-2 xl:grid-cols-1">
              <div className="rounded-xl border border-line/70 bg-raise/40 px-3 py-2.5">
                <p className="font-mono text-sm text-ivory">Securities-backed line</p>
                <p className="mt-1 font-mono text-xs tabular-nums text-dim">{(s.sbl.apr * 100).toFixed(0)}% APR · up to {(s.sbl.max_ltv * 100).toFixed(0)}% LTV · max {usd(s.sbl.maximum_draw_cents, 0)}</p>
                <p className="mt-1 text-xs leading-relaxed text-dim">Pledged lots are locked and cannot be sold. A 30% drop past {(s.sbl.max_ltv * 100).toFixed(0)}% LTV is a margin call.</p>
              </div>
              <div className="rounded-xl border border-line/70 bg-raise/40 px-3 py-2.5">
                <p className="font-mono text-sm text-ivory">0%-intro card</p>
                <p className="mt-1 font-mono text-xs tabular-nums text-dim">Up to {usd(s.intro_card.limit_cents, 0)}</p>
                <p className="mt-1 text-xs leading-relaxed text-dim">{(s.intro_card.intro_apr * 100).toFixed(0)}% for {s.intro_card.intro_days} days, then {(s.intro_card.apr_after * 100).toFixed(0)}% APR.</p>
              </div>
            </div>
          </div>
        </div>
        <div className="mt-4 overflow-x-auto rounded-2xl border border-line bg-raise p-4" data-testid="liquid-lots">
          <p className="kicker">Taxable stock lots</p>
          <p className="mt-1 text-xs text-faint">If sold over the 90-day window. Money columns are right-aligned.</p>
          <table className="mt-3 w-full min-w-[640px] border-separate border-spacing-0 text-sm">
            <colgroup>
              <col className="w-[18%]" />
              <col className="w-[14%]" />
              <col className="w-[14%]" />
              <col className="w-[14%]" />
              <col className="w-[20%]" />
              <col className="w-[20%]" />
            </colgroup>
            <thead>
              <tr className="text-faint">
                <th className="py-2 pr-4 text-left font-normal">Lot</th>
                <th className="py-2 px-3 text-right font-normal">Value</th>
                <th className="py-2 px-3 text-right font-normal">Paid</th>
                <th className="py-2 px-3 text-right font-normal">Gain</th>
                <th className="py-2 px-3 text-right font-normal">Tax</th>
                <th className="py-2 pl-3 text-right font-normal">Dividend lost</th>
              </tr>
            </thead>
            <tbody className="font-mono tabular-nums">
              {s.lots.map((lot) => {
                const gain = lot.market_value_cents - lot.basis_cents;
                const locked = !lot.sellable || s.constraints.protect.includes(lot.symbol);
                const rate = lot.holding_period === "short" ? s.assumptions.short_term_gain_rate : s.assumptions.long_term_gain_rate;
                const taxLabel = `${lot.holding_period === "short" ? "short" : "long"} ${(rate * 100).toFixed(0)}%`;
                const taxCents = Math.round(Math.max(0, gain) * rate);
                const divLost = lot.dividend_yield ? Math.round(lot.market_value_cents * lot.dividend_yield * (s.horizon_days / 365)) : 0;
                return (
                  <tr key={lot.lot_id} className="border-t border-line">
                    <td className="border-t border-line py-2.5 pr-4 align-top">
                      <span className="font-sans text-ivory">{lot.symbol}</span>
                      {locked ? <span className="mt-1 block w-fit rounded bg-amber/15 px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-amber">protected</span> : null}
                    </td>
                    <td className="border-t border-line py-2.5 px-3 text-right text-ivory">{usd(lot.market_value_cents, 0)}</td>
                    <td className="border-t border-line py-2.5 px-3 text-right text-dim">{usd(lot.basis_cents, 0)}</td>
                    <td className="border-t border-line py-2.5 px-3 text-right text-ivory">{usd(gain, 0)}</td>
                    <td className="border-t border-line py-2.5 px-3 text-right text-dim">
                      <span className="block text-ivory">{usd(taxCents, 0)}</span>
                      <span className="block text-[11px]">{taxLabel}</span>
                    </td>
                    <td className="border-t border-line py-2.5 pl-3 text-right text-dim">{lot.dividend_yield ? usd(divLost, 0) : "—"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p className="mt-3 text-sm text-dim">Total assets {usd(totalAssetsCents(s), 0)} (credit excluded; pledged lots are still assets, just locked). Purchase {usd(s.purchase_cents, 0)} over {s.horizon_days} days. No source is free.</p>
      </section>

      <section className="mt-14" data-testid="liquid-compare">
        <h2 className="font-serif text-3xl tracking-tight text-ivory">GeneAI&apos;s policy vs a single GPT-5.5 answer</h2>
        <p className="mt-2 max-w-3xl text-sm text-dim">Both are funding <em>policies</em>, run through the same deterministic evaluator on the default $6,000 purchase. Single attempt vs evolutionary search, not compute-matched.</p>
        <div className="mt-6 grid gap-5 lg:grid-cols-2">
          {geneai ? <CandidatePanel c={geneai} accent /> : null}
          {oneShot ? <CandidatePanel c={oneShot} /> : null}
        </div>
      </section>

      <Scoreboard file={file} />
    </>
  );
}

function Card({ title, value, note }: { title: string; value: string; note: string }) {
  return (
    <div className="rounded-2xl border border-line bg-raise p-4">
      <p className="kicker">{title}</p>
      <p className="mt-2 font-mono text-2xl text-ivory">{value}</p>
      <p className="mt-1 text-sm text-dim">{note}</p>
    </div>
  );
}

function CandidatePanel({ c, accent }: { c: LiquidCandidate; accent?: boolean }) {
  const meta = c.oneshot_meta;
  return (
    <article className={`rounded-2xl border p-5 ${accent ? "border-teal/60 bg-teal/5" : "border-line bg-raise"}`} data-testid={`liquid-panel-${c.key}`}>
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="font-serif text-2xl text-ivory">{c.label}</h3>
        <span className="font-mono text-xs text-faint">{c.provider}{meta && typeof meta.latency_ms === "number" ? ` · ${(meta.latency_ms / 1000).toFixed(1)}s` : ""}</span>
      </div>
      {!c.valid || !c.base ? (
        <p className="mt-4 text-sm text-danger">Invalid policy, not repaired: {c.invalid_reason ?? "not recorded"}</p>
      ) : (
        <>
          <p className="mt-3 text-base leading-relaxed text-ivory" data-testid={`liquid-summary-${c.key}`}>{c.summary}</p>
          <AllocationBar e={c.base} />
          <Ledger e={c.base} />
          <Stress e={c.base} />
        </>
      )}
    </article>
  );
}

function AllocationBar({ e }: { e: ScenarioEval }) {
  const total = Object.values(e.allocation_cents).reduce((a, b) => a + b, 0) || 1;
  const entries = Object.entries(e.allocation_cents).filter(([, v]) => v > 0);
  return (
    <div className="mt-5">
      <p className="kicker">Where the money comes from</p>
      <div className="mt-2 flex h-9 w-full overflow-hidden rounded-lg border border-line" role="img" aria-label="Funding allocation">
        {entries.map(([k, v]) => (
          <div key={k} style={{ width: `${(v / total) * 100}%`, background: SOURCE_COLOR[k] ?? "var(--prov-other)" }} className="flex items-center justify-center overflow-hidden whitespace-nowrap px-1 font-mono text-[11px] text-black/80" title={`${SOURCE_LABEL[k] ?? k} ${usd(v, 0)}`}>
            {v / total > 0.12 ? `${SOURCE_LABEL[k] ?? k} ${usd(v, 0)}` : ""}
          </div>
        ))}
      </div>
      <p className="mt-1.5 flex flex-wrap gap-x-4 gap-y-1 font-mono text-xs text-dim">
        {entries.map(([k, v]) => (
          <span key={k}><span className="mr-1 inline-block h-2 w-2 rounded-full" style={{ background: SOURCE_COLOR[k] }} />{SOURCE_LABEL[k] ?? k} {usd(v, 0)}</span>
        ))}
      </p>
    </div>
  );
}

function Ledger({ e }: { e: ScenarioEval }) {
  const rows: [string, number][] = [
    ["Tax on realized gains", e.costs_cents.estimated_tax],
    ["Savings interest given up", e.costs_cents.foregone_savings_interest],
    ["Dividends given up on sold lots", e.costs_cents.dividends_forgone],
    ["Securities-backed line interest", e.costs_cents.sbl_interest],
    ["Card interest (after the 0% period)", e.costs_cents.card_interest],
    ["Fees", e.costs_cents.fees],
  ];
  return (
    <div className="mt-5">
      <p className="kicker">Modeled cost over 90 days</p>
      <table className="mt-2 w-full table-fixed text-sm">
        <colgroup>
          <col className="w-[70%]" />
          <col className="w-[30%]" />
        </colgroup>
        <tbody className="font-mono tabular-nums">
          {rows.map(([label, cents]) => (
            <tr key={label} className="border-t border-line">
              <td className="py-1.5 pr-4 font-sans text-dim">{label}</td>
              <td className="py-1.5 text-right text-ivory">{usd(cents)}</td>
            </tr>
          ))}
          <tr className="border-t border-ivory/30">
            <td className="py-2 pr-4 font-sans text-ivory">Total modeled cost</td>
            <td className="py-2 text-right text-lg text-ivory">{usd(e.total_cost_cents)}</td>
          </tr>
        </tbody>
      </table>
      {e.stress.post_purchase.pledged_lots?.length ? <p className="mt-1 text-xs text-faint">Pledged as collateral (locked, not sold): {e.stress.post_purchase.pledged_lots.map((l) => SOURCE_LABEL[l] ?? l).join(", ")} · {usd(e.stress.post_purchase.pledged_cents ?? 0, 0)}.</p> : null}
    </div>
  );
}

function Stress({ e }: { e: ScenarioEval }) {
  const items = [
    { k: "rate", t: e.stress.rate_shock.label, ok: e.stress.rate_shock.passed, d: e.stress.rate_shock.passed ? "no rate exposure" : `+${usd(e.stress.rate_shock.extra_interest_cents)} interest` },
    { k: "market", t: e.stress.market_drop.label, ok: e.stress.market_drop.passed, d: e.stress.market_drop.loan_to_value_after_drop == null ? "nothing borrowed against stock" : `line at ${(e.stress.market_drop.loan_to_value_after_drop * 100).toFixed(0)}% LTV after the drop${e.stress.market_drop.margin_call ? " · margin call" : ""}` },
    { k: "emergency", t: e.stress.emergency.label, ok: e.stress.emergency.passed, d: `${usd(e.stress.emergency.cash_above_reserve_cents, 0)} cash above the reserve` },
  ];
  return (
    <div className="mt-5">
      <p className="kicker">Stress tests · risk penalty {usd(e.risk_penalty_cents)}</p>
      <ul className="mt-2 grid gap-1.5 text-sm">
        {items.map((i) => (
          <li key={i.k} className="flex items-baseline gap-2">
            <span className={`font-mono text-base ${i.ok ? "text-teal" : "text-danger"}`}>{i.ok ? "✓" : "✗"}</span>
            <span className="text-ivory">{i.t}</span>
            <span className="text-dim">· {i.d}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function Scoreboard({ file }: { file: LiquidFile }) {
  const rows = file.candidates;
  const cols: { key: string; label: string; hint: string; better: "lower" | "higher"; get: (c: LiquidCandidate) => number | null | undefined; fmt: (v: number) => string }[] = [
    { key: "cost", label: "Modeled cost", hint: "default $6,000", better: "lower", get: (c) => c.base?.total_cost_cents, fmt: (v) => usd(v) },
    { key: "risk", label: "Risk penalty", hint: "default $6,000", better: "lower", get: (c) => c.base?.risk_penalty_cents, fmt: (v) => usd(v) },
    { key: "stress", label: "Stress passed", hint: "of 3", better: "higher", get: (c) => c.base?.stress_passes, fmt: (v) => `${v}/3` },
    { key: "val", label: "Validation fitness", hint: `${file.variants.validation?.length ?? 0} scenarios`, better: "higher", get: (c) => c.splits?.validation.fitness, fmt: (v) => v.toFixed(2) },
    { key: "test", label: "Held-out test fitness", hint: `${file.variants.test?.length ?? 0} unseen scenarios`, better: "higher", get: (c) => c.splits?.test.fitness, fmt: (v) => v.toFixed(2) },
  ];
  const ranks = cols.map((col) => rankColumn(rows.map((r) => (r.valid ? col.get(r) : null)), col.better));
  return (
    <section className="mt-14" data-testid="liquid-scoreboard">
      <h2 className="font-serif text-3xl tracking-tight text-ivory">Every policy, same evaluator</h2>
      <p className="mt-2 max-w-3xl text-sm text-dim">Fitness = −(modeled cost + risk penalty) in dollars, averaged over scenario variants; 0 is perfect. Fitness is a score, not a bill.</p>
      <div className="mt-5 overflow-x-auto">
        <table className="w-full min-w-[760px] border-separate border-spacing-0 text-sm">
          <thead>
            <tr className="text-left text-faint">
              <th className="py-2 pr-4 font-normal">Policy</th>
              {cols.map((c) => (
                <th key={c.key} className="px-3 py-2 text-right font-normal">{c.label}<span className="block text-[11px] text-faint/80">{c.hint}</span></th>
              ))}
            </tr>
          </thead>
          <tbody className="font-mono tabular-nums">
            {rows.map((r, ri) => (
              <tr key={r.key} className={`border-t border-line ${r.key === "geneai" ? GENEAI_ROW : ""}`}>
                <td className="border-t border-line py-2.5 pr-4 font-sans text-ivory">{r.label}<span className="block text-[11px] text-faint">{r.provider}</span></td>
                {cols.map((c, ci) => {
                  const v = r.valid ? c.get(r) : null;
                  const rank = ranks[ci][ri];
                  return (
                    <td key={c.key} className="border-t border-line px-3 py-2.5 text-right">
                      {v == null ? <span className="text-faint">{r.valid ? "Not recorded" : "invalid"}</span> : <span className={RANK_CLASS[rank]}>{rank === "best" ? "✓ " : ""}{c.fmt(v)}</span>}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <RankLegend note="Same evaluator and constraints for every row." />
      <p className="mt-4 max-w-3xl text-base text-ivory" data-testid="liquid-takeaway">{takeaway(file)}</p>
    </section>
  );
}

function Evolution({ file, exp, reduced }: { file: LiquidFile; exp: RunExport | null; reduced: boolean }) {
  const layout = useMemo(() => (exp ? layoutRun(exp) : null), [exp]);
  const projection = useMemo(() => {
    if (!exp || exp.events.length === 0) return null;
    return projectRun(exp, exp.events[exp.events.length - 1]?.sequence ?? 0);
  }, [exp]);
  const genomes = useMemo(() => (exp ? genomeMap(exp) : new Map()), [exp]);
  const ancestry = useMemo(() => ancestryIds(exp?.champion_id ?? null, genomes), [exp?.champion_id, genomes]);
  const origin = file.provenance.behavior_origin;
  const counts = Object.entries(file.provenance.genomes_by_provider).sort((a, b) => b[1] - a[1]);
  return (
    <section className="mt-16" data-testid="liquid-evolution">
      <h2 className="font-serif text-3xl tracking-tight text-ivory">How the population evolved</h2>
      <p className="mt-2 max-w-3xl text-sm leading-relaxed text-dim">
        Run <span className="font-mono">{file.run.id}</span> · {file.run.generations_completed} generations · population {file.run.population_size} · {file.run.model_calls} model calls · {file.run.wall_seconds != null ? `${Math.round(file.run.wall_seconds)}s` : "time not recorded"}. Champion picked on validation scenarios, then scored once on held-out test scenarios.
      </p>
      <p className="mt-2 max-w-3xl text-sm text-dim" data-testid="liquid-provenance">
        Who wrote the genomes: {counts.map(([p, n]) => `${PROVIDER_LABEL[p] ?? p} ${n}`).join(" · ")}. The champion <span className="font-mono">{origin.id}</span> came from {PROVIDER_LABEL[origin.provider] ?? origin.provider} ({origin.operator}, generation {origin.generation}).
      </p>
      {projection ? <FitnessStrip projection={projection} /> : null}
      {exp && projection && layout ? (
        <LineageSvg
          layout={layout}
          projection={projection}
          ancestry={ancestry}
          follow={false}
          filter="all"
          family="all"
          selectedId={exp.champion_id}
          onSelect={() => undefined}
          focusGeneration={null}
          beatId="final"
          beatPhase="frozen"
          playing={false}
          reduced={reduced}
        />
      ) : (
        <p className="mt-4 text-sm text-dim">Family tree not loaded.</p>
      )}
      <div className="mt-6">
        <p className="kicker">Champion ancestry</p>
        <ol className="mt-2 grid gap-1.5 text-sm">
          {file.champion_lineage.map((n) => (
            <li key={n.id} className="flex flex-wrap gap-x-2 text-dim">
              <span className="font-mono text-ivory">{n.id}</span>
              <span>gen {n.generation} · {n.operator} · {PROVIDER_LABEL[n.provider] ?? n.provider}{n.parent_ids.length ? ` · from ${n.parent_ids.join(" + ")}` : ""}</span>
              <span className="font-mono">{n.train_fitness != null ? n.train_fitness.toFixed(2) : ""}</span>
              <span className="w-full pl-4 text-faint">{n.hypothesis}</span>
            </li>
          ))}
        </ol>
      </div>
    </section>
  );
}

function FitnessStrip({ projection }: { projection: Projection }) {
  const best = new Map<number, number>();
  for (const view of projection.views) {
    if (view.trainFitness == null) continue;
    const current = best.get(view.genome.generation);
    if (current == null || view.trainFitness > current) best.set(view.genome.generation, view.trainFitness);
  }
  const points = [...best.entries()].sort((a, b) => a[0] - b[0]).map(([generation, fitness]) => ({ generation, fitness }));
  if (points.length < 2) return null;
  const width = 1200;
  const height = 260;
  const pad = 40;
  const padX = 60;
  const min = Math.min(...points.map((p) => p.fitness));
  const max = Math.max(...points.map((p) => p.fitness));
  const flat = max - min < 1e-9;
  const span = Math.max(1e-6, max - min);
  const xOf = (i: number) => padX + (i / (points.length - 1)) * (width - padX * 2);
  const yOf = (f: number) => (flat ? height / 2 : pad + (1 - (f - min) / span) * (height - pad * 2));
  const d = points.map((p, i) => `${i === 0 ? "M" : "L"}${xOf(i)},${yOf(p.fitness)}`).join(" ");
  return (
    <figure className="mt-5 mb-3" data-testid="fitness-generations">
      <figcaption className="kicker">Best training fitness in each generation. Higher is better. The champion is chosen on validation, so it need not be the training best.{flat ? " The best training score did not change in this run." : ""}</figcaption>
      <svg viewBox={`0 0 ${width} ${height}`} className="mt-2 h-auto w-full" role="img" aria-label="Best fitness by generation">
        <path d={d} fill="none" stroke="var(--teal)" strokeWidth="3" strokeLinejoin="round" strokeLinecap="round" />
        {points.map((p, i) => (
          <g key={p.generation}>
            <circle cx={xOf(i)} cy={yOf(p.fitness)} r="7" fill="var(--teal)" />
            <text x={xOf(i)} y={yOf(p.fitness) - 14} textAnchor="middle" fill="var(--ivory)" fontSize="16" fontFamily="IBM Plex Mono, ui-monospace, monospace">{p.fitness.toFixed(2)}</text>
            <text x={xOf(i)} y={height - 4} textAnchor="middle" fill="var(--faint)" fontSize="15" fontFamily="IBM Plex Mono, ui-monospace, monospace">gen {p.generation}</text>
          </g>
        ))}
      </svg>
    </figure>
  );
}

function Assumptions({ file }: { file: LiquidFile }) {
  return (
    <section className="mt-16 rounded-2xl border border-line bg-raise p-5 text-sm leading-relaxed text-dim" data-testid="liquid-assumptions">
      <p className="kicker">Assumptions</p>
      <ul className="mt-2 list-disc space-y-1 pl-5">
        <li>Cost: {String(file.model.cost_formula)}.</li>
        <li>Risk (shown separately): {String(file.model.risk_formula)}.</li>
        <li>{String(file.model.fitness_formula)}</li>
        <li>Simple interest over 90 days. Flat 30% short-term and 20% long-term gains rates. Wash sales, state tax, price moves on unsold stock, and expected stock returns are not modeled. The card&apos;s 24% applies only to the 30 days after its 0% period inside the horizon.</li>
        <li>Train, validation, and test are fictional variants: {Object.entries(file.variants).map(([split, rows]) => `${split}: ${rows.map((r) => r.note).join("; ")}`).join(" · ")}.</li>
        <li>Evaluator {file.run.evaluator_version}. Fictional data. Simplified tax model. Not financial advice. Selling appreciated stock can trigger tax; borrowing creates debt.</li>
      </ul>
    </section>
  );
}
