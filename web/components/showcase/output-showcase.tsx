"use client";

import { useEffect, useMemo, useState } from "react";
import {
  budgetLine,
  cloudScoreboard,
  interpretShowcaseResponse,
  isManualBaseline,
  isSampleShowcase,
  keyScoreLines,
  originCredit,
  type CloudPolicy,
  type ScoreRow,
  type OutfitCard,
  type ShowcaseFile,
  type ShowcaseLoad,
} from "@/lib/showcase";
import { cloudPolicySummary, outfitSummary, plainWhy, SHOPPING_WEIGHTS, shopperRequest, type Mark } from "@/lib/plain";

const JOB_COLOR: Record<string, string> = {
  interactive: "var(--job-interactive)",
  batch: "var(--job-batch)",
  gpu: "var(--job-gpu)",
  highmem: "var(--job-highmem)",
};

export interface RunFacts {
  /** Highest generation index in the export (generation 0 is the seed population). */
  lastGeneration: number;
  championId: string | null;
  championGeneration: number | null;
}

export function OutputShowcase({ src, facts }: { src: string; facts?: RunFacts | null }) {
  const [load, setLoad] = useState<ShowcaseLoad | { status: "loading" }>({ status: "loading" });

  useEffect(() => {
    const controller = new AbortController();
    setLoad({ status: "loading" });
    fetch(src, { signal: controller.signal })
      .then(async (response) => {
        if (response.status === 404) return interpretShowcaseResponse(404, null);
        const body = await response.json().catch(() => null);
        return interpretShowcaseResponse(response.status, body);
      })
      .then((next) => setLoad(next))
      .catch((error: unknown) => {
        if (error instanceof DOMException && error.name === "AbortError") return;
        setLoad({ status: "invalid", message: "The showcase file could not be fetched." });
      });
    return () => controller.abort();
  }, [src]);

  if (load.status === "loading") {
    return <p className="mt-8 text-sm text-dim">Reading what each system produced…</p>;
  }
  if (load.status === "missing") {
    return (
      <p className="mt-8 max-w-2xl text-sm leading-relaxed text-dim" data-testid="showcase-missing">
        No showcase file for this run yet. The lineage below is the recorded export.
      </p>
    );
  }
  if (load.status === "invalid") {
    return <p className="mt-8 text-sm text-danger">{load.message}</p>;
  }
  return <ShowcaseBody file={load.file} facts={facts ?? null} />;
}

function ShowcaseBody({ file, facts }: { file: ShowcaseFile; facts: RunFacts | null }) {
  const sample = isSampleShowcase(file);
  return (
    <section className="mt-8" data-testid="output-showcase">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <p className="kicker">What they produced</p>
        {sample ? (
          <span className="rounded-full border border-amber/50 px-3 py-1 font-mono text-[10px] uppercase tracking-[0.16em] text-amber" data-testid="showcase-sample">
            Sample
          </span>
        ) : null}
      </div>
      {file.domain === "shopping" && file.shopping ? <ShoppingShowcase file={file} /> : null}
      {file.domain === "cloud_scheduling" && file.cloud ? <CloudShowcaseView file={file} /> : null}
      {file.domain === "cloud_scheduling" && file.cloud ? <Scoreboard file={file} facts={facts} /> : <ScoreLines file={file} />}
    </section>
  );
}

function CloudShowcaseView({ file }: { file: ShowcaseFile }) {
  const cloud = file.cloud!;
  const policies = cloud.policies.filter((policy) => !isManualBaseline(policy.key, policy.provider));
  if (policies.length === 0) return <p className="mt-4 text-sm text-dim">This showcase has no agentic policies.</p>;
  // One time axis for every panel so the charts compare directly.
  const span = Math.max(1, ...policies.flatMap((policy) => policy.schedule.map((bar) => bar.end_min)), ...cloud.jobs.map((job) => job.deadline_min));
  return (
    <div className="mt-4 grid gap-4 lg:grid-cols-2">
      {policies.map((policy) => (
        <article key={policy.key} className="glass rounded-3xl p-4">
          <h2 className="font-serif text-2xl text-ivory">{policy.label}</h2>
          <div className="mt-3">
            <Gantt cloud={cloud} policy={policy} span={span} />
          </div>
          <div className="mt-4">
            <PolicyDetail policy={policy} />
          </div>
        </article>
      ))}
    </div>
  );
}

function PolicyDetail({ policy }: { policy: CloudPolicy }) {
  const credit = originCredit(policy.provider, policy.behavior_origin);
  return (
    <div>
      {credit ? <p className="text-sm text-ivory" data-testid="behavior-origin">{credit}</p> : null}
      {cloudPolicySummary(policy) ? (
        <p className="mt-2 text-base leading-relaxed text-ivory" data-testid="policy-summary">{cloudPolicySummary(policy)}</p>
      ) : null}
      {policy.description ? <p className="mt-2 text-sm text-dim">{plainDescription(policy)}</p> : null}
      {!policy.feasible && policy.infeasible_reason ? (
        <p className="mt-2 text-sm text-danger">{policy.infeasible_reason}</p>
      ) : null}
      {policy.rules_plain && policy.rules_plain.length > 0 ? (
        <ul className="mt-3 space-y-1 text-sm text-dim" aria-label="Exact rules">
          {policy.rules_plain.map((rule) => (
            <li key={rule} className="flex gap-2"><span className="text-teal">·</span><span>{rule}</span></li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}

function Gantt({ cloud, policy, span }: { cloud: NonNullable<ShowcaseFile["cloud"]>; policy: CloudPolicy; span: number }) {
  const jobs = new Map(cloud.jobs.map((job) => [job.id, job]));
  const late = policy.schedule.filter((bar) => bar.missed_deadline).length;
  const label = 108;
  const row = 34;
  const chart = 720;
  const width = label + chart;
  const height = cloud.machines.length * row + 26;
  return (
    <div className="overflow-x-auto" data-testid="schedule-gantt">
      <svg viewBox={`0 0 ${width} ${height}`} className="min-w-[640px] w-full" role="img" aria-label={`${policy.label} schedule`}>
        {cloud.machines.map((machine, index) => {
          const y = 18 + index * row;
          return (
            <g key={machine.id}>
              <text x={0} y={y + 16} fill="var(--dim)" fontSize="11" fontFamily="IBM Plex Mono, ui-monospace, monospace">{machine.id}</text>
              <line x1={label} x2={width} y1={y + row - 6} y2={y + row - 6} stroke="var(--line)" />
              {policy.schedule.filter((bar) => bar.machine_id === machine.id).map((bar) => {
                const job = jobs.get(bar.job_id);
                const x = label + (bar.start_min / span) * chart;
                const w = Math.max(4, ((bar.end_min - bar.start_min) / span) * chart);
                const fill = bar.missed_deadline ? "var(--danger)" : JOB_COLOR[job?.class || ""] || "var(--faint)";
                return (
                  <g key={`${bar.job_id}-${bar.start_min}`} data-late={bar.missed_deadline ? "true" : undefined}>
                    <rect x={x} y={y} width={w} height={20} rx={4} fill={fill} opacity={bar.missed_deadline ? 0.95 : 0.55} stroke={bar.missed_deadline ? "var(--ivory)" : "transparent"} strokeWidth={bar.missed_deadline ? 1.6 : 0} strokeDasharray={bar.missed_deadline ? "3 2" : undefined} />
                    <text x={x + 6} y={y + 14} fill={bar.missed_deadline ? "var(--bg)" : "var(--ivory)"} fontSize="10" fontWeight={bar.missed_deadline ? 700 : 400} fontFamily="IBM Plex Mono, ui-monospace, monospace">{bar.job_id}{bar.missed_deadline ? " LATE" : ""}</text>
                  </g>
                );
              })}
            </g>
          );
        })}
        <text x={label} y={height - 2} fill="var(--faint)" fontSize="10" fontFamily="IBM Plex Mono, ui-monospace, monospace">0</text>
        <text x={width} y={height - 2} textAnchor="end" fill="var(--faint)" fontSize="10" fontFamily="IBM Plex Mono, ui-monospace, monospace">{compactMin(span)} min</text>
      </svg>
      <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-dim" data-testid="gantt-legend">
        <span className="flex items-center gap-1.5"><span className="inline-block h-3 w-5 rounded-sm border border-dashed border-ivory bg-danger" />Late: finished after its deadline ({late})</span>
        <span className="flex items-center gap-1.5"><span className="inline-block h-3 w-5 rounded-sm bg-[var(--job-batch)] opacity-60" />On time</span>
        <span className="text-faint">Same 0–{compactMin(span)} min axis in both panels.</span>
      </div>
    </div>
  );
}

function ShoppingShowcase({ file }: { file: ShowcaseFile }) {
  const shopping = file.shopping!;
  const outfits = shopping.outfits.filter((outfit) => !isManualBaseline(outfit.key, outfit.provider));
  const shopper = shopping.shopper;
  return (
    <div className="mt-4">
      <article className="glass rounded-3xl p-5" data-testid="shopper-request">
        <p className="kicker">The request</p>
        <p className="mt-2 font-serif text-2xl leading-snug text-ivory">{shopperRequest(shopper)}</p>
        <ul className="mt-3 flex flex-wrap gap-2 text-xs text-dim">
          <li className="rounded-full border border-line px-3 py-1">Budget ${shopper.budget_usd}</li>
          {shopper.style ? <li className="rounded-full border border-line px-3 py-1">Palette: {shopper.style}</li> : null}
          {(shopper.occasions ?? []).map((occasion) => <li key={occasion} className="rounded-full border border-line px-3 py-1">Occasion: {occasion}</li>)}
          {shopper.apparel_size ? <li className="rounded-full border border-line px-3 py-1">Size {shopper.apparel_size}</li> : null}
          {shopper.shoe_size != null ? <li className="rounded-full border border-line px-3 py-1">Shoe {shopper.shoe_size}</li> : null}
          {(shopper.likes ?? []).map((like) => <li key={like} className="rounded-full border border-teal/40 px-3 py-1 text-teal">Likes {like}</li>)}
          {(shopper.dislikes ?? []).map((dislike) => <li key={dislike} className="rounded-full border border-danger/40 px-3 py-1 text-danger">Dislikes {dislike}</li>)}
        </ul>
      </article>
      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        {outfits.map((outfit) => (
          <Outfit key={outfit.key} outfit={outfit} budget={shopper.budget_usd} />
        ))}
      </div>
      <p className="mt-3 text-sm text-dim" data-testid="score-weights">{SHOPPING_WEIGHTS}</p>
    </div>
  );
}

function ScoreLines({ file }: { file: ShowcaseFile }) {
  const lines = useMemo(() => keyScoreLines(file), [file]);
  if (lines.length === 0) return null;
  return (
    <ul className="mt-4 space-y-1" data-testid="key-scores">
      {lines.map((line) => (
        <li key={line} className="text-base leading-relaxed text-ivory">{line}</li>
      ))}
    </ul>
  );
}

function Outfit({ outfit, budget }: { outfit: OutfitCard; budget: number }) {
  const credit = originCredit(outfit.provider, outfit.behavior_origin);
  return (
    <article className={`rounded-2xl border p-4 ${outfit.feasible ? "border-line bg-raise/40" : "border-danger/50 bg-danger/5"}`} data-testid="outfit-card">
      <p className="font-serif text-2xl text-ivory">{outfit.label}</p>
      <p className="mt-1 font-mono text-[10px] uppercase tracking-widest text-faint">
        {outfit.provider || "provider"}{outfit.genome_id ? ` · ${outfit.genome_id}` : ""}
      </p>
      {credit ? <p className="mt-2 text-sm text-ivory" data-testid="behavior-origin">{credit}</p> : null}
      <p className="mt-3 text-base leading-relaxed text-ivory" data-testid="outfit-summary">{outfitSummary(outfit, budget)}</p>
      <p className={`mt-2 text-sm ${outfit.feasible ? "text-dim" : "text-danger"}`}>{budgetLine(outfit.total_usd, budget)}</p>
      {!outfit.feasible && outfit.infeasible_reason && !(outfit.why ?? []).includes(outfit.infeasible_reason) ? (
        <p className="mt-2 text-sm leading-relaxed text-danger">{plainWhy([outfit.infeasible_reason])[0].text}</p>
      ) : null}
      <ul className="mt-4 space-y-3">
        {outfit.items.map((item) => (
          <li key={item.id} className="flex gap-3">
            {item.image_url ? (
              // Real showcase files may point at shop CDNs. next/image would need every host.
              // eslint-disable-next-line @next/next/no-img-element
              <img src={item.image_url} alt="" className="h-16 w-14 rounded-md border border-line object-cover" />
            ) : (
              <span className="h-16 w-14 rounded-md border border-line bg-raise-2" />
            )}
            <span>
              {item.url ? (
                <a href={item.url} className="text-sm text-ivory underline-offset-2 hover:underline" target="_blank" rel="noreferrer">{item.name}</a>
              ) : (
                <span className="text-sm text-ivory">{item.name}</span>
              )}
              <span className="mt-1 block font-mono text-[10px] text-faint">{item.role || "item"} · ${item.price_usd}{item.source ? ` · ${item.source}` : ""}</span>
            </span>
          </li>
        ))}
      </ul>
      {outfit.why && outfit.why.length > 0 ? (
        <ul className="mt-4 space-y-1.5 text-sm" data-testid="plain-why">
          {plainWhy(outfit.why).map((line) => (
            <li key={`${line.label}-${line.text}`} className="flex gap-2">
              <MarkIcon mark={line.mark} />
              <span><span className="font-medium text-ivory">{line.label}:</span> <span className="text-dim">{line.text}</span></span>
            </li>
          ))}
        </ul>
      ) : null}
      {outfit.score != null ? <p className="mt-3 text-sm text-ivory" data-testid="outfit-score">Score {outfit.score.toFixed(2)} out of 1</p> : null}
    </article>
  );
}

function compactMin(value: number): string {
  return value.toFixed(value >= 10 ? 0 : 1);
}

/** Engine operator tags like "local perturbation" mean nothing to a judge. */
function plainDescription(policy: CloudPolicy): string {
  const text = (policy.description || "").trim();
  if (/^local perturbation$/i.test(text)) {
    return `${policy.genome_id ?? "This genome"} is a small random tweak (a mutation) of that genome, kept because it scored as well or better. The rules below are what it runs.`;
  }
  return text;
}

const WIN_STYLE: Record<ScoreRow["winner"], { gene: string; other: string }> = {
  gene: { gene: "bg-teal/15 text-teal font-semibold", other: "text-danger" },
  other: { gene: "text-danger", other: "bg-teal/15 text-teal font-semibold" },
  tie: { gene: "text-ivory", other: "text-ivory" },
};

function Scoreboard({ file, facts }: { file: ShowcaseFile; facts: RunFacts | null }) {
  const board = useMemo(() => cloudScoreboard(file), [file]);
  if (!board || board.rows.length === 0) return <ScoreLines file={file} />;
  const champion = file.cloud!.policies.find((policy) => policy.key === "champion");
  const origin = champion?.behavior_origin;
  const originId = origin ? (origin as { id?: string | null }).id ?? origin.genome_id : null;
  const groups: { id: ScoreRow["group"]; title: string }[] = [
    { id: "case", title: `This case (${file.case_id})` },
    { id: "heldout", title: board.cases ? `All ${board.cases} held-out test cases` : "Held-out test cases" },
  ];
  return (
    <div className="mt-6" data-testid="cloud-scoreboard">
      <div className="glass overflow-x-auto rounded-3xl p-4">
        <table className="w-full min-w-[680px] border-separate border-spacing-0 text-left text-sm">
          <caption className="pb-3 text-left font-serif text-2xl text-ivory">Who wins each metric</caption>
          <thead>
            <tr className="font-mono text-[10px] uppercase tracking-[0.16em] text-faint">
              <th className="border-b border-line px-3 py-2" scope="col">Metric</th>
              <th className="border-b border-line px-3 py-2" scope="col">{board.geneLabel}</th>
              <th className="border-b border-line px-3 py-2" scope="col">{board.otherLabel}</th>
              <th className="border-b border-line px-3 py-2" scope="col">Winner</th>
            </tr>
          </thead>
          {groups.map((group) => {
            const rows = board.rows.filter((row) => row.group === group.id);
            if (rows.length === 0) return null;
            return (
              <tbody key={group.id}>
                <tr>
                  <th colSpan={4} scope="colgroup" className="px-3 pt-4 pb-1 font-mono text-[10px] uppercase tracking-[0.16em] text-faint">{group.title}</th>
                </tr>
                {rows.map((row) => {
                  const style = WIN_STYLE[row.winner];
                  const u = row.unit && row.unit !== "deadlines" ? ` ${row.unit}` : "";
                  const winnerName = row.winner === "gene" ? board.geneLabel : row.winner === "other" ? board.otherLabel : "Tie";
                  return (
                    <tr key={row.key} data-testid={`score-${row.key}`} data-winner={row.winner}>
                      <th scope="row" className="border-b border-line px-3 py-2.5 font-normal text-ivory">
                        {row.metric}
                        <span className="ml-2 font-mono text-[10px] uppercase tracking-widest text-faint">{row.better} is better</span>
                      </th>
                      <td className="border-b border-line px-1 py-1.5 tabular-nums"><span className={`inline-block rounded-lg px-2 py-1 ${style.gene}`}>{row.winner === "gene" ? "✓ " : ""}{fmtCell(row.gene, row.unit)}{u}</span></td>
                      <td className="border-b border-line px-1 py-1.5 tabular-nums"><span className={`inline-block rounded-lg px-2 py-1 ${style.other}`}>{row.winner === "other" ? "✓ " : ""}{fmtCell(row.other, row.unit)}{u}</span></td>
                      <td className="border-b border-line px-3 py-2.5 text-dim">
                        <span className={row.winner === "tie" ? "text-ivory" : "font-medium text-teal"}>{winnerName}</span>
                        <span className="ml-2 text-xs">{row.sentence.split(" — ")[1] ?? ""}</span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            );
          })}
        </table>
        <p className="mt-4 text-base leading-relaxed text-ivory" data-testid="score-takeaway">
          <span className="font-semibold">Takeaway: </span>
          {board.takeaway}
        </p>
        {facts ? (
          <p className="mt-2 text-sm leading-relaxed text-dim" data-testid="architecture-line">
            <span className="font-medium text-ivory">Why the architecture worked: </span>
            Evolved from a seed population through generation {facts.lastGeneration}.
            {origin && origin.provider ? ` ${providerName(origin.provider)} proposed the ${origin.operator ?? "child"}${originId ? ` ${originId}` : ""} in generation ${origin.generation ?? 0} that became the champion${facts.championId && facts.championId !== originId ? ` (kept as ${facts.championId})` : ""}.` : ""}
            {" "}Validation picked it, and the test split was read once, after the freeze.
          </p>
        ) : null}
        <p className="mt-2 text-xs text-faint">Green ✓ marks the better number on each row; red is the other side. Lower is better for deadlines, time and energy; higher is better for the overall score.</p>
      </div>
    </div>
  );
}

function fmtCell(value: number, unit: string): string {
  if (unit === "deadlines") return value.toFixed(0);
  if (unit === "") return value.toFixed(3);
  return value.toFixed(2);
}

function providerName(provider: string): string {
  const key = provider.toLowerCase();
  if (key === "creao") return "CREAO";
  if (key === "veri") return "Veri (Qwen2.5-7B)";
  if (key === "baseline") return "A hand-written baseline";
  return provider;
}

function MarkIcon({ mark }: { mark: Mark }) {
  if (mark === "yes") return <span className="w-4 shrink-0 text-teal" aria-label="yes">✓</span>;
  if (mark === "partial") return <span className="w-4 shrink-0 text-amber" aria-label="partly">◐</span>;
  return <span className="w-4 shrink-0 text-danger" aria-label="no">✗</span>;
}
