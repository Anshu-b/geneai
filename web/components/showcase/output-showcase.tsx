"use client";

import { useEffect, useMemo, useState } from "react";
import {
  budgetLine,
  interpretShowcaseResponse,
  isManualBaseline,
  isSampleShowcase,
  keyScoreLines,
  originCredit,
  type CloudPolicy,
  type OutfitCard,
  type ShowcaseFile,
  type ShowcaseLoad,
} from "@/lib/showcase";

const JOB_COLOR: Record<string, string> = {
  interactive: "var(--job-interactive)",
  batch: "var(--job-batch)",
  gpu: "var(--job-gpu)",
  highmem: "var(--job-highmem)",
};

export function OutputShowcase({ src }: { src: string }) {
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
  return <ShowcaseBody file={load.file} />;
}

function ShowcaseBody({ file }: { file: ShowcaseFile }) {
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
      <ScoreLines file={file} />
    </section>
  );
}

function CloudShowcaseView({ file }: { file: ShowcaseFile }) {
  const cloud = file.cloud!;
  const policies = cloud.policies.filter((policy) => !isManualBaseline(policy.key, policy.provider));
  if (policies.length === 0) return <p className="mt-4 text-sm text-dim">This showcase has no agentic policies.</p>;
  return (
    <div className="mt-4 grid gap-4 lg:grid-cols-2">
      {policies.map((policy) => (
        <article key={policy.key} className="glass rounded-3xl p-4">
          <h2 className="font-serif text-2xl text-ivory">{policy.label}</h2>
          <PolicyDetail policy={policy} />
          <div className="mt-4">
            <Gantt cloud={cloud} policy={policy} />
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
      {policy.description ? <p className="mt-2 text-sm text-dim">{policy.description}</p> : null}
      {!policy.feasible && policy.infeasible_reason ? (
        <p className="mt-2 text-sm text-danger">{policy.infeasible_reason}</p>
      ) : null}
      {policy.rules_plain && policy.rules_plain.length > 0 ? (
        <ul className="mt-3 space-y-1 text-sm text-ivory">
          {policy.rules_plain.map((rule) => (
            <li key={rule} className="flex gap-2"><span className="text-teal">·</span><span>{rule}</span></li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}

function Gantt({ cloud, policy }: { cloud: NonNullable<ShowcaseFile["cloud"]>; policy: CloudPolicy }) {
  const jobs = new Map(cloud.jobs.map((job) => [job.id, job]));
  const span = Math.max(1, ...policy.schedule.map((bar) => bar.end_min), ...cloud.jobs.map((job) => job.deadline_min));
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
                const fill = JOB_COLOR[job?.class || ""] || "var(--faint)";
                return (
                  <g key={`${bar.job_id}-${bar.start_min}`}>
                    <rect x={x} y={y} width={w} height={20} rx={4} fill={fill} opacity={bar.missed_deadline ? 0.45 : 0.92} stroke={bar.missed_deadline ? "var(--danger)" : "transparent"} strokeWidth={bar.missed_deadline ? 1.6 : 0} />
                    <text x={x + 6} y={y + 14} fill="var(--bg)" fontSize="10" fontFamily="IBM Plex Mono, ui-monospace, monospace">{bar.job_id}{bar.missed_deadline ? " late" : ""}</text>
                  </g>
                );
              })}
            </g>
          );
        })}
        <text x={label} y={height - 2} fill="var(--faint)" fontSize="10" fontFamily="IBM Plex Mono, ui-monospace, monospace">0</text>
        <text x={width} y={height - 2} textAnchor="end" fill="var(--faint)" fontSize="10" fontFamily="IBM Plex Mono, ui-monospace, monospace">{span} min</text>
      </svg>
      <p className="mt-2 text-[11px] text-faint">Bars are the jobs on each machine. A red edge means the job finished after its deadline.</p>
    </div>
  );
}

function ShoppingShowcase({ file }: { file: ShowcaseFile }) {
  const shopping = file.shopping!;
  const outfits = shopping.outfits.filter((outfit) => !isManualBaseline(outfit.key, outfit.provider));
  return (
    <div className="mt-4 grid gap-4 lg:grid-cols-2">
      {outfits.map((outfit) => (
        <Outfit key={outfit.key} outfit={outfit} budget={shopping.shopper.budget_usd} />
      ))}
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
      <p className={`mt-3 text-sm ${outfit.feasible ? "text-ivory" : "text-danger"}`}>{budgetLine(outfit.total_usd, budget)}</p>
      {!outfit.feasible && outfit.infeasible_reason ? <p className="mt-2 text-sm leading-relaxed text-danger">{outfit.infeasible_reason}</p> : null}
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
        <ul className="mt-4 space-y-1 text-sm text-dim">
          {outfit.why.map((reason) => (
            <li key={reason}>{reason}</li>
          ))}
        </ul>
      ) : null}
      {outfit.score != null ? <p className="mt-3 font-mono text-[10px] text-faint">score {outfit.score} · secondary to the outfit above</p> : null}
    </article>
  );
}
