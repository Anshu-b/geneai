"use client";

import { useEffect, useMemo, useState } from "react";
import {
  budgetLine,
  cloudVerdict,
  exampleMissLine,
  interpretShowcaseResponse,
  isSampleShowcase,
  originCredit,
  shoppingSplitLine,
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
    <section className="glass mt-8 rounded-3xl p-4 md:p-6" data-testid="output-showcase">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="kicker">What each system actually produced</p>
          <h2 className="mt-2 font-serif text-3xl tracking-tight text-ivory md:text-4xl">The output, not just the score.</h2>
        </div>
        {sample ? (
          <span className="rounded-full border border-amber/50 px-3 py-1 font-mono text-[10px] uppercase tracking-[0.16em] text-amber" data-testid="showcase-sample">
            Sample
          </span>
        ) : null}
      </div>
      <p className="mt-3 max-w-3xl text-sm leading-relaxed text-dim">{file.case_note}</p>
      <p className="mt-1 font-mono text-[10px] text-faint">
        {file.split || "split"} · {file.case_id} · {file.run_id}
      </p>
      {file.domain === "shopping" && file.shopping ? <ShoppingShowcase file={file} /> : null}
      {file.domain === "cloud_scheduling" && file.cloud ? <CloudShowcaseView file={file} /> : null}
    </section>
  );
}

function CloudShowcaseView({ file }: { file: ShowcaseFile }) {
  const cloud = file.cloud!;
  const [key, setKey] = useState(cloud.policies[0]?.key ?? "");
  const policy = cloud.policies.find((item) => item.key === key) ?? cloud.policies[0];
  const verdict = useMemo(() => cloudVerdict(cloud.policies), [cloud.policies]);
  const example = useMemo(() => exampleMissLine(cloud.policies), [cloud.policies]);
  const splitCount = cloud.policies.find((item) => item.aggregate?.n_cases)?.aggregate?.n_cases;
  if (!policy) return <p className="mt-4 text-sm text-dim">This showcase has no policies.</p>;
  return (
    <div className="mt-6 space-y-5">
      <p className="text-base leading-relaxed text-ivory" data-testid="showcase-verdict">{verdict}</p>
      {splitCount ? (
        <div data-testid="showcase-aggregate">
          <p className="kicker">Across all {splitCount} held-out test cases</p>
          <div className="mt-3 grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
            {cloud.policies.filter((item) => item.aggregate).map((item) => (
              <div key={item.key} className="rounded-2xl border border-line bg-raise/40 px-3 py-3">
                <p className="font-serif text-lg text-ivory">{item.label}</p>
                <p className="mt-2 text-sm text-dim">
                  {item.aggregate!.sla_misses} deadlines missed in total
                </p>
                <p className="mt-1 font-mono text-[11px] text-faint">
                  mean finish {item.aggregate!.avg_completion_min} min · {item.aggregate!.energy_kwh} kWh · {item.aggregate!.hard_deadline_misses} hard
                </p>
              </div>
            ))}
          </div>
        </div>
      ) : null}
      <div>
        <p className="kicker">One example case</p>
        <p className="mt-2 text-sm leading-relaxed text-ivory" data-testid="showcase-example">{example}</p>
      </div>
      <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
        {cloud.policies.map((item) => (
          <button
            key={item.key}
            type="button"
            onClick={() => setKey(item.key)}
            aria-pressed={item.key === policy.key}
            className={`lift rounded-2xl border px-3 py-3 text-left ${item.key === policy.key ? "border-teal bg-teal/10" : "border-line bg-raise/40"}`}
          >
            <span className="block font-serif text-lg text-ivory">{item.label}</span>
            <span className="mt-1 block font-mono text-[10px] uppercase tracking-widest text-faint">{item.provider || "provider"} {item.genome_id ? `· ${item.genome_id}` : ""}</span>
            {item.feasible && item.summary ? (
              <span className="mt-2 block text-sm text-dim">
                {item.summary.avg_completion_min} min avg · {item.summary.hard_deadline_misses} hard misses · {item.summary.energy_kwh} kWh
              </span>
            ) : (
              <span className="mt-2 block text-sm text-danger">Infeasible</span>
            )}
          </button>
        ))}
      </div>
      <PolicyDetail policy={policy} />
      <Gantt cloud={cloud} policy={policy} />
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
  const shopper = shopping.shopper;
  const split = shoppingSplitLine(shopping.outfits);
  const shoppers = shopping.outfits.find((outfit) => outfit.aggregate?.n_shoppers)?.aggregate?.n_shoppers;
  return (
    <div className="mt-6 space-y-5">
      {split ? <p className="text-base leading-relaxed text-ivory" data-testid="showcase-verdict">{split}</p> : null}
      {shoppers ? (
        <div data-testid="showcase-aggregate">
          <p className="kicker">Across all {shoppers} held-out shoppers</p>
          <div className="mt-3 grid gap-2 md:grid-cols-3">
            {shopping.outfits.filter((outfit) => outfit.aggregate).map((outfit) => (
              <div key={outfit.key} className="rounded-2xl border border-line bg-raise/40 px-3 py-3">
                <p className="font-serif text-lg text-ivory">{outfit.label}</p>
                <p className="mt-2 text-sm text-dim">
                  feasible {outfit.aggregate!.feasible_count} of {outfit.aggregate!.n_shoppers}
                </p>
                <p className="mt-1 font-mono text-[11px] text-faint">
                  mean ${outfit.aggregate!.total_usd_mean} of ${outfit.aggregate!.budget_usd_mean}
                  {outfit.aggregate!.score_mean != null ? ` · score ${outfit.aggregate!.score_mean}` : ""}
                </p>
              </div>
            ))}
          </div>
        </div>
      ) : null}
      <p className="kicker">One example shopper</p>
      <article className="rounded-2xl border border-line bg-raise/50 p-4">
        <p className="kicker">Shopper {shopper.id}</p>
        <h3 className="mt-2 font-serif text-2xl text-ivory">{shopper.style || "Shopper profile"}</h3>
        {shopper.notes ? <p className="mt-2 max-w-3xl text-sm leading-relaxed text-dim">{shopper.notes}</p> : null}
        <dl className="mt-3 flex flex-wrap gap-x-6 gap-y-2 text-sm">
          <div><dt className="kicker">Budget</dt><dd className="text-ivory">${shopper.budget_usd}</dd></div>
          <div><dt className="kicker">Size</dt><dd className="text-ivory">{shopper.apparel_size || "—"}{shopper.shoe_size != null ? ` · shoe ${shopper.shoe_size}` : ""}</dd></div>
          {shopper.occasions?.length ? <div><dt className="kicker">Occasions</dt><dd className="text-ivory">{shopper.occasions.join(", ")}</dd></div> : null}
          {shopper.likes?.length ? <div><dt className="kicker">Likes</dt><dd className="text-ivory">{shopper.likes.join(", ")}</dd></div> : null}
          {shopper.dislikes?.length ? <div><dt className="kicker">Dislikes</dt><dd className="text-ivory">{shopper.dislikes.join(", ")}</dd></div> : null}
        </dl>
      </article>
      <div className="grid gap-4 lg:grid-cols-3">
        {shopping.outfits.map((outfit) => (
          <Outfit key={outfit.key} outfit={outfit} budget={shopper.budget_usd} />
        ))}
      </div>
    </div>
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
