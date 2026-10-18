"use client";

import { useEffect, useMemo, useState } from "react";
import {
  TSP_COLOR,
  TSP_KEYS,
  bestAt,
  elapsedFromFraction,
  formatDuration,
  formatGap,
  formatLength,
  heldoutLine,
  hitsOptimum,
  interpretTspResponse,
  isSampleTsp,
  provenanceLine,
  maxRaceMs,
  tspVerdict,
  type TspFile,
  type TspLoad,
  type TspMethod,
} from "@/lib/tsp";
import { RankLegend } from "@/components/rank-legend";
import { GENEAI_ROW, RANK_CLASS, rankColumn, type Rank } from "@/lib/rank";

const RACE_MS = 9000;

export function TspShowdown() {
  const [load, setLoad] = useState<TspLoad | { status: "loading" }>({ status: "loading" });

  useEffect(() => {
    const controller = new AbortController();
    fetch("/fixtures/tsp_benchmark.json", { signal: controller.signal })
      .then(async (response) => {
        if (response.status === 404) return interpretTspResponse(404, null);
        const body = await response.json().catch(() => null);
        return interpretTspResponse(response.status, body);
      })
      .then((next) => setLoad(next))
      .catch((error: unknown) => {
        if (error instanceof DOMException && error.name === "AbortError") return;
        setLoad({ status: "invalid", message: "The benchmark file could not be fetched." });
      });
    return () => controller.abort();
  }, []);

  return (
    <div data-testid="tsp-showdown">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="kicker">A known map</p>
          <h2 className="mt-3 max-w-3xl font-serif text-4xl tracking-tight text-ivory md:text-6xl">Traveling salesman, 15 cities</h2>
        </div>
        {load.status === "ready" && isSampleTsp(load.file) ? (
          <span className="rounded-full border border-amber/50 px-3 py-1 font-mono text-[10px] uppercase tracking-[0.16em] text-amber" data-testid="tsp-sample">
            Sample
          </span>
        ) : null}
      </div>
      <p className="mt-4 max-w-2xl text-base leading-relaxed text-dim">
        Four methods draw the same cities. Held-Karp&apos;s route is the shortest possible. The other maps keep that route as a dashed ghost, and the clock is part of the result.
      </p>
      {load.status === "loading" ? <p className="mt-8 text-sm text-dim">Reading the 15-city benchmark…</p> : null}
      {load.status === "missing" ? <p className="mt-8 text-sm text-dim">No traveling salesman benchmark yet.</p> : null}
      {load.status === "invalid" ? <p className="mt-8 text-sm text-danger">{load.message}</p> : null}
      {load.status === "ready" ? <ShowdownBody file={load.file} /> : null}
    </div>
  );
}

function ShowdownBody({ file }: { file: TspFile }) {
  const reduced = usePrefersReducedMotion();
  const max = useMemo(() => maxRaceMs(file), [file]);
  const [playing, setPlaying] = useState(false);
  const [elapsed, setElapsed] = useState<number | null>(null);
  const [race, setRace] = useState(0);
  const [intro, setIntro] = useState(reduced ? 1 : 0);

  useEffect(() => {
    if (reduced) return;
    let frame = 0;
    const start = performance.now();
    const tick = (now: number) => {
      const portion = Math.min(1, (now - start) / 1400);
      setIntro(portion);
      if (portion < 1) frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [reduced]);

  useEffect(() => {
    if (!playing) return;
    if (reduced) {
      setPlaying(false);
      setElapsed(null);
      return;
    }
    let frame = 0;
    const start = performance.now();
    const tick = (now: number) => {
      const portion = Math.min(1, (now - start) / RACE_MS);
      setElapsed(elapsedFromFraction(portion, max, false));
      if (portion < 1) frame = requestAnimationFrame(tick);
      else {
        setPlaying(false);
        setElapsed(null);
      }
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [playing, race, max, reduced]);


  return (
    <div className="mt-8">
      <p className="max-w-3xl text-base leading-relaxed text-ivory" data-testid="tsp-verdict">{tspVerdict(file)}</p>
      <div className="mt-5 flex flex-wrap items-center gap-2">
        <button
          type="button"
          className="btn-ink lift rounded-full px-4 py-2 text-sm font-medium"
          data-testid="tsp-race"
          aria-pressed={playing}
          onClick={() => {
            setElapsed(0);
            setRace((value) => value + 1);
            setPlaying(true);
          }}
        >
          {playing ? "Racing" : "Race"}
        </button>
        <p className="font-mono text-[11px] uppercase tracking-[0.16em] text-faint" data-testid="tsp-clock">
          {elapsed == null ? "Full routes" : formatDuration(elapsed)}
        </p>
      </div>
      <div className="mt-6 grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        {file.methods.map((method) => (
          <MapPanel key={method.key} file={file} method={method} elapsed={elapsed} intro={intro} />
        ))}
      </div>
      <p className="mt-3 max-w-3xl text-sm leading-relaxed text-dim">
        The solid line is the route that method handed in. While the race is running, each pen moves on the real clock, and the number above is the best length found by that moment.
      </p>
      <ComparisonTable file={file} />
    </div>
  );
}

function MapPanel({ file, method, elapsed, intro }: { file: TspFile; method: TspMethod; elapsed: number | null; intro: number }) {
  const live = elapsed == null ? method.length : bestAt(method, elapsed);
  const done = elapsed == null || elapsed + 1e-6 >= method.time_ms;
  const draw = elapsed == null ? intro : Math.min(1, Math.max(0, elapsed / method.time_ms));
  const crowned = hitsOptimum(method) && done;
  const ghost = method.key !== "held_karp";
  const color = method.valid ? TSP_COLOR[method.key] : "var(--danger)";
  return (
    <article className={`glass flex flex-col rounded-3xl p-3 ${method.valid ? "" : "border-danger/40"}`} data-testid={`tsp-map-${method.key}`}>
      <div className="flex items-start justify-between gap-2 px-1">
        <div>
          <h3 className="font-serif text-xl leading-tight text-ivory">{method.label}</h3>
          <p className="mt-1 font-mono text-[10px] uppercase tracking-[0.16em] text-faint">{method.provider}</p>
        </div>
        <GapBadge method={method} crowned={crowned} live={done ? method.gap_pct : null} />
      </div>
      <div className="mt-3 overflow-hidden rounded-2xl border border-line" style={{ background: "var(--stage)" }}>
        <RouteMap file={file} method={method} draw={draw} ghost={ghost} color={color} />
      </div>
      <p className="mt-3 px-1 font-mono text-[11px] text-dim tabular-nums">
        {live == null ? "Searching" : formatLength(live)}
        <span className="text-faint"> · {formatDuration(method.time_ms)} · {method.llm_calls} calls</span>
      </p>
      {!method.valid && method.invalid_reason ? (
        <p className="mt-2 px-1 text-sm leading-relaxed text-danger" data-testid="tsp-invalid-reason">{method.invalid_reason}</p>
      ) : null}
      {method.key === "held_karp" ? <p className="mt-2 px-1 text-sm leading-relaxed text-dim">Exact. The time beside the length is how long the proof took.</p> : null}
    </article>
  );
}

function GapBadge({ method, crowned, live }: { method: TspMethod; crowned: boolean; live: number | null }) {
  if (!method.valid) {
    return <span className="shrink-0 rounded-full border border-danger/40 px-2 py-1 font-mono text-[10px] uppercase tracking-widest text-danger">Not a tour</span>;
  }
  if (crowned) {
    return (
      <span className="inline-flex shrink-0 items-center gap-1 rounded-full border border-amber/50 px-2 py-1 font-mono text-[10px] uppercase tracking-widest text-amber" data-testid={`tsp-crown-${method.key}`}>
        <Crown />
        Optimal
      </span>
    );
  }
  return <span className="shrink-0 rounded-full border border-line px-2 py-1 font-mono text-[10px] uppercase tracking-widest text-dim">{live == null ? "…" : `+${formatGap(live)}`}</span>;
}

function Crown() {
  return (
    <svg viewBox="0 0 24 16" className="h-3 w-4" aria-hidden>
      <path d="M2 14 L5 5 L9 9 L12 2 L15 9 L19 5 L22 14 Z" fill="currentColor" />
    </svg>
  );
}

function RouteMap({ file, method, draw, ghost, color }: { file: TspFile; method: TspMethod; draw: number; ghost: boolean; color: string }) {
  const bounds = mapBounds(file);
  const visits = new Map<string, number>();
  for (const stop of method.tour) visits.set(stop, (visits.get(stop) || 0) + 1);
  const ghostPath = pathOf(file, file.optimum.tour, true);
  const routePath = pathOf(file, method.tour, method.valid);
  return (
    <svg viewBox={bounds} className="h-auto w-full" role="img" aria-label={`${method.label} on 15 cities`}>
      {ghost ? (
        <path data-testid={`tsp-ghost-${method.key}`} d={ghostPath} fill="none" stroke="var(--ivory)" strokeOpacity="0.45" strokeWidth="0.8" strokeDasharray="1.4 1.3" strokeLinejoin="round" />
      ) : null}
      <path d={routePath} fill="none" stroke={color} strokeWidth="2.4" strokeOpacity="0.28" strokeLinejoin="round" strokeLinecap="round" pathLength={1} strokeDasharray={1} strokeDashoffset={1 - draw} />
      <path d={routePath} fill="none" stroke={color} strokeWidth="1.05" strokeLinejoin="round" strokeLinecap="round" pathLength={1} strokeDasharray={1} strokeDashoffset={1 - draw} />
      {file.cities.map((city) => {
        const count = visits.get(city.id) || 0;
        const missed = count === 0;
        return (
          <g key={city.id}>
            <circle cx={city.x} cy={city.y} r={missed ? 1.7 : 1.35} fill={missed ? "transparent" : "var(--ivory)"} stroke={missed ? "var(--danger)" : "var(--bg)"} strokeWidth={missed ? 0.45 : 0.35} />
            {count > 1 ? <circle cx={city.x} cy={city.y} r="2.5" fill="none" stroke="var(--danger)" strokeWidth="0.4" /> : null}
            <text x={city.x + 1.8} y={city.y - 1.6} fill="var(--dim)" fontSize="3.1" fontFamily="IBM Plex Mono, ui-monospace, monospace">
              {city.id.replace(/^c/, "")}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function ComparisonTable({ file }: { file: TspFile }) {
  const methods = file.methods;
  const mapRanks = {
    length: rankColumn(methods.map((method) => (method.valid ? method.length : null)), "lower"),
    gap: rankColumn(methods.map((method) => (method.valid ? method.gap_pct : null)), "lower"),
    time: rankColumn(methods.map((method) => method.time_ms), "lower"),
    valid: rankColumn(methods.map((method) => (method.valid ? 1 : 0)), "higher"),
  };
  const held = TSP_KEYS.map((key) => file.heldout.summary[key]);
  const heldRanks = {
    gap: rankColumn(held.map((row) => row.mean_gap_pct), "lower"),
    time: rankColumn(held.map((row) => row.mean_time_ms), "lower"),
    hits: rankColumn(held.map((row) => row.optimal_hits), "higher"),
    valid: rankColumn(held.map((row) => row.valid_count), "higher"),
  };
  return (
    <div className="mt-8 overflow-x-auto" data-testid="tsp-table">
      <table className="w-full min-w-[720px] border-separate border-spacing-0 text-left text-sm">
        <caption className="sr-only">Tour length, gap, time, and held-out means for the four methods</caption>
        <thead>
          <tr>
            <th className="border-b border-line px-3 py-3 font-mono text-[10px] uppercase tracking-[0.16em] text-faint" scope="col">This map</th>
            <th className="border-b border-line px-3 py-3 font-mono text-[10px] uppercase tracking-[0.16em] text-faint" scope="col">Tour length</th>
            <th className="border-b border-line px-3 py-3 font-mono text-[10px] uppercase tracking-[0.16em] text-faint" scope="col">Above optimal</th>
            <th className="border-b border-line px-3 py-3 font-mono text-[10px] uppercase tracking-[0.16em] text-faint" scope="col">Time</th>
            <th className="border-b border-line px-3 py-3 font-mono text-[10px] uppercase tracking-[0.16em] text-faint" scope="col">LLM calls</th>
            <th className="border-b border-line px-3 py-3 font-mono text-[10px] uppercase tracking-[0.16em] text-faint" scope="col">Valid</th>
          </tr>
        </thead>
        <tbody>
          {file.methods.map((method, index) => (
            <tr key={method.key} data-testid={`tsp-row-${method.key}`} className={method.key === "geneai" ? GENEAI_ROW : undefined}>
              <th className="border-b border-line px-3 py-3 font-serif text-base font-normal text-ivory" scope="row">{method.label}</th>
              <RankCell rank={mapRanks.length[index]}>{formatLength(method.length)}</RankCell>
              <RankCell rank={mapRanks.gap[index]}>{method.valid ? formatGap(method.gap_pct) : "—"}</RankCell>
              <RankCell rank={mapRanks.time[index]}>{formatDuration(method.time_ms)}</RankCell>
              <td className="border-b border-line px-3 py-3 tabular-nums text-dim">{method.llm_calls}</td>
              <RankCell rank={mapRanks.valid[index]}>{method.valid ? "Yes" : "No"}</RankCell>
            </tr>
          ))}
        </tbody>
        <tbody data-testid="tsp-heldout">
          <tr>
            <th className="px-3 pt-6 pb-2 font-mono text-[10px] uppercase tracking-[0.16em] text-faint" scope="colgroup" colSpan={6}>
              Held-out, {file.heldout.n} maps
            </th>
          </tr>
          <tr>
            <th className="border-b border-line px-3 py-2 font-mono text-[10px] uppercase tracking-[0.16em] text-faint" scope="col">Method</th>
            <th className="border-b border-line px-3 py-2 font-mono text-[10px] uppercase tracking-[0.16em] text-faint" scope="col" />
            <th className="border-b border-line px-3 py-2 font-mono text-[10px] uppercase tracking-[0.16em] text-faint" scope="col">Mean gap</th>
            <th className="border-b border-line px-3 py-2 font-mono text-[10px] uppercase tracking-[0.16em] text-faint" scope="col">Mean time</th>
            <th className="border-b border-line px-3 py-2 font-mono text-[10px] uppercase tracking-[0.16em] text-faint" scope="col">Optimal hits</th>
            <th className="border-b border-line px-3 py-2 font-mono text-[10px] uppercase tracking-[0.16em] text-faint" scope="col">Valid</th>
          </tr>
          {TSP_KEYS.map((key, index) => {
            const method = file.methods.find((item) => item.key === key)!;
            const row = file.heldout.summary[key];
            return (
              <tr key={key} data-testid={`tsp-heldout-row-${key}`} className={key === "geneai" ? GENEAI_ROW : undefined}>
                <th className="border-b border-line px-3 py-3 font-serif text-base font-normal text-ivory" scope="row">{method.label}</th>
                <td className="border-b border-line px-3 py-3 text-faint">—</td>
                <RankCell rank={heldRanks.gap[index]}>{formatGap(row.mean_gap_pct)}</RankCell>
                <RankCell rank={heldRanks.time[index]}>{formatDuration(row.mean_time_ms)}</RankCell>
                <RankCell rank={heldRanks.hits[index]}>{row.optimal_hits} of {file.heldout.n}</RankCell>
                <RankCell rank={heldRanks.valid[index]}>{row.valid_count} of {file.heldout.n}</RankCell>
              </tr>
            );
          })}
        </tbody>
      </table>
      <RankLegend note="Lower is better for length, gap and time; higher for valid and optimal hits. LLM calls are for information only." />
      <p className="mt-3 max-w-3xl text-sm leading-relaxed text-dim" data-testid="tsp-heldout-line">{heldoutLine(file)} Mean gap counts only valid tours.</p>
      {provenanceLine(file) ? (
        <p className="mt-2 max-w-3xl text-sm leading-relaxed text-dim" data-testid="tsp-provenance">{provenanceLine(file)}</p>
      ) : null}
      {file.notes ? (
        <details className="mt-2 max-w-3xl text-sm text-dim" data-testid="tsp-notes">
          <summary className="cursor-pointer text-faint">How this benchmark was run</summary>
          <p className="mt-2 leading-relaxed">{file.notes}</p>
        </details>
      ) : null}
    </div>
  );
}


function mapBounds(file: TspFile): string {
  const xs = file.cities.map((city) => city.x);
  const ys = file.cities.map((city) => city.y);
  const minX = Math.min(...xs) - 8;
  const minY = Math.min(...ys) - 8;
  const maxX = Math.max(...xs) + 12;
  const maxY = Math.max(...ys) + 8;
  return `${minX} ${minY} ${maxX - minX} ${maxY - minY}`;
}

function pathOf(file: TspFile, tour: string[], close: boolean): string {
  const byId = new Map(file.cities.map((city) => [city.id, city]));
  const points = tour.map((id) => byId.get(id)).filter((city): city is TspFile["cities"][number] => Boolean(city));
  if (points.length === 0) return "";
  const commands = points.map((city, index) => `${index === 0 ? "M" : "L"}${city.x} ${city.y}`);
  if (close) commands.push(`L${points[0].x} ${points[0].y}`);
  return commands.join(" ");
}

function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    const media = window.matchMedia("(prefers-reduced-motion: reduce)");
    const apply = () => setReduced(media.matches);
    apply();
    media.addEventListener("change", apply);
    return () => media.removeEventListener("change", apply);
  }, []);
  return reduced;
}

function RankCell({ rank, children }: { rank: Rank; children: React.ReactNode }) {
  return (
    <td className="border-b border-line px-1 py-1.5 tabular-nums" data-rank={rank}>
      <span className={`inline-block ${RANK_CLASS[rank]}`}>
        {rank === "best" ? "✓ " : ""}
        {children}
      </span>
    </td>
  );
}
