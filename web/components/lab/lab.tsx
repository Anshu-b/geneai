"use client";

import gsap from "gsap";
import dynamic from "next/dynamic";
import { useEffect, useMemo, useRef, useState } from "react";
import { Header } from "@/components/header";
import { LineageSvg } from "@/components/lab/lineage-svg";
import { Scrubber } from "@/components/lab/scrubber";
import { SidePanel } from "@/components/lab/side-panel";
import { fitnessHelp } from "@/lib/domain";
import { formatFitness, shortHash } from "@/lib/format";
import { presentationMode, type LoadSource } from "@/lib/mode";
import { parseExport } from "@/lib/parse";
import {
  ancestryIds,
  buildBeats,
  genomeMap,
  isHandwrittenBaseline,
  layoutRun,
  projectRun,
  providerBucket,
  PROVIDER_COLOR,
  type Beat,
} from "@/lib/project";
import { geneaiApiBase, openRunStream, type StreamHandle } from "@/lib/sse";
import type { RunExport } from "@/lib/types";

const Lineage3D = dynamic(() => import("@/components/lab/lineage-3d"), {
  ssr: false,
  loading: () => <p className="p-6 text-sm text-dim">Preparing the 3D lineage…</p>,
});

type Filter = "all" | "living" | "pruned" | "infeasible" | "ancestry";

export function Lab({ fixture, title }: { fixture: string; title: string }) {
  const [exp, setExp] = useState<RunExport | null>(null);
  const [beats, setBeats] = useState<Beat[]>([]);
  const [beatIndex, setBeatIndex] = useState(0);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [source, setSource] = useState<LoadSource>("bundled");
  const [sourceLabel, setSourceLabel] = useState(fixture);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<1 | 2 | 4>(1);
  const [follow, setFollow] = useState(false);
  const [filter, setFilter] = useState<Filter>("all");
  const [family, setFamily] = useState("all");
  const [show3d, setShow3d] = useState(false);
  const [live, setLive] = useState(false);
  const [streamNote, setStreamNote] = useState<string | null>(null);
  const [reduced, setReduced] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const streamRef = useRef<StreamHandle | null>(null);
  const expRef = useRef<RunExport | null>(null);
  const heroRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const media = window.matchMedia("(prefers-reduced-motion: reduce)");
    setReduced(media.matches);
    const onChange = () => setReduced(media.matches);
    media.addEventListener("change", onChange);
    return () => media.removeEventListener("change", onChange);
  }, []);

  useEffect(() => {
    if (reduced) return;
    const ctx = gsap.context(() => {
      gsap.from(".hero-rise", { y: 14, autoAlpha: 0, stagger: 0.08, duration: 0.7, ease: "power2.out" });
    }, heroRef);
    return () => ctx.revert();
  }, [reduced, fixture]);

  function accept(next: RunExport, nextSource: LoadSource, label: string) {
    expRef.current = next;
    const nextBeats = buildBeats(next.events);
    setExp(next);
    setBeats(nextBeats);
    setBeatIndex(Math.max(0, nextBeats.length - 1));
    setSelectedId(next.champion_id);
    setSource(nextSource);
    setSourceLabel(label);
    setError(null);
    setPlaying(false);
    setLive(false);
    setFollow(false);
    setFamily("all");
    streamRef.current?.close();
    streamRef.current = null;
  }

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    const runUrl = new URLSearchParams(window.location.search).get("run");
    (async () => {
      try {
        const target = runUrl || fixture;
        const response = await fetch(target);
        if (!response.ok) throw new Error(`Could not fetch the export (${response.status}).`);
        const json: unknown = await response.json();
        if (cancelled) return;
        accept(parseExport(json), runUrl ? "url" : "bundled", runUrl || fixture);
      } catch (err) {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : "Could not read the export.");
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [fixture]);

  useEffect(() => {
    if (!playing) return;
    if (reduced) {
      setBeatIndex(Math.max(0, beats.length - 1));
      setPlaying(false);
      return;
    }
    const delay = 880 / speed;
    const timer = window.setTimeout(() => {
      setBeatIndex((current) => {
        if (current >= beats.length - 1) {
          setPlaying(false);
          return current;
        }
        return current + 1;
      });
    }, delay);
    return () => window.clearTimeout(timer);
  }, [playing, beatIndex, speed, beats.length, reduced]);

  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      const target = event.target;
      if (target instanceof HTMLElement && (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable)) {
        return;
      }
      if (event.key === "ArrowRight") {
        event.preventDefault();
        setPlaying(false);
        setBeatIndex((current) => Math.min(beats.length - 1, current + 1));
      } else if (event.key === "ArrowLeft") {
        event.preventDefault();
        setPlaying(false);
        setBeatIndex((current) => Math.max(0, current - 1));
      } else if (event.key === " ") {
        event.preventDefault();
        if (beats.length === 0) return;
        if (beatIndex >= beats.length - 1) {
          setBeatIndex(0);
          setPlaying(true);
          return;
        }
        setPlaying((current) => !current);
      } else if (event.key === "f" && exp) {
        const beat = beats[beatIndex];
        if (!beat) return;
        const projection = projectRun(exp, beat.sequence);
        if (projection.championFrozen) setFollow((current) => !current);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [beats, beatIndex, exp]);

  useEffect(() => () => streamRef.current?.close(), []);

  const beat = beats[beatIndex];
  const layout = useMemo(() => (exp ? layoutRun(exp) : null), [exp]);
  const projection = useMemo(() => (exp && beat ? projectRun(exp, beat.sequence) : null), [exp, beat]);
  const genomes = useMemo(() => (exp ? genomeMap(exp) : new Map()), [exp]);
  const ancestry = useMemo(
    () => ancestryIds(projection?.championFrozen ? exp?.champion_id ?? null : null, genomes),
    [projection?.championFrozen, exp?.champion_id, genomes],
  );
  const presentation = exp ? presentationMode(exp.run.mode, live) : null;
  const selected = selectedId && projection ? projection.byId.get(selectedId) ?? null : null;
  const families = layout?.lanes.map((lane) => lane.family) ?? [];
  const providerCounts = useMemo(() => {
    const counts = new Map<string, number>();
    for (const genome of exp?.genomes ?? []) {
      const bucket = providerBucket(genome.provider);
      counts.set(bucket, (counts.get(bucket) ?? 0) + 1);
    }
    return counts;
  }, [exp]);

  async function onFile(file: File) {
    try {
      const text = await file.text();
      accept(parseExport(JSON.parse(text) as unknown), "upload", file.name);
    } catch (err) {
      setError(err instanceof Error ? err.message : "That file is not a GeneAI export.");
    }
  }

  function attachStream() {
    const base = geneaiApiBase();
    if (!base || !exp) return;
    streamRef.current?.close();
    setStreamNote(null);
    streamRef.current = openRunStream(base, exp.run.id, {
      onStatus: (status, detail) => {
        setLive(status === "open");
        if (status === "error") setStreamNote(detail || "The stream failed. The loaded export is unchanged.");
        if (status === "closed") setStreamNote("Stream closed.");
      },
      onEvent: (event) => {
        const current = expRef.current;
        if (!current) return;
        if (event.run_id && event.run_id !== current.run.id) {
          setStreamNote(`Ignored an event for ${event.run_id}. This view is ${current.run.id}.`);
          return;
        }
        if (current.events.some((item) => item.sequence === event.sequence)) return;
        const events = [...current.events, event].sort((a, b) => a.sequence - b.sequence);
        const next = { ...current, events };
        expRef.current = next;
        setExp(next);
        setBeats(buildBeats(events));
      },
    });
  }

  const api = geneaiApiBase();
  const living = projection?.views.filter((view) => view.status === "kept" || view.status === "elite" || (view.status === "champion" && !view.laterPruned)) ?? [];

  return (
    <div
      onDragOver={(event) => event.preventDefault()}
      onDrop={(event) => {
        event.preventDefault();
        const file = event.dataTransfer.files?.[0];
        if (file) void onFile(file);
      }}
    >
      <Header presentation={presentation} />
      <main className="mx-auto max-w-[1600px] px-4 py-5 md:px-6">
        <div ref={heroRef} className="max-w-3xl">
          <p className="hero-rise kicker">{title}</p>
          <h1 className="hero-rise mt-2 font-serif text-4xl tracking-tight text-ivory md:text-6xl">Intelligence that evolves.</h1>
          <p className="hero-rise mt-3 max-w-2xl text-base leading-relaxed text-dim md:text-lg">
            One engine grows a population, scores it, and prunes what does not survive. This page plays an export. It is not a live claim.
          </p>
        </div>

        {loading ? (
          <p className="mt-16 text-sm text-dim" role="status">Reading the export…</p>
        ) : error && !exp ? (
          <p className="mt-16 max-w-xl text-sm text-danger" role="alert">{error}</p>
        ) : exp && projection && layout && beat ? (
          <>
            {error ? <p className="mt-4 text-sm text-danger" role="alert">{error}</p> : null}
            <div className="mt-6 grid gap-4 xl:grid-cols-[300px_minmax(0,1fr)_380px]">
              <section className="ghost-scroll space-y-4 xl:max-h-[calc(100vh-8.5rem)] xl:overflow-auto xl:pr-2">
                <div className="panel p-3">
                  <p className="kicker">Run</p>
                  <p className="mt-1 font-mono text-sm">{exp.run.id}</p>
                  <p className="mt-1 text-xs leading-relaxed text-dim">
                    {source === "bundled" ? "Bundled export" : source === "upload" ? "Uploaded file" : source === "url" ? "Loaded from URL" : "Live stream"} · {sourceLabel}
                  </p>
                  <p className="mt-2 text-xs text-dim">
                    {exp.run.domain} · seed {exp.run.seed} · {exp.run.status || "status unknown"} · {exp.run.stop_reason || "no stop reason"}
                  </p>
                  <p className="mt-2 break-all font-mono text-[10px] text-faint" title={exp.integrity_sha256}>
                    integrity {shortHash(exp.integrity_sha256, 14)}
                  </p>
                  <p className="mt-1 text-[11px] leading-relaxed text-faint">
                    The hash is the engine hash of the full export. Bundled files omit event payload.task, so this page does not recompute it.
                  </p>
                  {exp.run.git_sha ? <p className="mt-1 font-mono text-[10px] text-faint">engine {exp.run.git_sha.slice(0, 12)}</p> : null}
                </div>
                <div>
                  <p className="kicker">Objective</p>
                  <p className="mt-2 text-sm leading-relaxed text-ivory/90">{exp.objective.prompt}</p>
                  <dl className="mt-3 space-y-1 text-xs text-dim">
                    {Object.entries(exp.objective.constraints).map(([key, value]) => (
                      <div key={key} className="flex justify-between gap-3">
                        <dt>{key}</dt>
                        <dd className="font-mono text-ivory">{String(value)}</dd>
                      </div>
                    ))}
                  </dl>
                </div>
                <div className="grid grid-cols-2 gap-2 text-sm">
                  <Stat label="Living" value={String(living.length)} />
                  <Stat label="Pruned" value={String(projection.prunedIds.length)} />
                  <Stat label="Born" value={String(projection.views.length)} />
                  <Stat label="Best train" value={formatFitness(projection.best?.fitness)} />
                </div>
                <p className="text-[11px] leading-relaxed text-faint">{fitnessHelp(exp.run.domain)}</p>
                <div className="space-y-2">
                  <label className="block text-xs text-dim">
                    Family lane
                    <select
                      className="mt-1 w-full rounded border border-line bg-raise px-2 py-1.5 text-sm text-ivory"
                      value={family}
                      onChange={(event) => setFamily(event.target.value)}
                    >
                      <option value="all">All families</option>
                      {families.map((name) => (
                        <option key={name} value={name}>{name}</option>
                      ))}
                    </select>
                  </label>
                  <label className="block text-xs text-dim">
                    Show
                    <select
                      className="mt-1 w-full rounded border border-line bg-raise px-2 py-1.5 text-sm text-ivory"
                      value={filter}
                      onChange={(event) => setFilter(event.target.value as Filter)}
                    >
                      <option value="all">Kept, pruned, and elites</option>
                      <option value="living">Living only</option>
                      <option value="pruned">Pruned only</option>
                      <option value="infeasible">Infeasible</option>
                      <option value="ancestry">Champion ancestry</option>
                    </select>
                  </label>
                  <button
                    type="button"
                    className="w-full rounded border border-line px-3 py-2 text-left text-sm disabled:opacity-40"
                    disabled={!projection.championFrozen}
                    aria-pressed={follow}
                    onClick={() => setFollow((current) => !current)}
                  >
                    {follow ? "Champion ancestry on" : "Follow champion ancestry"}
                    <span className="mt-1 block text-xs text-dim">
                      {projection.championFrozen ? "Dims every branch that is not a parent of the frozen champion." : "Available after champion.frozen."}
                    </span>
                  </button>
                </div>
                <div>
                  <p className="kicker">Living at this point</p>
                  <ul className="mt-2 max-h-48 space-y-1 overflow-auto">
                    {projection.championFrozen && projection.championId ? (
                      <li>
                        <button type="button" className="text-left text-sm text-amber" onClick={() => setSelectedId(projection.championId)}>
                          {projection.championId} · champion
                        </button>
                      </li>
                    ) : null}
                    {living
                      .filter((view) => view.genome.id !== projection.championId)
                      .slice()
                      .sort((a, b) => (b.trainFitness ?? -Infinity) - (a.trainFitness ?? -Infinity))
                      .map((view) => (
                        <li key={view.genome.id}>
                          <button type="button" className="w-full text-left text-sm" onClick={() => setSelectedId(view.genome.id)}>
                            <span className="font-mono text-teal">{view.genome.id}</span>
                            <span className="text-dim"> {view.status} · {formatFitness(view.trainFitness)}</span>
                          </button>
                        </li>
                      ))}
                  </ul>
                </div>
                <div>
                  <p className="kicker">Providers in this export</p>
                  <ul className="mt-2 space-y-1 text-sm">
                    {(["veri", "creao", "baseline", "mock", "random"] as const).map((bucket) => (
                      <li key={bucket} className="flex items-center justify-between">
                        <span className="flex items-center gap-2">
                          <span className="inline-block h-2.5 w-2.5" style={{ background: PROVIDER_COLOR[bucket], borderRadius: bucket === "mock" ? 99 : 2 }} />
                          {bucket}
                        </span>
                        <span className="font-mono text-xs text-dim">{providerCounts.get(bucket) ?? 0}</span>
                      </li>
                    ))}
                  </ul>
                  <p className="mt-2 text-[11px] leading-relaxed text-faint">
                    Color is the recorded provider. Mock is not veri.studio or CREAO. Squares are baselines, diamonds veri, hexagons CREAO, triangles random. A rose mark is an immigrant.
                  </p>
                  <p className="mt-1 text-[11px] leading-relaxed text-faint">
                    Node size is train fitness. Ghosts were pruned. An ivory ring is an elite carried unchanged. Gold is the frozen champion.
                  </p>
                </div>
                <div className="space-y-2">
                  <p className="kicker">Load another export</p>
                  <button type="button" className="rounded border border-line px-3 py-1.5 text-sm" onClick={() => fileRef.current?.click()}>
                    Choose JSON
                  </button>
                  <input
                    ref={fileRef}
                    type="file"
                    accept="application/json,.json"
                    className="sr-only"
                    onChange={(event) => {
                      const file = event.target.files?.[0];
                      if (file) void onFile(file);
                    }}
                  />
                  <form
                    className="flex gap-2"
                    onSubmit={(event) => {
                      event.preventDefault();
                      const data = new FormData(event.currentTarget);
                      const url = String(data.get("run") || "").trim();
                      if (!url) return;
                      setLoading(true);
                      fetch(url)
                        .then(async (response) => {
                          if (!response.ok) throw new Error(`Could not fetch the export (${response.status}).`);
                          return response.json() as Promise<unknown>;
                        })
                        .then((json) => accept(parseExport(json), "url", url))
                        .catch((err: unknown) => {
                          setError(err instanceof Error ? `${err.message} If this is a remote file, the host may be blocking browser reads. Download it and choose the JSON instead.` : "Could not open that URL.");
                        })
                        .finally(() => setLoading(false));
                    }}
                  >
                    <label className="sr-only" htmlFor="run-url">Export URL</label>
                    <input id="run-url" name="run" placeholder="https://…/export.json" className="min-w-0 flex-1 rounded border border-line bg-raise px-2 py-1.5 text-sm" />
                    <button type="submit" className="rounded border border-line px-2 text-sm">Open</button>
                  </form>
                  <button type="button" className="text-sm text-teal disabled:text-faint" disabled={!api || !exp} onClick={attachStream}>
                    Attach live stream
                  </button>
                  <p className="text-[11px] leading-relaxed text-faint">
                    {api ? `Stream URL uses ${api}.` : "Live stream needs NEXT_PUBLIC_GENEAI_API. Recorded exports still play."}
                  </p>
                  {streamNote ? <p className="text-xs text-amber">{streamNote}</p> : null}
                </div>
                <div className="flex items-center justify-between gap-3">
                  <p className="text-xs text-dim">← → step · space play · f ancestry</p>
                  <button
                    type="button"
                    className="rounded border border-line px-2 py-1 font-mono text-[10px] uppercase tracking-widest"
                    aria-pressed={show3d}
                    onClick={() => setShow3d((current) => !current)}
                  >
                    {show3d ? "2D lineage" : "3D lineage"}
                  </button>
                </div>
                <p className="text-[11px] text-faint">
                  Hand-written baselines in this file: {exp.genomes.filter(isHandwrittenBaseline).length}. Edges are mutate, crossover, or repair. Immigrants have no parents.
                </p>
              </section>

              <section>
                {show3d ? (
                  <Lineage3D
                    views={projection.views.filter((view) => family === "all" || view.genome.metadata.family === family)}
                    layout={layout}
                    onSelect={setSelectedId}
                    reduced={reduced}
                  />
                ) : (
                  <LineageSvg
                    layout={layout}
                    projection={projection}
                    ancestry={ancestry}
                    follow={follow}
                    filter={filter}
                    family={family}
                    selectedId={selectedId}
                    onSelect={setSelectedId}
                    focusGeneration={beat.focusGeneration}
                    beatId={beat.id}
                    beatPhase={beat.phase}
                    playing={playing}
                    reduced={reduced}
                  />
                )}
                <Scrubber
                  beats={beats}
                  index={beatIndex}
                  playing={playing}
                  speed={speed}
                  onIndex={setBeatIndex}
                  onPlaying={setPlaying}
                  onSpeed={setSpeed}
                />
              </section>

              <SidePanel exp={exp} projection={projection} selected={selected} onSelect={setSelectedId} />
            </div>
          </>
        ) : null}
      </main>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded border border-line px-2 py-2">
      <p className="kicker">{label}</p>
      <p className="mt-1 font-mono text-sm">{value}</p>
    </div>
  );
}
