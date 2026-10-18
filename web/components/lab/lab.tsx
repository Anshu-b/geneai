"use client";

import { useEffect, useMemo, useState } from "react";
import { Header } from "@/components/header";
import { LineageSvg } from "@/components/lab/lineage-svg";
import { OutputShowcase } from "@/components/showcase/output-showcase";
import { objectiveSentence } from "@/lib/domain";
import { presentationMode } from "@/lib/mode";
import { parseExport } from "@/lib/parse";
import { ancestryIds, genomeMap, layoutRun, projectRun, type Projection } from "@/lib/project";
import type { RunExport } from "@/lib/types";

export function Lab({ fixture, title, showcase }: { fixture: string; title: string; showcase?: string }) {
  const [exp, setExp] = useState<RunExport | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [reduced, setReduced] = useState(false);

  useEffect(() => {
    const media = window.matchMedia("(prefers-reduced-motion: reduce)");
    const apply = () => setReduced(media.matches);
    apply();
    media.addEventListener("change", apply);
    return () => media.removeEventListener("change", apply);
  }, []);

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
        setExp(parseExport(json));
      } catch (err) {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : "Could not read the export.");
        setExp(null);
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [fixture]);

  const layout = useMemo(() => (exp ? layoutRun(exp) : null), [exp]);
  const projection = useMemo(() => {
    if (!exp || exp.events.length === 0) return null;
    const sequence = exp.events[exp.events.length - 1]?.sequence ?? 0;
    return projectRun(exp, sequence);
  }, [exp]);
  const genomes = useMemo(() => (exp ? genomeMap(exp) : new Map()), [exp]);
  const ancestry = useMemo(
    () => ancestryIds(projection?.championFrozen ? exp?.champion_id ?? null : null, genomes),
    [projection?.championFrozen, exp?.champion_id, genomes],
  );
  const presentation = exp ? presentationMode(exp.run.mode, false) : null;

  return (
    <div>
      <Header presentation={presentation} />
      <main className="mx-auto max-w-[1600px] px-4 py-8 md:px-6">
        <p className="kicker">{title}</p>
        {loading ? <p className="mt-6 text-sm text-dim" role="status">Reading the export…</p> : null}
        {error && !exp ? <p className="mt-6 max-w-xl text-sm text-danger" role="alert">{error}</p> : null}
        {exp ? (
          <h1 className="mt-3 max-w-3xl font-serif text-4xl tracking-tight text-ivory md:text-5xl" data-testid="objective">
            {objectiveSentence(exp.objective.prompt)}
          </h1>
        ) : null}

        {showcase ? <OutputShowcase src={showcase} /> : null}

        {exp && projection && layout ? (
          <section className="mt-10">
            <h2 className="font-serif text-3xl tracking-tight text-ivory">How the population moved</h2>
            <p className="mt-2 max-w-2xl text-sm leading-relaxed text-dim">
              Best training score each generation, then the family tree. Generations run left to right. Color is who wrote the genome. Size is its score.
            </p>
            <FitnessStrip projection={projection} />
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
              compact
            />
          </section>
        ) : null}
      </main>
    </div>
  );
}

function FitnessStrip({ projection }: { projection: Projection }) {
  const points = fitnessByGeneration(projection);
  if (points.length < 2) return null;
  const width = 640;
  const height = 72;
  const pad = 8;
  const min = Math.min(...points.map((point) => point.fitness));
  const max = Math.max(...points.map((point) => point.fitness));
  const span = Math.max(1e-6, max - min);
  const xOf = (index: number) => pad + (index / (points.length - 1)) * (width - pad * 2);
  const yOf = (fitness: number) => pad + (1 - (fitness - min) / span) * (height - pad * 2);
  const d = points.map((point, index) => `${index === 0 ? "M" : "L"}${xOf(index)},${yOf(point.fitness)}`).join(" ");
  return (
    <figure className="mt-4 mb-3" data-testid="fitness-generations">
      <figcaption className="kicker">Best fitness by generation. Higher is better.</figcaption>
      <svg viewBox={`0 0 ${width} ${height}`} className="mt-2 h-16 w-full max-w-3xl" role="img" aria-label="Best fitness by generation">
        <path d={d} fill="none" stroke="var(--teal)" strokeWidth="2.5" strokeLinejoin="round" strokeLinecap="round" />
        {points.map((point, index) => (
          <g key={point.generation}>
            <circle cx={xOf(index)} cy={yOf(point.fitness)} r="3.5" fill="var(--teal)" />
            <text x={xOf(index)} y={height - 1} textAnchor="middle" fill="var(--faint)" fontSize="11" fontFamily="IBM Plex Mono, ui-monospace, monospace">
              {point.generation}
            </text>
          </g>
        ))}
      </svg>
    </figure>
  );
}

function fitnessByGeneration(projection: Projection): { generation: number; fitness: number }[] {
  const best = new Map<number, number>();
  for (const view of projection.views) {
    if (view.trainFitness == null) continue;
    const current = best.get(view.genome.generation);
    if (current == null || view.trainFitness > current) best.set(view.genome.generation, view.trainFitness);
  }
  return [...best.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([generation, fitness]) => ({ generation, fitness }));
}
