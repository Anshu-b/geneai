"use client";

import gsap from "gsap";
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { familyOf, formatFitness } from "@/lib/format";
import {
  PROVIDER_COLOR,
  operatorLabel,
  providerBucket,
  type LayoutModel,
  type Projection,
} from "@/lib/project";

const LABEL_X = 200;

interface Props {
  layout: LayoutModel;
  projection: Projection;
  ancestry: Set<string>;
  follow: boolean;
  filter: "all" | "living" | "pruned" | "infeasible" | "ancestry";
  family: string;
  selectedId: string | null;
  onSelect: (id: string) => void;
  focusGeneration: number | null;
  beatId: string;
  beatPhase: string;
  playing: boolean;
  reduced: boolean;
  compact?: boolean;
}

function shapePoints(kind: string, r: number): string | null {
  if (kind === "baseline") {
    return null;
  }
  if (kind === "veri") {
    return `0,${-r} ${r},0 0,${r} ${-r},0`;
  }
  if (kind === "random") {
    return `0,${-r} ${r * 0.9},${r * 0.75} ${-r * 0.9},${r * 0.75}`;
  }
  if (kind === "creao") {
    const pts: string[] = [];
    for (let i = 0; i < 6; i += 1) {
      const a = (Math.PI / 3) * i - Math.PI / 6;
      pts.push(`${Math.cos(a) * r},${Math.sin(a) * r}`);
    }
    return pts.join(" ");
  }
  return null;
}

export function LineageSvg({
  layout,
  projection,
  ancestry,
  follow,
  filter,
  family,
  selectedId,
  onSelect,
  focusGeneration,
  beatId,
  beatPhase,
  playing,
  reduced,
  compact = false,
}: Props) {
  const scrollerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const userScaled = useRef(false);
  const [hover, setHover] = useState<string | null>(null);
  const [scale, setScale] = useState(0.7);

  const visible = useMemo(() => {
    return projection.views.filter((view) => {
      if (family !== "all" && familyOf(view.genome.metadata) !== family) return false;
      if (filter === "living") return view.status === "kept" || view.status === "elite" || view.status === "champion";
      if (filter === "pruned") return view.status === "pruned" || view.laterPruned;
      if (filter === "infeasible") return view.feasible === false;
      if (filter === "ancestry") return ancestry.has(view.genome.id);
      return true;
    });
  }, [projection.views, family, filter, ancestry]);

  const visibleIds = useMemo(() => new Set(visible.map((view) => view.genome.id)), [visible]);

  const labelIds = useMemo(() => {
    const ids = new Set<string>();
    if (hover) ids.add(hover);
    if (selectedId) ids.add(selectedId);
    if (follow) ancestry.forEach((id) => ids.add(id));
    if (focusGeneration != null) {
      for (const view of visible) {
        if (view.genome.generation === focusGeneration) ids.add(view.genome.id);
      }
    }
    for (const view of visible) {
      const label = operatorLabel(view.genome);
      if (label === "crossover" || label === "repair") ids.add(view.genome.id);
    }
    return ids;
  }, [hover, selectedId, follow, ancestry, focusGeneration, visible]);

  useEffect(() => {
    const scroller = scrollerRef.current;
    if (!scroller || !playing) return;
    const generation = focusGeneration ?? layout.generations[layout.generations.length - 1];
    if (generation == null) return;
    const target = layout.xFor(generation) * scale - scroller.clientWidth * 0.32;
    if (reduced) {
      scroller.scrollLeft = Math.max(0, target);
      return;
    }
    const tween = gsap.to(scroller, {
      scrollLeft: Math.max(0, target),
      duration: 0.55,
      ease: "power2.inOut",
      overwrite: true,
    });
    return () => {
      tween.kill();
    };
  }, [playing, focusGeneration, beatId, scale, reduced, layout]);

  useEffect(() => {
    if (reduced) return;
    if (beatPhase !== "frozen" && beatPhase !== "tested") return;
    const ctx = gsap.context(() => {
      gsap.fromTo(
        ".champion-halo",
        { opacity: 0.2 },
        { opacity: 1, duration: 0.65, yoyo: true, repeat: 3, ease: "sine.inOut" },
      );
    }, svgRef);
    return () => ctx.revert();
  }, [beatId, beatPhase, reduced]);

  function fitView() {
    const scroller = scrollerRef.current;
    if (!scroller) return;
    const fitW = (scroller.clientWidth - 16) / layout.width;
    const fitH = (scroller.clientHeight - 16) / layout.height;
    const narrow = scroller.clientWidth < 760;
    const next = narrow ? Math.min(1, Math.max(0.5, fitW)) : Math.min(1, Math.max(0.32, Math.min(fitW, fitH)));
    userScaled.current = false;
    setScale(next);
  }

  useLayoutEffect(() => {
    if (userScaled.current) return;
    fitView();
    // fit once for this plate; later zoom is explicit
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [layout.width, layout.height]);

  const keptByGeneration = new Map<number, number>();
  for (const view of projection.views) {
    if (view.status === "kept" || view.status === "elite") {
      keptByGeneration.set(view.genome.generation, (keptByGeneration.get(view.genome.generation) ?? 0) + 1);
    }
  }

  return (
    <div className="relative">
      {compact ? null : <div className="absolute right-3 top-3 z-10 flex gap-1">
        <button type="button" className="panel px-2 py-1 font-mono text-[10px] uppercase tracking-widest" onClick={() => { userScaled.current = true; setScale((value) => Math.max(0.32, value - 0.1)); }}>
          Smaller
        </button>
        <button type="button" className="panel px-2 py-1 font-mono text-[10px] uppercase tracking-widest" onClick={() => { userScaled.current = true; setScale((value) => Math.min(1.6, value + 0.1)); }}>
          Larger
        </button>
        <button type="button" className="panel px-2 py-1 font-mono text-[10px] uppercase tracking-widest" onClick={fitView}>
          Fit
        </button>
      </div>}
      <div ref={scrollerRef} className={`lineage-stage ghost-scroll overflow-auto rounded-md border border-line ${compact ? "h-[340px]" : "h-[68vh] min-h-[520px]"}`} data-testid="lineage-svg">
        <svg
          ref={svgRef}
          width={layout.width * scale}
          height={layout.height * scale}
          viewBox={`0 0 ${layout.width} ${layout.height}`}
          role="img"
          aria-label="Lineage of genomes by generation and family"
        >
          <defs>
            <filter id="champ-glow" x="-40%" y="-40%" width="180%" height="180%">
              <feGaussianBlur stdDeviation="2.2" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>
          {layout.lanes.map((lane, index) => (
            <g key={lane.family}>
              <rect
                x={0}
                y={lane.y}
                width={layout.width}
                height={lane.height}
                fill={index % 2 === 0 ? "var(--lane)" : "transparent"}
              />
              <text
                x={LABEL_X}
                y={lane.y + lane.height / 2}
                textAnchor="end"
                dominantBaseline="middle"
                fill="var(--lane-label)"
                fontSize="11"
                fontFamily="IBM Plex Mono, ui-monospace, monospace"
              >
                {lane.family}
                <title>{lane.family}</title>
              </text>
            </g>
          ))}
          {layout.generations.map((generation) => {
            const x = layout.xFor(generation);
            const focused = focusGeneration === generation;
            const kept = keptByGeneration.get(generation) ?? 0;
            const championHere =
              projection.championFrozen &&
              projection.views.some(
                (view) => view.status === "champion" && view.genome.generation === generation,
              );
            return (
              <g key={generation} data-gen-col={generation}>
                {focused ? (
                  <rect x={x - 46} y={36} width={92} height={layout.height - 48} fill="var(--focus-col)" />
                ) : null}
                <text x={x} y={22} textAnchor="middle" fill={focused ? "var(--gen-focus)" : "var(--gen-label)"} fontSize="12" fontFamily="IBM Plex Mono, ui-monospace, monospace">
                  G{generation}
                </text>
                <text x={x} y={38} textAnchor="middle" fill="var(--gen-meta)" fontSize="9" fontFamily="IBM Plex Mono, ui-monospace, monospace">
                  {championHere ? "champion" : `${kept} kept`}
                </text>
              </g>
            );
          })}
          {visible.map((view) =>
            view.genome.parent_ids.map((parentId) => {
              if (!visibleIds.has(parentId)) return null;
              const parent = layout.nodes.get(parentId);
              const child = layout.nodes.get(view.genome.id);
              if (!parent || !child) return null;
              const label = operatorLabel(view.genome);
              const midX = (parent.x + child.x) / 2;
              const midY = (parent.y + child.y) / 2 - 8;
              const onPath = follow && ancestry.has(view.genome.id) && ancestry.has(parentId);
              const selected = selectedId === view.genome.id || selectedId === parentId;
              const faded = view.status === "pruned" && !onPath && !selected;
              const stroke =
                label === "crossover" ? "var(--edge-cross)" : label === "repair" || label === "immigrant" ? "var(--edge-rose)" : "var(--edge-mutate)";
              return (
                <g key={`${parentId}-${view.genome.id}`}>
                  <path
                    d={`M ${parent.x} ${parent.y} C ${midX} ${parent.y}, ${midX} ${child.y}, ${child.x} ${child.y}`}
                    fill="none"
                    stroke={onPath ? "var(--edge-path)" : stroke}
                    strokeWidth={onPath || selected ? 1.7 : 1}
                    strokeDasharray={label === "repair" ? "3 3" : undefined}
                    opacity={faded ? 0.12 : onPath || selected ? 0.95 : 0.45}
                  />
                  {labelIds.has(view.genome.id) && label !== "seed" ? (
                    <text
                      x={midX}
                      y={midY}
                      textAnchor="middle"
                      fontSize="8"
                      fontFamily="IBM Plex Mono, ui-monospace, monospace"
                      fill={stroke}
                      stroke="var(--label-halo)"
                      strokeWidth="3"
                      paintOrder="stroke"
                      style={{ pointerEvents: "none" }}
                    >
                      {label}
                    </text>
                  ) : null}
                </g>
              );
            }),
          )}
          {visible.map((view) => {
            const pos = layout.nodes.get(view.genome.id);
            if (!pos) return null;
            const bucket = providerBucket(view.genome.provider);
            const color = PROVIDER_COLOR[bucket];
            const dimmed = follow && projection.championFrozen && !ancestry.has(view.genome.id);
            const opacity = view.status === "pruned" ? "var(--pruned-opacity)" : dimmed ? 0.16 : 1;
            const points = shapePoints(bucket, pos.r);
            const immigrant = operatorLabel(view.genome) === "immigrant";
            const selected = selectedId === view.genome.id;
            return (
              <g
                key={view.genome.id}
                transform={`translate(${pos.x} ${pos.y})`}
                data-genome={view.genome.id}
                className="node"
                style={{
                  cursor: "pointer",
                  opacity,
                  transition: reduced ? undefined : "opacity 380ms ease",
                }}
                onMouseEnter={() => setHover(view.genome.id)}
                onMouseLeave={() => setHover((current) => (current === view.genome.id ? null : current))}
                onClick={() => onSelect(view.genome.id)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    onSelect(view.genome.id);
                  }
                }}
                role="button"
                tabIndex={0}
                aria-pressed={selected}
                aria-label={`${view.genome.id}, generation ${view.genome.generation}, ${view.status}, ${view.genome.provider}, ${familyOf(view.genome.metadata)}`}
              >
                {view.status === "champion" ? (
                  <circle className="champion-halo" r={pos.r + 8} fill="none" stroke="var(--champ)" strokeWidth="1.6" filter="url(#champ-glow)" />
                ) : null}
                {view.status === "elite" || (view.status === "champion" && projection.eliteIds.includes(view.genome.id)) ? (
                  <circle r={pos.r + 4.5} fill="none" stroke="var(--elite-ring)" strokeWidth="1.1" />
                ) : null}
                {selected ? <circle r={pos.r + 6.5} fill="none" stroke="var(--select-ring)" strokeWidth="1" /> : null}
                {view.status === "pruned" ? (
                  <circle r={pos.r + 2.4} fill="none" stroke={color} strokeWidth="0.8" strokeDasharray="2 2" />
                ) : null}
                {bucket === "baseline" ? (
                  <rect x={-pos.r} y={-pos.r} width={pos.r * 2} height={pos.r * 2} fill={view.feasible === false ? "none" : color} stroke={color} strokeWidth="1.2" />
                ) : points ? (
                  <polygon points={points} fill={view.feasible === false ? "none" : color} stroke={color} strokeWidth="1.2" />
                ) : (
                  <circle r={pos.r} fill={view.feasible === false ? "none" : color} stroke={color} strokeWidth="1.2" />
                )}
                {view.feasible === false ? (
                  <line x1={-pos.r} y1={pos.r} x2={pos.r} y2={-pos.r} stroke="var(--infeasible)" strokeWidth="1.2" />
                ) : null}
                {immigrant ? (
                  <polygon points={`0,${-pos.r - 7} 4,${-pos.r - 1} -4,${-pos.r - 1}`} fill="var(--edge-rose)" />
                ) : null}
                {immigrant && (labelIds.has(view.genome.id) || hover === view.genome.id) ? (
                  <text y={pos.r + 12} textAnchor="middle" fontSize="8" fill="var(--edge-rose)" fontFamily="IBM Plex Mono, ui-monospace, monospace">
                    immigrant
                  </text>
                ) : null}
              </g>
            );
          })}
        </svg>
      </div>
      {hover ? <LineageTooltip id={hover} projection={projection} /> : null}
    </div>
  );
}

function LineageTooltip({ id, projection }: { id: string; projection: Projection }) {
  const view = projection.byId.get(id);
  if (!view) return null;
  return (
    <div
      className="pointer-events-none absolute bottom-3 left-3 z-10 max-w-xs rounded border border-line bg-raise px-3 py-2"
      data-testid="lineage-tooltip"
      role="tooltip"
    >
      <p className="font-mono text-[11px] text-ivory">{view.genome.id}</p>
      <p className="text-[11px] text-dim">
        {view.status} · {view.genome.provider} · gen {view.genome.generation}
        {view.trainFitness != null ? ` · train ${formatFitness(view.trainFitness)}` : ""}
      </p>
    </div>
  );
}
