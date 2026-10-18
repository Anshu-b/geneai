"use client";

import { useEffect, useState } from "react";

const TEAL = "#1f8a72";
const VIOLET = "#6d5bd0";
const GOLD = "#c4893a";
const ROSE = "#c45b78";
const BLUE = "#3c88b5";

type Tag = "parent" | "child" | "immigrant" | "dying";

interface Beast {
  id: string;
  genes: string[];
  tag?: Tag;
}

const PARENT_A: Beast = { id: "a", genes: [TEAL, TEAL, GOLD, TEAL, BLUE] };
const PARENT_B: Beast = { id: "b", genes: [VIOLET, GOLD, TEAL, ROSE, GOLD] };
const WEAK: Beast = { id: "c", genes: [ROSE, ROSE, VIOLET, BLUE, VIOLET] };
const MID: Beast = { id: "d", genes: [BLUE, TEAL, GOLD, VIOLET, TEAL] };
const OTHER: Beast = { id: "e", genes: [GOLD, VIOLET, ROSE, BLUE, ROSE] };
const CHILD_COPY: Beast = { id: "child", genes: [...PARENT_A.genes], tag: "child" };
const CHILD_MIX: Beast = { id: "child", genes: [TEAL, TEAL, GOLD, ROSE, GOLD], tag: "child" };
const CHILD_MUT: Beast = { id: "child", genes: [TEAL, ROSE, GOLD, ROSE, GOLD], tag: "child" };
const NEWCOMER: Beast = { id: "imm", genes: ["#e6c15a", "#2a9d8f", "#e07a5f", "#e6c15a", "#2a9d8f"], tag: "immigrant" };

const PHASES: { id: string; title: string; copy: string; beasts: Beast[] }[] = [
  {
    id: "variation",
    title: "Variation",
    copy: "No two creatures match. Each stripe is a gene, and the mix is already different before anyone breeds.",
    beasts: [PARENT_A, PARENT_B, WEAK, MID, OTHER],
  },
  {
    id: "selection",
    title: "Selection",
    copy: "The world keeps what fits. Outlines mark the two that get to breed. The pale one is a poor match.",
    beasts: [
      { ...PARENT_A, tag: "parent" },
      { ...PARENT_B, tag: "parent" },
      { ...WEAK, tag: "dying" },
      MID,
      OTHER,
    ],
  },
  {
    id: "inheritance",
    title: "Inheritance",
    copy: "A child starts as a copy of a parent. The stripes pass on. Nothing new has been invented yet.",
    beasts: [{ ...PARENT_A, tag: "parent" }, { ...PARENT_B, tag: "parent" }, CHILD_COPY, MID],
  },
  {
    id: "crossover",
    title: "Breeding",
    copy: "Two parents mix. The child takes the first stripes from one and the last stripes from the other. That is crossover.",
    beasts: [{ ...PARENT_A, tag: "parent" }, { ...PARENT_B, tag: "parent" }, CHILD_MIX, MID],
  },
  {
    id: "mutation",
    title: "Mutation",
    copy: "Copying is sloppy. One stripe comes out the wrong color. A mutation is a random copy error, not a plan.",
    beasts: [{ ...PARENT_A, tag: "parent" }, { ...PARENT_B, tag: "parent" }, CHILD_MUT, MID],
  },
  {
    id: "immigration",
    title: "Immigration",
    copy: "A newcomer walks in carrying colors the locals do not have. That outside blood is gene flow.",
    beasts: [PARENT_A, CHILD_MUT, NEWCOMER, MID],
  },
  {
    id: "death",
    title: "Death",
    copy: "The poorest fit dies off. Who remains is who gets another turn.",
    beasts: [PARENT_A, CHILD_MUT, NEWCOMER, { ...WEAK, tag: "dying" }],
  },
];

function fit(genes: string[]) {
  return genes.filter((gene) => gene === TEAL || gene === GOLD).length;
}

export function StripedBody({ genes, faint = false, flash = -1, scale = 1 }: { genes: string[]; faint?: boolean; flash?: number; scale?: number }) {
  const width = 72 * scale;
  const height = 128 * scale;
  return (
    <div className={`relative shrink-0 ${faint ? "opacity-35" : ""}`} style={{ width, height }} aria-hidden>
      <div className="absolute left-1/2 top-[2%] h-[12%] w-[22%] -translate-x-1/2 rounded-full border border-line bg-ivory/90" />
      <div className="absolute inset-x-[8%] top-[16%] bottom-[4%] overflow-hidden rounded-full border border-ivory/40 shadow-[0_16px_40px_rgba(0,0,0,0.22)]">
        <div className="flex h-full flex-col">
          {genes.map((gene, index) => (
            <span
              key={`${gene}-${index}`}
              className="flex-1"
              style={{
                background: gene,
                boxShadow: index === flash ? "inset 0 0 0 2px #fffaf0" : undefined,
              }}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

export function NaturalSelection() {
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const phase = PHASES[index];

  useEffect(() => {
    if (!playing) return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const timer = window.setInterval(() => setIndex((value) => (value + 1) % PHASES.length), 1600);
    return () => window.clearInterval(timer);
  }, [playing]);

  return (
    <div data-testid="natural-selection">
      <div className="glass rounded-3xl p-4 md:p-6">
        <div className="flex min-h-52 flex-wrap items-end justify-center gap-5 md:gap-10">
          {phase.beasts.map((beast) => (
            <figure key={beast.id + (beast.tag || "")} className="text-center">
              <div className={beast.tag === "parent" ? "rounded-full ring-2 ring-teal ring-offset-4 ring-offset-bg" : ""}>
                <StripedBody genes={beast.genes} faint={beast.tag === "dying"} flash={phase.id === "mutation" && beast.tag === "child" ? 1 : -1} />
              </div>
              <figcaption className="mt-3 font-mono text-[10px] uppercase tracking-widest text-faint">
                {beast.tag === "parent" ? "parent" : beast.tag === "child" ? "child" : beast.tag === "immigrant" ? "newcomer" : beast.tag === "dying" ? "fading" : `fit ${fit(beast.genes)}`}
              </figcaption>
            </figure>
          ))}
        </div>
        <p className="mt-4 text-center font-mono text-[10px] uppercase tracking-[0.16em] text-faint">The world favors teal and gold. Fit is how many of those stripes a creature carries.</p>
        <div className="mt-5 flex flex-wrap items-end justify-between gap-4">
          <p className="font-serif text-4xl text-ivory" data-testid="selection-phase">{phase.title}</p>
          <div className="flex gap-2">
            <button type="button" className="lift rounded-full border border-line px-4 py-2 text-sm" aria-pressed={playing} data-testid="selection-play" onClick={() => setPlaying((value) => !value)}>
              {playing ? "Pause" : "Play"}
            </button>
            <button
              type="button"
              className="lift rounded-full border border-line px-4 py-2 text-sm"
              data-testid="selection-step"
              onClick={() => {
                setPlaying(false);
                setIndex((value) => (value + 1) % PHASES.length);
              }}
            >
              Step
            </button>
          </div>
        </div>
      </div>
      <p className="mt-4 max-w-2xl text-base leading-relaxed text-ivory">{phase.copy}</p>
    </div>
  );
}

const GA_MAP = [
  ["Organism", "Candidate solution"],
  ["Gene", "One piece of that solution"],
  ["Fitness", "The score"],
  ["Mutation", "A random tweak"],
  ["Crossover", "Two candidates mixed"],
  ["Immigration", "A fresh random candidate"],
  ["Death", "Drop the low scores"],
];

export function GeneticAlgorithms() {
  return (
    <div className="grid items-center gap-8 lg:grid-cols-[auto_1fr]">
      <div className="flex items-center gap-4">
        <StripedBody genes={CHILD_MUT.genes} />
        <span className="font-serif text-3xl text-faint" aria-hidden>→</span>
        <div className="glass w-40 overflow-hidden rounded-2xl">
          <div className="flex h-3">
            {CHILD_MUT.genes.map((gene, index) => (
              <span key={`${gene}-${index}`} className="flex-1" style={{ background: gene }} />
            ))}
          </div>
          <p className="px-3 py-3 font-serif text-lg text-ivory">Candidate</p>
          <p className="px-3 pb-3 font-mono text-[10px] text-faint">fitness 3</p>
        </div>
      </div>
      <ul className="grid gap-2 sm:grid-cols-2">
        {GA_MAP.map(([bio, algo]) => (
          <li key={bio} className="flex items-center justify-between gap-3 rounded-2xl border border-line bg-raise/40 px-3 py-2 text-sm">
            <span className="text-dim">{bio}</span>
            <span className="font-serif text-ivory">{algo}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

const BRIDGE = [
  { bio: "Mutation", geneai: "Veri mutate", genes: CHILD_MUT.genes, note: "The random copy error. Qwen on Veri does most of these." },
  { bio: "Breeding", geneai: "Crossover", genes: CHILD_MIX.genes, note: "Two parents mix. CREAO makes a few of these calls." },
  { bio: "Immigration", geneai: "Immigrants", genes: NEWCOMER.genes, note: "A newcomer with no parents, so the pool does not go stale." },
  { bio: "Death", geneai: "Prune", genes: WEAK.genes, note: "The unfit leave the living set. They stay in the log." },
  { bio: "Survival", geneai: "Elites", genes: PARENT_A.genes, note: "The fittest are copied forward unchanged, under the same id." },
  { bio: "A broken child", geneai: "Repair", genes: [ROSE, TEAL, GOLD, BLUE, VIOLET], note: "One fix if the child breaks the rules. Then it is dropped." },
  { bio: "A twin", geneai: "Dedup", genes: PARENT_A.genes, note: "Two identical payloads are not both kept." },
];

export function GeneBridge() {
  return (
    <div className="mt-8 grid gap-3 sm:grid-cols-2 xl:grid-cols-3" data-testid="gene-bridge">
      {BRIDGE.map((item) => (
        <article key={item.geneai} className="glass overflow-hidden rounded-3xl">
          <div className="flex h-2.5" aria-hidden>
            {item.genes.map((gene, index) => (
              <span key={`${item.geneai}-${index}`} className="flex-1" style={{ background: gene }} />
            ))}
          </div>
          <div className="flex items-end gap-4 p-4">
            <StripedBody genes={item.genes} faint={item.geneai === "Prune"} scale={0.82} />
            <div className="min-w-0 pb-1">
              <p className="font-mono text-[10px] uppercase tracking-[0.16em] text-faint">{item.bio}</p>
              <h3 className="mt-1 font-serif text-2xl text-ivory">{item.geneai}</h3>
              <p className="mt-2 text-sm leading-relaxed text-dim">{item.note}</p>
            </div>
          </div>
        </article>
      ))}
    </div>
  );
}
