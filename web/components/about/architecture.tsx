"use client";

import { useState } from "react";

type NodeId =
  | "domain"
  | "creao"
  | "veri"
  | "ops"
  | "scorer"
  | "select"
  | "test"
  | "export"
  | "site"
  | "baselines";

type ArchNode = {
  id: NodeId;
  step: string;
  title: string;
  where: string;
  line: string;
  bullets: string[];
  accent: string;
};

const NODES: Record<NodeId, ArchNode> = {
  domain: {
    id: "domain",
    step: "01",
    title: "Domain adapter",
    where: "Python · geneai_core/domains",
    line: "Each problem plugs in a genome schema, its constraints, train/val/test cases, and a scorer.",
    accent: "var(--dim)",
    bullets: [
      "Four adapters on one engine: cloud scheduling, TSP, shopping, and Liquid (funding policy).",
      "A genome is plain JSON: a scheduling policy, a city tour, an outfit, or a funding policy.",
      "Shopping catalog: Lululemon pages collected with the Browser Use Cloud API, tagged by a deterministic script, then frozen to a file.",
      "Liquid works in integer cents, so money never drifts on floating point.",
    ],
  },
  creao: {
    id: "creao",
    step: "02",
    title: "CREAO strategist",
    where: "CREAO cloud · GeneAI Mutator app",
    line: "An agent receives the best parents and their scores, then proposes mutate or crossover children.",
    accent: "var(--prov-creao)",
    bullets: [
      "Called through CREAO's app API trigger: create a run, then poll until it completes.",
      "Default of 3 strategist calls per generation, so it steers the search without paying for every child.",
      "Its output is validated against the domain schema and structurally repaired. Anything still illegal is rejected.",
      "CREAO does not report which model sits under the agent, so we don't name one.",
    ],
  },
  veri: {
    id: "veri",
    step: "03",
    title: "Veri · Qwen2.5-7B",
    where: "veri.studio · vLLM on 1× NVIDIA L4",
    line: "Our own Qwen/Qwen2.5-7B-Instruct deployment writes new genomes from the parents it is shown.",
    accent: "var(--prov-veri)",
    bullets: [
      "An OpenAI-compatible endpoint at api.veri.studio/v1/deployments/{id}, served by vLLM on one 24 GB L4 GPU.",
      "At most 8 requests in flight. 429 and 503 (deployment waking) are retried using Retry-After.",
      "About 16 tokens/s, so a 400-token genome takes about 25 s. Billing is about $1.05/hr while the deployment is up.",
      "Every genome records the provider that produced it, which is how the graph colors its nodes.",
    ],
  },
  ops: {
    id: "ops",
    step: "04",
    title: "Operators & repair",
    where: "Python · operators.py, selection.py",
    line: "Plain code handles the classic GA work: selection, crossover, mutation, repair, and local polish.",
    accent: "var(--teal)",
    bullets: [
      "Diversity-aware tournament selection, with elites carried forward unchanged.",
      "TSP: city swap and 2-opt reversal mutations, order crossover, repair of broken tours, and a short 2-opt polish.",
      "Hard budgets per run: 800 model calls, 900 evaluations, and a wall-clock cap.",
    ],
  },
  scorer: {
    id: "scorer",
    step: "05",
    title: "Deterministic scorer",
    where: "Python · adapter.evaluate()",
    line: "The same genome always gets the same fitness. No LLM is involved in grading.",
    accent: "var(--amber)",
    bullets: [
      "Scores are cached by (genome content hash, split), so a repeated genome is never scored twice.",
      "Infeasible answers, such as over budget or a broken tour, are flagged. They don't receive a good score.",
      "Liquid example: fitness = −(cost + risk), checked against an APR +5pp shock, a −30% market, and a $2k emergency.",
    ],
  },
  select: {
    id: "select",
    step: "06",
    title: "Champion on validation",
    where: "EvolutionController",
    line: "Evolution climbs on train. The top finalists are then re-scored on validation, and the best one is frozen.",
    accent: "var(--teal)",
    bullets: [
      "Train fitness drives every generation. Six finalists are compared on validation cases they never trained on.",
      "Choosing the winner on separate data catches policies that only memorised the train cases.",
      "The champion is frozen with a champion.frozen event before test is touched.",
    ],
  },
  test: {
    id: "test",
    step: "07",
    title: "Sealed test",
    where: "EvolutionController · read once",
    line: "The frozen champion is scored once on held-out test cases. Those are the numbers we report.",
    accent: "var(--amber)",
    bullets: [
      "Test runs only after the champion is frozen, so no choice was ever tuned on it.",
      "It is read exactly once. No retries, no picking the best of several test runs.",
      "GPT-5.5 and the traditional algorithms are graded on these same test cases by the same scorer.",
    ],
  },
  export: {
    id: "export",
    step: "08",
    title: "Event log → fixture",
    where: "runs/<run_id>/ → web/public/fixtures",
    line: "Every step is an immutable event, and the finished run is exported as hashed JSON.",
    accent: "var(--violet)",
    bullets: [
      "Each run writes events.jsonl, checkpoint.json, export.json, and summary.json.",
      "Lineage events record each genome's parents, its operator, and the provider that produced it.",
      "export.json carries a sha256 over canonical JSON. build_showcase.py re-scores the featured held-out case with the domain evaluator.",
      "Trimmed copies are committed as web/public/fixtures/*.json.",
    ],
  },
  site: {
    id: "site",
    step: "09",
    title: "Next.js replay site",
    where: "Next.js App Router · browser",
    line: "The site you are on loads those JSON fixtures and replays the recorded run.",
    accent: "var(--ivory)",
    bullets: [
      "Each page fetches its own file: /cloud, /shopping, /liquid, and the TSP showdown on the home page.",
      "No model calls happen during the demo, which is why the badge says Recorded rather than Live.",
      "The evolution graph is drawn straight from the lineage events: color is the provider, gold is the champion.",
    ],
  },
  baselines: {
    id: "baselines",
    step: "↔",
    title: "Comparison arms",
    where: "Same scorer · same test cases",
    line: "Rivals are graded by the exact scorer GeneAI is graded by, on the same test cases.",
    accent: "var(--danger)",
    bullets: [
      "GPT-5.5 one-shot: a single request to OpenAI gpt-5.5 (Responses API), then scored. One attempt against a search, not compute-matched.",
      "TSP traditional: DFS branch-and-bound, an exact solver that proves the optimum, plus a classic genetic algorithm.",
      "Results are attached to the same export, so the site compares them side by side.",
    ],
  },
};

const LOOP: NodeId[] = ["creao", "veri", "ops", "scorer", "select"];

function Arrow({ dir = "down" }: { dir?: "down" | "right" }) {
  if (dir === "right") {
    return (
      <svg aria-hidden viewBox="0 0 24 12" className="hidden h-3 w-5 shrink-0 md:block" style={{ color: "var(--faint)" }}>
        <path d="M0 6h20m-5-5 5 5-5 5" fill="none" stroke="currentColor" strokeWidth="1.4" />
      </svg>
    );
  }
  return (
    <div className="flex justify-center py-1.5" aria-hidden>
      <svg viewBox="0 0 12 24" className="h-6 w-3" style={{ color: "var(--faint)" }}>
        <path d="M6 0v20m-5-5 5 5 5-5" fill="none" stroke="currentColor" strokeWidth="1.4" />
      </svg>
    </div>
  );
}

function Node({ node, active, onPick, compact }: { node: ArchNode; active: boolean; onPick: (id: NodeId) => void; compact?: boolean }) {
  return (
    <button
      type="button"
      onMouseEnter={() => onPick(node.id)}
      onFocus={() => onPick(node.id)}
      onClick={() => onPick(node.id)}
      aria-pressed={active}
      className={`lift group relative w-full min-w-0 self-stretch rounded-xl border text-left ${compact ? "px-3 py-3" : "px-4 py-3.5"} ${
        active ? "bg-raise-2" : "bg-raise/50 hover:bg-raise"
      }`}
      style={{ borderColor: active ? node.accent : "var(--line)", boxShadow: active ? `0 0 0 1px ${node.accent}` : undefined }}
    >
      <span className="flex items-center gap-2">
        <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: node.accent }} />
        <span className="font-mono text-[10px] tracking-widest text-faint">{node.step}</span>
        <span className="text-sm font-medium text-ivory">{node.title}</span>
      </span>
      <span className="mt-1 block font-mono text-[10px] text-faint">{node.where}</span>
      <span className={`mt-2 block text-xs leading-snug text-dim ${compact ? "" : "md:text-[13px]"}`}>{node.line}</span>
    </button>
  );
}

export function Architecture() {
  const [active, setActive] = useState<NodeId>("veri");
  const pick = (id: NodeId) => setActive(id);
  const current = NODES[active];

  return (
    <section data-testid="architecture" className="mx-auto max-w-6xl px-4 pb-24 md:px-6">
      <p className="kicker">System architecture</p>
      <h2 className="mt-3 font-serif text-4xl tracking-tight text-ivory">How a run moves through the system</h2>
      <p className="mt-3 max-w-3xl text-base leading-relaxed text-dim">
        A Python engine runs the evolution offline and calls two hosted model services. Every step goes into a log, and this site replays that log. Hover or tap a box to see the details.
      </p>

      <div className="mt-10 grid gap-6 lg:grid-cols-[minmax(0,1fr)_320px]">
        <div className="min-w-0">
          <div className="mx-auto max-w-md">
            <Node node={NODES.domain} active={active === "domain"} onPick={pick} />
          </div>
          <Arrow />

          <div className="relative rounded-2xl border border-dashed p-4 md:p-5" style={{ borderColor: "color-mix(in srgb, var(--teal) 45%, transparent)" }}>
            <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
              <span className="kicker" style={{ color: "var(--teal)" }}>Evolution controller · one generation</span>
              <span className="font-mono text-[10px] text-faint">↻ repeats for N generations · train split</span>
            </div>
            <div className="flex flex-col gap-2 md:flex-row md:items-stretch">
              {LOOP.map((id, i) => (
                <div key={id} className="flex min-w-0 flex-1 items-center gap-1.5">
                  <Node node={NODES[id]} active={active === id} onPick={pick} compact />
                  {i < LOOP.length - 1 ? <Arrow dir="right" /> : null}
                </div>
              ))}
            </div>
            <svg aria-hidden viewBox="0 0 1000 28" preserveAspectRatio="none" className="mt-2 hidden h-6 w-full md:block" style={{ color: "color-mix(in srgb, var(--teal) 60%, transparent)" }}>
              <path d="M900 2v14H100V2" fill="none" stroke="currentColor" strokeWidth="1.4" strokeDasharray="5 5" vectorEffect="non-scaling-stroke" />
              <path d="M94 9l6-8 6 8" fill="none" stroke="currentColor" strokeWidth="1.4" vectorEffect="non-scaling-stroke" />
            </svg>
            <p className="text-center font-mono text-[10px] text-faint md:-mt-1">survivors become the next generation&apos;s parents</p>
          </div>

          <div className="mt-4 grid gap-4 md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
            <div>
              <Arrow />
              <Node node={NODES.test} active={active === "test"} onPick={pick} />
            </div>
            <div>
              <div className="flex justify-center py-1.5 font-mono text-[10px] text-faint" aria-hidden>
                <span className="h-6 leading-6">↔ same scorer</span>
              </div>
              <div className="rounded-xl border border-dashed" style={{ borderColor: "color-mix(in srgb, var(--danger) 45%, transparent)" }}>
                <Node node={NODES.baselines} active={active === "baselines"} onPick={pick} />
              </div>
            </div>
          </div>
          <Arrow />
          <div className="flex flex-col gap-2 md:flex-row md:items-stretch">
            <div className="flex min-w-0 flex-1 items-center gap-1.5">
              <Node node={NODES.export} active={active === "export"} onPick={pick} />
              <Arrow dir="right" />
            </div>
            <div className="flex-1">
              <Node node={NODES.site} active={active === "site"} onPick={pick} />
            </div>
          </div>

          <div className="mt-5 flex flex-wrap gap-x-5 gap-y-2 font-mono text-[10px] text-faint">
            <span><span className="mr-1.5 inline-block h-2 w-2 rounded-full" style={{ background: "var(--prov-creao)" }} />hosted agent (CREAO)</span>
            <span><span className="mr-1.5 inline-block h-2 w-2 rounded-full" style={{ background: "var(--prov-veri)" }} />hosted GPU model (veri.studio)</span>
            <span><span className="mr-1.5 inline-block h-2 w-2 rounded-full" style={{ background: "var(--teal)" }} />engine code</span>
            <span><span className="mr-1.5 inline-block h-2 w-2 rounded-full" style={{ background: "var(--amber)" }} />grading</span>
            <span><span className="mr-1.5 inline-block h-2 w-2 rounded-full" style={{ background: "var(--violet)" }} />recorded artifacts</span>
          </div>
        </div>

        <aside className="glass h-fit rounded-2xl p-5 lg:sticky lg:top-24" aria-live="polite">
          <div className="flex items-center gap-2">
            <span className="h-2.5 w-2.5 rounded-full" style={{ background: current.accent }} />
            <span className="font-mono text-[10px] tracking-widest text-faint">{current.step}</span>
            <h3 className="text-base font-medium text-ivory">{current.title}</h3>
          </div>
          <p className="mt-1 font-mono text-[10px] text-faint">{current.where}</p>
          <p className="mt-3 text-sm leading-relaxed text-ivory">{current.line}</p>
          <ul className="mt-4 space-y-2.5">
            {current.bullets.map((b) => (
              <li key={b} className="flex gap-2.5 text-[13px] leading-snug text-dim">
                <span className="mt-[7px] h-1 w-1 shrink-0 rounded-full" style={{ background: current.accent }} />
                <span>{b}</span>
              </li>
            ))}
          </ul>
        </aside>
      </div>
    </section>
  );
}
