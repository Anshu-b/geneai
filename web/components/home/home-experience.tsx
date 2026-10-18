"use client";

import { AnimatePresence, motion, useReducedMotion, useScroll, useSpring } from "framer-motion";
import Link from "next/link";
import { useState } from "react";
import { Header } from "@/components/header";
import { LivingField } from "@/components/home/living-field";
import { GeneBridge, GeneticAlgorithms, NaturalSelection } from "@/components/home/story";
import { TspShowdown } from "@/components/home/tsp-showdown";

const STAGES = [
  { id: "seed", title: "Seed", copy: "The first generation drops in any hand-written baselines, then fills the remaining slots from the model." },
  { id: "evaluate", title: "Evaluate", copy: "Each candidate is scored on the training cases. That score is the only signal selection uses." },
  { id: "select", title: "Select", copy: "A tournament of three picks parents. A small novelty bonus and a family cap keep one family from taking over." },
  { id: "vary", title: "Mutate / cross", copy: "Mutation is a Veri mutate, a random copy error done in bulk. Breeding is crossover: two parents mix, and CREAO makes a few of those calls." },
  { id: "repair", title: "Repair", copy: "If a child breaks the rules, it gets one repair attempt. If that fails, the child is dropped." },
  { id: "dedup", title: "Dedup", copy: "A twin is not kept twice. Two genomes with the same payload are not both kept." },
  { id: "prune", title: "Prune", copy: "Death is prune. The unfit leave the living population and stay in the log." },
  { id: "elites", title: "Elites", copy: "The fittest survive as elites, copied forward with the same id. Immigration is a newcomer with no parents." },
];

/** Verified against the live Veri deployment (bfe84d51e141), the CREAO provider config, and run exports. */
const MODELS = [
  {
    role: "Bulk mutation · Veri.studio",
    name: "Qwen/Qwen2.5-7B-Instruct",
    detail: "Deployment geneai-qwen7b, served with vLLM on 1× NVIDIA L4 (24 GB) on AWS, 8 concurrent requests.",
  },
  {
    role: "Strategist · CREAO",
    name: "CREAO agent (agent.creao.ai)",
    detail: "Called as an app-scoped CREAO agent. CREAO does not report which model it runs underneath, so we do not name one.",
  },
  {
    role: "One-shot comparison · OpenAI",
    name: "gpt-5.5 (gpt-5.5-2026-04-23)",
    detail: "One Responses API call per run, reasoning effort low. The recorded cloud run predates that setting and used the API default.",
  },
];

const rise = {
  hidden: { opacity: 0, y: 28 },
  show: { opacity: 1, y: 0, transition: { duration: 0.7, ease: [0.22, 1, 0.36, 1] as const } },
};

function LoopRing({
  stage,
  reduced,
  onPick,
}: {
  stage: string;
  reduced: boolean;
  onPick: (stage: (typeof STAGES)[number]) => void;
}) {
  const radius = 138;
  return (
    <div className="relative mx-auto mt-10 w-full max-w-xl">
      <svg viewBox="0 0 400 400" className="h-auto w-full" role="img" aria-label="Evolutionary loop">
        <circle cx="200" cy="200" r={radius} fill="none" stroke="var(--line)" strokeWidth="1" />
        <circle
          cx="200"
          cy="200"
          r={radius}
          fill="none"
          stroke="var(--teal)"
          strokeWidth="2"
          strokeDasharray="36 240"
          className={reduced ? undefined : "loop-orbit"}
          style={{ transformOrigin: "200px 200px" }}
        />
        {STAGES.map((item, index) => {
          const angle = (index / STAGES.length) * Math.PI * 2 - Math.PI / 2;
          const x = 200 + Math.cos(angle) * radius;
          const y = 200 + Math.sin(angle) * radius;
          const active = item.id === stage;
          return (
            <g key={item.id} onClick={() => onPick(item)} className="cursor-pointer">
              <circle cx={x} cy={y} r={active ? 15 : 9} fill={active ? "var(--teal)" : "var(--raise)"} stroke="var(--ivory)" strokeWidth="1" />
              <text x={x} y={y + (index < 4 ? -20 : 24)} textAnchor="middle" fill="var(--dim)" fontSize="11" fontFamily="IBM Plex Mono, ui-monospace, monospace">
                {item.title}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

export function HomeExperience() {
  const reduced = useReducedMotion();
  const { scrollYProgress } = useScroll();
  const progress = useSpring(scrollYProgress, { stiffness: 120, damping: 24, restDelta: 0.001 });
  const [stage, setStage] = useState(STAGES[0]);
  const [frozen, setFrozen] = useState(false);

  return (
    <div data-testid="architecture-home">
      <motion.div className="fixed inset-x-0 top-0 z-40 h-[3px] origin-left bg-teal" style={{ scaleX: reduced ? 1 : progress }} />
      <Header />
      <section className="relative min-h-[100svh] overflow-hidden">
        <LivingField />
        <div className="pointer-events-none absolute inset-0 bg-gradient-to-r from-bg from-0% via-bg/80 via-34% to-transparent to-62%" />
        <div className="pointer-events-none absolute inset-x-0 bottom-0 h-32 bg-gradient-to-t from-bg to-transparent" />
        <div className="relative z-10 mx-auto flex min-h-[100svh] max-w-[1600px] flex-col justify-end px-4 pb-16 pt-28 md:px-6 md:pb-24">
          <p className="kicker">GeneAI · the engine</p>
          <h1 className="mt-4 max-w-4xl font-serif text-[clamp(3.4rem,8vw,7.4rem)] leading-[0.9] tracking-tight text-ivory">
            One loop.
            <span className="mt-1 block italic text-ivory/80">Better agents.</span>
          </h1>
          <p className="mt-6 max-w-xl text-lg leading-relaxed text-dim">
            A population of policies is scored, crossed, and pruned. The survivor is chosen on validation. The test is read once.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <Link href="/cloud" data-testid="hero-cta" className="btn-ink lift rounded-full px-5 py-2.5 text-sm font-medium">Open the cloud run</Link>
            <Link href="/shopping" className="lift rounded-full border border-line bg-raise/40 px-5 py-2.5 text-sm text-ivory backdrop-blur">See a shopping run</Link>
          </div>
        </div>
      </section>

      <main className="mx-auto max-w-[1600px] px-4 md:px-6">
        <motion.section className="py-24 md:py-32" initial={reduced ? false : "hidden"} whileInView="show" viewport={{ once: true, margin: "-12%" }} variants={rise}>
          <p className="kicker">In the wild</p>
          <h2 className="mt-3 max-w-3xl font-serif text-4xl tracking-tight text-ivory md:text-6xl">Natural selection</h2>
          <p className="mt-4 max-w-2xl text-base leading-relaxed text-dim">
            Genes pass on. Some copies are wrong, some children mix two parents, strangers arrive, and the unfit die off. What fits the world is what remains.
          </p>
          <div className="mt-8">
            <NaturalSelection />
          </div>
        </motion.section>

        <motion.section className="pb-24 md:pb-32" initial={reduced ? false : "hidden"} whileInView="show" viewport={{ once: true, margin: "-12%" }} variants={rise}>
          <p className="kicker">Borrowed in the 1970s</p>
          <h2 className="mt-3 max-w-3xl font-serif text-4xl tracking-tight text-ivory md:text-6xl">Genetic algorithms</h2>
          <p className="mt-4 max-w-2xl text-base leading-relaxed text-dim">
            John Holland wrote this loop down as a search. A genome is a candidate solution. A fitness function is the score. The operators stay the same.
          </p>
          <div className="mt-8">
            <GeneticAlgorithms />
          </div>
        </motion.section>

        <motion.section
          className="pb-24 md:pb-32"
          initial={reduced ? false : "hidden"}
          whileInView="show"
          viewport={{ once: true, margin: "-12%" }}
          variants={rise}
        >
          <p className="kicker">The same loop, on policies</p>
          <h2 className="mt-3 max-w-3xl font-serif text-4xl tracking-tight text-ivory md:text-6xl">Same steps, every agentic generation.</h2>
          <p className="mt-4 max-w-2xl text-base leading-relaxed text-dim">
            The creature becomes a policy. The stripes are the rules it carries. Each biological step has an operator in the engine.
          </p>
          <GeneBridge />
          <LoopRing stage={stage.id} reduced={!!reduced} onPick={setStage} />
          <div className="mt-8 grid grid-cols-2 gap-2 md:grid-cols-4 xl:grid-cols-8">
            {STAGES.map((item, index) => {
              const active = item.id === stage.id;
              return (
                <button
                  key={item.id}
                  type="button"
                  data-testid={`loop-${item.id}`}
                  aria-pressed={active}
                  onClick={() => setStage(item)}
                  className={`lift relative rounded-2xl border px-3 py-4 text-left ${active ? "border-teal bg-teal/10" : "border-line bg-raise/30"}`}
                >
                  <span className="font-mono text-[10px] text-faint">0{index + 1}</span>
                  <span className="mt-2 block font-serif text-xl text-ivory">{item.title}</span>
                </button>
              );
            })}
          </div>
          <AnimatePresence mode="wait">
            <motion.p
              key={stage.id}
              data-testid="stage-copy"
              initial={reduced ? false : { opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={reduced ? undefined : { opacity: 0, y: -8 }}
              transition={{ duration: 0.28 }}
              className="glass mt-4 max-w-3xl rounded-2xl px-5 py-4 text-base leading-relaxed text-ivory"
            >
              {stage.copy}
            </motion.p>
          </AnimatePresence>
        </motion.section>

        <motion.section className="pb-24 md:pb-32" initial={reduced ? false : "hidden"} whileInView="show" viewport={{ once: true, margin: "-12%" }} variants={rise}>
          <p className="kicker">Who does what</p>
          <h2 className="mt-3 max-w-3xl font-serif text-4xl tracking-tight text-ivory md:text-6xl">A few sharp calls. A lot of small ones.</h2>
          <div className="mt-10 grid gap-4 lg:grid-cols-3">
            <article className="glass lift rounded-3xl p-6">
              <p className="kicker">Strategist</p>
              <h3 className="mt-3 font-serif text-3xl text-ivory">CREAO</h3>
              <p className="mt-3 text-sm leading-relaxed text-dim">A few high-level calls each generation, crossover first, then mutation. The default cap is three, and that batch runs together.</p>
              <div className="mt-6 flex gap-2" aria-hidden>
                {[0, 1, 2].map((dot) => (
                  <span key={dot} className="h-3 w-3 rounded-full bg-violet" />
                ))}
              </div>
            </article>
            <article className="glass lift rounded-3xl p-6 lg:translate-y-6">
              <p className="kicker">Bulk mutation</p>
              <h3 className="mt-3 font-serif text-3xl text-ivory">Qwen2.5-7B on Veri</h3>
              <p className="mt-3 text-sm leading-relaxed text-dim">veri.studio hosts the model that writes most mutations and the immigrants. Most of the population comes from here, not from CREAO.</p>
              <div className="mt-6 flex flex-wrap gap-1.5" aria-hidden>
                {Array.from({ length: 18 }, (_, index) => (
                  <span key={index} className="h-2 w-2 rounded-full bg-teal" style={{ opacity: 0.45 + (index % 5) * 0.1 }} />
                ))}
              </div>
            </article>
            <article className="glass lift rounded-3xl p-6">
              <p className="kicker">Outside the loop</p>
              <h3 className="mt-3 font-serif text-3xl text-ivory">GPT-5.5</h3>
              <p className="mt-3 text-sm leading-relaxed text-dim">One request, scored on the same test. That is a single attempt against the search. It is not compute-matched, and it is not a parent in the lineage.</p>
              <div className="mt-6 h-3 w-3 rounded-full bg-amber" aria-hidden />
            </article>
          </div>
          <div className="glass mt-12 rounded-3xl p-6" data-testid="models-in-build">
            <p className="kicker">Models in this build</p>
            <dl className="mt-4 grid gap-5 md:grid-cols-3">
              {MODELS.map((model) => (
                <div key={model.role}>
                  <dt className="font-mono text-[10px] uppercase tracking-[0.16em] text-faint">{model.role}</dt>
                  <dd className="mt-1 font-serif text-2xl text-ivory">{model.name}</dd>
                  <dd className="mt-1 text-sm leading-relaxed text-dim">{model.detail}</dd>
                </div>
              ))}
            </dl>
            <p className="mt-4 text-xs text-faint">From the Veri deployment record, the CREAO app config, and the model id the OpenAI API returned in the recorded runs.</p>
          </div>
        </motion.section>

        <motion.section className="pb-24 md:pb-32" initial={reduced ? false : "hidden"} whileInView="show" viewport={{ once: true, margin: "-12%" }} variants={rise}>
          <p className="kicker">Train, validation, test</p>
          <h2 className="mt-3 max-w-3xl font-serif text-4xl tracking-tight text-ivory md:text-6xl">The test stays sealed.</h2>
          <p className="mt-4 max-w-2xl text-base leading-relaxed text-dim">Train is what the loop sees. Validation picks the champion. Test is not read until that champion is frozen, and then only once.</p>
          <div className="mt-8 grid gap-4 md:grid-cols-3">
            <SplitPane label="Train" detail="Seen every generation. Selection uses only this score." status="Open" open dots={18} />
            <SplitPane label="Validation" detail="Picks the champion (amber) from the finalists." status="Open" open dots={8} highlight="first" />
            <SplitPane
              label="Test"
              detail={frozen ? "Champion scored once, after the freeze. Nothing changes after this." : "Locked. No genome is scored here until the champion is frozen."}
              status={frozen ? "Read once" : "Sealed"}
              open={frozen}
              dots={8}
              highlight={frozen ? "first" : undefined}
            />
          </div>
          <div className="mt-4 flex min-h-10 flex-wrap items-center gap-3">
            <button
              type="button"
              data-testid="freeze-champion"
              className={`lift h-10 rounded-full border px-4 text-sm ${frozen ? "border-teal text-teal" : "border-line"}`}
              aria-pressed={frozen}
              disabled={frozen}
              onClick={() => setFrozen(true)}
            >
              {frozen ? "✓ Champion frozen, test read once" : "Freeze the champion"}
            </button>
            {frozen ? (
              <button type="button" className="h-10 text-sm text-dim underline underline-offset-4" onClick={() => setFrozen(false)}>
                Replay
              </button>
            ) : null}
          </div>
        </motion.section>

        <motion.section className="pb-24 md:pb-32" initial={reduced ? false : "hidden"} whileInView="show" viewport={{ once: true, margin: "-12%" }} variants={rise}>
          <p className="kicker">Domain adapters</p>
          <h2 className="mt-3 max-w-3xl font-serif text-4xl tracking-tight text-ivory md:text-6xl">Same engine. Swap the score.</h2>
          <p className="mt-4 max-w-2xl text-base leading-relaxed text-dim">The controller does not import a domain. Cloud, shopping, or quant is a plugin that supplies the genome, the rules, and the score.</p>
          <div className="mt-8 grid gap-4 md:grid-cols-[1.2fr_1fr_1fr_1fr] md:items-stretch">
            <div className="glass flex items-center rounded-3xl p-6">
              <div>
                <p className="kicker">Controller</p>
                <p className="mt-2 font-serif text-3xl text-ivory">One loop</p>
                <p className="mt-2 text-sm text-dim">Operators, elites, and the lineage stay put.</p>
              </div>
            </div>
            <Adapter href="/cloud" name="Cloud" detail="Schedule jobs on a mixed pool." />
            <Adapter href="/shopping" name="Shopping" detail="Build an outfit for one shopper." />
            <div className="rounded-3xl border border-dashed border-line p-6 opacity-70">
              <p className="kicker">Soon</p>
              <p className="mt-2 font-serif text-3xl text-ivory">Quant</p>
              <p className="mt-2 text-sm text-dim">The plug is reserved. The adapter is not built.</p>
            </div>
          </div>
        </motion.section>

        <motion.section className="pb-24 md:pb-32" initial={reduced ? false : "hidden"} whileInView="show" viewport={{ once: true, margin: "-12%" }} variants={rise}>
          <TspShowdown />
        </motion.section>

        <section className="pb-28">
          <div className="glass flex flex-col items-start justify-between gap-6 rounded-3xl px-6 py-8 md:flex-row md:items-center md:px-10">
            <div>
              <p className="kicker">Recorded runs</p>
              <h2 className="mt-2 font-serif text-4xl text-ivory">Watch the search, then read what it produced.</h2>
            </div>
            <div className="flex flex-wrap gap-3">
              <Link href="/cloud" className="btn-ink lift rounded-full px-5 py-2.5 text-sm font-medium">Cloud scheduling</Link>
              <Link href="/shopping" className="lift rounded-full border border-line px-5 py-2.5 text-sm">Shopping</Link>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

function SplitPane({
  label,
  detail,
  status,
  open,
  dots,
  highlight,
}: {
  label: string;
  detail: string;
  status: string;
  open: boolean;
  dots: number;
  highlight?: "first";
}) {
  const accent = highlight === "first";
  return (
    <article
      data-testid={`split-${label.toLowerCase()}`}
      className={`flex min-h-[148px] flex-col rounded-3xl border p-5 transition-colors ${accent ? "border-teal/50 bg-teal/5" : "border-line bg-raise/30"} ${open ? "" : "border-dashed"}`}
    >
      <div className="flex items-center justify-between gap-3">
        <p className="kicker">{label}</p>
        <span className={`rounded-full border px-2 py-0.5 font-mono text-[10px] uppercase tracking-[0.14em] ${open ? "border-teal/40 text-teal" : "border-line text-faint"}`}>
          {open ? status : `🔒 ${status}`}
        </span>
      </div>
      <p className={`mt-3 text-sm leading-relaxed ${open ? "text-ivory" : "text-dim"}`}>{detail}</p>
      <div className="mt-auto flex flex-wrap gap-1.5 pt-4" aria-hidden>
        {Array.from({ length: dots }, (_, index) => (
          <span
            key={index}
            className={`h-2 w-2 rounded-full ${accent && index === 0 ? "bg-amber" : open ? "bg-ivory/70" : "bg-ivory/20"}`}
          />
        ))}
      </div>
    </article>
  );
}

function Adapter({ href, name, detail }: { href: string; name: string; detail: string }) {
  return (
    <Link href={href} className="glass lift block rounded-3xl p-6">
      <p className="kicker">Adapter</p>
      <p className="mt-2 font-serif text-3xl text-ivory">{name}</p>
      <p className="mt-2 text-sm text-dim">{detail}</p>
    </Link>
  );
}
