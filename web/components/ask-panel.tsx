"use client";

import { useState, type FormEvent } from "react";
import { Header } from "@/components/header";
import { classifyGoal, launchBody } from "@/lib/ask";
import { geneaiApiBase } from "@/lib/sse";

const CHIPS = [
  "Build an outfit under $250",
  "Reduce cloud queueing time",
  "Explore historical trading strategies",
];

export function AskPanel() {
  const [prompt, setPrompt] = useState("");
  const [pending, setPending] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const [fault, setFault] = useState<string | null>(null);
  const api = geneaiApiBase();
  const classification = classifyGoal(prompt);
  const canLaunch = Boolean(api) && classification.supported && !pending;

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    if (!canLaunch || (classification.domain !== "shopping" && classification.domain !== "cloud_scheduling")) return;
    setPending(true);
    setFault(null);
    setResult(null);
    try {
      const response = await fetch(`${api.replace(/\/$/, "")}/api/runs`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(launchBody(prompt, classification.domain)),
      });
      const text = await response.text();
      setResult(`${response.status}\n${text.slice(0, 4000)}`);
      if (!response.ok) setFault("The API did not accept the run. Nothing on this page was invented to fill the gap.");
    } catch (err) {
      setFault(err instanceof Error ? err.message : "The request failed.");
    } finally {
      setPending(false);
    }
  }

  return (
    <>
      <Header />
      <main className="mx-auto max-w-3xl px-4 py-14 md:px-6">
        <p className="kicker">Ask GeneAI</p>
        <h1 className="mt-3 font-serif text-5xl tracking-tight">What would you like GeneAI to optimize?</h1>
        <p className="mt-4 max-w-2xl text-base leading-relaxed text-dim">
          The composer only names a registered adapter. It does not evolve an answer in the browser, and it will not invent a fitness for a domain that has no evaluator.
        </p>
        <form onSubmit={onSubmit} className="mt-8" data-testid="ask-composer">
          <label htmlFor="goal" className="sr-only">Goal</label>
          <textarea
            id="goal"
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            rows={4}
            placeholder="A measurable goal with a domain GeneAI already knows how to score."
            className="w-full rounded-md border border-line bg-raise px-4 py-3 text-lg outline-none"
          />
          <div className="mt-3 flex flex-wrap gap-2">
            {CHIPS.map((chip) => (
              <button key={chip} type="button" className="rounded-full border border-line px-3 py-1.5 text-sm text-dim hover:text-ivory" onClick={() => setPrompt(chip)}>
                {chip}
              </button>
            ))}
          </div>
          <div className="mt-6 rounded border border-line p-4">
            <p className="kicker">{classification.supported ? "Adapter" : "Not launched"}</p>
            <p className="mt-2 font-serif text-2xl">{classification.title}</p>
            <p className="mt-2 text-sm leading-relaxed text-dim">{classification.reason}</p>
          </div>
          <button
            type="submit"
            disabled={!canLaunch}
            className="mt-4 rounded-full border border-teal/50 px-4 py-2 text-sm text-teal disabled:border-line disabled:text-faint"
          >
            {pending ? "Sending…" : "Launch micro run"}
          </button>
          {!api ? (
            <p className="mt-3 text-sm text-dim">
              Live launch is off until NEXT_PUBLIC_GENEAI_API points at the GeneAI service. Cloud and Shopping replays still open.
            </p>
          ) : (
            <p className="mt-3 text-sm text-dim">POST {api.replace(/\/$/, "")}/api/runs · population 4 · 2 generations. The response is shown raw.</p>
          )}
          {fault ? <p className="mt-3 text-sm text-danger" role="alert">{fault}</p> : null}
          {result ? <pre className="mt-4 max-h-80 overflow-auto rounded border border-line bg-black/30 p-3 font-mono text-xs">{result}</pre> : null}
        </form>
      </main>
    </>
  );
}
