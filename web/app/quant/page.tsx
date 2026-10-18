import type { Metadata } from "next";
import Link from "next/link";
import { Header } from "@/components/header";

export const metadata: Metadata = { title: "Quant" };

export default function QuantPage() {
  return (
    <>
      <Header />
      <main className="mx-auto max-w-3xl px-4 py-16 md:px-6">
        <p className="kicker">Quant</p>
        <h1 className="mt-3 font-serif text-5xl tracking-tight">Coming soon.</h1>
        <p className="mt-4 text-lg leading-relaxed text-dim">
          The engine has no quant adapter in this build, so this tab does not show a strategy, a backtest, or a score. Historical trading stays off until a real evaluator exists.
        </p>
        <div className="mt-8 flex gap-3 text-sm">
          <Link href="/cloud" className="rounded-full border border-teal/50 px-4 py-2 text-teal">Open the cloud run</Link>
          <Link href="/shopping" className="rounded-full border border-line px-4 py-2">Open shopping</Link>
        </div>
        <p className="mt-10 text-sm text-faint" data-testid="quant-placeholder">Coming soon</p>
      </main>
    </>
  );
}
