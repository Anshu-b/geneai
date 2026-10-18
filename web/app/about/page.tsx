import type { Metadata } from "next";
import { Header } from "@/components/header";

export const metadata: Metadata = { title: "About" };

export default function AboutPage() {
  return (
    <>
      <Header />
      <main className="mx-auto max-w-3xl px-4 py-16 md:px-6">
        <p className="kicker">About</p>
        <h1 className="mt-3 font-serif text-5xl tracking-tight">Your agents shouldn&apos;t just execute. They should evolve.</h1>
        <div className="mt-8 space-y-5 text-base leading-relaxed text-dim">
          <p>
            GeneAI is one evolutionary engine. A domain adapter supplies the genome, the constraints, and the score. Shopping and cloud scheduling in this lab are two exports from that same engine, commit 6bbf16d, mode mock.
          </p>
          <p>
            The bundled runs are playback. Provider calls in them are mock, including the mock agent path. They are not veri.studio or CREAO executions, and the badge stays Mock. A run is Live only while a stream is actually open.
          </p>
          <p>
            The champion is frozen on validation, then scored once on a held-out test. That test stays hidden until <span className="text-ivory">champion.frozen</span>. The one-shot arm in these files was skipped. The comparison is a single attempt versus evolutionary search, and it is not compute-matched.
          </p>
          <p>
            Shopping items whose ids start with <span className="font-mono text-ivory">fx_</span> are fixture catalog rows. They have no store link and are not purchasable. Quant is not implemented.
          </p>
          <p>
            Node color is the recorded provider. Rows are families. Size is train fitness. Ghosts were pruned and stay in the log. An ivory ring is an elite carried with the same id. Gold is the frozen champion, even if a later generation pruned it from the living population.
          </p>
        </div>
      </main>
    </>
  );
}
