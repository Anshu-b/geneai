export function RankLegend({ note }: { note?: string }) {
  return (
    <p className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-dim" data-testid="rank-legend">
      <span>
        <span className="rounded bg-teal/15 px-1.5 py-0.5 font-semibold text-teal">✓ green</span> = best in column (ties all green)
      </span>
      <span>
        <span className="text-danger/80">red</span> = worst
      </span>
      <span>
        <span className="inline-block h-3 w-3 bg-teal/10 align-middle shadow-[inset_3px_0_0_var(--teal)]" /> GeneAI
      </span>
      {note ? <span className="text-faint">{note}</span> : null}
    </p>
  );
}
