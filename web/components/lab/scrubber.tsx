"use client";

import { formatFitness } from "@/lib/format";
import type { Beat } from "@/lib/project";

interface Props {
  beats: Beat[];
  index: number;
  playing: boolean;
  speed: 1 | 2 | 4;
  onIndex: (index: number) => void;
  onPlaying: (playing: boolean) => void;
  onSpeed: (speed: 1 | 2 | 4) => void;
}

export function Scrubber({ beats, index, playing, speed, onIndex, onPlaying, onSpeed }: Props) {
  const beat = beats[index];
  const generations = [...new Set(beats.filter((item) => item.phase === "settled").map((item) => item.generation))];
  if (!beat) return null;
  const stats = beat.stats;
  const bits = [
    stats.births != null ? `${stats.births} born` : null,
    stats.prunes != null ? `${stats.prunes} pruned` : null,
    stats.elites != null ? `${stats.elites} elites` : null,
    stats.living != null ? `${stats.living} living` : null,
    stats.rejected ? `${stats.rejected} rejected` : null,
    stats.finalists != null ? `${stats.finalists} finalists` : null,
    stats.bestFitness != null ? `best train ${formatFitness(stats.bestFitness)}` : null,
    stats.bestGenomeId ? stats.bestGenomeId : null,
  ].filter(Boolean);

  return (
    <div className="panel mt-3 p-3" data-testid="scrubber">
      <div className="flex flex-wrap items-center gap-2">
        <button
          type="button"
          className="rounded-full border border-teal/50 px-3 py-1.5 text-sm text-teal"
          onClick={() => {
            if (playing) {
              onPlaying(false);
              return;
            }
            if (index >= beats.length - 1) onIndex(0);
            onPlaying(true);
          }}
        >
          {playing ? "Pause" : index >= beats.length - 1 ? "Replay" : "Play"}
        </button>
        <div className="flex rounded-full border border-line p-0.5" role="group" aria-label="Playback speed">
          {([1, 2, 4] as const).map((value) => (
            <button
              key={value}
              type="button"
              aria-pressed={speed === value}
              className={`rounded-full px-2 py-1 font-mono text-[11px] ${speed === value ? "bg-ivory/10 text-ivory" : "text-dim"}`}
              onClick={() => onSpeed(value)}
            >
              {value}×
            </button>
          ))}
        </div>
        <p className="min-w-0 flex-1 truncate font-serif text-lg text-ivory">{beat.title}</p>
        <button
          type="button"
          className="font-mono text-[10px] uppercase tracking-widest text-dim"
          onClick={() => {
            onPlaying(false);
            onIndex(beats.length - 1);
          }}
        >
          Jump to end
        </button>
      </div>
      <label className="mt-3 block">
        <span className="sr-only">Replay position</span>
        <input
          type="range"
          min={0}
          max={Math.max(0, beats.length - 1)}
          value={index}
          onChange={(event) => {
            onPlaying(false);
            onIndex(Number(event.target.value));
          }}
        />
      </label>
      <div className="mt-2 flex gap-1 overflow-x-auto pb-1">
        {generations.map((generation) => {
          const settled = beats.findIndex((item) => item.phase === "settled" && item.generation === generation);
          const active = beat.generation === generation && (beat.phase === "settled" || beat.phase === "prunes" || beat.phase === "births" || beat.phase === "elites");
          return (
            <button
              key={generation}
              type="button"
              className={`shrink-0 rounded px-2 py-1 font-mono text-[10px] ${active ? "bg-teal/15 text-teal" : "text-dim"}`}
              onClick={() => {
                onPlaying(false);
                if (settled >= 0) onIndex(settled);
              }}
            >
              G{generation}
            </button>
          );
        })}
        {beats.some((item) => item.phase === "frozen") ? (
          <button
            type="button"
            className={`shrink-0 rounded px-2 py-1 font-mono text-[10px] ${beat.phase === "frozen" || beat.phase === "tested" || beat.phase === "complete" ? "text-amber" : "text-dim"}`}
            onClick={() => {
              onPlaying(false);
              const frozen = beats.findIndex((item) => item.phase === "frozen");
              if (frozen >= 0) onIndex(frozen);
            }}
          >
            Freeze
          </button>
        ) : null}
      </div>
      <p className="mt-1 font-mono text-[11px] text-dim">{bits.join(" · ") || "Stepping the recorded event log."}</p>
    </div>
  );
}
