export type LoadSource = "bundled" | "upload" | "url" | "live";

export type ModeBadge = "Live" | "Recorded" | "Fixture" | "Mock";

export interface Presentation {
  badge: ModeBadge;
  detail: string;
  runMode: string;
}

/**
 * Badge comes from the export's run.mode, except Live, which is only used
 * while an SSE stream is actually open. Playback of a live or hybrid export
 * stays Recorded so a file never looks like it is computing.
 */
export function presentationMode(runMode: string | undefined, liveConnected: boolean): Presentation {
  const mode = (runMode || "").trim();
  const lower = mode.toLowerCase();
  if (liveConnected) {
    return {
      badge: "Live",
      runMode: mode || "unknown",
      detail: "A stream is open on the configured API. This is not inferred from the file alone.",
    };
  }
  if (lower === "mock") {
    return {
      badge: "Mock",
      runMode: mode,
      detail: "run.mode is mock. Provider calls in this export are not sponsor execution.",
    };
  }
  if (lower === "fixture") {
    return {
      badge: "Fixture",
      runMode: mode,
      detail: "run.mode is fixture. This is not a live or recorded sponsor run.",
    };
  }
  if (lower === "recorded") {
    return {
      badge: "Recorded",
      runMode: mode,
      detail: "Playback of a recorded export. Nothing is computing now.",
    };
  }
  if (lower === "live" || lower === "hybrid") {
    return {
      badge: "Recorded",
      runMode: mode,
      detail: `Playback of a ${lower} export. The run is not executing now.`,
    };
  }
  return {
    badge: "Recorded",
    runMode: mode || "unknown",
    detail: mode
      ? `Playback. The export says run.mode=${mode}.`
      : "Playback. The export did not include a run mode.",
  };
}
