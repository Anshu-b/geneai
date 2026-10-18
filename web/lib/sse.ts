import type { GeneEvent } from "./types";

export interface StreamHandlers {
  onEvent: (event: GeneEvent) => void;
  onStatus: (status: "open" | "error" | "closed", detail?: string) => void;
}

export interface StreamHandle {
  close: () => void;
}

/** SSE hook for a live run. No-op unless NEXT_PUBLIC_GENEAI_API is set by the caller. */
export function openRunStream(apiBase: string, runId: string, handlers: StreamHandlers): StreamHandle {
  const base = apiBase.replace(/\/$/, "");
  const url = `${base}/api/runs/${encodeURIComponent(runId)}/stream`;
  const source = new EventSource(url);
  source.onopen = () => handlers.onStatus("open");
  source.onerror = () => handlers.onStatus("error", "The event stream reported an error.");
  source.onmessage = (message) => {
    try {
      const data = JSON.parse(message.data) as Partial<GeneEvent>;
      if (!data || typeof data.type !== "string" || typeof data.sequence !== "number") {
        handlers.onStatus("error", "A stream message was not a GeneAI event.");
        return;
      }
      handlers.onEvent(data as GeneEvent);
    } catch {
      handlers.onStatus("error", "A stream message was not JSON.");
    }
  };
  return {
    close() {
      source.close();
      handlers.onStatus("closed");
    },
  };
}

export function geneaiApiBase(): string {
  return process.env.NEXT_PUBLIC_GENEAI_API?.trim() || "";
}
