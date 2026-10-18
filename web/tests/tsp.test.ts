import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  heldoutLine,
  hitsOptimum,
  interpretTspResponse,
  isSampleTsp,
  parseTsp,
  timeFraction,
  tspVerdict,
  type TspFile,
} from "../lib/tsp";

function load(): TspFile {
  return parseTsp(JSON.parse(readFileSync(join(process.cwd(), "public/fixtures/tsp_benchmark.json"), "utf8")));
}

function walkLength(file: TspFile, tour: string[], close: boolean): number {
  const byId = new Map(file.cities.map((city) => [city.id, city]));
  let sum = 0;
  for (let index = 0; index < tour.length - 1; index += 1) {
    const from = byId.get(tour[index])!;
    const to = byId.get(tour[index + 1])!;
    sum += Math.hypot(from.x - to.x, from.y - to.y);
  }
  if (close) {
    const from = byId.get(tour[tour.length - 1])!;
    const to = byId.get(tour[0])!;
    sum += Math.hypot(from.x - to.x, from.y - to.y);
  }
  return sum;
}

describe("tsp benchmark", () => {
  it("parses the sample as four methods with Held-Karp as the optimum", () => {
    const file = load();
    expect(file.schema).toBe("geneai-tsp-v1");
    expect(file.cities).toHaveLength(15);
    expect(isSampleTsp(file)).toBe(true);
    expect(file.methods.map((method) => method.key)).toEqual(["held_karp", "classic_ga", "llm_one_shot", "geneai"]);
    expect(file.methods.map((method) => method.label)).toEqual([
      "Traditional: dynamic programming (Held-Karp)",
      "Classic genetic algorithm",
      "GPT-5.5 single attempt",
      "GeneAI",
    ]);
    const held = file.methods[0];
    expect(held.gap_pct).toBe(0);
    expect(held.time_ms).toBeGreaterThan(0);
    expect(file.optimum.tour).toEqual(held.tour);
    expect(file.optimum.length).toBe(held.length);
    expect(hitsOptimum(held)).toBe(true);
    expect(Math.abs(walkLength(file, held.tour, true) - held.length)).toBeLessThan(0.02);
    const classic = file.methods[1];
    expect(classic.gap_pct).toBeGreaterThan(4);
    expect(hitsOptimum(classic)).toBe(false);
    expect(hitsOptimum(file.methods[3])).toBe(true);
    const shot = file.methods[2];
    expect(shot.valid).toBe(false);
    expect(shot.invalid_reason).toMatch(/never visits c7/);
    expect(shot.tour.filter((id) => id === "c2").length).toBeGreaterThan(1);
    expect(shot.tour).not.toContain("c7");
    expect(tspVerdict(file)).toContain("Held-Karp proves a tour");
    expect(tspVerdict(file)).toContain("GeneAI matches that tour");
    expect(tspVerdict(file)).toContain(shot.invalid_reason!);
  });

  it("keeps the held-out summary consistent and rejects a heuristic arm", () => {
    const raw = JSON.parse(readFileSync(join(process.cwd(), "public/fixtures/tsp_benchmark.json"), "utf8"));
    const file = parseTsp(raw);
    expect(file.heldout.n).toBe(20);
    expect(file.heldout.summary.held_karp.optimal_hits).toBe(20);
    expect(file.heldout.summary.geneai.optimal_hits).toBe(16);
    expect(file.heldout.summary.llm_one_shot.valid_count).toBe(6);
    expect(heldoutLine(file)).toContain("Across 20 held-out maps");
    expect(heldoutLine(file)).toContain("16 of 20");
    for (const key of ["held_karp", "classic_ga", "llm_one_shot", "geneai"] as const) {
      const rows = raw.heldout.instances.map((instance: { methods: Record<string, { gap_pct: number | null; valid: boolean; time_ms: number }> }) => instance.methods[key]);
      const valid = rows.filter((row: { valid: boolean; gap_pct: number | null }) => row.valid && row.gap_pct != null);
      const meanGap = valid.reduce((sum: number, row: { gap_pct: number }) => sum + row.gap_pct, 0) / valid.length;
      expect(file.heldout.summary[key].mean_gap_pct).toBeCloseTo(meanGap, 1);
      expect(file.heldout.summary[key].optimal_hits).toBe(rows.filter((row: { valid: boolean; gap_pct: number | null }) => row.valid && row.gap_pct === 0).length);
    }
    expect(timeFraction(10, 1000, true)).toBeGreaterThan(timeFraction(10, 1000, false));
    expect(interpretTspResponse(404, null)).toEqual({ status: "missing" });
    raw.methods[0].key = "heuristic";
    expect(interpretTspResponse(200, raw).status).toBe("invalid");
  });
});