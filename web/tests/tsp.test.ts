import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  heldoutLine,
  hitsOptimum,
  interpretTspResponse,
  isSampleTsp,
  parseTsp,
  provenanceLine,
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
  it("parses the recorded file as four methods with Held-Karp as the optimum", () => {
    const file = load();
    expect(file.schema).toBe("geneai-tsp-v1");
    expect(file.cities).toHaveLength(15);
    expect(isSampleTsp(file)).toBe(false);
    expect(file.methods.map((method) => method.key)).toEqual(["held_karp", "classic_ga", "llm_one_shot", "geneai"]);
    const held = file.methods[0];
    expect(held.gap_pct).toBe(0);
    expect(held.time_ms).toBeGreaterThan(0);
    expect(file.optimum.tour).toEqual(held.tour);
    expect(file.optimum.length).toBe(held.length);
    expect(hitsOptimum(held)).toBe(true);
    expect(Math.abs(walkLength(file, held.tour, true) - held.length!)).toBeLessThan(0.02);
    for (const method of file.methods) {
      if (method.valid) {
        expect(method.length!).toBeGreaterThanOrEqual(held.length! - 0.01);
        expect(new Set(method.tour).size).toBe(15);
      } else {
        expect(method.invalid_reason).toBeTruthy();
      }
    }
    const verdict = tspVerdict(file);
    expect(verdict).toContain("Held-Karp proves a tour");
    const gene = file.methods[3];
    if (gene.provenance_counts) {
      for (const count of Object.values(gene.provenance_counts)) expect(provenanceLine(file)).toContain(`${count} from`);
    }
    const shot = file.methods[2];
    if (!shot.valid) expect(verdict).toContain(shot.invalid_reason!);
  });

  it("keeps the held-out summary consistent and rejects a heuristic arm", () => {
    const raw = JSON.parse(readFileSync(join(process.cwd(), "public/fixtures/tsp_benchmark.json"), "utf8"));
    const file = parseTsp(raw);
    const n = file.heldout.n;
    expect(raw.heldout.instances).toHaveLength(n);
    expect(file.heldout.summary.held_karp.optimal_hits).toBe(n);
    expect(heldoutLine(file)).toContain(`Across ${n} held-out maps`);
    for (const key of ["held_karp", "classic_ga", "llm_one_shot", "geneai"] as const) {
      type Row = { gap_pct: number | null; valid: boolean };
      const rows: Row[] = raw.heldout.instances.map((instance: { by_method?: Record<string, Row>; methods?: Record<string, Row> }) => (instance.by_method ?? instance.methods)![key]);
      const valid = rows.filter((row) => row.valid && row.gap_pct != null);
      expect(file.heldout.summary[key].valid_count).toBe(rows.filter((row) => row.valid).length);
      if (valid.length > 0) {
        const meanGap = valid.reduce((sum, row) => sum + row.gap_pct!, 0) / valid.length;
        expect(file.heldout.summary[key].mean_gap_pct).toBeCloseTo(meanGap, 1);
      } else {
        expect(file.heldout.summary[key].mean_gap_pct).toBeNull();
      }
      expect(file.heldout.summary[key].optimal_hits).toBe(rows.filter((row) => row.valid && row.gap_pct === 0).length);
    }
    expect(interpretTspResponse(404, null)).toEqual({ status: "missing" });
    raw.methods[0].key = "heuristic";
    expect(interpretTspResponse(200, raw).status).toBe("invalid");
  });
});
