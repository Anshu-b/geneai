import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { cloudScoreboard, parseShowcase } from "../lib/showcase";

const cloud = parseShowcase(JSON.parse(readFileSync(join(process.cwd(), "public/fixtures/cloud_scheduling_showcase.json"), "utf8")));

describe("cloud scoreboard", () => {
  it("marks the winner of each metric from the data", () => {
    const board = cloudScoreboard(cloud)!;
    expect(board.geneLabel).toBe("GeneAI");
    expect(board.otherLabel).toBe("GPT-5.5");
    const champ = cloud.cloud!.policies.find((p) => p.key === "champion")!;
    const gpt = cloud.cloud!.policies.find((p) => p.key === "gpt55_one_shot")!;
    for (const row of board.rows) {
      const lowerWins = row.gene < row.other ? "gene" : row.gene > row.other ? "other" : "tie";
      const expected = row.better === "lower" ? lowerWins : lowerWins === "gene" ? "other" : lowerWins === "other" ? "gene" : "tie";
      expect(row.winner).toBe(expected);
      expect(row.sentence).toContain("GeneAI");
      expect(row.sentence).toContain("GPT-5.5");
    }
    const all = board.rows.find((row) => row.key === "all_missed")!;
    expect(all.gene).toBe(champ.aggregate!.sla_misses);
    expect(all.other).toBe(gpt.aggregate!.sla_misses);
    const caseMissed = board.rows.find((row) => row.key === "case_missed")!;
    expect(caseMissed.gene).toBe(champ.summary!.sla_misses);
    expect(board.cases).toBe(champ.aggregate!.n_cases);
    for (const row of board.rows.filter((r) => r.winner === "other")) expect(board.takeaway).toContain(`GPT-5.5 wins on`);
    expect(board.takeaway).toContain(`Across ${board.cases} held-out cases`);
  });
});
