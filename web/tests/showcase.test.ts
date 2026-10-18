import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { objectiveSentence } from "../lib/domain";
import {
  cloudVerdict,
  exampleMissLine,
  interpretShowcaseResponse,
  isManualBaseline,
  isSampleShowcase,
  keyScoreLines,
  originCredit,
  parseShowcase,
  shoppingSplitLine,
} from "../lib/showcase";

function load(name: string) {
  return parseShowcase(JSON.parse(readFileSync(join(process.cwd(), "public/fixtures", name), "utf8")));
}

describe("showcase fixtures", () => {
  it("parses the recorded cloud and shopping files", () => {
    const cloud = load("cloud_scheduling_showcase.json");
    const shop = load("shopping_showcase.json");
    expect(cloud.schema).toBe("geneai-showcase-v1");
    expect(shop.domain).toBe("shopping");
    expect(isSampleShowcase(cloud)).toBe(false);
    expect(isSampleShowcase(shop)).toBe(false);
    expect(cloud.cloud?.policies).toHaveLength(4);
    expect(shop.shopping?.outfits.length).toBeGreaterThan(0);
  });

  it("leads the cloud verdict with the held-out split and keeps the case as an example", () => {
    const cloud = load("cloud_scheduling_showcase.json");
    const policies = cloud.cloud!.policies;
    const champ = policies.find((policy) => policy.key === "champion")!;
    expect(cloudVerdict(policies)).toContain(`Across all ${champ.aggregate!.n_cases} held-out test cases`);
    expect(cloudVerdict(policies)).toContain(`GeneAI champion missed ${champ.aggregate!.sla_misses}`);
    const example = exampleMissLine(policies);
    for (const policy of policies) {
      expect(example).toContain(`${policy.label} missed ${policy.summary!.sla_misses}`);
    }
    const champion = policies.find((policy) => policy.key === "champion")!;
    expect(originCredit(champion.provider, champion.behavior_origin)).toMatch(/^Found by CREAO crossover g\d+/);
    const robin = policies.find((policy) => policy.key === "round_robin")!;
    expect(originCredit(robin.provider, robin.behavior_origin)).toBeNull();
  });

  it("scores only the agentic arms and keeps baselines out of that comparison", () => {
    const cloud = load("cloud_scheduling_showcase.json");
    const shop = load("shopping_showcase.json");
    expect(isManualBaseline("round_robin", "baseline")).toBe(true);
    expect(isManualBaseline("deadline_weighted", "baseline")).toBe(true);
    expect(isManualBaseline("best_baseline", "baseline")).toBe(true);
    expect(isManualBaseline("champion", "veri")).toBe(false);
    expect(isManualBaseline("gpt55_one_shot", "openai")).toBe(false);
    const policies = cloud.cloud!.policies;
    const champ = policies.find((p) => p.key === "champion")!;
    const gpt = policies.find((p) => p.key === "gpt55_one_shot")!;
    const cloudLines = keyScoreLines(cloud);
    expect(cloudLines[0]).toContain(`Missed deadlines: ${champ.summary!.sla_misses} vs ${gpt.summary!.sla_misses}`);
    expect(cloudLines.join(" ")).not.toContain("Round robin");
    expect(cloudLines.join(" ")).toContain(`${champ.aggregate!.sla_misses} vs ${gpt.aggregate!.sla_misses}`);
    const outfits = shop.shopping!.outfits;
    const gene = outfits.find((o) => o.key === "champion")!;
    const one = outfits.find((o) => o.key === "gpt55_one_shot")!;
    const shopLines = keyScoreLines(shop);
    expect(shopLines[0]).toContain(`$${Math.round(gene.total_usd)} vs $${Math.round(one.total_usd)}`);
    expect(shopLines.join(" ")).toContain(`${gene.aggregate!.feasible_count} of ${gene.aggregate!.n_shoppers}`);
    expect(shopLines.join(" ")).not.toContain("Neutral");
    expect(objectiveSentence("Schedule jobs onto a heterogeneous pool: CPU nodes, one high-memory node, and one GPU. Hard constraint: never exceed.")).toBe(
      "Schedule jobs onto a heterogeneous pool: CPU nodes, one high-memory node, and one GPU.",
    );
  });

  it("summarizes shopping across shoppers and degrades when the file is missing", () => {
    const shop = load("shopping_showcase.json");
    const n = shop.shopping!.outfits[0].aggregate!.n_shoppers;
    expect(shoppingSplitLine(shop.shopping!.outfits)).toContain(`Across all ${n} held-out shoppers`);
    expect(interpretShowcaseResponse(404, null)).toEqual({ status: "missing" });
    expect(interpretShowcaseResponse(200, { schema: "nope" }).status).toBe("invalid");
  });
});
