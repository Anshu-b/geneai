import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  cloudVerdict,
  exampleMissLine,
  interpretShowcaseResponse,
  isSampleShowcase,
  originCredit,
  parseShowcase,
  shoppingSplitLine,
} from "../lib/showcase";

function load(name: string) {
  return parseShowcase(JSON.parse(readFileSync(join(process.cwd(), "public/fixtures", name), "utf8")));
}

describe("showcase fixtures", () => {
  it("parses the sample cloud and shopping files", () => {
    const cloud = load("cloud_scheduling_showcase.json");
    const shop = load("shopping_showcase.json");
    expect(cloud.schema).toBe("geneai-showcase-v1");
    expect(shop.domain).toBe("shopping");
    expect(isSampleShowcase(cloud)).toBe(true);
    expect(isSampleShowcase(shop)).toBe(true);
    expect(cloud.cloud?.policies).toHaveLength(4);
    expect(shop.shopping?.outfits.length).toBeGreaterThan(0);
  });

  it("leads the cloud verdict with the held-out split and keeps the case as an example", () => {
    const cloud = load("cloud_scheduling_showcase.json");
    const policies = cloud.cloud!.policies;
    expect(cloudVerdict(policies)).toContain("Across all 12 held-out test cases");
    expect(cloudVerdict(policies)).toContain("GeneAI champion missed 31");
    const example = exampleMissLine(policies);
    expect(example).toContain("GeneAI champion missed 3");
    expect(example).toContain("GPT-5.5 one shot missed 6");
    expect(example).toContain("Round robin missed 8");
    const champion = policies.find((policy) => policy.key === "champion")!;
    expect(originCredit(champion.provider, champion.behavior_origin)).toBe(
      "Found by CREAO crossover g0051, rediscovered in gen 5.",
    );
    const robin = policies.find((policy) => policy.key === "round_robin")!;
    expect(originCredit(robin.provider, robin.behavior_origin)).toBeNull();
  });

  it("summarizes shopping across shoppers and degrades when the file is missing", () => {
    const shop = load("shopping_showcase.json");
    expect(shoppingSplitLine(shop.shopping!.outfits)).toContain("Across all 20 held-out shoppers");
    expect(interpretShowcaseResponse(404, null)).toEqual({ status: "missing" });
    expect(interpretShowcaseResponse(200, { schema: "nope" }).status).toBe("invalid");
  });
});
