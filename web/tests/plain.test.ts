import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { cloudPolicySummary, outfitSummary, plainWhy, shopperRequest } from "../lib/plain";
import { parseShowcase } from "../lib/showcase";

const load = (name: string) => parseShowcase(JSON.parse(readFileSync(join(process.cwd(), "public/fixtures", name), "utf8")));

describe("plain-English text", () => {
  it("summarizes cloud policies from their real rules", () => {
    const cloud = load("cloud_scheduling_showcase.json");
    for (const policy of cloud.cloud!.policies) {
      const text = cloudPolicySummary(policy)!;
      const window = policy.rules_plain!.map((rule) => /first (\d+) waiting/.exec(rule)?.[1]).find(Boolean);
      expect(text).toContain(`the next ${window} waiting jobs`);
    }
    const champ = cloud.cloud!.policies.find((p) => p.key === "champion")!;
    expect(cloudPolicySummary(champ)).toBe(
      "GeneAI looks at the next 24 waiting jobs and runs hard-deadline jobs and short jobs first. It sends each job to the machine type it fits and packs already-busy CPUs a little tighter.",
    );
  });

  it("writes the shopper request and outfit summaries from the fixture", () => {
    const shop = load("shopping_showcase.json");
    const shopper = shop.shopping!.shopper;
    const request = shopperRequest(shopper);
    expect(request).toContain(`Shopper ${shopper.id}`);
    expect(request).toContain(`under $${shopper.budget_usd}`);
    for (const outfit of shop.shopping!.outfits) {
      const text = outfitSummary(outfit, shopper.budget_usd);
      for (const item of outfit.items) expect(text).toContain(item.name);
    }
  });

  it("rewrites engine score bullets without raw floats or lists", () => {
    const lines = plainWhy([
      "Style match 0.00: 0/3 items share bright tags ['athletic', 'bright']",
      "Occasion fit 0.67: 2/3 items match ['dinner', 'gym']",
      "Bright colors: 3/3 items (component 1.00)",
      "Availability confidence 1.00",
      "Under budget by $86.00 (spend $254.00 of $340.00, efficiency 0.25)",
    ]);
    expect(lines.map((line) => line.label)).toEqual(["Style", "Occasion", "Colors", "In stock", "Budget"]);
    expect(lines[0]).toEqual({ label: "Style", mark: "no", text: "0 of 3 items are tagged athletic or bright." });
    expect(lines[1].mark).toBe("partial");
    expect(lines[4].text).toBe("$254 of $340, with $86 left over.");
    for (const line of lines) {
      expect(line.text).not.toMatch(/\[|\]|0\.\d\d/);
    }
  });
});
