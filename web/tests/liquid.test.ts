import { describe, expect, it } from "vitest";
import { bestIndices, parseLiquid, totalAssetsCents, usd } from "../lib/liquid";
import fixture from "../public/fixtures/liquid.json";

describe("liquid", () => {
  it("marks ties and ignores nulls", () => {
    expect([...bestIndices([3, 1, 1, null], "lower")]).toEqual([1, 2]);
    expect([...bestIndices([-5, 0, null], "higher")]).toEqual([1]);
    expect(bestIndices([null, undefined], "lower").size).toBe(0);
  });
  it("formats cents", () => {
    expect(usd(1479)).toBe("$14.79");
    expect(usd(300000, 0)).toBe("$3,000");
  });
  it("reads the recorded run and keeps credit out of assets", () => {
    const file = parseLiquid(fixture);
    expect(file.mode).toBe("saved_run_replay");
    expect(totalAssetsCents(file.scenario)).toBe(3_350_000);
    const geneai = file.candidates.find((c) => c.key === "geneai");
    expect(geneai?.valid).toBe(true);
    for (const c of file.candidates) {
      if (!c.valid || !c.base) continue;
      const sum = Object.values(c.base.allocation_cents).reduce((a, b) => a + b, 0);
      expect(sum).toBe(file.scenario.purchase_cents);
      expect(c.base.allocation_cents["nvda-1"] ?? 0).toBe(0);
    }
  });
  it("rejects the wrong schema", () => {
    expect(() => parseLiquid({ schema: "x" })).toThrow();
  });
});
