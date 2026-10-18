import { describe, expect, it } from "vitest";
import { rankColumn } from "../lib/rank";

describe("rankColumn", () => {
  it("marks every tie as best and the worst only with three distinct values", () => {
    expect(rankColumn([308.5, 311.4, 310.3, 308.5], "lower")).toEqual(["best", "worst", "mid", "best"]);
    expect(rankColumn([5, 5, 0, 5], "higher")).toEqual(["best", "best", "mid", "best"]);
    expect(rankColumn([1, null, 2], "lower")).toEqual(["best", "none", "mid"]);
    expect(rankColumn([0.57, 0, 2.6, 0], "lower")).toEqual(["mid", "best", "worst", "best"]);
  });
});
