import { describe, expect, it } from "vitest";
import { MAX_GENERATION, PARTICLES } from "../components/home/living-field";

describe("hero field", () => {
  it("keeps the hero calm and the generation counter bounded", () => {
    expect(PARTICLES).toBeGreaterThanOrEqual(150);
    expect(PARTICLES).toBeLessThanOrEqual(200);
    expect(MAX_GENERATION).toBe(60);
  });
});
