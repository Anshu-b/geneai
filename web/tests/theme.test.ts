import { describe, expect, it } from "vitest";
import { resolveInitialTheme } from "../lib/theme";

describe("resolveInitialTheme", () => {
  it("stays dark when nothing is stored and the OS does not ask for light", () => {
    expect(resolveInitialTheme(null, false)).toBe("dark");
    expect(resolveInitialTheme(undefined, false)).toBe("dark");
    expect(resolveInitialTheme("", false)).toBe("dark");
    expect(resolveInitialTheme("system", false)).toBe("dark");
  });

  it("follows prefers-color-scheme on the first visit", () => {
    expect(resolveInitialTheme(null, true)).toBe("light");
    expect(resolveInitialTheme(undefined, true)).toBe("light");
  });

  it("lets a stored choice beat the OS preference", () => {
    expect(resolveInitialTheme("light", false)).toBe("light");
    expect(resolveInitialTheme("dark", true)).toBe("dark");
  });
});
