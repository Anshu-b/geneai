import { expect, test, type Page } from "@playwright/test";

async function contrastRatio(page: Page, testId: string) {
  return page.getByTestId(testId).evaluate((element) => {
    const style = getComputedStyle(element);
    const parse = (value: string) => {
      const match = value.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/);
      if (!match) return null;
      return [Number(match[1]), Number(match[2]), Number(match[3])] as const;
    };
    const channel = (c: number) => {
      const s = c / 255;
      return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
    };
    const lum = (rgb: readonly [number, number, number]) => 0.2126 * channel(rgb[0]) + 0.7152 * channel(rgb[1]) + 0.0722 * channel(rgb[2]);
    const fg = parse(style.color);
    const bg = parse(style.backgroundColor);
    if (!fg || !bg) return 0;
    const lighter = Math.max(lum(fg), lum(bg));
    const darker = Math.min(lum(fg), lum(bg));
    return (lighter + 0.05) / (darker + 0.05);
  });
}

test("home explains the engine and does not mount the lineage", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: /One loop/ })).toBeVisible();
  await expect(page.getByTestId("living-field")).toBeVisible();
  await expect(page.getByTestId("lineage-svg")).toHaveCount(0);
  const natural = page.getByRole("heading", { name: "Natural selection" });
  await natural.scrollIntoViewIfNeeded();
  await expect(natural).toBeVisible();
  await page.getByTestId("selection-step").click();
  await expect(page.getByTestId("selection-phase")).toHaveText("Selection");
  const algorithms = page.getByRole("heading", { name: "Genetic algorithms" });
  await algorithms.scrollIntoViewIfNeeded();
  await expect(algorithms).toBeVisible();
  const agentic = page.getByRole("heading", { name: "Same steps, every agentic generation." });
  await agentic.scrollIntoViewIfNeeded();
  await expect(agentic).toBeVisible();
  await page.getByTestId("loop-repair").click();
  await expect(page.getByTestId("stage-copy")).toContainText("one repair attempt");
  await page.getByTestId("toy-step").click();
  await expect(page.getByTestId("toy-landscape")).toBeVisible();
  await expect(page.getByRole("link", { name: "Cloud scheduling" })).toBeVisible();
  const cta = page.getByTestId("hero-cta");
  await expect(cta).toHaveAccessibleName("Open the cloud run");
  await expect(cta).toBeVisible();
  expect(await contrastRatio(page, "hero-cta")).toBeGreaterThanOrEqual(4.5);
});

test("tsp showdown races Held-Karp against the other three methods", async ({ page }) => {
  await page.goto("/");
  const heading = page.getByRole("heading", { name: "Traveling salesman, 15 cities" });
  await heading.scrollIntoViewIfNeeded();
  await expect(heading).toBeVisible();
  await expect(page.getByTestId("tsp-sample")).toHaveCount(0);
  await expect(page.getByTestId("tsp-map-held_karp")).toBeVisible();
  await expect(page.getByTestId("tsp-map-classic_ga")).toBeVisible();
  await expect(page.getByTestId("tsp-map-llm_one_shot")).toBeVisible();
  await expect(page.getByTestId("tsp-map-geneai")).toBeVisible();
  await expect(page.getByTestId("tsp-map-heuristic")).toHaveCount(0);
  await expect(page.getByTestId("tsp-ghost-classic_ga")).toBeVisible();
  await expect(page.getByTestId("tsp-ghost-geneai")).toBeVisible();
  await expect(page.getByTestId("tsp-crown-held_karp")).toBeVisible();
  await expect(page.getByTestId("tsp-invalid-reason")).toContainText("empty output");
  await expect(page.getByTestId("tsp-heldout")).toContainText("5 of 5");
  await page.getByTestId("tsp-race").click();
  await expect(page.getByTestId("tsp-race")).toHaveAttribute("aria-pressed", "true");
  await page.getByTestId("tsp-log").click();
  await expect(page.getByTestId("tsp-log")).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByTestId("tsp-chart")).toContainText("log scale");
});

test.describe("light hero", () => {
  test.use({ colorScheme: "light" });

  test("primary CTA stays readable on floral white", async ({ page }) => {
    await page.goto("/");
    const cta = page.getByTestId("hero-cta");
    await expect(cta).toHaveAccessibleName("Open the cloud run");
    await expect(cta).toBeVisible();
    expect(await contrastRatio(page, "hero-cta")).toBeGreaterThanOrEqual(4.5);
    const colors = await cta.evaluate((element) => {
      const style = getComputedStyle(element);
      return { color: style.color, background: style.backgroundColor };
    });
    expect(colors.background).toBe("rgb(26, 24, 20)");
    expect(colors.color).toBe("rgb(255, 250, 240)");
  });
});
