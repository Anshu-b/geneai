import { expect, test } from "@playwright/test";

test("cloud lab renders a recorded lineage without console errors", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  await page.goto("/cloud");
  await expect(page.getByRole("heading", { name: "Intelligence that evolves." })).toBeVisible();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await expect(page.getByRole("button", { name: "Toggle color theme" })).toBeVisible();
  await expect(page.getByTestId("mode-badge")).toHaveText(/Recorded/);
  await expect(page.getByTestId("output-showcase")).toBeVisible();
  await expect(page.getByTestId("showcase-sample")).toBeVisible();
  await expect(page.getByTestId("showcase-verdict")).toContainText("Across all");
  await expect(page.getByTestId("showcase-example")).toContainText("missed 3");
  await expect(page.getByTestId("behavior-origin")).toContainText("Found by CREAO crossover g0051");
  await expect(page.getByTestId("schedule-gantt")).toBeVisible();
  await expect(page.getByTestId("lineage-svg")).toBeVisible({ timeout: 30_000 });
  const champion = page.locator('[data-genome="g0060"]');
  await champion.scrollIntoViewIfNeeded();
  await expect(champion).toBeVisible();
  await page.getByRole("tab", { name: "Results" }).click();
  await expect(page.getByTestId("results-panel")).toContainText("single attempt vs evolutionary search");
  await expect(page.getByTestId("results-panel")).toContainText("Not compute-matched");
  await page.getByRole("tab", { name: "Provenance" }).click();
  await page.locator('[data-genome="g0060"]').click();
  await expect(page.getByTestId("provenance-panel")).toContainText("g0060");
  expect(errors.filter((line) => !line.includes("favicon"))).toEqual([]);
});

test("shopping shows fixture badges and no product links", async ({ page }) => {
  await page.goto("/shopping");
  await expect(page.getByTestId("lineage-svg")).toBeVisible({ timeout: 30_000 });
  await page.getByRole("tab", { name: "Outfit" }).click();
  await expect(page.getByTestId("output-showcase")).toBeVisible();
  await expect(page.getByTestId("outfit-card").first()).toBeVisible();
  await expect(page.getByText("over the $220 budget")).toBeVisible();
  await expect(page.getByTestId("fixture-badge").first()).toBeVisible();
  await expect(page.getByRole("link", { name: /product/i })).toHaveCount(0);
});

test("quant is a placeholder and ask stays disabled without an API", async ({ page }) => {
  await page.goto("/quant");
  await expect(page.getByTestId("quant-placeholder")).toHaveText("Coming soon");
  await page.goto("/ask");
  await expect(page.getByTestId("ask-composer")).toBeVisible();
  await expect(page.getByRole("button", { name: "Launch micro run" })).toBeDisabled();
  await expect(page.getByRole("button", { name: "Toggle color theme" })).toBeVisible();
});
