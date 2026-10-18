import { expect, test } from "@playwright/test";

test("cloud lab renders a recorded lineage without console errors", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  await page.goto("/cloud");
  await expect(page.getByTestId("objective")).toContainText("Schedule jobs onto a heterogeneous pool");
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await expect(page.getByRole("button", { name: "Toggle color theme" })).toBeVisible();
  await expect(page.getByTestId("mode-badge")).toHaveText(/Recorded/);
  await expect(page.getByTestId("output-showcase")).toBeVisible();
  await expect(page.getByTestId("showcase-sample")).toBeVisible();
  await expect(page.getByTestId("key-scores")).toContainText("Missed deadlines: 3 vs 6");
  await expect(page.getByText("Round robin")).toHaveCount(0);
  await expect(page.getByText("Deadline weighted")).toHaveCount(0);
  await expect(page.getByTestId("behavior-origin")).toContainText("Found by CREAO crossover g0051");
  await expect(page.getByTestId("schedule-gantt").first()).toBeVisible();
  await expect(page.getByTestId("fitness-generations")).toBeVisible();
  await expect(page.getByTestId("lineage-svg")).toBeVisible({ timeout: 30_000 });
  await expect(page.getByText("integrity")).toHaveCount(0);
  await expect(page.getByTestId("results-panel")).toHaveCount(0);
  const champion = page.locator('[data-genome="g0060"]');
  await champion.scrollIntoViewIfNeeded();
  await expect(champion).toBeVisible();
  expect(errors.filter((line) => !line.includes("favicon"))).toEqual([]);
});

test("shopping shows the agentic outfits and hides the manual baseline", async ({ page }) => {
  await page.goto("/shopping");
  await expect(page.getByTestId("objective")).toContainText("Build one complete outfit");
  await expect(page.getByTestId("lineage-svg")).toBeVisible({ timeout: 30_000 });
  await expect(page.getByTestId("output-showcase")).toBeVisible();
  await expect(page.getByTestId("outfit-card")).toHaveCount(2);
  await expect(page.getByText("over the $220 budget")).toBeVisible();
  await expect(page.getByText("Neutral outfit")).toHaveCount(0);
  await expect(page.getByTestId("key-scores")).toContainText("18 of 20");
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
