import { expect, test } from "@playwright/test";

test("home explains the engine and does not mount the lineage", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: /One loop/ })).toBeVisible();
  await expect(page.getByTestId("living-field")).toBeVisible();
  await expect(page.getByTestId("lineage-svg")).toHaveCount(0);
  await page.getByTestId("loop-repair").click();
  await expect(page.getByTestId("stage-copy")).toContainText("one repair attempt");
  await page.getByTestId("toy-step").click();
  await expect(page.getByTestId("toy-landscape")).toBeVisible();
  await expect(page.getByRole("link", { name: "Cloud scheduling" })).toBeVisible();
});
