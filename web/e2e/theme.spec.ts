import { expect, test } from "@playwright/test";

test.use({ colorScheme: "light" });

test("light theme uses floral white and the toggle persists", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  await expect(page.locator(".theme-icon-moon")).toBeVisible();
  await expect(page.locator(".theme-icon-sun")).toBeHidden();

  const background = await page.evaluate(() => getComputedStyle(document.body).backgroundColor);
  expect(background).toBe("rgb(255, 250, 240)");

  const tokens = await page.evaluate(() => {
    const style = getComputedStyle(document.documentElement);
    return {
      elite: style.getPropertyValue("--elite-ring").trim().toLowerCase(),
      mock: style.getPropertyValue("--prov-mock").trim().toLowerCase(),
      raise: style.getPropertyValue("--raise").trim().toLowerCase(),
      ivory: style.getPropertyValue("--ivory").trim().toLowerCase(),
    };
  });
  expect(tokens).toEqual({
    elite: "#1a1814",
    mock: "#0e7a62",
    raise: "#f7f1e3",
    ivory: "#1a1814",
  });

  const toggle = page.getByTestId("theme-toggle");
  const badge = page.getByTestId("mode-badge");
  await expect(badge).toBeVisible();
  await expect.poll(async () => {
    const toggleBox = await toggle.boundingBox();
    const badgeBox = await badge.boundingBox();
    if (!toggleBox || !badgeBox) return -1;
    return badgeBox.x - (toggleBox.x + toggleBox.width);
  }).toBeGreaterThanOrEqual(0);
  await expect.poll(async () => {
    const toggleBox = await toggle.boundingBox();
    const badgeBox = await badge.boundingBox();
    if (!toggleBox || !badgeBox) return 999;
    return Math.abs(toggleBox.y - badgeBox.y);
  }).toBeLessThan(20);

  await toggle.click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await expect(page.locator(".theme-icon-sun")).toBeVisible();
  await expect.poll(() => page.evaluate(() => localStorage.getItem("geneai-theme"))).toBe("dark");
  await page.reload();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");

  await toggle.click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  await page.reload();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  expect(await page.evaluate(() => localStorage.getItem("geneai-theme"))).toBe("light");
  expect(await page.evaluate(() => getComputedStyle(document.body).backgroundColor)).toBe("rgb(255, 250, 240)");
});
