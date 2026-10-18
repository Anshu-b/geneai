"use client";

import { applyTheme } from "@/lib/theme";

export function ThemeToggle() {
  return (
    <button
      type="button"
      data-testid="theme-toggle"
      aria-label="Toggle color theme"
      className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-line text-ivory"
      onClick={() => {
        const next = document.documentElement.dataset.theme === "light" ? "dark" : "light";
        applyTheme(next);
      }}
    >
      <svg className="theme-icon theme-icon-sun" viewBox="0 0 24 24" aria-hidden="true">
        <circle cx="12" cy="12" r="4" fill="none" stroke="currentColor" strokeWidth="1.6" />
        <path
          d="M12 2.5v2.2M12 19.3v2.2M2.5 12h2.2M19.3 12h2.2M4.8 4.8l1.6 1.6M17.6 17.6l1.6 1.6M19.2 4.8l-1.6 1.6M6.4 17.6l-1.6 1.6"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.6"
          strokeLinecap="round"
        />
      </svg>
      <svg className="theme-icon theme-icon-moon" viewBox="0 0 24 24" aria-hidden="true">
        <path
          d="M16.5 13.2A6.2 6.2 0 0 1 10.7 5.5 6.8 6.8 0 1 0 16.5 13.2Z"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.6"
          strokeLinejoin="round"
        />
      </svg>
    </button>
  );
}
