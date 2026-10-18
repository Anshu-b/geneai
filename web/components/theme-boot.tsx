"use client";

import { useLayoutEffect } from "react";
import { resolveInitialTheme, THEME_KEY } from "@/lib/theme";

/** Re-apply after hydration without writing storage, so an OS preference is not locked in. */
export function ThemeBoot() {
  useLayoutEffect(() => {
    let stored: string | null = null;
    try {
      stored = localStorage.getItem(THEME_KEY);
    } catch {
      stored = null;
    }
    const theme = resolveInitialTheme(stored, window.matchMedia("(prefers-color-scheme: light)").matches);
    document.documentElement.dataset.theme = theme;
  }, []);
  return null;
}
