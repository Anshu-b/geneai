export const THEME_KEY = "geneai-theme";

export type ThemeName = "light" | "dark";

/** Stored choice wins. Otherwise follow the OS, and stay dark when it does not ask for light. */
export function resolveInitialTheme(stored: string | null | undefined, prefersLight: boolean): ThemeName {
  if (stored === "light" || stored === "dark") return stored;
  return prefersLight ? "light" : "dark";
}

export function applyTheme(theme: ThemeName): void {
  document.documentElement.dataset.theme = theme;
  try {
    localStorage.setItem(THEME_KEY, theme);
  } catch {
    /* storage can be blocked; the attribute still updates this view */
  }
}

export interface ThemePaint {
  providers: Record<"veri" | "creao" | "baseline" | "mock" | "random" | "other", string>;
  edge: string;
  edgeOpacity: number;
  stage: string;
  elite: string;
  champ: string;
  pruned: number;
  light: boolean;
}

const DARK_PAINT: ThemePaint = {
  providers: {
    veri: "#8ecbff",
    creao: "#c3b2ff",
    baseline: "#e4c27a",
    mock: "#3fbea5",
    random: "#e7a3c0",
    other: "#d9d3c7",
  },
  edge: "#c9ddd6",
  edgeOpacity: 0.38,
  stage: "#070908",
  elite: "#f3efe4",
  champ: "#e4c27a",
  pruned: 0.28,
  light: false,
};

function readNumber(value: string, fallback: number): number {
  const parsed = Number.parseFloat(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

/** Resolved paint for WebGL, which cannot consume CSS variables directly. */
export function readThemePaint(): ThemePaint {
  if (typeof document === "undefined") return DARK_PAINT;
  const style = getComputedStyle(document.documentElement);
  const pick = (name: string, fallback: string) => style.getPropertyValue(name).trim() || fallback;
  return {
    providers: {
      veri: pick("--prov-veri", DARK_PAINT.providers.veri),
      creao: pick("--prov-creao", DARK_PAINT.providers.creao),
      baseline: pick("--prov-baseline", DARK_PAINT.providers.baseline),
      mock: pick("--prov-mock", DARK_PAINT.providers.mock),
      random: pick("--prov-random", DARK_PAINT.providers.random),
      other: pick("--prov-other", DARK_PAINT.providers.other),
    },
    edge: pick("--edge-3d", DARK_PAINT.edge),
    edgeOpacity: readNumber(pick("--edge-3d-opacity", String(DARK_PAINT.edgeOpacity)), DARK_PAINT.edgeOpacity),
    stage: pick("--stage-3d", DARK_PAINT.stage),
    elite: pick("--elite-ring", DARK_PAINT.elite),
    champ: pick("--champ", DARK_PAINT.champ),
    pruned: readNumber(pick("--pruned-opacity", String(DARK_PAINT.pruned)), DARK_PAINT.pruned),
    light: document.documentElement.dataset.theme === "light",
  };
}

/** Runs before hydration so the first paint matches the saved theme. */
export const THEME_BOOT = `(function(){try{var k=${JSON.stringify(THEME_KEY)};var s=localStorage.getItem(k);var light=window.matchMedia("(prefers-color-scheme: light)").matches;var t=(s==="light"||s==="dark")?s:(light?"light":"dark");document.documentElement.setAttribute("data-theme",t);}catch(e){document.documentElement.setAttribute("data-theme","dark");}})();`;
