/**
 * Plain-English text generated deterministically from showcase data.
 * Nothing here is invented: every phrase comes from a rule, weight, item, or metric in the file.
 */
import type { CloudPolicy, OutfitCard, ShopperProfile } from "./showcase";

interface Pref {
  phrase: string;
  weight: number;
}

const RULE = /^(Prefer|Avoid) (.+?) \(weight (-?[\d.]+)\)\.?$/;

function jobPhrase(target: string, positive: boolean): string | null {
  const t = target.toLowerCase();
  if (t.includes("hard deadline")) return positive ? "hard-deadline jobs" : "deferring hard-deadline jobs";
  if (t.includes("close to their deadline")) return positive ? "jobs close to their deadline" : null;
  if (t === "short jobs") return positive ? "short jobs" : "long jobs";
  if (t.includes("already waited")) return positive ? "jobs that have waited longest" : null;
  return null;
}

function machinePhrase(target: string, positive: boolean): string | null {
  const t = target.toLowerCase();
  if (t.includes("machine class that matches")) return positive ? "sends each job to the machine type it fits" : "ignores machine type";
  if (t.includes("busy cpus")) return positive ? "packs already-busy CPUs a little tighter" : "avoids piling onto busy CPUs";
  if (t.includes("busy memory")) return positive ? "packs already-busy memory" : "avoids machines whose memory is already busy";
  if (t.includes("lower-power")) return positive ? "favors lower-power machines" : "favors faster, higher-power machines";
  if (t.includes("free cpu")) return positive ? "favors machines with free CPU" : null;
  if (t.includes("next machine in rotation")) return positive ? "rotates through machines in turn" : "avoids simple round-robin rotation";
  return null;
}

function list(items: string[]): string {
  if (items.length <= 1) return items.join("");
  return `${items.slice(0, -1).join(", ")} and ${items[items.length - 1]}`;
}

function nameOf(label: string): string {
  if (/gpt/i.test(label)) return "GPT-5.5";
  if (/geneai/i.test(label)) return "GeneAI";
  return label;
}

/** e.g. "GeneAI looks at the next 24 waiting jobs and runs hard-deadline and short jobs first. It sends each job to the machine type it fits and packs already-busy CPUs a little tighter." */
export function cloudPolicySummary(policy: CloudPolicy): string | null {
  const rules = policy.rules_plain ?? [];
  if (rules.length === 0) return null;
  const who = nameOf(policy.label);
  const window = rules.map((rule) => /first (\d+) waiting jobs/i.exec(rule)?.[1]).find(Boolean);
  const prefs: Pref[] = [];
  for (const rule of rules) {
    const match = RULE.exec(rule.trim());
    if (!match) continue;
    const weight = Number(match[3]);
    if (!Number.isFinite(weight)) continue;
    prefs.push({ phrase: match[2], weight });
  }
  prefs.sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight));
  const jobs = prefs.map((pref) => jobPhrase(pref.phrase, pref.weight > 0)).filter((value): value is string => Boolean(value)).slice(0, 3);
  const machines = prefs.map((pref) => machinePhrase(pref.phrase, pref.weight > 0)).filter((value): value is string => Boolean(value)).slice(0, 3);
  const first = `${who} looks at ${window ? `the next ${window} waiting jobs` : "the waiting jobs"}${jobs.length ? ` and runs ${list(jobs)} first` : ""}.`;
  const second = machines.length ? ` It ${list(machines)}.` : "";
  const extra = rules.filter((rule) => /^For \w+ jobs/i.test(rule)).length;
  const third = extra ? ` It also has ${extra} special case${extra === 1 ? "" : "s"} for GPU or high-memory jobs.` : "";
  return `${first}${second}${third}`;
}

function money(value: number): string {
  return `$${Math.round(value)}`;
}

/** e.g. "GeneAI builds a 3-piece look: Repper Short-Sleeve Shirt (top), … for $254 of the $340 budget, leaving $86." */
export function outfitSummary(outfit: OutfitCard, budget: number): string {
  const who = nameOf(outfit.label);
  const pieces = outfit.items.map((item) => `${item.name}${item.role ? ` (${item.role})` : ""}`);
  const count = outfit.items.length;
  const left = budget - outfit.total_usd;
  if (count === 0) return `${who} did not return a usable outfit.`;
  const head = `${who} builds a ${count}-piece outfit: ${list(pieces)}.`;
  if (!outfit.feasible && left < 0) return `${head} It costs ${money(outfit.total_usd)}, ${money(-left)} over the ${money(budget)} budget, so the outfit is rejected.`;
  if (!outfit.feasible) return `${head} It breaks a rule, so it is rejected${outfit.infeasible_reason ? `: ${outfit.infeasible_reason}` : "."}`;
  return `${head} It costs ${money(outfit.total_usd)} of the ${money(budget)} budget, leaving ${money(left)}.`;
}

const STYLE_WANTED: Record<string, string[]> = { bright: ["bright", "athletic"], neutral: ["neutral", "relaxed"] };

/** e.g. "Shopper s11 wants a bright, athletic outfit for dinner and the gym…" */
export function shopperRequest(shopper: ShopperProfile): string {
  const style = shopper.style ? (STYLE_WANTED[shopper.style] ?? [shopper.style]).join(", ") : "";
  const occasions = (shopper.occasions ?? []).map((occasion) => (occasion === "gym" ? "the gym" : occasion));
  const sizes = [shopper.apparel_size ? `apparel size ${shopper.apparel_size}` : "", shopper.shoe_size != null ? `shoe size ${shopper.shoe_size}` : ""].filter(Boolean);
  return `Shopper ${shopper.id} wants a ${style ? `${style} ` : ""}outfit${occasions.length ? ` for ${list(occasions)}` : ""}: one top, one bottom and one pair of shoes${sizes.length ? ` in ${list(sizes)}` : ""}, all under $${shopper.budget_usd}.`;
}

export type Mark = "yes" | "partial" | "no";

export interface PlainLine {
  label: string;
  mark: Mark;
  text: string;
}

function hitMark(hits: number, count: number): Mark {
  if (hits >= count && count > 0) return "yes";
  return hits > 0 ? "partial" : "no";
}

/** Rewrites the engine's score bullets (outfit_why) into short labelled lines. Unknown lines pass through. */
export function plainWhy(lines: string[]): PlainLine[] {
  const out: PlainLine[] = [];
  for (const raw of lines) {
    const line = raw.trim();
    let m = /^Style match [\d.]+: (\d+)\/(\d+) items share (\w+) tags \[(.*)\]/.exec(line);
    if (m) {
      const tags = m[4].replace(/['"]/g, "").split(",").map((tag) => tag.trim()).filter(Boolean);
      out.push({ label: "Style", mark: hitMark(+m[1], +m[2]), text: `${m[1]} of ${m[2]} items are tagged ${tags.join(" or ")}.` });
      continue;
    }
    m = /^Occasion fit [\d.]+: (\d+)\/(\d+) items match \[(.*)\]/.exec(line);
    if (m) {
      const occ = m[3].replace(/['"]/g, "").split(",").map((tag) => tag.trim()).filter(Boolean);
      out.push({ label: "Occasion", mark: hitMark(+m[1], +m[2]), text: `${m[1]} of ${m[2]} items suit ${occ.join(" or ")}.` });
      continue;
    }
    m = /^(Bright|Neutral) colors: (\d+)\/(\d+) items/.exec(line);
    if (m) {
      out.push({ label: "Colors", mark: hitMark(+m[2], +m[3]), text: `${m[2]} of ${m[3]} items are ${m[1].toLowerCase()} colors.` });
      continue;
    }
    m = /^Availability confidence ([\d.]+)/.exec(line);
    if (m) {
      const value = Number(m[1]);
      out.push({ label: "In stock", mark: value >= 0.999 ? "yes" : value > 0 ? "partial" : "no", text: value >= 0.999 ? "Every item was in stock with a live product link when recorded." : `${Math.round(value * 100)}% of items were verified in stock.` });
      continue;
    }
    m = /^(Under|Over) budget by \$([\d.]+) \(spend \$([\d.]+) of \$([\d.]+)/.exec(line);
    if (m) {
      const under = m[1] === "Under";
      out.push({ label: "Budget", mark: under ? "yes" : "no", text: under ? `$${Math.round(+m[3])} of $${Math.round(+m[4])}, with $${Math.round(+m[2])} left over.` : `$${Math.round(+m[3])} is $${Math.round(+m[2])} over the $${Math.round(+m[4])} budget.` });
      continue;
    }
    m = /^over budget ([\d.]+) > ([\d.]+)/i.exec(line);
    if (m) {
      out.push({ label: "Budget", mark: "no", text: `$${Math.round(+m[1])} is over the $${Math.round(+m[2])} budget, so the outfit is rejected.` });
      continue;
    }
    out.push({ label: "Note", mark: "no", text: line });
  }
  return out;
}

export const SHOPPING_WEIGHTS =
  "The score runs from 0 to 1 (higher fits the shopper better): 30% style, 25% occasion, 20% colors, 15% in stock, 10% money left over. An outfit over budget or missing a top, bottom or shoes scores nothing.";
