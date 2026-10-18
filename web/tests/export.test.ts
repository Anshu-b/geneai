import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { classifyGoal } from "../lib/ask";
import { isFixtureProduct, outfitItems } from "../lib/domain";
import { presentationMode } from "../lib/mode";
import { parseExport } from "../lib/parse";
import {
  ancestryIds,
  buildBeats,
  genomeMap,
  isHandwrittenBaseline,
  layoutRun,
  operatorLabel,
  projectRun,
} from "../lib/project";
import type { RunExport } from "../lib/types";

function load(name: string): RunExport {
  const raw = readFileSync(join(process.cwd(), "public/fixtures", name), "utf8");
  return parseExport(JSON.parse(raw));
}

describe("fixture contract", () => {
  it("parses both recorded engine exports", () => {
    const cloud = load("cloud_scheduling.json");
    const shop = load("shopping.json");
    expect(cloud.schema_version).toBe("1.0");
    expect(shop.schema_version).toBe("1.0");
    expect(cloud.run.mode).toBe("hybrid");
    expect(shop.run.mode).toBe("hybrid");
    expect(cloud.champion_id).toBeTruthy();
    expect(shop.champion_id).toBeTruthy();
    expect(cloud.integrity_sha256).toMatch(/^[0-9a-f]{64}$/);
    expect(shop.integrity_sha256).toMatch(/^[0-9a-f]{64}$/);
    expect(cloud.one_shot_baseline?.compute_matched).toBe(false);
    expect(cloud.one_shot_baseline?.comparison_label).toBe("single attempt vs evolutionary search");
    expect(cloud.events.some((event) => "task" in event.payload)).toBe(false);
    expect(shop.events.some((event) => "task" in event.payload)).toBe(false);
    const cloudProviders = new Set(cloud.genomes.map((genome) => genome.provider));
    expect(cloudProviders.has("veri") || cloudProviders.has("creao")).toBe(true);
  });

  it("does not invent shopping links for fixture products", () => {
    const shop = load("shopping.json");
    for (const genome of shop.genomes) {
      for (const item of outfitItems(genome.payload)) {
        if (isFixtureProduct(item)) {
          expect(item.product_url).toBeUndefined();
          expect(item.image_url).toBeUndefined();
        }
      }
    }
  });
});

describe("playback", () => {
  it("hides the held-out test until the champion is frozen and tested", () => {
    const cloud = load("cloud_scheduling.json");
    const champion = cloud.champion_id!;
    const beats = buildBeats(cloud.events);
    const settled = beats.filter((beat) => beat.phase === "settled");
    const completed = cloud.events.filter((event) => event.type === "generation.completed").map((event) => Number(event.payload.generation));
    expect(settled.map((beat) => beat.generation)).toEqual(completed);
    const last = settled[settled.length - 1];
    const mid = projectRun(cloud, last.sequence);
    expect(mid.testVisible).toBe(false);
    expect(mid.championFrozen).toBe(false);
    expect(mid.byId.get(champion)?.test ?? null).toBeNull();

    const frozen = beats.find((beat) => beat.phase === "frozen")!;
    const atFreeze = projectRun(cloud, frozen.sequence);
    expect(atFreeze.championFrozen).toBe(true);
    expect(atFreeze.testVisible).toBe(false);
    expect(atFreeze.byId.get(champion)?.status).toBe("champion");
    expect(atFreeze.byId.get(champion)?.validation?.fitness).toBeTypeOf("number");

    const tested = beats.find((beat) => beat.phase === "tested")!;
    const done = projectRun(cloud, tested.sequence);
    expect(done.testVisible).toBe(true);
    expect(done.byId.get(champion)?.test?.fitness).toBeTypeOf("number");
  });

  it("keeps parent edges and distinguishes baselines at generation 0", () => {
    const cloud = load("cloud_scheduling.json");
    const beats = buildBeats(cloud.events);
    const gen0 = beats.find((beat) => beat.phase === "settled" && beat.generation === 0)!;
    const view = projectRun(cloud, gen0.sequence);
    const baseline = cloud.genomes.find((genome) => isHandwrittenBaseline(genome))!;
    expect(view.byId.get(baseline.id)).toBeDefined();
    const layout = layoutRun(cloud);
    const child = cloud.genomes.find((genome) => genome.parent_ids.length > 0 && layout.nodes.has(genome.parent_ids[0]))!;
    expect(layout.nodes.get(child.id)!.x).toBeGreaterThan(layout.nodes.get(child.parent_ids[0])!.x);
    const ancestors = ancestryIds(cloud.champion_id, genomeMap(cloud));
    const champion = genomeMap(cloud).get(cloud.champion_id!)!;
    for (const parent of champion.parent_ids) expect(ancestors.has(parent)).toBe(true);
  });

  it("labels operators from the genome and badges recorded runs", () => {
    const shop = load("shopping.json");
    for (const genome of shop.genomes) expect(operatorLabel(genome)).not.toBe("unknown");
    expect(presentationMode(shop.run.mode, false).badge).toBe("Recorded");
    expect(presentationMode(shop.run.mode, false).runMode).toBe("hybrid");
    expect(presentationMode("mock", false).badge).toBe("Mock");
    expect(presentationMode("live", false).badge).toBe("Recorded");
    expect(presentationMode("fixture", false).badge).toBe("Fixture");
    expect(presentationMode("mock", true).badge).toBe("Live");
  });
});

describe("ask", () => {
  it("refuses to invent a quant run", () => {
    expect(classifyGoal("Explore historical trading strategies").supported).toBe(false);
    expect(classifyGoal("Reduce cloud queueing time").domain).toBe("cloud_scheduling");
    expect(classifyGoal("Build an outfit under $250").domain).toBe("shopping");
    expect(classifyGoal("Write a poem about moss").supported).toBe(false);
  });
});
