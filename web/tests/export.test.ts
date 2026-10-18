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
  it("parses both engine exports", () => {
    const cloud = load("cloud_scheduling.json");
    const shop = load("shopping.json");
    expect(cloud.schema_version).toBe("1.0");
    expect(shop.schema_version).toBe("1.0");
    expect(cloud.run.mode).toBe("mock");
    expect(shop.run.mode).toBe("mock");
    expect(cloud.champion_id).toBe("g0206");
    expect(shop.champion_id).toBe("g0236");
    expect(cloud.integrity_sha256).toBe("88d82a9766baefc1ac6c9ecba638a7ff50095562f9c23cf3f457163b6ac89e51");
    expect(shop.integrity_sha256).toBe("21970cbd3eb7f92964f0753b487a07c21cbcf6d367c52ad46081a467e81eeacd");
    expect(cloud.one_shot_baseline?.status).toBe("skipped");
    expect(cloud.one_shot_baseline?.compute_matched).toBe(false);
    expect(cloud.one_shot_baseline?.comparison_label).toBe("single attempt vs evolutionary search");
    expect(cloud.events.some((event) => "task" in event.payload)).toBe(false);
    expect(shop.events.some((event) => "task" in event.payload)).toBe(false);
  });

  it("does not invent shopping links", () => {
    const shop = load("shopping.json");
    for (const genome of shop.genomes) {
      for (const item of outfitItems(genome.payload)) {
        expect(item.product_url).toBeUndefined();
        expect(item.image_url).toBeUndefined();
        expect(isFixtureProduct(item)).toBe(true);
      }
    }
  });
});

describe("playback", () => {
  it("hides the held-out test until the champion is frozen and tested", () => {
    const cloud = load("cloud_scheduling.json");
    const beats = buildBeats(cloud.events);
    const settled = beats.filter((beat) => beat.phase === "settled");
    expect(settled.map((beat) => beat.generation)).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    const gen12 = settled.find((beat) => beat.generation === 12)!;
    const mid = projectRun(cloud, gen12.sequence);
    expect(mid.testVisible).toBe(false);
    expect(mid.championFrozen).toBe(false);
    expect(mid.byId.get("g0206")?.test).toBeNull();
    expect(mid.byId.get("g0206")?.validation).toBeNull();
    expect(mid.byId.get("g0301")?.genome.generation).toBe(12);

    const frozen = beats.find((beat) => beat.phase === "frozen")!;
    const atFreeze = projectRun(cloud, frozen.sequence);
    expect(atFreeze.championFrozen).toBe(true);
    expect(atFreeze.testVisible).toBe(false);
    expect(atFreeze.byId.get("g0206")?.status).toBe("champion");
    expect(atFreeze.byId.get("g0206")?.validation?.fitness).toBeCloseTo(-1.579686522088974, 6);

    const tested = beats.find((beat) => beat.phase === "tested")!;
    const done = projectRun(cloud, tested.sequence);
    expect(done.testVisible).toBe(true);
    expect(done.byId.get("g0206")?.test?.fitness).toBeCloseTo(-1.571329317083924, 6);
    expect(done.byId.get("g0206")?.laterPruned).toBe(true);
    expect(done.byId.get("g0001")?.status).toBe("pruned");
    expect(done.populationIds).toContain("g0267");
    expect(done.populationIds).not.toContain("g0206");
  });

  it("keeps parent edges and distinguishes elites at generation 0", () => {
    const cloud = load("cloud_scheduling.json");
    const beats = buildBeats(cloud.events);
    const gen0 = beats.find((beat) => beat.phase === "settled" && beat.generation === 0)!;
    const view = projectRun(cloud, gen0.sequence);
    expect(view.byId.has("g0206")).toBe(false);
    expect(view.byId.get("g0001")?.status).toBe("kept");
    expect(isHandwrittenBaseline(view.byId.get("g0001")!.genome)).toBe(true);
    const layout = layoutRun(cloud);
    const child = layout.nodes.get("g0206")!;
    const parent = layout.nodes.get("g0177")!;
    expect(child.x).toBeGreaterThan(parent.x);
    const ancestors = ancestryIds("g0206", genomeMap(cloud));
    expect(ancestors.has("g0013")).toBe(true);
    expect(ancestors.has("g0087")).toBe(true);
  });

  it("labels shopping repair and immigrant operators from the genome", () => {
    const shop = load("shopping.json");
    const repair = shop.genomes.find((genome) => genome.id === "g0046")!;
    const immigrant = shop.genomes.find((genome) => genome.metadata.provider_role === "immigrant")!;
    expect(operatorLabel(repair)).toBe("repair");
    expect(operatorLabel(immigrant)).toBe("immigrant");
    const ancestors = ancestryIds(shop.champion_id, genomeMap(shop));
    expect(ancestors.has("g0026")).toBe(true);
    expect(ancestors.has("g0045")).toBe(true);
    expect(presentationMode(shop.run.mode, false).badge).toBe("Mock");
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
