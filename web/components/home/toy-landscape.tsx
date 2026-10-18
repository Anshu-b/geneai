"use client";

import { useEffect, useRef, useState } from "react";

interface Dot {
  x: number;
  y: number;
  fit: number;
}

function fitness(x: number, y: number): number {
  const peak = (cx: number, cy: number, spread: number) => Math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / spread);
  return peak(0.35, 0.5, 0.045) + 0.72 * peak(-0.45, -0.2, 0.08) + 0.5 * peak(-0.05, -0.6, 0.035);
}

function seedPopulation(count: number): Dot[] {
  return Array.from({ length: count }, (_, index) => {
    const x = ((index * 0.37) % 1.6) - 0.8;
    const y = ((index * 0.53) % 1.6) - 0.8;
    return { x, y, fit: fitness(x, y) };
  });
}

function nextGeneration(population: Dot[]): Dot[] {
  const ranked = [...population].sort((a, b) => b.fit - a.fit);
  const elites = ranked.slice(0, 4).map((dot) => ({ ...dot }));
  const children: Dot[] = [];
  while (elites.length + children.length < population.length) {
    const parent = elites[children.length % elites.length];
    const other = elites[(children.length + 1) % elites.length];
    const mix = children.length % 2 === 0;
    const x = (mix ? parent.x : (parent.x + other.x) / 2) + (Math.random() - 0.5) * 0.22;
    const y = (mix ? parent.y : (parent.y + other.y) / 2) + (Math.random() - 0.5) * 0.22;
    const clamped = { x: Math.max(-1, Math.min(1, x)), y: Math.max(-1, Math.min(1, y)), fit: 0 };
    clamped.fit = fitness(clamped.x, clamped.y);
    children.push(clamped);
  }
  return [...elites, ...children];
}

export function ToyLandscape() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [population, setPopulation] = useState<Dot[]>(() => seedPopulation(18));
  const [generation, setGeneration] = useState(0);
  const [playing, setPlaying] = useState(false);
  const best = population.reduce((top, dot) => (dot.fit > top.fit ? dot : top), population[0]);

  useEffect(() => {
    if (!playing) return;
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduced) return;
    const timer = window.setInterval(() => {
      setPopulation((current) => nextGeneration(current));
      setGeneration((value) => value + 1);
    }, 520);
    return () => window.clearInterval(timer);
  }, [playing]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 1.75);
    canvas.width = Math.floor(rect.width * dpr);
    canvas.height = Math.floor(rect.height * dpr);
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    const width = rect.width;
    const height = rect.height;
    const style = getComputedStyle(document.documentElement);
    const light = document.documentElement.dataset.theme === "light";
    context.clearRect(0, 0, width, height);
    const image = context.createImageData(Math.floor(width), Math.floor(height));
    const step = 4;
    for (let y = 0; y < height; y += step) {
      for (let x = 0; x < width; x += step) {
        const fx = (x / width) * 2 - 1;
        const fy = 1 - (y / height) * 2;
        const value = fitness(fx, fy);
        const tone = Math.max(0, Math.min(1, value / 1.35));
        const r = light ? 255 - tone * 40 : 8 + tone * 40;
        const g = light ? 250 - tone * 70 : 10 + tone * 90;
        const b = light ? 240 - tone * 80 : 9 + tone * 50;
        for (let oy = 0; oy < step; oy += 1) {
          for (let ox = 0; ox < step; ox += 1) {
            const px = x + ox;
            const py = y + oy;
            if (px >= width || py >= height) continue;
            const index = (py * Math.floor(width) + px) * 4;
            image.data[index] = r;
            image.data[index + 1] = g;
            image.data[index + 2] = b;
            image.data[index + 3] = 255;
          }
        }
      }
    }
    const off = document.createElement("canvas");
    off.width = Math.floor(width);
    off.height = Math.floor(height);
    off.getContext("2d")?.putImageData(image, 0, 0);
    context.drawImage(off, 0, 0);
    const ink = style.getPropertyValue("--ivory").trim() || "#f3efe4";
    const gold = style.getPropertyValue("--champ").trim() || "#e4c27a";
    for (const dot of population) {
      const x = ((dot.x + 1) / 2) * width;
      const y = ((1 - dot.y) / 2) * height;
      const elite = dot === best;
      context.beginPath();
      context.fillStyle = elite ? gold : ink;
      context.arc(x, y, elite ? 6 : 3.5, 0, Math.PI * 2);
      context.fill();
    }
  }, [population, best]);

  function stepOnce() {
    setPlaying(false);
    setPopulation((current) => nextGeneration(current));
    setGeneration((value) => value + 1);
  }

  return (
    <div>
      <div className="overflow-hidden rounded-2xl border border-line">
        <canvas ref={canvasRef} data-testid="toy-landscape" className="h-72 w-full md:h-80" aria-label="Toy population on a fitness landscape" />
      </div>
      <div className="mt-3 flex flex-wrap items-center gap-2">
        <button type="button" className="lift rounded-full border border-line px-3 py-1.5 text-sm" onClick={() => setPlaying((value) => !value)} aria-pressed={playing}>
          {playing ? "Pause" : "Play"}
        </button>
        <button type="button" className="lift rounded-full border border-line px-3 py-1.5 text-sm" data-testid="toy-step" onClick={stepOnce}>
          Step
        </button>
        <button
          type="button"
          className="lift rounded-full border border-line px-3 py-1.5 text-sm"
          onClick={() => {
            setPlaying(false);
            setPopulation(seedPopulation(18));
            setGeneration(0);
          }}
        >
          Reset
        </button>
        <p className="font-mono text-[11px] text-faint">generation {generation} · best {best.fit.toFixed(2)}</p>
      </div>
      <p className="mt-2 text-sm leading-relaxed text-dim">
        A toy in this browser. Dots climb a fake landscape by keeping a few elites and nudging the rest. It does not call a model, and it is not the GeneAI engine.
      </p>
    </div>
  );
}
