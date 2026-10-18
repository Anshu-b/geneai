"use client";

import { useEffect, useRef, useState } from "react";

type Kind = 0 | 1 | 2;

interface Particle {
  x: number;
  y: number;
  z: number;
  vx: number;
  vy: number;
  r: number;
  kind: Kind;
  life: number;
  state: 0 | 1 | 2;
  parent: number;
}

interface Hud {
  generation: number;
  fitness: number;
  history: number[];
}

/** Calm hero: a few hundred dots at most, and a run that ends. */
export const PARTICLES = 180;
export const MAX_GENERATION = 60;

function spawn(count: number): Particle[] {
  return Array.from({ length: count }, (_, index) => {
    const right = index % 5 !== 0;
    const x = right ? 0.38 + Math.random() * 0.6 : Math.random() * 0.42;
    const roll = Math.random();
    const kind: Kind = roll < 0.68 ? 0 : roll < 0.86 ? 1 : 2;
    return {
      x,
      y: Math.random(),
      z: Math.random(),
      vx: (Math.random() - 0.5) * 0.00015,
      vy: (Math.random() - 0.5) * 0.00015,
      r: 0.8 + Math.random() * 2.2,
      kind,
      life: 0.55 + Math.random() * 0.45,
      state: index % 23 === 0 ? 1 : 0,
      parent: -1,
    };
  });
}

export function LivingField() {
  const ref = useRef<HTMLCanvasElement>(null);
  const [hud, setHud] = useState<Hud>({ generation: 1, fitness: 0.42, history: [0.34, 0.38, 0.41, 0.42] });

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const context = canvas.getContext("2d", { alpha: true });
    if (!context) return;

    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const pointer = { x: 0.68, y: 0.46, on: false };
    let width = 1;
    let height = 1;
    let dpr = 1;
    let particles = spawn(PARTICLES);
    let generation = 1;
    let frame = 0;
    let visible = true;
    let scroll = 0;
    let peakX = 0.72;
    let peakY = 0.4;
    const history = [0.34, 0.37, 0.4, 0.42];
    const palette = { veri: "#8ecbff", creao: "#c3b2ff", baseline: "#e4c27a", ember: "#e08a78", light: false };

    const readPalette = () => {
      const style = getComputedStyle(document.documentElement);
      const pick = (name: string, fallback: string) => style.getPropertyValue(name).trim() || fallback;
      palette.veri = pick("--prov-veri", "#8ecbff");
      palette.creao = pick("--prov-creao", "#c3b2ff");
      palette.baseline = pick("--prov-baseline", "#e4c27a");
      palette.ember = pick("--danger", "#e08a78");
      palette.light = document.documentElement.dataset.theme === "light";
    };

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      width = Math.max(1, rect.width);
      height = Math.max(1, rect.height);
      dpr = Math.min(window.devicePixelRatio || 1, 1.25);
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      context.setTransform(dpr, 0, 0, dpr, 0, 0);
      const next = Math.min(PARTICLES, Math.max(120, Math.floor((width * height) / 6500)));
      if (Math.abs(next - particles.length) > 20) particles = spawn(next);
    };

    const colorOf = (kind: Kind) => (kind === 0 ? palette.veri : kind === 1 ? palette.creao : palette.baseline);

    const breed = () => {
      const count = particles.length;
      peakX = 0.55 + 0.35 * (0.5 + 0.5 * Math.sin(generation * 0.7));
      peakY = 0.35 + 0.4 * (0.5 + 0.5 * Math.cos(generation * 0.45));
      const fitness = new Float32Array(count);
      const order = particles.map((_, index) => index);
      for (let index = 0; index < count; index += 1) {
        const particle = particles[index];
        const dist = Math.hypot(particle.x - peakX, particle.y - peakY);
        fitness[index] = (1 - Math.min(1, dist / 0.85)) * (0.65 + particle.z * 0.35) * particle.life;
      }
      order.sort((a, b) => fitness[b] - fitness[a]);
      const eliteCut = Math.max(8, Math.floor(count * 0.035));
      const pruneFrom = Math.floor(count * 0.78);
      for (let rank = 0; rank < count; rank += 1) {
        const particle = particles[order[rank]];
        if (rank < eliteCut) {
          particle.state = 1;
          particle.life = 1;
        } else if (rank >= pruneFrom && particle.state !== 2) {
          particle.state = 2;
        }
      }
      const births = Math.floor(count * 0.07);
      for (let born = 0; born < births; born += 1) {
        const slot = particles[order[count - 1 - (born % Math.max(1, count - pruneFrom))]];
        const parentIndex = order[born % eliteCut];
        const parent = particles[parentIndex];
        const roll = Math.random();
        slot.x = Math.min(0.98, Math.max(0.02, parent.x + (Math.random() - 0.5) * 0.055));
        slot.y = Math.min(0.98, Math.max(0.02, parent.y + (Math.random() - 0.5) * 0.055));
        slot.z = Math.min(1, Math.max(0, parent.z + (Math.random() - 0.5) * 0.25));
        slot.vx = (Math.random() - 0.5) * 0.0002;
        slot.vy = (Math.random() - 0.5) * 0.0002;
        slot.r = 0.9 + Math.random() * 2;
        slot.kind = roll < 0.7 ? parent.kind : roll < 0.88 ? 0 : roll < 0.96 ? 1 : 2;
        slot.life = 1;
        slot.state = 0;
        slot.parent = parentIndex;
      }
      generation += 1;
      let top = 0;
      for (let index = 0; index < eliteCut; index += 1) top += fitness[order[index]];
      const best = top / eliteCut;
      history.push(Number(best.toFixed(3)));
      if (history.length > 28) history.shift();
      setHud({ generation, fitness: best, history: history.slice() });
    };

    const draw = () => {
      context.clearRect(0, 0, width, height);
      context.globalCompositeOperation = "source-over";
      context.lineWidth = 1;
      context.strokeStyle = palette.light ? "rgba(26,24,20,0.18)" : "rgba(243,239,228,0.22)";
      context.beginPath();
      for (let index = 0; index < particles.length; index += 1) {
        const particle = particles[index];
        const parent = particle.parent;
        if (parent < 0 || particle.life < 0.25) continue;
        const from = particles[parent];
        if (!from || from.life < 0.2) continue;
        const depth = (particle.z + from.z) * 0.5;
        const ox = (pointer.x - 0.5) * depth * 46;
        const oy = (pointer.y - 0.5) * depth * 28 + scroll * depth * 36;
        context.moveTo(from.x * width + ox, from.y * height + oy);
        context.lineTo(particle.x * width + ox, particle.y * height + oy);
      }
      context.globalAlpha = palette.light ? 0.2 : 0.3;
      context.stroke();
      context.globalAlpha = 1;

      if (!palette.light) context.globalCompositeOperation = "lighter";
      for (let layer = 0; layer < 3; layer += 1) {
        for (const particle of particles) {
          const bucket = Math.min(2, Math.floor(particle.z * 3));
          if (bucket !== layer || particle.life <= 0.02) continue;
          const depth = 0.35 + particle.z * 0.65;
          const ox = (pointer.x - 0.5) * particle.z * 56;
          const oy = (pointer.y - 0.5) * particle.z * 34 + scroll * particle.z * 42;
          const x = particle.x * width + ox;
          const y = particle.y * height + oy;
          const pruning = particle.state === 2;
          const fill = pruning ? palette.ember : colorOf(particle.kind);
          const radius = particle.r * (0.65 + depth) * (pruning ? 0.45 + particle.life * 0.55 : 1);
          context.fillStyle = fill;
          context.globalAlpha = (palette.light ? 0.07 : 0.1) * particle.life * depth;
          context.beginPath();
          context.arc(x, y, radius * 4.2, 0, Math.PI * 2);
          context.fill();
          context.globalAlpha = (palette.light ? 0.5 : 0.55) * particle.life;
          context.beginPath();
          context.arc(x, y, Math.max(0.6, radius), 0, Math.PI * 2);
          context.fill();
          if (particle.state === 1 && particle.life > 0.6) {
            context.globalAlpha = 0.8 * particle.life;
            context.strokeStyle = palette.light ? "#1a1814" : "#f3efe4";
            context.stroke();
          }
        }
      }
      context.globalAlpha = 1;
      context.globalCompositeOperation = "source-over";
    };

    const step = () => {
      if (!visible) {
        frame = requestAnimationFrame(step);
        return;
      }
      if (!reduced) {
        for (const particle of particles) {
          if (particle.state === 2) particle.life = Math.max(0, particle.life - 0.006);
          if (pointer.on) {
            const dx = pointer.x - particle.x;
            const dy = pointer.y - particle.y;
            const dist = Math.hypot(dx, dy) || 0.0001;
            if (dist < 0.24) {
              const force = dist < 0.045 ? -0.0005 : 0.00018;
              particle.vx += (dx / dist) * force * (0.5 + particle.z);
              particle.vy += (dy / dist) * force * (0.5 + particle.z);
            }
          }
          particle.vx += (Math.random() - 0.5) * 0.00004;
          particle.vy += (Math.random() - 0.5) * 0.00004;
          particle.x += particle.vx;
          particle.y += particle.vy;
          if (particle.x < 0.02 || particle.x > 0.98) particle.vx *= -1;
          if (particle.y < 0.02 || particle.y > 0.98) particle.vy *= -1;
          particle.x = Math.min(0.98, Math.max(0.02, particle.x));
          particle.y = Math.min(0.98, Math.max(0.02, particle.y));
          particle.vx *= 0.988;
          particle.vy *= 0.988;
        }
      }
      draw();
      frame = requestAnimationFrame(step);
    };

    let breedTimer = 0;
    const scheduleBreed = () => {
      if (reduced) return;
      breed();
      // Stop at the cap: the hero shows a finished run, not an endless counter.
      if (generation >= MAX_GENERATION) return;
      breedTimer = window.setTimeout(scheduleBreed, 1600);
    };

    resize();
    readPalette();
    draw();
    const observer = new IntersectionObserver(([entry]) => {
      visible = entry.isIntersecting;
    });
    observer.observe(canvas);
    const onPointer = (event: PointerEvent) => {
      const rect = canvas.getBoundingClientRect();
      pointer.x = (event.clientX - rect.left) / rect.width;
      pointer.y = (event.clientY - rect.top) / rect.height;
      pointer.on = pointer.x >= 0 && pointer.x <= 1 && pointer.y >= 0 && pointer.y <= 1;
    };
    const onScroll = () => {
      scroll = Math.min(1.2, window.scrollY / Math.max(1, window.innerHeight));
    };
    window.addEventListener("resize", resize);
    window.addEventListener("pointermove", onPointer);
    window.addEventListener("scroll", onScroll, { passive: true });
    const theme = new MutationObserver(() => {
      readPalette();
      draw();
    });
    theme.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    if (!reduced) {
      frame = requestAnimationFrame(step);
      breedTimer = window.setTimeout(scheduleBreed, 700);
    }

    return () => {
      cancelAnimationFrame(frame);
      window.clearTimeout(breedTimer);
      observer.disconnect();
      theme.disconnect();
      window.removeEventListener("resize", resize);
      window.removeEventListener("pointermove", onPointer);
      window.removeEventListener("scroll", onScroll);
    };
  }, []);

  const points = hud.history
    .map((value, index) => {
      const x = (index / Math.max(1, hud.history.length - 1)) * 120;
      const y = 32 - Math.min(1, value) * 28;
      return `${x},${y}`;
    })
    .join(" ");

  return (
    <div className="absolute inset-0">
      <canvas ref={ref} data-testid="living-field" className="absolute inset-0 h-full w-full" aria-hidden />
      <div className="pointer-events-none absolute bottom-6 right-4 z-20 w-52 rounded-2xl border border-line bg-raise/75 p-3 backdrop-blur-md md:right-8" data-testid="field-hud">
        <p className="font-mono text-[10px] uppercase tracking-[0.16em] text-faint">Generation {Math.min(hud.generation, MAX_GENERATION)} of {MAX_GENERATION}{hud.generation >= MAX_GENERATION ? " · done" : ""}</p>
        <p className="mt-1 font-serif text-3xl leading-none text-ivory">{hud.fitness.toFixed(2)}</p>
        <svg viewBox="0 0 120 36" className="mt-2 h-9 w-full" aria-hidden>
          <polyline fill="none" stroke="var(--teal)" strokeWidth="1.6" points={points} />
        </svg>
        <ul className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-[10px] uppercase tracking-widest text-dim">
          <li className="flex items-center gap-1"><span className="h-2 w-2 rounded-full" style={{ background: "var(--prov-veri)" }} />Veri</li>
          <li className="flex items-center gap-1"><span className="h-2 w-2 rounded-full" style={{ background: "var(--prov-creao)" }} />CREAO</li>
          <li className="flex items-center gap-1"><span className="h-2 w-2 rounded-full" style={{ background: "var(--prov-baseline)" }} />Baseline</li>
        </ul>
      </div>
    </div>
  );
}
