"use client";

import { useEffect, useRef } from "react";

interface Node {
  x: number;
  y: number;
  vx: number;
  vy: number;
  r: number;
  kind: "teal" | "gold" | "violet" | "dim";
  elite: boolean;
}

function makeNodes(count: number): Node[] {
  const kinds: Node["kind"][] = ["teal", "teal", "teal", "gold", "violet", "dim"];
  return Array.from({ length: count }, (_, index) => ({
    x: (index * 0.173) % 1,
    y: (index * 0.291) % 1,
    vx: ((index % 7) - 3) * 0.00008,
    vy: ((index % 5) - 2) * 0.00007,
    r: index % 11 === 0 ? 3.4 : 1.6 + (index % 4) * 0.35,
    kind: kinds[index % kinds.length],
    elite: index % 9 === 0,
  }));
}

export function LivingField() {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;
    const nodes = makeNodes(72);
    const pointer = { x: 0.72, y: 0.42, on: false };
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let frame = 0;
    let visible = true;
    let scroll = 0;

    const colors = () => {
      const style = getComputedStyle(document.documentElement);
      const pick = (name: string) => style.getPropertyValue(name).trim();
      return {
        teal: pick("--teal") || "#7ee0c6",
        gold: pick("--champ") || "#e4c27a",
        violet: pick("--violet") || "#c4b5fd",
        dim: pick("--faint") || "#6f695f",
        light: document.documentElement.dataset.theme === "light",
      };
    };

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 1.75);
      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));
      context.setTransform(dpr, 0, 0, dpr, 0, 0);
    };

    const draw = (time: number) => {
      const rect = canvas.getBoundingClientRect();
      const width = rect.width;
      const height = rect.height;
      const palette = colors();
      context.clearRect(0, 0, width, height);
      const drift = reduced ? 0 : scroll * 0.04;
      for (let i = 0; i < nodes.length; i += 1) {
        for (let j = i + 1; j < nodes.length; j += 1) {
          const a = nodes[i];
          const b = nodes[j];
          const dx = a.x - b.x;
          const dy = a.y - b.y;
          const dist = Math.hypot(dx, dy);
          if (dist > 0.16) continue;
          context.strokeStyle = palette.light ? "rgba(26,24,20,0.14)" : "rgba(243,239,228,0.16)";
          context.globalAlpha = (1 - dist / 0.16) * 0.7;
          context.beginPath();
          context.moveTo(a.x * width, a.y * height + drift);
          context.lineTo(b.x * width, b.y * height + drift);
          context.stroke();
        }
      }
      context.globalAlpha = 1;
      for (const node of nodes) {
        const color = palette[node.kind === "dim" ? "dim" : node.kind];
        const x = node.x * width;
        const y = node.y * height + drift;
        context.beginPath();
        context.fillStyle = color;
        context.globalAlpha = node.kind === "dim" ? 0.45 : 0.9;
        context.arc(x, y, node.r, 0, Math.PI * 2);
        context.fill();
        if (node.elite) {
          context.globalAlpha = 0.9;
          context.strokeStyle = palette.light ? "#1a1814" : "#f3efe4";
          context.lineWidth = 1;
          context.beginPath();
          context.arc(x, y, node.r + 3.5, 0, Math.PI * 2);
          context.stroke();
        }
      }
      context.globalAlpha = 1;
      if (!reduced) {
        const pulse = nodes[0];
        context.strokeStyle = palette.gold;
        context.globalAlpha = 0.25 + 0.15 * Math.sin(time / 700);
        context.beginPath();
        context.arc(pulse.x * width, pulse.y * height + drift, 16 + 4 * Math.sin(time / 500), 0, Math.PI * 2);
        context.stroke();
        context.globalAlpha = 1;
      }
    };

    const step = (time: number) => {
      if (!visible) {
        frame = requestAnimationFrame(step);
        return;
      }
      if (!reduced) {
        for (const node of nodes) {
          if (pointer.on) {
            const dx = pointer.x - node.x;
            const dy = pointer.y - node.y;
            const dist = Math.hypot(dx, dy);
            if (dist < 0.22 && dist > 0.001) {
              node.vx += (dx / dist) * 0.00009;
              node.vy += (dy / dist) * 0.00009;
            }
          }
          node.x += node.vx;
          node.y += node.vy;
          if (node.x < 0.02 || node.x > 0.98) node.vx *= -1;
          if (node.y < 0.02 || node.y > 0.98) node.vy *= -1;
          node.x = Math.min(0.98, Math.max(0.02, node.x));
          node.y = Math.min(0.98, Math.max(0.02, node.y));
          node.vx *= 0.992;
          node.vy *= 0.992;
        }
      }
      draw(time);
      frame = requestAnimationFrame(step);
    };

    const onPointer = (event: PointerEvent) => {
      const rect = canvas.getBoundingClientRect();
      pointer.x = (event.clientX - rect.left) / rect.width;
      pointer.y = (event.clientY - rect.top) / rect.height;
      pointer.on = pointer.x > 0 && pointer.x < 1 && pointer.y > 0 && pointer.y < 1;
    };
    const onLeave = () => {
      pointer.on = false;
    };
    const onScroll = () => {
      scroll = Math.min(1, window.scrollY / Math.max(1, window.innerHeight));
    };
    const observer = new IntersectionObserver(([entry]) => {
      visible = entry.isIntersecting;
    });
    resize();
    draw(0);
    if (!reduced) frame = requestAnimationFrame(step);
    observer.observe(canvas);
    window.addEventListener("resize", resize);
    window.addEventListener("pointermove", onPointer);
    window.addEventListener("scroll", onScroll, { passive: true });
    canvas.addEventListener("pointerleave", onLeave);
    const theme = new MutationObserver(() => draw(performance.now()));
    theme.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      theme.disconnect();
      window.removeEventListener("resize", resize);
      window.removeEventListener("pointermove", onPointer);
      window.removeEventListener("scroll", onScroll);
      canvas.removeEventListener("pointerleave", onLeave);
    };
  }, []);

  return <canvas ref={ref} data-testid="living-field" className="absolute inset-0 h-full w-full" aria-hidden />;
}
