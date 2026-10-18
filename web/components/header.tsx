"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { Presentation } from "@/lib/mode";
import { ThemeToggle } from "@/components/theme-toggle";

const LINKS = [
  { href: "/", label: "Evolution Lab" },
  { href: "/cloud", label: "Cloud" },
  { href: "/shopping", label: "Shopping" },
  { href: "/quant", label: "Quant" },
  { href: "/ask", label: "Ask GeneAI" },
  { href: "/about", label: "About" },
];

export function Header({ presentation }: { presentation?: Presentation | null }) {
  const path = usePathname();
  return (
    <header className="sticky top-0 z-30 border-b border-line bg-bg/85 backdrop-blur-md">
      <div className="mx-auto flex max-w-[1600px] items-center gap-4 px-4 py-3 md:px-6">
        <Link href="/" className="shrink-0">
          <span className="font-serif text-2xl tracking-tight text-ivory">GeneAI</span>
        </Link>
        <nav className="flex min-w-0 flex-1 gap-1 overflow-x-auto" aria-label="Primary">
          {LINKS.map((link) => {
            const active = link.href === "/" ? path === "/" : path === link.href;
            return (
              <Link
                key={link.href}
                href={link.href}
                aria-current={active ? "page" : undefined}
                className={`shrink-0 rounded-full px-3 py-1.5 text-sm ${
                  active ? "bg-ivory/10 text-ivory" : "text-dim hover:text-ivory"
                }`}
              >
                {link.label}
              </Link>
            );
          })}
        </nav>
        <div className="flex shrink-0 items-start gap-2">
          <ThemeToggle />
          {presentation ? (
          <div className="text-right" title={presentation.detail}>
            <div
              data-testid="mode-badge"
              className="inline-flex items-center gap-2 rounded-full border border-line px-2.5 py-1 font-mono text-[10px] uppercase tracking-[0.16em]"
            >
              <span
                className={`h-1.5 w-1.5 rounded-full ${
                  presentation.badge === "Live"
                    ? "bg-teal"
                    : presentation.badge === "Mock"
                      ? "bg-amber"
                      : presentation.badge === "Fixture"
                        ? "bg-violet"
                        : "bg-ivory"
                }`}
                aria-hidden
              />
              {presentation.badge}
            </div>
            <p className="mt-1 hidden font-mono text-[10px] text-faint sm:block">run.mode {presentation.runMode}</p>
          </div>
          ) : null}
        </div>
      </div>
    </header>
  );
}
