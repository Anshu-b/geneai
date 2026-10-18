import type { Metadata } from "next";
import { IBM_Plex_Mono, Instrument_Sans, Instrument_Serif } from "next/font/google";
import { ThemeBoot } from "@/components/theme-boot";
import { THEME_BOOT } from "@/lib/theme";
import "./globals.css";

const sans = Instrument_Sans({
  subsets: ["latin"],
  variable: "--font-instrument-sans",
});

const serif = Instrument_Serif({
  subsets: ["latin"],
  weight: "400",
  style: ["normal", "italic"],
  variable: "--font-instrument-serif",
});

const mono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500"],
  variable: "--font-plex",
});

export const metadata: Metadata = {
  title: {
    default: "GeneAI",
    template: "%s · GeneAI",
  },
  description:
    "A living view of evolutionary search. Replay GeneAI lineages, pruning, and held-out scores from engine exports.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" data-theme="dark" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_BOOT }} />
      </head>
      <body className={`${sans.variable} ${serif.variable} ${mono.variable} antialiased`}>
        <ThemeBoot />
        {children}
      </body>
    </html>
  );
}
