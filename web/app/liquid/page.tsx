import type { Metadata } from "next";
import { LiquidPage } from "@/components/liquid/liquid-page";

export const metadata: Metadata = { title: "Liquid" };

export default function Liquid() {
  return <LiquidPage src="/fixtures/liquid.json" runSrc="/fixtures/liquid_run.json" />;
}
