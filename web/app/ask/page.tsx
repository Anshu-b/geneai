import type { Metadata } from "next";
import { AskPanel } from "@/components/ask-panel";

export const metadata: Metadata = { title: "Ask GeneAI" };

export default function AskPage() {
  return <AskPanel />;
}
