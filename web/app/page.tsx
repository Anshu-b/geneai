import type { Metadata } from "next";
import { HomeExperience } from "@/components/home/home-experience";

export const metadata: Metadata = {
  title: "Evolution Lab",
  description: "How the GeneAI engine searches, freezes a champion, and swaps domains.",
};

export default function HomePage() {
  return <HomeExperience />;
}
