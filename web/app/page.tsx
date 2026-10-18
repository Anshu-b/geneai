import type { Metadata } from "next";
import { Lab } from "@/components/lab/lab";

export const metadata: Metadata = { title: "Evolution Lab" };

export default function HomePage() {
  return <Lab fixture="/fixtures/cloud_scheduling.json" title="Evolution lab · cloud scheduling" />;
}
