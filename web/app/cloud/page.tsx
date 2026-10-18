import type { Metadata } from "next";
import { Lab } from "@/components/lab/lab";

export const metadata: Metadata = { title: "Cloud" };

export default function CloudPage() {
  return <Lab fixture="/fixtures/cloud_scheduling.json" title="Cloud scheduling" />;
}
