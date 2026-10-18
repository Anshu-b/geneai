import type { Metadata } from "next";
import { Lab } from "@/components/lab/lab";

export const metadata: Metadata = { title: "Shopping" };

export default function ShoppingPage() {
  return <Lab fixture="/fixtures/shopping.json" title="Shopping" />;
}
