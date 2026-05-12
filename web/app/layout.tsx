import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Manufacturing GraphRAG · Chat",
  description:
    "Conversational copilot for manufacturing operations — auto-corrects jargon, asks clarifying questions, grounded in your documents and knowledge graph.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-cream-50 text-ink-800 antialiased">
        {children}
      </body>
    </html>
  );
}
