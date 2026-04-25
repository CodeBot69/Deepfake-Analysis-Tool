import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "DeepScan AI — Deepfake Analysis Tool",
  description:
    "High-precision deepfake detection using multi-model ensemble AI analysis. Detect manipulated media with spatial, frequency, and lip-sync analysis.",
  keywords: ["deepfake", "detection", "AI", "forensics", "analysis"],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
