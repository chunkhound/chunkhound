/**
 * Converts OG image SVGs to PNG for social platform compatibility.
 *
 * Social platforms (Twitter/X, Facebook, LinkedIn, Discord, etc.)
 * do not support SVG as og:image. This script uses resvg-js to
 * render the SVGs (including their embedded woff2 fonts) to PNG.
 *
 * Env override: CHUNKHOUND_PUBLIC_DIR — set to a custom public dir
 * for hermetic testing.
 */
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { resolve, join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { Resvg } from "@resvg/resvg-js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const publicDir = process.env.CHUNKHOUND_PUBLIC_DIR
  ? resolve(process.env.CHUNKHOUND_PUBLIC_DIR)
  : join(__dirname, "..", "public");

const svgFiles = ["og-image-dark.svg", "og-image-light.svg"];

for (const svgFile of svgFiles) {
  const svgPath = join(publicDir, svgFile);
  const pngPath = join(publicDir, svgFile.replace(/\.svg$/, ".png"));

  if (!existsSync(svgPath)) {
    throw new Error(`OG SVG not found at ${svgPath}`);
  }

  const svg = readFileSync(svgPath, "utf-8");

  let pngBuffer;
  try {
    const resvg = new Resvg(svg, {
      fitTo: { mode: "width", value: 1200 },
    });
    const pngData = resvg.render();
    pngBuffer = pngData.asPng();
  } catch (err) {
    throw new Error(`Failed to render ${svgPath}: ${err}`);
  }

  writeFileSync(pngPath, pngBuffer);
  console.log(`\u2713 Generated ${pngPath} (${(pngBuffer.length / 1024).toFixed(1)} KB)`);
}
