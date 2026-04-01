/**
 * LEGO Layout Visualizer — Color palette generation.
 *
 * Generates distinct tile colors and gradient palettes for layout visualization.
 */

/**
 * Generate a palette of distinct colors for tiles.
 * Uses HSL with golden-angle rotation for maximum visual separation.
 */
export function generateTilePalette(numTiles) {
  if (numTiles <= 0) return [];
  if (numTiles === 1) return ['hsl(210, 70%, 75%)'];

  const colors = [];
  const goldenAngle = 137.508; // degrees
  for (let i = 0; i < numTiles; i++) {
    const hue = (i * goldenAngle) % 360;
    colors.push(`hsl(${hue}, 70%, 75%)`);
  }
  return colors;
}

/**
 * Generate a sequential gradient palette for non-tiled layouts.
 * Maps flat index values to a cool-to-warm color ramp.
 */
export function gradientColor(value, total) {
  const t = total > 1 ? value / (total - 1) : 0;
  // Cool blue -> warm orange
  const h = 220 - t * 190; // 220 (blue) -> 30 (orange)
  const s = 65 + t * 10;
  const l = 72 - t * 5;
  return `hsl(${h}, ${s}%, ${l}%)`;
}

/**
 * Compute tile ID for a coordinate given tile dimensions.
 * Returns -1 if not a tiled layout.
 */
export function computeTileId(i, j, shape, tileDims) {
  if (!tileDims || tileDims.length < 2) return -1;
  const [BM, BN] = tileDims;
  if (BM <= 0 || BN <= 0) return -1;
  const ti = Math.floor(i / BM);
  const tj = Math.floor(j / BN);
  const numTilesN = Math.ceil(shape[1] / BN);
  return ti * numTilesN + tj;
}

/**
 * Get color for a cell based on its coordinates and layout type.
 */
export function getCellColor(i, j, flat, shape, tileDims, palette) {
  const tileId = computeTileId(i, j, shape, tileDims);
  if (tileId >= 0 && palette && palette.length > 0) {
    return palette[tileId % palette.length];
  }
  // Fallback: gradient by flat index
  const total = shape[0] * shape[1];
  return gradientColor(flat, total);
}
