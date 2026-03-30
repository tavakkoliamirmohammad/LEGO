/**
 * LEGO Layout Visualizer — SVG Renderer.
 *
 * Renders the dual-view visualization:
 *  - Logical grid (2D) on the left
 *  - Physical memory strip (1D) on the right
 *  - Connecting Bezier curves between them
 */

import { generateTilePalette, getCellColor, computeTileId } from './colors.js';

const SVG_NS = 'http://www.w3.org/2000/svg';

/** Create an SVG element with attributes. */
function svgEl(tag, attrs = {}) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    el.setAttribute(k, v);
  }
  return el;
}

/**
 * Main render function. Called with mapping data from the server.
 *
 * @param {Object} data - {mapping: [[i,j,flat],...], shape: [M,N], total: int}
 * @param {Object} opts - {tileDims: [BM,BN] or null}
 */
export function render(data, opts = {}) {
  const { mapping, shape } = data;
  const [M, N] = shape;
  const total = M * N;
  const tileDims = opts.tileDims || null;

  // Compute tile palette
  let numTiles = 0;
  if (tileDims && tileDims[0] > 0 && tileDims[1] > 0) {
    numTiles = Math.ceil(M / tileDims[0]) * Math.ceil(N / tileDims[1]);
  }
  const palette = numTiles > 0 ? generateTilePalette(numTiles) : [];

  // Build lookup: (i,j) -> flat, and flat -> (i,j)
  const fwdMap = {};  // "i,j" -> flat
  const invMap = {};  // flat -> {i, j}
  for (const entry of mapping) {
    const [i, j, flat] = entry;
    fwdMap[`${i},${j}`] = flat;
    invMap[flat] = { i, j };
  }

  // Sizing
  const maxCellSize = 50;
  const minCellSize = 20;
  const cellSize = Math.max(minCellSize, Math.min(maxCellSize, Math.floor(500 / Math.max(M, N))));
  const pad = 4; // padding around the grid
  const gridW = N * cellSize + pad * 2;
  const gridH = M * cellSize + pad * 2;
  const stripCellH = Math.max(15, Math.min(30, Math.floor(600 / total)));
  const stripW = 50 + pad * 2;
  const stripH = total * stripCellH + pad * 2;
  const connW = 120;

  // Container
  const container = document.getElementById('viz-container');
  container.innerHTML = '';

  // --- Logical Grid ---
  const logicalSection = document.createElement('div');
  logicalSection.className = 'viz-section';
  logicalSection.innerHTML = '<h4>Logical View</h4>';

  const logicalSvg = svgEl('svg', {
    width: gridW, height: gridH,
    viewBox: `0 0 ${gridW} ${gridH}`,
  });

  const logicalCells = {};
  for (let i = 0; i < M; i++) {
    for (let j = 0; j < N; j++) {
      const flat = fwdMap[`${i},${j}`];
      const color = getCellColor(i, j, flat, shape, tileDims, palette);
      const x = pad + j * cellSize;
      const y = pad + i * cellSize;

      const g = svgEl('g', { class: 'cell', 'data-i': i, 'data-j': j, 'data-flat': flat });

      g.appendChild(svgEl('rect', {
        x, y, width: cellSize, height: cellSize,
        fill: color, rx: 2,
      }));

      const fontSize = cellSize < 30 ? 9 : 11;
      const text = svgEl('text', {
        x: x + cellSize / 2, y: y + cellSize / 2,
        'font-size': fontSize,
      });
      text.textContent = flat;
      g.appendChild(text);

      logicalSvg.appendChild(g);
      logicalCells[`${i},${j}`] = {
        el: g, x: x + cellSize / 2, y: y + cellSize / 2, flat, color
      };
    }
  }

  logicalSection.appendChild(logicalSvg);
  container.appendChild(logicalSection);

  // --- Connections SVG ---
  const connSection = document.createElement('div');
  connSection.className = 'viz-section';
  connSection.innerHTML = '<h4>&nbsp;</h4>';

  const vizH = Math.max(gridH, stripH);
  const connSvg = svgEl('svg', {
    width: connW, height: vizH,
    viewBox: `0 0 ${connW} ${vizH}`,
    class: 'connections-svg',
  });

  const connections = {};
  const showAllLines = total <= 64;

  for (const entry of mapping) {
    const [i, j, flat] = entry;
    const logCell = logicalCells[`${i},${j}`];
    const logX = 0;
    const logY = logCell.y;
    const phyX = connW;
    const phyY = pad + flat * stripCellH + stripCellH / 2;

    const midX1 = connW * 0.35;
    const midX2 = connW * 0.65;
    const d = `M${logX},${logY} C${midX1},${logY} ${midX2},${phyY} ${phyX},${phyY}`;

    const path = svgEl('path', {
      d,
      class: 'connection',
      stroke: logCell.color,
      'data-i': i, 'data-j': j, 'data-flat': flat,
    });

    if (!showAllLines) {
      path.style.opacity = '0';
    }

    connSvg.appendChild(path);
    connections[`${i},${j}`] = path;
  }

  connSection.appendChild(connSvg);
  container.appendChild(connSection);

  // --- Physical Strip ---
  const physSection = document.createElement('div');
  physSection.className = 'viz-section';
  physSection.innerHTML = '<h4>Physical Memory</h4>';

  const physSvg = svgEl('svg', {
    width: stripW, height: stripH,
    viewBox: `0 0 ${stripW} ${stripH}`,
  });

  const physCells = {};
  for (let f = 0; f < total; f++) {
    const src = invMap[f];
    if (!src) continue;
    const color = getCellColor(src.i, src.j, f, shape, tileDims, palette);
    const y = pad + f * stripCellH;

    const g = svgEl('g', { class: 'cell', 'data-flat': f, 'data-i': src.i, 'data-j': src.j });

    g.appendChild(svgEl('rect', {
      x: pad, y, width: stripW - pad * 2, height: stripCellH,
      fill: color, rx: 2,
    }));

    const fontSize = stripCellH < 20 ? 8 : 10;
    const text = svgEl('text', {
      x: stripW / 2, y: y + stripCellH / 2,
      'font-size': fontSize,
    });
    text.textContent = f;
    g.appendChild(text);

    physSvg.appendChild(g);
    physCells[f] = { el: g, i: src.i, j: src.j };
  }

  physSection.appendChild(physSvg);
  container.appendChild(physSection);

  // --- Hover interaction ---
  function highlight(i, j) {
    const flat = fwdMap[`${i},${j}`];

    // Highlight logical cell
    for (const [key, cell] of Object.entries(logicalCells)) {
      if (key === `${i},${j}`) {
        cell.el.classList.add('highlighted');
        cell.el.classList.remove('dimmed');
      } else {
        cell.el.classList.remove('highlighted');
        cell.el.classList.add('dimmed');
      }
    }

    // Highlight physical cell
    for (const [f, cell] of Object.entries(physCells)) {
      if (parseInt(f) === flat) {
        cell.el.classList.add('highlighted');
        cell.el.classList.remove('dimmed');
      } else {
        cell.el.classList.remove('highlighted');
        cell.el.classList.add('dimmed');
      }
    }

    // Highlight connection
    for (const [key, path] of Object.entries(connections)) {
      if (key === `${i},${j}`) {
        path.classList.add('highlighted');
        path.classList.remove('dimmed');
        path.style.opacity = '';
      } else {
        path.classList.remove('highlighted');
        path.classList.add('dimmed');
        if (!showAllLines) path.style.opacity = '';
      }
    }
  }

  function unhighlight() {
    for (const cell of Object.values(logicalCells)) {
      cell.el.classList.remove('highlighted', 'dimmed');
    }
    for (const cell of Object.values(physCells)) {
      cell.el.classList.remove('highlighted', 'dimmed');
    }
    for (const [key, path] of Object.entries(connections)) {
      path.classList.remove('highlighted', 'dimmed');
      if (!showAllLines) path.style.opacity = '0';
    }
  }

  // Attach hover events to logical grid
  logicalSvg.addEventListener('mouseover', (e) => {
    const cell = e.target.closest('.cell');
    if (cell) {
      highlight(parseInt(cell.dataset.i), parseInt(cell.dataset.j));
    }
  });

  logicalSvg.addEventListener('mouseout', () => unhighlight());

  // Attach hover events to physical strip
  physSvg.addEventListener('mouseover', (e) => {
    const cell = e.target.closest('.cell');
    if (cell) {
      highlight(parseInt(cell.dataset.i), parseInt(cell.dataset.j));
    }
  });

  physSvg.addEventListener('mouseout', () => unhighlight());
}
