/**
 * LEGO Layout Visualizer — Main application logic.
 *
 * Handles editor, dimension controls, presets, and server communication.
 */

import { render } from './renderer.js';

// ============================================================================
// Presets
// ============================================================================

const PRESETS = {
  'row-major': {
    label: 'Row-Major',
    code: 'L = OrderBy(Row(M, N)).GroupBy([(M, N)])',
    needsTile: false,
  },
  'col-major': {
    label: 'Col-Major',
    code: 'L = OrderBy(Col(M, N)).GroupBy([(M, N)])',
    needsTile: false,
  },
  'tiled': {
    label: 'Tiled',
    code: 'L = OrderBy(Row(M, N)).TileBy([M//BM, N//BN], [BM, BN])',
    needsTile: true,
  },
  'transposed': {
    label: 'Transposed',
    code: 'L = OrderBy(RegP((M, N), (1, 0))).GroupBy([(M, N)])',
    needsTile: false,
  },
};

// ============================================================================
// DOM references
// ============================================================================

const editor = document.getElementById('code-editor');
const runBtn = document.getElementById('run-btn');
const errorOutput = document.getElementById('error-output');
const dimM = document.getElementById('dim-M');
const dimN = document.getElementById('dim-N');
const dimBM = document.getElementById('dim-BM');
const dimBN = document.getElementById('dim-BN');
const presetBtns = document.querySelectorAll('.preset-btn');
const tileFields = document.getElementById('tile-fields');
const formulaOutput = document.getElementById('formula-output');

// ============================================================================
// State
// ============================================================================

let currentPreset = 'row-major';
let lastMapping = null;

// ============================================================================
// Preset handling
// ============================================================================

function selectPreset(name) {
  const preset = PRESETS[name];
  if (!preset) return;

  currentPreset = name;
  editor.value = preset.code;

  // Update active button
  presetBtns.forEach(btn => {
    btn.classList.toggle('active', btn.dataset.preset === name);
  });

  // Show/hide tile dimension fields
  if (tileFields) {
    tileFields.style.display = preset.needsTile ? 'grid' : 'none';
  }
}

presetBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    selectPreset(btn.dataset.preset);
    runVisualization();
  });
});

// ============================================================================
// Compilation & visualization
// ============================================================================

async function runVisualization() {
  const code = editor.value.trim();
  if (!code) return;

  const M = parseInt(dimM.value) || 8;
  const N = parseInt(dimN.value) || 8;
  const BM = parseInt(dimBM.value) || 4;
  const BN = parseInt(dimBN.value) || 4;

  runBtn.disabled = true;
  runBtn.textContent = 'Compiling...';
  errorOutput.classList.remove('visible');
  errorOutput.textContent = '';

  try {
    const response = await fetch('/compile', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        code,
        shape: [M, N],
        extra_dims: { BM, BN },
      }),
    });

    const contentType = response.headers.get('Content-Type') || '';

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.error || 'Compilation failed');
    }

    let data;

    if (contentType.includes('application/wasm')) {
      // WASM response: instantiate and compute mapping
      const wasmBytes = await response.arrayBuffer();
      data = await computeMappingFromWasm(wasmBytes, [M, N]);
    } else {
      // JSON response: mapping data directly
      data = await response.json();
    }

    // Determine tile dims for coloring
    const tileDims = code.includes('TileBy') ? [BM, BN] : null;

    // Render
    render(data, { tileDims });
    lastMapping = data;

    // Update formula display
    if (formulaOutput) {
      formulaOutput.textContent = `${M}x${N} grid | ${data.total} elements`;
      if (tileDims) {
        formulaOutput.textContent += ` | ${BM}x${BN} tiles`;
      }
    }

  } catch (e) {
    errorOutput.textContent = e.message;
    errorOutput.classList.add('visible');
  } finally {
    runBtn.disabled = false;
    runBtn.textContent = 'Visualize';
  }
}

/**
 * Compute mapping by calling WASM apply() for each coordinate.
 */
async function computeMappingFromWasm(wasmBytes, shape) {
  const [M, N] = shape;
  const total = M * N;

  const { instance } = await WebAssembly.instantiate(wasmBytes, {
    env: { memory: new WebAssembly.Memory({ initial: 1 }) },
  });

  const apply = instance.exports.apply;
  if (!apply) {
    throw new Error('WASM module does not export an "apply" function');
  }

  const mapping = [];
  for (let i = 0; i < M; i++) {
    for (let j = 0; j < N; j++) {
      const flat = apply(i, j);
      mapping.push([i, j, flat]);
    }
  }

  return { mapping, shape, total };
}

// ============================================================================
// Event listeners
// ============================================================================

runBtn.addEventListener('click', runVisualization);

// Ctrl+Enter to run
editor.addEventListener('keydown', (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
    e.preventDefault();
    runVisualization();
  }
  // Tab inserts spaces
  if (e.key === 'Tab') {
    e.preventDefault();
    const start = editor.selectionStart;
    const end = editor.selectionEnd;
    editor.value = editor.value.substring(0, start) + '    ' + editor.value.substring(end);
    editor.selectionStart = editor.selectionEnd = start + 4;
  }
});

// ============================================================================
// Initialization
// ============================================================================

selectPreset('row-major');
// Auto-run on load
runVisualization();
