/**
 * LEGO Layout Visualizer — Main application logic.
 *
 * Handles editor, autocomplete, dimension controls, presets,
 * and server communication.
 */

import { render } from './renderer.js';

// ============================================================================
// Presets
// ============================================================================

const PRESETS = {
  'row-major': {
    label: 'Row-Major',
    code: 'M, N = 8, 8\nL = OrderBy(Row(M, N)).GroupBy([(M, N)])',
  },
  'col-major': {
    label: 'Col-Major',
    code: 'M, N = 8, 8\nL = OrderBy(Col(M, N)).GroupBy([(M, N)])',
  },
  'tiled': {
    label: 'Tiled',
    code: 'M, N = 8, 8\nBM, BN = 4, 4\nL = OrderBy(Row(M, N)).TileBy([M//BM, N//BN], [BM, BN])',
  },
  'transposed': {
    label: 'Transposed',
    code: 'M, N = 8, 8\nL = OrderBy(RegP((M, N), (1, 0))).GroupBy([(M, N)])',
  },
};

// ============================================================================
// Autocomplete dictionary
// ============================================================================

const COMPLETIONS = [
  { text: 'OrderBy',  hint: 'OrderBy(perm1, perm2, ...)' },
  { text: 'Row',      hint: 'Row(dim1, dim2, ...)' },
  { text: 'Col',      hint: 'Col(dim1, dim2, ...)' },
  { text: 'RegP',     hint: 'RegP((dims), (perm))' },
  { text: 'GenP',     hint: 'GenP((dims), f_apply, f_inv)' },
  { text: 'GroupBy',  hint: 'GroupBy([(dim1, dim2)])' },
  { text: 'TileBy',   hint: 'TileBy([outer_dims], [tile_sizes])' },
];

// ============================================================================
// DOM references
// ============================================================================

const editor = document.getElementById('code-editor');
const runBtn = document.getElementById('run-btn');
const errorOutput = document.getElementById('error-output');
const presetBtns = document.querySelectorAll('.preset-btn');
const formulaOutput = document.getElementById('formula-output');

// ============================================================================
// State
// ============================================================================

let currentPreset = 'row-major';
let lastMapping = null;

// ============================================================================
// Autocomplete
// ============================================================================

// Create autocomplete dropdown
const acDropdown = document.createElement('div');
acDropdown.id = 'autocomplete-dropdown';
document.body.appendChild(acDropdown);

let acVisible = false;
let acIndex = 0;
let acFiltered = [];
let acWordStart = 0;

function getWordAtCursor() {
  const pos = editor.selectionStart;
  const text = editor.value;
  let start = pos;
  while (start > 0 && /[a-zA-Z_]/.test(text[start - 1])) start--;
  return { word: text.substring(start, pos), start };
}

function showAutocomplete() {
  const { word, start } = getWordAtCursor();
  if (word.length < 1) { hideAutocomplete(); return; }

  acFiltered = COMPLETIONS.filter(c =>
    c.text.toLowerCase().startsWith(word.toLowerCase()) && c.text !== word
  );
  if (acFiltered.length === 0) { hideAutocomplete(); return; }

  acWordStart = start;
  acIndex = 0;
  acVisible = true;

  // Position dropdown near cursor
  const rect = editor.getBoundingClientRect();
  // Estimate cursor position (rough but works for monospace)
  const linesBefore = editor.value.substring(0, editor.selectionStart).split('\n');
  const lineNum = linesBefore.length - 1;
  const colNum = linesBefore[linesBefore.length - 1].length;
  const charW = 7.8; // approximate char width at 13px monospace
  const lineH = 19.5; // approximate line height

  acDropdown.style.left = `${rect.left + colNum * charW + 10}px`;
  acDropdown.style.top = `${rect.top + (lineNum + 1) * lineH + 10}px`;

  renderDropdown();
}

function renderDropdown() {
  acDropdown.innerHTML = '';
  acDropdown.style.display = 'block';

  acFiltered.forEach((item, i) => {
    const row = document.createElement('div');
    row.className = 'ac-item' + (i === acIndex ? ' ac-active' : '');
    row.innerHTML = `<span class="ac-text">${item.text}</span><span class="ac-hint">${item.hint}</span>`;
    row.addEventListener('mousedown', (e) => {
      e.preventDefault();
      acceptCompletion(i);
    });
    acDropdown.appendChild(row);
  });
}

function hideAutocomplete() {
  acVisible = false;
  acDropdown.style.display = 'none';
  acDropdown.innerHTML = '';
}

function acceptCompletion(index) {
  const item = acFiltered[index];
  if (!item) return;

  const pos = editor.selectionStart;
  const before = editor.value.substring(0, acWordStart);
  const after = editor.value.substring(pos);
  editor.value = before + item.text + after;
  editor.selectionStart = editor.selectionEnd = acWordStart + item.text.length;
  editor.focus();
  hideAutocomplete();
}

editor.addEventListener('input', () => {
  showAutocomplete();
});

editor.addEventListener('keydown', (e) => {
  if (acVisible) {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      acIndex = (acIndex + 1) % acFiltered.length;
      renderDropdown();
      return;
    }
    if (e.key === 'ArrowUp') {
      e.preventDefault();
      acIndex = (acIndex - 1 + acFiltered.length) % acFiltered.length;
      renderDropdown();
      return;
    }
    if (e.key === 'Tab' || e.key === 'Enter') {
      if (acFiltered.length > 0) {
        e.preventDefault();
        acceptCompletion(acIndex);
        return;
      }
    }
    if (e.key === 'Escape') {
      hideAutocomplete();
      return;
    }
  }

  // Auto-close brackets
  const PAIRS = { '(': ')', '[': ']', '{': '}' };
  const CLOSERS = new Set([')', ']', '}']);
  if (PAIRS[e.key]) {
    e.preventDefault();
    const start = editor.selectionStart;
    const end = editor.selectionEnd;
    const selected = editor.value.substring(start, end);
    const open = e.key;
    const close = PAIRS[open];
    editor.value = editor.value.substring(0, start) + open + selected + close + editor.value.substring(end);
    editor.selectionStart = editor.selectionEnd = start + 1 + selected.length;
    // If there was a selection, place cursor after it; otherwise between brackets
    if (selected.length === 0) {
      editor.selectionStart = editor.selectionEnd = start + 1;
    }
    return;
  }
  // Skip over closing bracket if it's already the next char
  if (CLOSERS.has(e.key)) {
    const pos = editor.selectionStart;
    if (editor.value[pos] === e.key) {
      e.preventDefault();
      editor.selectionStart = editor.selectionEnd = pos + 1;
      return;
    }
  }
  // Backspace: delete matching bracket pair
  if (e.key === 'Backspace') {
    const pos = editor.selectionStart;
    if (pos > 0 && editor.selectionStart === editor.selectionEnd) {
      const before = editor.value[pos - 1];
      const after = editor.value[pos];
      if (PAIRS[before] && PAIRS[before] === after) {
        e.preventDefault();
        editor.value = editor.value.substring(0, pos - 1) + editor.value.substring(pos + 1);
        editor.selectionStart = editor.selectionEnd = pos - 1;
        return;
      }
    }
  }

  // Ctrl+Enter to run
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
    e.preventDefault();
    hideAutocomplete();
    runVisualization();
    return;
  }

  // Tab inserts spaces (when autocomplete not active)
  if (e.key === 'Tab' && !acVisible) {
    e.preventDefault();
    const start = editor.selectionStart;
    const end = editor.selectionEnd;
    editor.value = editor.value.substring(0, start) + '    ' + editor.value.substring(end);
    editor.selectionStart = editor.selectionEnd = start + 4;
  }
});

editor.addEventListener('blur', () => {
  setTimeout(hideAutocomplete, 150);
});

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

  runBtn.disabled = true;
  runBtn.textContent = 'Compiling...';
  errorOutput.classList.remove('visible');
  errorOutput.textContent = '';

  try {
    const response = await fetch('/compile', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ code }),
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
      data = await computeMappingFromWasm(wasmBytes, data.shape);
    } else {
      // JSON response: mapping data directly
      data = await response.json();
    }

    // Use server-inferred tile dims (from layout object), not hardcoded BM/BN
    const tileDims = data.tile_info || null;

    // Render
    render(data, { tileDims });
    lastMapping = data;

    // Update formula display
    if (formulaOutput) {
      const [M, N] = data.shape;
      let info = `${M}\u00d7${N} grid \u00b7 ${data.total} elements`;
      if (tileDims) {
        info += ` \u00b7 ${tileDims[0]}\u00d7${tileDims[1]} tiles`;
      }
      formulaOutput.textContent = info;
    }

    // Show compiler outputs
    for (const [id, key] of [
      ['symbolic-output', 'symbolic'],
      ['mlir-lego', 'mlir_lego'],
      ['mlir-arith', 'mlir_arith'],
      ['mlir-llvm', 'mlir_llvm'],
    ]) {
      const el = document.getElementById(id);
      if (el) el.textContent = data[key] || '';
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

// ============================================================================
// Initialization
// ============================================================================

selectPreset('row-major');
// Auto-run on load
runVisualization();
