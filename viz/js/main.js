/**
 * LEGO Layout Visualizer — Main application logic.
 *
 * Uses the WASM-compiled LEGO compiler (lego_driver.wasm) to run
 * the full MLIR pipeline in the browser.  No Python server needed.
 */

import { render } from './renderer.js';

// ============================================================================
// WASM Compiler
// ============================================================================

let wasmCompiler = null;   // Emscripten module instance
let compilerReady = false;

async function initCompiler() {
  try {
    const module = await LEGOCompiler();
    wasmCompiler = {
      compile: module.cwrap('lego_compile', 'number', ['string', 'number', 'number']),
      free:    module.cwrap('lego_free', null, ['number']),
      UTF8ToString: module.UTF8ToString.bind(module),
    };
    compilerReady = true;
    document.getElementById('run-btn').textContent = 'Visualize';
  } catch (e) {
    console.error('Failed to load WASM compiler:', e);
    document.getElementById('error-output').textContent =
      'Failed to load compiler: ' + e.message;
    document.getElementById('error-output').classList.add('visible');
  }
}

function compileMLIR(mlirText, M, N) {
  if (!compilerReady) throw new Error('Compiler not loaded yet');
  const ptr = wasmCompiler.compile(mlirText, M || 0, N || 0);
  const json = wasmCompiler.UTF8ToString(ptr);
  wasmCompiler.free(ptr);
  return JSON.parse(json);
}

// ============================================================================
// DSL → Layout AST (sandbox-based evaluation)
// ============================================================================

/**
 * DSL functions that build a layout AST when evaluated.
 * The user's Python-like code is preprocessed and run in this sandbox.
 */
function _createDSLSandbox(vars) {
  function Row(...dims)  { return { op: 'row', dims }; }
  function Col(...dims)  { return { op: 'col', dims }; }
  function RegP(dims, perm) { return { op: 'reg_p', dims: [...dims], perm: [...perm] }; }

  function OrderBy(...perms) {
    const node = { op: 'order_by', perms };
    node.GroupBy = (groups) => ({
      op: 'group_by', inner: node, dims: groups[0] ? [...groups[0]] : [],
    });
    node.TileBy = (...groups) => ({
      op: 'tile_by', inner: node,
      groups: groups.map(g => [...g]),  // [[g0_dims], [g1_dims], ...]
    });
    return node;
  }

  return { ...vars, Row, Col, RegP, OrderBy, Math };
}

/**
 * Preprocess Python-like syntax to valid JavaScript:
 *  - // (int div) → Math.floor(a/b)
 *  - RegP((a, b), (c, d)) → RegP([a, b], [c, d])
 *  - GroupBy([(a, b)]) → GroupBy([[a, b]])
 */
function _pythonToJS(expr) {
  // Integer division: a // b → Math.floor(a/b)
  expr = expr.replace(/(\b[\w.]+(?:\([^)]*\))?)\s*\/\/\s*(\b[\w.]+(?:\([^)]*\))?)/g,
    'Math.floor($1/$2)');
  // RegP((a, b, ...), (c, d, ...)) → RegP([a, b, ...], [c, d, ...])
  expr = expr.replace(/RegP\s*\(\s*\(([^)]+)\)\s*,\s*\(([^)]+)\)\s*\)/g,
    'RegP([$1], [$2])');
  // GroupBy([(a, b, ...)]) → GroupBy([[a, b, ...]])
  expr = expr.replace(/GroupBy\s*\(\s*\[\s*\(([^)]+)\)\s*\]\s*\)/g,
    'GroupBy([[$1]])');
  // TileBy((a, b), ...) → TileBy([a, b], ...) — convert tuple args to arrays.
  // Find .TileBy( and replace inner (...) tuples with [...] arrays.
  const tileIdx = expr.indexOf('.TileBy(');
  if (tileIdx >= 0) {
    const before = expr.slice(0, tileIdx + 8); // up to and including '('
    let rest = expr.slice(tileIdx + 8);
    // Replace (x, y, ...) with [x, y, ...] in the TileBy arguments
    rest = rest.replace(/\(([^()]+)\)/g, '[$1]');
    expr = before + rest;
  }
  return expr;
}

/**
 * Parse the DSL code and return { layout, vars, shape, tileInfo }.
 *
 * Handles arbitrary expressions, constants, and variable references:
 *   M, N = 8, 8
 *   BM = M // 4
 *   L = OrderBy(Row(M, N)).TileBy([M//BM, N//BN], [BM, BN])
 */
function parseDSL(code) {
  const vars = {};
  let layoutAST = null;

  for (const line of code.trim().split('\n')) {
    const t = line.trim();
    if (!t || t.startsWith('#')) continue;

    // Layout definition: L = ...
    const lm = t.match(/^L\s*=\s*(.+)$/);
    if (lm) {
      const sandbox = _createDSLSandbox(vars);
      const jsExpr = _pythonToJS(lm[1].trim());
      try {
        const keys = Object.keys(sandbox);
        const vals = Object.values(sandbox);
        layoutAST = new Function(...keys, `return ${jsExpr}`)(...vals);
      } catch (e) {
        throw new Error(`Failed to evaluate layout: ${e.message}`);
      }
      continue;
    }

    // Variable assignment: M, N = 8, 8  or  BM = M // 4
    const am = t.match(/^(\w+(?:\s*,\s*\w+)*)\s*=\s*(.+)$/);
    if (am) {
      const names = am[1].split(',').map(s => s.trim());
      const rhsParts = am[2].split(',');

      if (names.length === rhsParts.length) {
        // Multi-assignment: M, N = 8, 8  or  BM, BN = M//4, N//4
        for (let i = 0; i < names.length; i++) {
          const jsExpr = _pythonToJS(rhsParts[i].trim());
          try {
            vars[names[i]] = new Function(...Object.keys(vars), `return ${jsExpr}`)
              (...Object.values(vars));
          } catch (e) {
            throw new Error(`Failed to evaluate ${names[i]}: ${e.message}`);
          }
        }
      } else {
        // Single assignment: BM = M // 4
        const jsExpr = _pythonToJS(am[2].trim());
        try {
          vars[names[0]] = new Function(...Object.keys(vars), 'Math', `return ${jsExpr}`)
            (...Object.values(vars), Math);
        } catch (e) {
          throw new Error(`Failed to evaluate ${names[0]}: ${e.message}`);
        }
      }
      continue;
    }
  }

  if (!layoutAST) throw new Error('No layout expression found (L = ...)');

  // Derive grid shape from the layout structure.
  let M, N;
  if (layoutAST.op === 'tile_by') {
    // Product of each dim across all groups: M = g0[0]*g1[0]*..., N = g0[1]*g1[1]*...
    const gs = layoutAST.groups;
    M = gs.reduce((p, g) => p * g[0], 1);
    N = gs.reduce((p, g) => p * g[1], 1);
  } else if (layoutAST.op === 'group_by') {
    M = layoutAST.dims[0];
    N = layoutAST.dims[1];
  } else {
    M = vars['M'];
    N = vars['N'];
  }
  if (M === undefined || N === undefined)
    throw new Error('Cannot determine grid dimensions');

  // Innermost tile group determines visual tile boundaries
  const tileInfo = layoutAST.op === 'tile_by' && layoutAST.groups.length > 1
    ? layoutAST.groups[layoutAST.groups.length - 1] : null;

  return {
    mlir:     layoutToMLIR(layoutAST),
    shape:    [M, N],
    tileInfo,
    layout:   layoutAST,
  };
}

// ============================================================================
// Layout AST → MLIR
// ============================================================================

function layoutToMLIR(layout) {
  // Collect all unique integer constants needed
  const consts = new Map();
  const ssaCounter = { n: 0 };
  const addConst = (v) => {
    v = Math.floor(v);
    if (!consts.has(v)) consts.set(v, `%c${v}`);
    return consts.get(v);
  };

  // Walk layout to collect all dimension values
  function collectDims(node) {
    if (!node || typeof node !== 'object') return;
    if (node.dims) node.dims.forEach(d => addConst(d));
    if (node.perm) {} // perm values are literal integers, not SSA
    if (node.groups) node.groups.forEach(g => g.forEach(d => addConst(d)));
    if (node.perms) node.perms.forEach(collectDims);
    if (node.inner) collectDims(node.inner);
  }
  collectDims(layout);

  const constLines = [...consts.entries()]
    .map(([v, name]) => `    ${name} = arith.constant ${v} : index`).join('\n');

  const ref = (v) => addConst(v);

  // Emit a layout op, return its SSA name
  function emitLayout(node) {
    const id = ssaCounter.n++;
    const name = id === 0 ? '%layout' : `%layout_${id}`;
    const dimRefs = node.dims.map(d => ref(d)).join(', ');

    if (node.op === 'row') {
      return { name, line: `    ${name} = lego.row [${dimRefs}] : !lego.layout` };
    } else if (node.op === 'col') {
      return { name, line: `    ${name} = lego.col [${dimRefs}] : !lego.layout` };
    } else if (node.op === 'reg_p') {
      const permStr = node.perm.join(', ');
      return { name, line: `    ${name} = lego.reg_p perm [${permStr}] dims [${dimRefs}] : !lego.layout` };
    }
    throw new Error(`Unknown layout op: ${node.op}`);
  }

  // Handle order_by: may contain multiple perms
  function emitOrderBy(node) {
    const parts = [];
    const permNames = [];
    for (const p of node.perms) {
      const { name, line } = emitLayout(p);
      parts.push(line);
      permNames.push(name);
    }
    const obName = `%ob_${ssaCounter.n++}`;
    parts.push(`    ${obName} = lego.order_by(${permNames.join(', ')}) : !lego.layout`);
    return { name: obName, lines: parts };
  }

  // Build the full module
  let bodyLines = [];
  let applyLayout;

  if (layout.op === 'group_by') {
    // GroupBy: simple 2D apply
    const ob = emitOrderBy(layout.inner);
    bodyLines.push(...ob.lines);
    // GroupBy wraps with dims
    const gbDimRefs = layout.dims.map(d => ref(d)).join(', ');
    const gbName = `%gb_${ssaCounter.n++}`;
    bodyLines.push(`    ${gbName} = lego.group_by [${gbDimRefs}](${ob.name}) : !lego.layout`);
    applyLayout = gbName;

    return `module {
  func.func @apply(%i: index, %j: index) -> index {
${constLines}
${bodyLines.join('\n')}
    %flat = lego.apply ${applyLayout}(%i, %j) : !lego.layout
    return %flat : index
  }
}`;

  } else if (layout.op === 'tile_by') {
    const gs = layout.groups;   // [[g0_dims], [g1_dims], ...]
    const nLevels = gs.length;
    const nDims = gs[0].length; // all groups must have same width (2 for 2D)

    // Refs for all group dims
    const groupRefs = gs.map(g => g.map(d => ref(d)));

    if (nLevels === 1) {
      // Single group: no decomposition, pass (i, j) directly
      const ob = emitOrderBy(layout.inner);
      bodyLines.push(...ob.lines);
      const tbName = `%tb_${ssaCounter.n++}`;
      bodyLines.push(`    ${tbName} = lego.tile_by ${ob.name} tile_dims [[${groupRefs[0].join(', ')}]] : !lego.layout`);
      applyLayout = tbName;

      return `module {
  func.func @apply(%i: index, %j: index) -> index {
${constLines}
${bodyLines.join('\n')}
    %flat = lego.apply ${applyLayout}(%i, %j) : !lego.layout
    return %flat : index
  }
}`;
    }

    // Multi-group: use a single reg_p to decompose 2D → N-D.
    //
    // Step 1: flatten (i,j) → flat_2d via row-major reg_p
    // Step 2: apply_inverse on a single N-D reg_p → all tiled coords
    // Step 3: pass to tile_by apply
    //
    // The N-D reg_p has:
    //   dims = level-grouped: [g0[0], g0[1], ..., g1[0], g1[1], ...]
    //   perm = interleave by dim: [0, d, 2d, ..., 1, d+1, 2d+1, ...]
    //          (all levels of dim 0 first, then dim 1, etc.)

    // Grid dims
    const gridM = gs.reduce((p, g) => p * g[0], 1);
    const gridN = gs.reduce((p, g) => p * g[1], 1);
    const gridMRef = ref(gridM);
    const gridNRef = ref(gridN);

    // Step 1: flatten
    const flattenName = `%grid_${ssaCounter.n++}`;
    bodyLines.push(`    ${flattenName} = lego.reg_p perm [0, 1] dims [${gridMRef}, ${gridNRef}] : !lego.layout`);
    bodyLines.push(`    %flat_2d = lego.apply ${flattenName}(%i, %j) : !lego.layout`);

    // Step 2: build N-D reg_p
    // dims in level-grouped order: [g0[0], g0[1], g1[0], g1[1], ...]
    const allDims = [];
    for (let l = 0; l < nLevels; l++)
      for (let d = 0; d < nDims; d++)
        allDims.push(groupRefs[l][d]);

    // perm: interleave by dimension
    // [0, d, 2d, ..., 1, d+1, 2d+1, ..., d-1, 2d-1, ...]
    const totalCoords = nLevels * nDims;
    const unflatPerm = [];
    for (let d = 0; d < nDims; d++)
      for (let l = 0; l < nLevels; l++)
        unflatPerm.push(l * nDims + d);

    const unflatName = `%unflatten_${ssaCounter.n++}`;
    bodyLines.push(`    ${unflatName} = lego.reg_p perm [${unflatPerm.join(', ')}] dims [${allDims.join(', ')}] : !lego.layout`);

    // apply_inverse → level-grouped coord names
    const coordNames = [];
    for (let l = 0; l < nLevels; l++)
      for (let d = 0; d < nDims; d++)
        coordNames.push(`%c${l}_${d}`);
    const retTypes = coordNames.map(() => 'index').join(', ');
    bodyLines.push(`    ${coordNames.join(', ')} = lego.apply_inverse ${unflatName}(%flat_2d) : !lego.layout -> ${retTypes}`);

    // Step 3: build tile_by + apply
    const ob = emitOrderBy(layout.inner);
    bodyLines.push(...ob.lines);
    const tileDimsStr = groupRefs.map(gr => `[${gr.join(', ')}]`).join(', ');
    const tbName = `%tb_${ssaCounter.n++}`;
    bodyLines.push(`    ${tbName} = lego.tile_by ${ob.name} tile_dims [${tileDimsStr}] : !lego.layout`);
    applyLayout = tbName;

    return `module {
  func.func @apply(%i: index, %j: index) -> index {
${constLines}
${bodyLines.join('\n')}
    %flat = lego.apply ${applyLayout}(${coordNames.join(', ')}) : !lego.layout
    return %flat : index
  }
}`;

  } else {
    // Bare layout (no GroupBy/TileBy wrapper) — just apply directly
    const { name, line } = emitLayout(layout);
    bodyLines.push(line);
    applyLayout = name;

    return `module {
  func.func @apply(%i: index, %j: index) -> index {
${constLines}
${bodyLines.join('\n')}
    %flat = lego.apply ${applyLayout}(%i, %j) : !lego.layout
    return %flat : index
  }
}`;
  }
}

// ============================================================================
// MLIR syntax highlighting
// ============================================================================

function esc(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function highlightMLIR(text) {
  if (!text) return '';
  return esc(text)
    .replace(/(\/\/.*)/g, '<span class="hl-comment">$1</span>')
    .replace(/("(?:[^"\\]|\\.)*")/g, '<span class="hl-string">$1</span>')
    .replace(/\b(lego|arith|scf|llvm|memref|func|builtin|cf|gpu)\.\w+/g, '<span class="hl-op">$&</span>')
    .replace(/(!lego\.\w+|(?<![%@])(?:\b(?:index|i1|i8|i16|i32|i64|f16|f32|f64)\b))/g, '<span class="hl-type">$1</span>')
    .replace(/(%[\w]+)/g, '<span class="hl-ssa">$1</span>')
    .replace(/\b(\d+)\b/g, '<span class="hl-num">$1</span>')
    .replace(/\b(func\.func|return|module|scf\.for|scf\.yield|step|to|public|attributes)\b/g, '<span class="hl-kw">$&</span>');
}

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
// Autocomplete
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
// Autocomplete logic
// ============================================================================

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

  const rect = editor.getBoundingClientRect();
  const linesBefore = editor.value.substring(0, editor.selectionStart).split('\n');
  const lineNum = linesBefore.length - 1;
  const colNum = linesBefore[linesBefore.length - 1].length;
  const charW = 7.8;
  const lineH = 19.5;

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
    if (selected.length === 0) {
      editor.selectionStart = editor.selectionEnd = start + 1;
    }
    return;
  }
  if (CLOSERS.has(e.key)) {
    const pos = editor.selectionStart;
    if (editor.value[pos] === e.key) {
      e.preventDefault();
      editor.selectionStart = editor.selectionEnd = pos + 1;
      return;
    }
  }
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

  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
    e.preventDefault();
    hideAutocomplete();
    runVisualization();
    return;
  }

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
// Theme toggle
// ============================================================================

const themeBtn = document.getElementById('theme-btn');
function setTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  themeBtn.textContent = theme === 'dark' ? '\u263E' : '\u2600';
  localStorage.setItem('lego-theme', theme);
}

themeBtn.addEventListener('click', () => {
  const current = document.documentElement.getAttribute('data-theme') || 'dark';
  setTheme(current === 'dark' ? 'light' : 'dark');
});

setTheme(localStorage.getItem('lego-theme') || 'dark');

// ============================================================================
// Line numbers
// ============================================================================

const lineNumbers = document.getElementById('line-numbers');

function updateLineNumbers() {
  const lines = editor.value.split('\n').length;
  lineNumbers.innerHTML = Array.from({ length: lines }, (_, i) =>
    `<div style="padding:0 4px">${i + 1}</div>`
  ).join('');
}

editor.addEventListener('input', () => { updateLineNumbers(); showAutocomplete(); });
editor.addEventListener('scroll', () => { lineNumbers.scrollTop = editor.scrollTop; });
updateLineNumbers();

// ============================================================================
// Preset handling
// ============================================================================

function selectPreset(name) {
  const preset = PRESETS[name];
  if (!preset) return;

  currentPreset = name;
  editor.value = preset.code;
  updateLineNumbers();

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
    // 1. Parse DSL → layout AST + MLIR + shape
    const { mlir, shape, tileInfo } = parseDSL(code);
    const [M, N] = shape;

    // 2. Compile MLIR + evaluate apply(i,j) for all cells — one call
    if (!compilerReady) throw new Error('Compiler not loaded yet');
    const compilerResult = compileMLIR(mlir, M, N);
    if (compilerResult.error) throw new Error(compilerResult.error);

    const data = {
      mapping: compilerResult.mapping,
      shape: compilerResult.shape || shape,
      total: M * N,
      tile_info: tileInfo,
    };

    // 4. Render visualization
    const vizContainer = document.getElementById('viz-container');
    if (data.mapping && data.mapping.length > 0) {
      const tileDims = tileInfo || null;
      render(data, { tileDims });
      lastMapping = data;

      if (formulaOutput) {
        const [M, N] = shape;
        let info = `${M}\u00d7${N} grid \u00b7 ${data.total} elements`;
        if (tileDims) {
          info += ` \u00b7 ${tileDims[0]}\u00d7${tileDims[1]} tiles`;
        }
        formulaOutput.textContent = info;
      }
    }

    // 5. Show compiler outputs
    // 5. Show compiler outputs
    for (const [id, key] of [
      ['mlir-lego', 'lego'],
      ['mlir-arith', 'arith'],
      ['mlir-llvm', 'llvm'],
    ]) {
      const el = document.getElementById(id);
      if (el) el.innerHTML = highlightMLIR(compilerResult[key] || '');
    }

  } catch (e) {
    errorOutput.textContent = e.message;
    errorOutput.classList.add('visible');
  } finally {
    runBtn.disabled = false;
    runBtn.textContent = 'Visualize';
  }
}

// ============================================================================
// Event listeners
// ============================================================================

runBtn.addEventListener('click', runVisualization);

// ============================================================================
// Initialization
// ============================================================================

selectPreset('row-major');
runBtn.textContent = 'Loading compiler...';
runBtn.disabled = true;

// Load WASM compiler, then auto-run
initCompiler().then(() => {
  runVisualization();
});
