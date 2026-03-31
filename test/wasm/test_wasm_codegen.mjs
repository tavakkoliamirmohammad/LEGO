/**
 * WASM codegen integration tests.
 *
 * Loads the Emscripten-compiled LEGO driver, compiles various MLIR layouts,
 * instantiates the generated WASM modules, and verifies that the exported
 * apply(i,j) function returns correct physical addresses.
 *
 * Usage:  node --experimental-vm-modules test_wasm_codegen.mjs [path/to/lego_driver.js]
 *
 * Requires Node >= 16 (BigInt + WebAssembly).
 */

import { createRequire } from 'module';
import { fileURLToPath } from 'url';
import path from 'path';

const require = createRequire(import.meta.url);

// ---------------------------------------------------------------------------
// Locate the driver
// ---------------------------------------------------------------------------

const driverPath = process.argv[2] ||
  path.resolve(path.dirname(fileURLToPath(import.meta.url)),
               '../../viz/wasm/lego_driver.js');

const LEGOCompiler = require(driverPath);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

let compiler;

async function init() {
  const mod = await LEGOCompiler();
  compiler = {
    compile: mod.cwrap('lego_compile', 'number', ['string', 'number', 'number']),
    free:    mod.cwrap('lego_free', null, ['number']),
    UTF8ToString: mod.UTF8ToString.bind(mod),
  };
}

function compileMLIR(mlir, M, N) {
  const ptr = compiler.compile(mlir, M, N);
  const json = compiler.UTF8ToString(ptr);
  compiler.free(ptr);
  return JSON.parse(json);
}

async function instantiateWasm(b64) {
  const bytes = Buffer.from(b64, 'base64');
  // LLVM's WebAssembly backend imports env.__linear_memory even for
  // pure arithmetic functions.  Provide a minimal memory instance.
  const imports = { env: { __linear_memory: new WebAssembly.Memory({ initial: 0 }) } };
  const { instance } = await WebAssembly.instantiate(bytes, imports);
  return instance.exports.apply;
}

function computeMapping(apply, M, N) {
  const mapping = [];
  for (let i = 0; i < M; i++)
    for (let j = 0; j < N; j++)
      mapping.push([i, j, Number(apply(BigInt(i), BigInt(j)))]);
  return mapping;
}

// ---------------------------------------------------------------------------
// Test cases
// ---------------------------------------------------------------------------

const tests = [];
let passed = 0, failed = 0;

function test(name, fn) { tests.push({ name, fn }); }

// --- Row-major: apply(i,j) = i*N + j ---
test('row-major 4x8', async () => {
  const M = 4, N = 8;
  const mlir = `module {
  func.func @apply(%i: index, %j: index) -> index {
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %layout = lego.row [%c4, %c8] : !lego.layout
    %flat = lego.apply %layout(%i, %j) : !lego.layout
    return %flat : index
  }
}`;
  const result = compileMLIR(mlir, M, N);
  assert(!result.error, `compile error: ${result.error}`);
  assert(result.wasmBinary, 'no wasmBinary in result');
  assert(result.lego, 'no lego IR');
  assert(result.arith, 'no arith IR');
  assert(result.llvm, 'no llvm IR');

  const apply = await instantiateWasm(result.wasmBinary);
  const mapping = computeMapping(apply, M, N);

  for (const [i, j, flat] of mapping) {
    const expected = i * N + j;
    assert(flat === expected,
      `row(${i},${j}): got ${flat}, expected ${expected}`);
  }
});

// --- Column-major: apply(i,j) = j*M + i ---
test('col-major 4x8', async () => {
  const M = 4, N = 8;
  const mlir = `module {
  func.func @apply(%i: index, %j: index) -> index {
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %layout = lego.col [%c4, %c8] : !lego.layout
    %flat = lego.apply %layout(%i, %j) : !lego.layout
    return %flat : index
  }
}`;
  const result = compileMLIR(mlir, M, N);
  assert(!result.error, `compile error: ${result.error}`);

  const apply = await instantiateWasm(result.wasmBinary);
  const mapping = computeMapping(apply, M, N);

  for (const [i, j, flat] of mapping) {
    const expected = j * M + i;
    assert(flat === expected,
      `col(${i},${j}): got ${flat}, expected ${expected}`);
  }
});

// --- Permuted: reg_p perm [1,0] (column-major via permutation) ---
test('reg_p perm [1,0] 4x8', async () => {
  const M = 4, N = 8;
  const mlir = `module {
  func.func @apply(%i: index, %j: index) -> index {
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %layout = lego.reg_p perm [1, 0] dims [%c4, %c8] : !lego.layout
    %flat = lego.apply %layout(%i, %j) : !lego.layout
    return %flat : index
  }
}`;
  const result = compileMLIR(mlir, M, N);
  assert(!result.error, `compile error: ${result.error}`);

  const apply = await instantiateWasm(result.wasmBinary);
  const mapping = computeMapping(apply, M, N);

  for (const [i, j, flat] of mapping) {
    const expected = j * M + i;
    assert(flat === expected,
      `reg_p[1,0](${i},${j}): got ${flat}, expected ${expected}`);
  }
});

// --- GroupBy (order_by + group_by) ---
test('group_by row 4x8', async () => {
  const M = 4, N = 8;
  const mlir = `module {
  func.func @apply(%i: index, %j: index) -> index {
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %r = lego.row [%c4, %c8] : !lego.layout
    %ob = lego.order_by(%r) : !lego.layout
    %gb = lego.group_by [%c4, %c8](%ob) : !lego.layout
    %flat = lego.apply %gb(%i, %j) : !lego.layout
    return %flat : index
  }
}`;
  const result = compileMLIR(mlir, M, N);
  assert(!result.error, `compile error: ${result.error}`);

  const apply = await instantiateWasm(result.wasmBinary);
  const mapping = computeMapping(apply, M, N);

  // GroupBy(OrderBy(Row(4,8))) with dims [4,8] is just row-major
  for (const [i, j, flat] of mapping) {
    const expected = i * N + j;
    assert(flat === expected,
      `group_by_row(${i},${j}): got ${flat}, expected ${expected}`);
  }
});

// --- Bijectivity: all outputs unique and in range [0, M*N) ---
test('bijectivity check col 6x5', async () => {
  const M = 6, N = 5;
  const mlir = `module {
  func.func @apply(%i: index, %j: index) -> index {
    %c6 = arith.constant 6 : index
    %c5 = arith.constant 5 : index
    %layout = lego.col [%c6, %c5] : !lego.layout
    %flat = lego.apply %layout(%i, %j) : !lego.layout
    return %flat : index
  }
}`;
  const result = compileMLIR(mlir, M, N);
  assert(!result.error, `compile error: ${result.error}`);

  const apply = await instantiateWasm(result.wasmBinary);
  const mapping = computeMapping(apply, M, N);
  const total = M * N;

  const seen = new Set();
  for (const [i, j, flat] of mapping) {
    assert(flat >= 0 && flat < total,
      `out of range: apply(${i},${j})=${flat}, total=${total}`);
    assert(!seen.has(flat),
      `duplicate: apply(${i},${j})=${flat} already seen`);
    seen.add(flat);
  }
  assert(seen.size === total, `expected ${total} unique values, got ${seen.size}`);
});

// --- IR stages present and non-empty ---
test('IR stages populated', async () => {
  const mlir = `module {
  func.func @apply(%i: index, %j: index) -> index {
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %layout = lego.row [%c2, %c3] : !lego.layout
    %flat = lego.apply %layout(%i, %j) : !lego.layout
    return %flat : index
  }
}`;
  const result = compileMLIR(mlir, 2, 3);
  assert(!result.error, `compile error: ${result.error}`);

  // LEGO stage should contain lego ops
  assert(result.lego.includes('lego.'), 'lego IR missing lego ops');
  // Arith stage should be lowered (no lego ops, has arith ops)
  assert(!result.arith.includes('lego.'), 'arith IR still has lego ops');
  assert(result.arith.includes('arith.'), 'arith IR missing arith ops');
  // LLVM stage should be fully lowered
  assert(result.llvm.includes('llvm.'), 'llvm IR missing llvm ops');
});

// --- Error handling ---
test('invalid MLIR returns error', () => {
  const result = compileMLIR('not valid mlir at all', 4, 4);
  assert(result.error, 'expected error for invalid MLIR');
});

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------

function assert(cond, msg) {
  if (!cond) throw new Error(msg || 'assertion failed');
}

async function run() {
  await init();
  console.log(`Running ${tests.length} WASM codegen tests...\n`);

  for (const { name, fn } of tests) {
    try {
      await fn();
      passed++;
      console.log(`  PASS  ${name}`);
    } catch (e) {
      failed++;
      console.log(`  FAIL  ${name}: ${e.message}`);
    }
  }

  console.log(`\n${passed} passed, ${failed} failed`);
  process.exit(failed > 0 ? 1 : 0);
}

run();
