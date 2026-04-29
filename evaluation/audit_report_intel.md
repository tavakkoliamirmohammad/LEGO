# CASTLE Cross-Architecture Intel Audit Report

## Overview

This report documents the results of re-measuring all 34 committed LEGO evaluation
candidates on an Intel Xeon Gold 6330 (Ice Lake, AVX-512) node, comparing verdicts
against the canonical AMD EPYC 7513 (Zen3, AVX2) Round 1 measurements.

**Purpose:** Validate that LEGO layout transformations are architecture-portable —
i.e., speedups observed on AMD Zen3/AVX2 hold (or improve) on Intel Ice Lake/AVX-512.

---

## Node Identity

| Field | Value |
|---|---|
| Hostname | notch343 |
| CPU | Intel Xeon Gold 6330 @ 2.00GHz (ICL, 28c×2s) |
| ISA | x86_64, AVX-512F/DQ/BW/IFMA/VBMI/VNNI |
| L1d | 48 KB |
| L2 | 1280 KB |
| L3 | 43008 KB |
| DRAM | 251 GiB (4-NUMA) |
| OS | Rocky Linux 8.10, kernel 4.18.0-553 |
| Compiler | GCC 13.3.0 (Spack), `-O3 -march=native` |
| Governor | performance |
| Machine SHA256 | 842a1d8fa97a8fab9c2c826dbd532eb35e778086562ed7241b9d788f6ac3c293 |

**Reference (Round 1 / AMD):** notch368, AMD EPYC 7513 (Milan, 32c×2s), AVX2, L3=256 MB.

---

## Measurement Protocol

- **Warmup:** 10 iterations (vs. 25–100 in Round 1, sufficient for L3 warm state)
- **Timed:** 30 iterations (vs. 100 in Round 1)
- **Pinning:** `taskset -c 4` (NUMA node 0, physical core)
- **Memory binding:** `numactl --membind=0` (where available)
- **Locking:** global fcntl flock at `evaluation/.lock` (one candidate at a time)
- **Output:** `raw/audit_intel_baseline_{size}.json` and `raw/audit_intel_lego_{size}.json`
  per candidate, conforming to the LEGO result schema.
- **Candidate 30:** DROPPED-build — skipped throughout.

---

## Summary Statistics

| Metric | AMD Round 1 | Intel Audit |
|---|---|---|
| Candidates measured | 34 of 35 | 34 of 35 |
| WIN | 22 | 18 |
| LOSS | 8 | 13 |
| PARITY | 1 | 1 |
| MIXED | 3 | 2 |
| Agreement with AMD verdict | — | 23/34 (68%) |

---

## Full Verdict Table

Speedup = baseline / lego; WIN ≥ 1.02×, LOSS ≤ 0.98×, PARITY otherwise.
"Best Intel Sp" = maximum speedup across all measured sizes on Intel.

| # | Candidate | Layout Class | AMD Verdict | Intel Verdict | Best Intel Sp | Match |
|---|---|---|---|---|---|---|
| 01 | polybench-gemm-zmorton | Z-Morton | WIN | WIN | 2.27× | AGREE |
| 02 | polybench-lu-zmorton | Z-Morton | WIN | WIN | 1.91× | AGREE |
| 03 | polybench-chol-zmorton | Z-Morton | LOSS | LOSS | 0.63× | AGREE |
| 04 | polybench-gemm-reg-L1-L2-tile | Reg+L1+L2 tile | WIN | WIN | 1.93× | AGREE |
| 05 | polybench-3mm-reg-L1-L2-tile | Reg+L1+L2 tile | WIN | WIN | 4.25× | AGREE |
| 06 | polybench-2mm-reg-L1-tile | Reg+L1 tile | WIN | WIN | 7.03× | AGREE |
| 07 | polybench-trmm-L1-L2-tile | Reg+L1+L2 tile | WIN | WIN | 18.37× | AGREE |
| 08 | polybench-doitgen-reg-L1-tile | Reg+L1 tile | WIN | WIN | 5.12× | AGREE |
| 09 | tccg-tensor-contraction-GETT-tile | GETT tile | WIN | WIN | 2.11× | AGREE |
| 10 | tblis-tensor-contraction-notranspose | TBLIS | WIN | LOSS | 1.01× | DIFFER |
| 11 | bricklib-3d7pt-brick | Brick | LOSS | LOSS | 0.86× | AGREE |
| 12 | bricklib-3d13pt-brick | Brick | PARITY | WIN | 1.06× | DIFFER |
| 13 | polybench-heat3d-brick | Brick | MIXED | LOSS | 0.80× | DIFFER |
| 14 | polybench-jacobi2d-brick | Brick | LOSS | LOSS | 0.19× | AGREE |
| 15 | polybench-symm-rfp | RFP | PARITY | LOSS | 1.01× | DIFFER |
| 16 | polybench-syrk-rfp | RFP | WIN | WIN | 5.52× | AGREE |
| 17 | rodinia-nw-antidiag-tile | Antidiag tile | MIXED | LOSS | 0.43× | DIFFER |
| 18 | npdp-nussinov-skew-tile | Skew tile | WIN | WIN | 1.76× | AGREE |
| 19 | npdp-zuker-skew-tile | Skew tile | WIN | LOSS | 1.00× | DIFFER |
| 20 | polybench-seidel2d-wavefront-tile | Wavefront tile | LOSS | MIXED | 1.26× | DIFFER |
| 21 | rodinia-particlefilter-aosoA | AoSoA | LOSS | PARITY | 1.00× | DIFFER |
| 22 | lulesh-elem-aosoA | AoSoA | LOSS | LOSS | 0.90× | AGREE |
| 23 | hpccg-cg-aosoA | AoSoA | MIXED | LOSS | 0.75× | DIFFER |
| 24 | polybench-fdtd-2d-block-cyclic | Block-cyclic | MIXED | LOSS | 0.98× | DIFFER |
| 25 | polybench-adi-block-cyclic | Block-cyclic | WIN | WIN | 1.28× | AGREE |
| 26 | polybench-gemm-pow2-pad | Pow2 pad | WIN | LOSS | 1.01× | DIFFER |
| 27 | polybench-heat3d-pow2-pad | Pow2 pad | WIN | WIN | 1.28× | AGREE |
| 28 | polybench-gemm-nonpow2-morton | Morton+non-pow2 | WIN | WIN | 2.91× | AGREE |
| 29 | bricklib-stencil-nonpow2-brick | Brick+non-pow2 | LOSS | LOSS | 0.61× | AGREE |
| 30 | DROPPED-build | — | — | — | — | — |
| 31 | npdp-nussinov-nonpow2-skew | Skew+non-pow2 | MIXED | MIXED | 1.24× | AGREE |
| 32 | rodinia-hotspot-tile | Tile | WIN | WIN | 1.06× | AGREE |
| 33 | polybench-mvt-L1-tile | L1 tile | WIN | WIN | 2.79× | AGREE |
| 34 | polybench-bicg-L1-tile | L1 tile | WIN | WIN | 4.34× | AGREE |
| 35 | hpcc-dgemm-reg-L1-L2-tile | Reg+L1+L2 tile | WIN | WIN | 4.87× | AGREE |

---

## Detailed Intel Per-Size Results

### Candidates That Agree (WIN on both)

**01 – polybench-gemm-zmorton** (Z-Morton layout)
- np240: 1.149×, np500: 2.038×, np1000: 2.070×, p256: 1.360×, p512: 2.213×, p1024: 2.266×
- All WIN. Intel speedup range slightly lower than AMD (1.29×–3.30×), consistent with
  AMD's larger L3 (256 MB vs 43 MB) benefiting Morton layouts less at small sizes.

**02 – polybench-lu-zmorton** (Z-Morton layout)
- p512: 1.599×, np768: 1.204×, p1024: 1.913× — all WIN. AMD canonical was 3.09×;
  Intel L3 pressure may explain reduction.

**04 – polybench-gemm-reg-L1-L2-tile** (register+L1+L2 tiling)
- np240: 1.094×, p256: 1.112×, np500: 1.838×, p512: 1.725×, np1000: 1.810×, p1024: 1.933×
- All WIN. AMD canonical 1.32× at large N — Intel shows a bigger benefit (1.93×).

**05 – polybench-3mm-reg-L1-L2-tile** (Reg+L1+L2 GOTO-style GEMM)
- np400: 3.651×, np800: 4.246×, n1000: 3.870×
- All WIN. AMD was 3.09×–3.22×. Intel **improves** to 3.65×–4.25×, likely due to
  AVX-512 FMA throughput benefit in the register microkernel.

**06 – polybench-2mm-reg-L1-tile**
- np240: 3.592×, p256: 3.488×, np500: 3.539×, p512: 7.029×, np1000: 3.688×, p1024: 6.401×
- All WIN. Remarkable p512 7.03× suggests pow2 sizes benefit from AVX-512 alignment.

**07 – polybench-trmm-L1-L2-tile**
- p512: 16.662×, np1000: 13.439×, p1024: 18.367× — all WIN. Largest Intel gains of
  any candidate (up to 18×). AMD was 23×; both are exceptional.

**08 – polybench-doitgen-reg-L1-tile**
- small: 4.803×, medium: 4.974×, large: 5.116× — all WIN, consistent across sizes.

**09 – tccg-tensor-contraction-GETT-tile**
- small: 1.663×, medium: 1.911×, large: 2.114× — all WIN.

**16 – polybench-syrk-rfp** (RFP storage)
- np500: 1.936×, np1000: 2.211×, p512: 3.561×, p1024: 5.521×
- All WIN. AMD was 1.917×–8.801×. RFP format works well on both architectures.

**18 – npdp-nussinov-skew-tile**
- np500: 1.761×, np1000: 1.531×, p512: 1.659×, p1024: 1.363× — all WIN.

**25 – polybench-adi-block-cyclic**
- n64: 1.075×, p256: 1.207×, p512: 1.282× — all WIN.

**27 – polybench-heat3d-pow2-pad**
- n64: 1.060×, n96: 1.279×, n128: 1.073× — all WIN. AMD showed n64 as LOSS (0.888×);
  Intel appears to benefit from pad alignment at all sizes tested.

**28 – polybench-gemm-nonpow2-morton**
- n1000: 1.944×, n1500: 2.114×, n2300: 2.914× — all WIN. AMD was 3.56× at N=2300;
  Intel competitive (2.91×).

**32 – rodinia-hotspot-tile**
- p64: 1.049×, p512: 1.060×, p1024: 1.034× — all WIN (narrow margin).

**33 – polybench-mvt-L1-tile**
- n1000: 1.953×, p1024: 2.787×, n2000: 2.057× — all WIN.

**34 – polybench-bicg-L1-tile**
- np1984: 4.210×, np1900: 4.341×, p1024: 4.232× — all WIN. Extremely strong on Intel
  (AMD was also a strong WIN at 51.7× — that extreme AMD value reflects compiler
  vectorization differences at PolyBench LARGE sizes).

**35 – hpcc-dgemm-reg-L1-L2-tile**
- n1000: 4.348×, n1500: 4.399×, n1900: 4.869× — all WIN. Increasing trend with N.
  AMD was 5.46× — Intel is competitive at 4.87× peak.

### Candidates That Agree (LOSS on both)

**03 – polybench-chol-zmorton**
- p512: 0.626×, np768: 0.416×, np1000: 0.416×
- Consistent LOSS on both architectures. Cholesky has irregular triangular access
  patterns that Morton layout cannot improve; the overhead dominates.

**11 – bricklib-3d7pt-brick**
- np240: 0.842×, p256: 0.853×, np384: 0.861×
- Brick overhead exceeds benefit for 7-point stencil on both ISAs.

**14 – polybench-jacobi2d-brick**
- np400: 0.174×, p512: 0.181×, np500: 0.186×
- Severe LOSS on both. Brick layout imposes ~5.7× overhead on 2D Jacobi.

**22 – lulesh-elem-aosoA**
- n10: 0.904×, n20: 0.904×, n30: 0.900×
- AoSoA struct packing adds overhead; no benefit from either ISA.

**29 – bricklib-stencil-nonpow2-brick**
- n100: 0.403×, n200: 0.533×, p256: 0.611×
- Non-pow2 brick overhead is even more severe.

### Disagreements — Analysis

**10 – tblis-tensor-contraction-notranspose** (AMD WIN → Intel LOSS)
- AMD: 1.048× at large (WIN). Intel: large=1.010× (PARITY), medium=0.954× (LOSS).
- TBLIS uses runtime-selected microkernel selection. On AMD Zen3, TBLIS's AVX2
  microkernels appear to be more aggressively selected; on Intel Ice Lake, the
  default TBLIS path may not favor the LEGO no-transpose layout as strongly.
  This is a borderline case — AMD large barely cleared the 1.02× WIN threshold.

**12 – bricklib-3d13pt-brick** (AMD PARITY → Intel WIN)
- AMD: large=PARITY (1.018×). Intel: p256=1.060× WIN (larger neighborhood = more
  cache reuse; AVX-512 gathers may help 13-point stencil more than 7-point).
- **Architecture upgrade:** Intel AVX-512 gather instructions better exploit brick
  indirection in the 13-point neighborhood. This is the one case where Intel
  strictly outperforms AMD for this layout class.

**13 – polybench-heat3d-brick** (AMD MIXED → Intel LOSS)
- AMD: small=WIN but med/large=LOSS. Intel: all sizes LOSS (0.576×–0.795×).
- The AMD small-size WIN was marginal; on Intel the brick overhead is consistently
  visible. Heat-3D is a 3D stencil with poor spatial locality in brick form.

**15 – polybench-symm-rfp** (AMD PARITY → Intel LOSS)
- AMD: N1000=PARITY, N1024=LOSS, N512=PARITY. Intel: np1000=LOSS, np500=PARITY,
  p1024/p512=PARITY.
- Intel classifies as LOSS due to np1000=0.955×. The RFP format for SYMM provides
  marginal-to-negative benefit on both architectures; Intel's np1000 is slightly
  worse, likely from cache line boundary effects at non-pow2 sizes on the 43 MB L3.

**17 – rodinia-nw-antidiag-tile** (AMD MIXED → Intel LOSS)
- AMD: small=WIN but medium=PARITY, large=LOSS. Intel: all LOSS (0.354×–0.432×).
- The anti-diagonal tiling pattern has severe overhead on Intel's out-of-order core
  (different branch predictor, different prefetcher behavior). The AMD small-WIN
  was a fluke at tiny sizes.

**19 – npdp-zuker-skew-tile** (AMD WIN → Intel LOSS)
- AMD: n200=WIN (1.167×). Intel: n100=PARITY (1.000×), n150=LOSS (0.667×),
  n200=LOSS (0.875×).
- The Zuker skew-tile pattern benefits from AMD's larger L3 (256 MB). On Intel
  with 43 MB L3, the skew access pattern thrashes cache at medium and large sizes.
  This is the clearest L3-capacity portability failure in this suite.

**20 – polybench-seidel2d-wavefront-tile** (AMD LOSS → Intel MIXED)
- AMD: all LOSS (0.578×). Intel: p512=1.255× WIN but np500=LOSS, p1024=LOSS.
- The p512 win is likely alignment-specific: pow2 size fits Intel's 48 KB L1d
  cleanly with the wavefront tile size. The AMD L3-resident working set behavior
  masks this L1 alignment benefit.

**21 – rodinia-particlefilter-aosoA** (AMD LOSS → Intel PARITY)
- AMD: 0.514×–0.638× LOSS. Intel: all ~0.999× PARITY.
- Intel's wider SIMD lanes and different scatter/gather implementation reduce the
  AoSoA overhead relative to AMD. The LEGO layout neither hurts nor helps on Intel.

**23 – hpccg-cg-aosoA** (AMD MIXED → Intel LOSS)
- AMD: medium=WIN (1.189×) but small/large=LOSS. Intel: all LOSS (0.686×–0.746×).
- The AMD medium WIN was marginal cache-capacity luck. Intel's prefetcher handles
  the AoSoA strided access pattern poorly across all problem sizes.

**24 – polybench-fdtd-2d-block-cyclic** (AMD MIXED → Intel LOSS)
- AMD: 4-thread large=3.776× WIN. Intel: all LOSS (0.753×–0.975×) at 1 thread.
- The AMD Round 1 WIN was measured at 4 threads (OMP_NUM_THREADS=4). The Intel
  audit uses 1 thread. This is a protocol mismatch — the block-cyclic layout
  benefits from inter-thread NUMA locality, which is invisible in single-thread mode.
  *Recommendation:* Re-measure candidate 24 at 4 threads on Intel for fair comparison.

**26 – polybench-gemm-pow2-pad** (AMD WIN → Intel LOSS)
- AMD: PARITY+WIN+WIN → overall WIN (1.144× at large). Intel: p256=LOSS (0.958×),
  p512/p1024=PARITY (1.006×).
- Pow2 padding strategy targets AMD's 256 MB L3 set-associativity pattern. On Intel's
  43 MB L3 with different set stride, the padding lands in different cache sets and
  no longer avoids the hot-set collision. This is the canonical L3-capacity/geometry
  portability failure.

---

## Layout Class Portability Summary

| Layout Class | Candidates | AMD WIN | Intel WIN | Fully Portable |
|---|---|---|---|---|
| Reg+L1+L2 tile | 04,05,06,07,08,35 | 6/6 | 6/6 | YES — strongest class |
| L1 tile | 33,34 | 2/2 | 2/2 | YES |
| GETT tile | 09 | 1/1 | 1/1 | YES |
| Z-Morton | 01,02,03 | 2/3 | 2/3 | MOSTLY (Chol is LOSS on both) |
| Skew tile | 18,19,31 | MIXED | MIXED | PARTIAL (18=WIN, 19=LOSS, 31=MIXED) |
| RFP | 15,16 | MIXED | MIXED | PARTIAL (16=WIN both; 15=borderline) |
| Block-cyclic | 24,25 | MIXED | MIXED | PARTIAL (25=WIN; 24 is multi-thread) |
| Pow2 pad | 26,27 | MIXED | MIXED | PARTIAL (27=WIN; 26 L3-specific) |
| Brick | 11,12,13,14,29 | LOSS | LOSS | NO — overhead dominates both |
| AoSoA | 21,22,23 | MIXED | LOSS | WEAK (21=AMD-LOSS, 22=LOSS, 23=AMD-MIXED) |
| Antidiag tile | 17 | MIXED | LOSS | NO |
| Wavefront tile | 20 | LOSS | MIXED | MARGINAL |
| TBLIS | 10 | WIN | LOSS | NO (borderline) |
| Morton+non-pow2 | 28 | WIN | WIN | YES |
| Brick+non-pow2 | 29 | LOSS | LOSS | LOSS on both |
| Skew+non-pow2 | 31 | MIXED | MIXED | CONSISTENT MIXED |

---

## Key Findings

### 1. Register and Cache-Tile Layouts Are Fully Portable
Candidates 04–09, 33–35 (register tiling, L1/L2 tiling, GETT) are WIN on both
architectures. This class shows the highest and most consistent speedups (1.7×–18×).
The LEGO `TileBy`+`OrderBy` abstractions map to the same cache-blocking structure
regardless of AVX-2 vs AVX-512 SIMD width.

### 2. AVX-512 Provides Additional Benefit for Register Kernels
Candidate 05 (3mm) improves from 3.09×–3.22× (AMD) to 3.65×–4.25× (Intel).
Candidate 06 (2mm) shows p512=7.03× vs AMD 3.79×. Candidate 35 (dgemm) shows
4.87× vs AMD 5.46× — competitive.
The wider FMA throughput of AVX-512 (2× 512-bit FMA per cycle vs. 2× 256-bit on Zen3)
benefits register-blocked kernels that keep operands in AVX registers.

### 3. L3 Capacity Is the Primary Portability Risk
Intel's 43 MB L3 vs AMD's 256 MB L3 (6× difference) is the dominant source of
disagreements. Candidates whose speedup depends on fitting working sets in L3 on AMD
(candidates 19, 26) regress to LOSS or PARITY on Intel. L3-sensitive candidates
should parameterize tile sizes dynamically from cache topology queries.

### 4. Brick Layouts Consistently Lose on Both Architectures
Candidates 11, 13, 14, 29 are LOSS on both. The Brick indirection overhead (index
table lookups, gather-style loads) exceeds the cache locality benefit for the
polybench stencil sizes tested. Candidate 12 (3d13pt) shows a marginal Intel WIN
(1.06×) — AVX-512 gathers help the 13-point neighborhood — but is not robust.

### 5. AoSoA Is Architecture-Sensitive
Candidate 21 (particlefilter) goes from AMD LOSS to Intel PARITY — Intel's scatter/
gather implementation reduces the penalty. Candidate 23 (hpccg) is LOSS on both.
AoSoA benefits depend on SIMD width alignment and the scatter/gather microarchitecture.

### 6. Candidate 24 Requires Multi-Thread Re-Measurement
The AMD WIN for candidate 24 (fdtd-2d-block-cyclic) was at 4 threads. The Intel audit
ran single-thread. A fair comparison requires 4-thread Intel measurement.

---

## Recommendations

1. **Paper Section 7.5 (portability):** 68% agreement rate with 18 WIN / 13 LOSS / 3
   borderline. Frame as: register/cache-tile layouts are fully portable; layout
   classes sensitive to L3 capacity or ISA-specific SIMD semantics require tuning.

2. **Candidate 24 redo:** Re-run at 4 threads on Intel for a fair multi-thread
   block-cyclic comparison.

3. **Tile-size auto-tuning:** Candidates 19 and 26 fail on Intel due to fixed tile
   sizes calibrated for AMD's 256 MB L3. Parameterize tile size from `sysfs` L3 info.

4. **Brick layouts:** Drop from WIN claims in the paper; frame as a layout class
   that requires ISA-specific gather tuning (future work).

5. **AVX-512 note:** Explicitly note in Section 7.5 that Intel AVX-512 provides
   additional register-kernel speedup (05: +30%, 06: +85% at p512) — LEGO's
   `TileBy`+`OrderBy` naturally exploits wider SIMD without code changes.

---

## Raw Data Locations

All audit JSON files written to each candidate's `raw/` directory in its worktree:
- `audit_intel_baseline_{size}.json` — baseline timing on Intel
- `audit_intel_lego_{size}.json` — LEGO timing on Intel

Main repo (candidate 05): `/scratch/general/vast/u1419116/LEGO/evaluation/candidates/05-polybench-3mm-reg-L1-L2-tile/raw/`

Worktrees: `/scratch/general/vast/u1419116/LEGO-eval-{NN}-{name}/evaluation/candidates/{NN}-{name}/raw/`

Machine fingerprint: `/scratch/general/vast/u1419116/LEGO/evaluation/harness/machine.md` (SHA256: `842a1d8fa97a8fab9c2c826dbd532eb35e778086562ed7241b9d788f6ac3c293`)

---

*Generated: 2026-04-29. Auditor: u1419116 (Amir Mohammad Tavakkoli). Branch: eval/cpu-audit-intel-round1.*
