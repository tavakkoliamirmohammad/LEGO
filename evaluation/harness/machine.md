# Measurement Node Fingerprint (Intel Xeon — audit pass)

This file captures the node every measurement in this round runs on.
It is generated once at scaffolding time. If the node changes (e.g. the
Slurm allocation moves), the harness's pre-measurement check refuses to
run and asks for a re-fingerprint.

NOTE: the original AMD EPYC 7513 fingerprint (used for the canonical
35-candidate Round 1 measurements) is preserved at
evaluation/harness/machine_amd_notch368.md. THIS file describes the
Intel Xeon Gold 6330 node where the cross-architecture audit pass is
being executed.

## Identity

- captured_at: 2026-04-29T03:57:06Z
- captured_by: u1419116
- hostname: notch343
- uname: Linux notch343 4.18.0-553.89.1.el8_10.x86_64 #1 SMP Fri Dec 12 10:42:53 UTC 2025 x86_64 x86_64 x86_64 GNU/Linux

## CPU

```
Architecture:        x86_64
CPU op-mode(s):      32-bit, 64-bit
Byte Order:          Little Endian
CPU(s):              112
On-line CPU(s) list: 0-111
Thread(s) per core:  2
Core(s) per socket:  28
Socket(s):           2
NUMA node(s):        4
Vendor ID:           GenuineIntel
CPU family:          6
Model:               106
Model name:          Intel(R) Xeon(R) Gold 6330 CPU @ 2.00GHz
Stepping:            6
CPU MHz:             2000.000
CPU max MHz:         3100.0000
CPU min MHz:         800.0000
BogoMIPS:            4000.00
Virtualization:      VT-x
L1d cache:           48K
L1i cache:           32K
L2 cache:            1280K
L3 cache:            43008K
NUMA node0 CPU(s):   0-13,56-69
NUMA node1 CPU(s):   14-27,70-83
NUMA node2 CPU(s):   28-41,84-97
NUMA node3 CPU(s):   42-55,98-111
Flags:               fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid dca sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb cat_l3 invpcid_single intel_ppin ssbd mba ibrs ibpb stibp ibrs_enhanced tpr_shadow vnmi flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb intel_pt avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local split_lock_detect wbnoinvd dtherm ida arat pln pts hwp hwp_act_window hwp_epp hwp_pkg_req avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg tme avx512_vpopcntdq la57 rdpid fsrm md_clear pconfig flush_l1d arch_capabilities
```

## NUMA

```
available: 4 nodes (0-3)
node 0 cpus: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 56 57 58 59 60 61 62 63 64 65 66 67 68 69
node 0 size: 64503 MB
node 0 free: 18588 MB
node 1 cpus: 14 15 16 17 18 19 20 21 22 23 24 25 26 27 70 71 72 73 74 75 76 77 78 79 80 81 82 83
node 1 size: 64058 MB
node 1 free: 15415 MB
node 2 cpus: 28 29 30 31 32 33 34 35 36 37 38 39 40 41 84 85 86 87 88 89 90 91 92 93 94 95 96 97
node 2 size: 64461 MB
node 2 free: 37478 MB
node 3 cpus: 42 43 44 45 46 47 48 49 50 51 52 53 54 55 98 99 100 101 102 103 104 105 106 107 108 109 110 111
node 3 size: 64501 MB
node 3 free: 29881 MB
node distances:
node   0   1   2   3 
  0:  10  20  20  20 
  1:  20  10  20  20 
  2:  20  20  10  20 
  3:  20  20  20  10 
```

## Memory

```
              total        used        free      shared  buff/cache   available
Mem:          251Gi       9.7Gi        98Gi       2.1Gi       142Gi       237Gi
Swap:          63Gi        59Mi        63Gi
```

## Frequency / governor / turbo (as observed; no sudo to modify)

- governor (cpu0): performance
- intel_pstate.no_turbo: 0
- cpufreq.scaling_max_freq: 3100000

## Compilers

- gcc: gcc (GCC) 8.5.0 20210514 (Red Hat 8.5.0-28)
- g++: g++ (GCC) 8.5.0 20210514 (Red Hat 8.5.0-28)
- gfortran: GNU Fortran (GCC) 8.5.0 20210514 (Red Hat 8.5.0-28)
- rustc: rustc 1.94.1 (e408947bf 2026-03-25)
- julia: unavailable

## GPU (recorded for opportunistic CUDA-C candidates only)

```
GPU 0: NVIDIA RTX A6000 (UUID: GPU-47efec96-e431-1ce7-aad4-29b74dbc8435)
```
