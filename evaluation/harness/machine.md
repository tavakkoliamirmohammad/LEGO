# Measurement Node Fingerprint

This file captures the node every measurement in this round runs on.
It is generated once at scaffolding time. If the node changes (e.g. the
Slurm allocation moves), the harness's pre-measurement check refuses to
run and asks for a re-fingerprint.

## Identity

- captured_at: 2026-04-28T18:08:10Z
- captured_by: u1419116
- hostname: notch368
- uname: Linux notch368 4.18.0-553.89.1.el8_10.x86_64 #1 SMP Fri Dec 12 10:42:53 UTC 2025 x86_64 x86_64 x86_64 GNU/Linux

## CPU

```
Architecture:        x86_64
CPU op-mode(s):      32-bit, 64-bit
Byte Order:          Little Endian
CPU(s):              128
On-line CPU(s) list: 0-127
Thread(s) per core:  2
Core(s) per socket:  32
Socket(s):           2
NUMA node(s):        8
Vendor ID:           AuthenticAMD
CPU family:          25
Model:               1
Model name:          AMD EPYC 7513 32-Core Processor
Stepping:            1
CPU MHz:             3673.223
CPU max MHz:         3681.6399
CPU min MHz:         1500.0000
BogoMIPS:            5199.73
Virtualization:      AMD-V
L1d cache:           32K
L1i cache:           32K
L2 cache:            512K
L3 cache:            32768K
NUMA node0 CPU(s):   0-7,64-71
NUMA node1 CPU(s):   8-15,72-79
NUMA node2 CPU(s):   16-23,80-87
NUMA node3 CPU(s):   24-31,88-95
NUMA node4 CPU(s):   32-39,96-103
NUMA node5 CPU(s):   40-47,104-111
NUMA node6 CPU(s):   48-55,112-119
NUMA node7 CPU(s):   56-63,120-127
Flags:               fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good nopl nonstop_tsc cpuid extd_apicid aperfmperf pni pclmulqdq monitor ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 invpcid_single hw_pstate ssbd mba ibrs ibpb stibp vmmcall fsgsbase bmi1 avx2 smep bmi2 invpcid cqm rdt_a rdseed adx smap clflushopt clwb sha_ni xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local clzero irperf xsaveerptr wbnoinvd amd_ppin brs arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold v_vmsave_vmload vgif v_spec_ctrl umip pku ospke vaes vpclmulqdq rdpid overflow_recov succor smca
```

## NUMA

```
available: 8 nodes (0-7)
node 0 cpus: 0 1 2 3 4 5 6 7 64 65 66 67 68 69 70 71
node 0 size: 31838 MB
node 0 free: 19332 MB
node 1 cpus: 8 9 10 11 12 13 14 15 72 73 74 75 76 77 78 79
node 1 size: 32203 MB
node 1 free: 25433 MB
node 2 cpus: 16 17 18 19 20 21 22 23 80 81 82 83 84 85 86 87
node 2 size: 32247 MB
node 2 free: 18745 MB
node 3 cpus: 24 25 26 27 28 29 30 31 88 89 90 91 92 93 94 95
node 3 size: 32235 MB
node 3 free: 21504 MB
node 4 cpus: 32 33 34 35 36 37 38 39 96 97 98 99 100 101 102 103
node 4 size: 32247 MB
node 4 free: 9040 MB
node 5 cpus: 40 41 42 43 44 45 46 47 104 105 106 107 108 109 110 111
node 5 size: 32247 MB
node 5 free: 12272 MB
node 6 cpus: 48 49 50 51 52 53 54 55 112 113 114 115 116 117 118 119
node 6 size: 32247 MB
node 6 free: 14538 MB
node 7 cpus: 56 57 58 59 60 61 62 63 120 121 122 123 124 125 126 127
node 7 size: 32235 MB
node 7 free: 16033 MB
node distances:
node   0   1   2   3   4   5   6   7 
  0:  10  12  12  12  32  32  32  32 
  1:  12  10  12  12  32  32  32  32 
  2:  12  12  10  12  32  32  32  32 
  3:  12  12  12  10  32  32  32  32 
  4:  32  32  32  32  10  12  12  12 
  5:  32  32  32  32  12  10  12  12 
  6:  32  32  32  32  12  12  10  12 
  7:  32  32  32  32  12  12  12  10 
```

## Memory

```
              total        used        free      shared  buff/cache   available
Mem:          251Gi        28Gi       133Gi        16Gi        89Gi       204Gi
Swap:          63Gi       2.8Gi        61Gi
```

## Frequency / governor / turbo (as observed; no sudo to modify)

- governor (cpu0): performance
- intel_pstate.no_turbo: unknown
- cpufreq.scaling_max_freq: 3681640

## Compilers

- gcc: gcc (GCC) 8.5.0 20210514 (Red Hat 8.5.0-28)
- g++: g++ (GCC) 8.5.0 20210514 (Red Hat 8.5.0-28)
- gfortran: GNU Fortran (GCC) 8.5.0 20210514 (Red Hat 8.5.0-28)
- rustc: rustc 1.94.1 (e408947bf 2026-03-25)
- julia: unavailable

## GPU (recorded for opportunistic CUDA-C candidates only)

```
GPU 0: NVIDIA RTX A6000 (UUID: GPU-5836d38a-9e30-2868-0e4d-dc0759c2c366)
```
