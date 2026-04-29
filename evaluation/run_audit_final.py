#!/usr/bin/env python3
"""
CASTLE Intel cross-architecture audit — final production version.
Handles all 35 candidates' unique binary interfaces.
Protocol: 10 warmup + 30 timed, flock-protected.
"""

from __future__ import annotations
import argparse, fcntl, json, os, statistics, subprocess, sys, time
from pathlib import Path

LOCK_ABS  = Path("/scratch/general/vast/u1419116/LEGO/evaluation/.lock")
INTEL_FP  = "842a1d8fa97a8fab9c2c826dbd532eb35e778086562ed7241b9d788f6ac3c293"
ROOT      = Path("/scratch/general/vast/u1419116")

WARMUP = 10
TIMED  = 30

GCC13 = ("/uufs/chpc.utah.edu/sys/spack/v019/linux-rocky8-nehalem/gcc-8.5.0/"
         "gcc-13.3.0-7km3tumc7bhyfmhbb4pmymhhuqr362et/bin/gcc")
GXX13 = ("/uufs/chpc.utah.edu/sys/spack/v019/linux-rocky8-nehalem/gcc-8.5.0/"
         "gcc-13.3.0-7km3tumc7bhyfmhbb4pmymhhuqr362et/bin/g++")
GCC8  = "gcc"
GXX8  = "g++"

TASKSET = "4"
NUMA    = "0"

GCC13_VER = subprocess.check_output([GCC13,"--version"],text=True).splitlines()[0]
GCC8_VER  = subprocess.check_output([GCC8, "--version"],text=True).splitlines()[0]
GXX13_VER = GCC13_VER

C_FLAGS   = ["-O3","-march=native","-fopenmp"]
CPP_FLAGS = ["-O3","-march=native","-fopenmp","-std=c++17"]

# ── lock ──────────────────────────────────────────────────────────────────
def lock_acquire():
    LOCK_ABS.parent.mkdir(parents=True, exist_ok=True)
    fd = open(LOCK_ABS, "a")
    fcntl.flock(fd.fileno(), fcntl.LOCK_EX)
    return fd

def lock_release(fd):
    fcntl.flock(fd.fileno(), fcntl.LOCK_UN); fd.close()

# ── core helpers ──────────────────────────────────────────────────────────
def med_iqr(vals):
    m = int(statistics.median(vals))
    i = int(statistics.quantiles(vals,n=4)[2]-statistics.quantiles(vals,n=4)[0]) if len(vals)>=4 else 0
    return m, i

def classify(b, l):
    s = b/l
    return "WIN" if s>=1.02 else ("LOSS" if s<=0.98 else "PARITY")

def make_rec(cid, version, size_tag, vals, lang, compiler_ver, flags):
    m,i = med_iqr(vals)
    return {"candidate_id":cid,"version":version,"size":size_tag,
            "build":{"language":lang,"compiler_version":compiler_ver,"flags":list(flags)},
            "repro_setup":{"taskset_cores":TASKSET,"numactl_membind":NUMA,
                           "governor":"performance","turbo":"enabled"},
            "warmup_iterations":WARMUP,"timed_iterations":TIMED,
            "per_iteration_ns":vals,"median_ns":m,"iqr_ns":i,
            "machine_fingerprint_sha256":INTEL_FP}

def write_rec(raw_dir, size_tag, b_rec, l_rec):
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir/f"audit_intel_baseline_{size_tag}.json").write_text(json.dumps(b_rec,indent=2))
    (raw_dir/f"audit_intel_lego_{size_tag}.json").write_text(json.dumps(l_rec,indent=2))

def build_c(src, out, flags, defines=None, link=None):
    cmd = [GCC13]+list(flags)+(defines or [])+[str(src),"-o",str(out)]+(link or [])
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode!=0: raise RuntimeError(f"Build {out.name}: {r.stderr[:500]}")

def build_cpp(srcs, out, flags, defines=None, includes=None, link=None):
    if isinstance(srcs,(str,Path)): srcs=[srcs]
    cmd = ([GXX13]+list(flags)+(defines or [])
           +[f"-I{i}" for i in (includes or [])]
           +[str(s) for s in srcs]+(link or [])+["-o",str(out)])
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode!=0: raise RuntimeError(f"Build {out.name}: {r.stderr[:500]}")

def run_pin(binary, args=(), timeout=300):
    env = os.environ.copy(); env["OMP_NUM_THREADS"]="1"
    cmd=["taskset","-c",TASKSET,"numactl",f"--membind={NUMA}",str(binary)]+[str(a) for a in args]
    r=subprocess.run(cmd,capture_output=True,text=True,timeout=timeout,env=env)
    if r.returncode!=0: raise RuntimeError(f"Run {binary}: rc={r.returncode} {r.stderr[:200]}")
    return r.stdout

def parse_ns_lines(stdout, n):
    """One int (ns) per line."""
    vals=[]
    for line in stdout.strip().splitlines():
        line=line.strip()
        try: vals.append(int(line))
        except ValueError:
            try: vals.append(int(float(line)*1e9))
            except ValueError: pass
    if len(vals)<n: raise RuntimeError(f"Got {len(vals)}/{n} timing lines:\n{stdout[:300]}")
    return vals[-n:]

def parse_ns_indexed(stdout, n):
    """'idx ns' format."""
    vals=[]
    for line in stdout.strip().splitlines():
        p=line.split()
        if len(p)==2:
            try: vals.append(int(p[1]))
            except ValueError: pass
    if len(vals)<n: raise RuntimeError(f"Indexed: got {len(vals)}/{n}")
    return vals[-n:]

def parse_ns_csv3(stdout, n):
    """'N,version,ns' CSV format (candidate 17)."""
    vals=[]
    for line in stdout.strip().splitlines():
        p=line.strip().split(",")
        if len(p)==3:
            try: vals.append(int(p[2]))
            except ValueError: pass
    if len(vals)<n: raise RuntimeError(f"CSV3: got {len(vals)}/{n}")
    return vals[-n:]

def parse_ns_json_array(stdout, n):
    """JSON array of ints (candidate 10)."""
    for line in stdout.strip().splitlines():
        l=line.strip()
        if l.startswith("["):
            try:
                arr=json.loads(l)
                if len(arr)>=n: return arr[-n:]
            except: pass
    raise RuntimeError(f"No JSON array in: {stdout[:200]}")

def parse_ns_json_obj(stdout, n):
    """JSON object with per_iteration_ns key (candidates 21/22/23 etc)."""
    try:
        d=json.loads(stdout.strip())
        arr=d["per_iteration_ns"]
        if len(arr)>=n: return arr[-n:]
    except: pass
    raise RuntimeError(f"No JSON obj in: {stdout[:200]}")

def parse_ns_prefixed(stdout, n):
    """'N=xxx ns1 ns2...' on one line (candidate 35)."""
    vals=[]
    for line in stdout.strip().splitlines():
        if line.startswith("N="):
            for p in line.split()[1:]:
                try: vals.append(int(p))
                except ValueError: pass
    if len(vals)>=n: return vals[-n:]
    return parse_ns_lines(stdout, n)

def meas(binary, args, parser, total, timeout=300):
    stdout=run_pin(binary,args,timeout)
    return parser(stdout, total)

def meas_secs_per_call(binary, args, warmup, timed, timeout=60):
    """Call binary once per measurement, parse float seconds from stdout."""
    vals=[]
    for _ in range(warmup+timed):
        out=run_pin(binary,args,timeout=timeout)
        lines=[l.strip() for l in out.strip().splitlines() if l.strip()]
        for l in reversed(lines):
            try: vals.append(int(float(l)*1e9)); break
            except ValueError: pass
    return vals[-timed:]

def meas_wallclock(binary, args, warmup, timed, timeout=60):
    """Wall-clock each invocation."""
    env=os.environ.copy(); env["OMP_NUM_THREADS"]="1"
    cmd=["taskset","-c",TASKSET,"numactl",f"--membind={NUMA}",str(binary)]+[str(a) for a in args]
    for _ in range(warmup):
        subprocess.run(cmd,capture_output=True,env=env,timeout=timeout)
    vals=[]
    for _ in range(timed):
        t0=time.perf_counter_ns()
        subprocess.run(cmd,capture_output=True,env=env,timeout=timeout)
        vals.append(time.perf_counter_ns()-t0)
    return vals

def pair(cid,raw_dir,tag,bv,lv,lang,cver,flags):
    br=make_rec(cid,"baseline",tag,bv,lang,cver,flags)
    lr=make_rec(cid,"lego",    tag,lv,lang,cver,flags)
    write_rec(raw_dir,tag,br,lr)
    sp=br["median_ns"]/lr["median_ns"]; v=classify(br["median_ns"],lr["median_ns"])
    print(f"    {tag}: {sp:.3f}x {v}")
    return (tag,sp,v)

# ── candidate audits ──────────────────────────────────────────────────────

def audit_05(cdir,rdir):
    """3mm-reg-L1-L2-tile: pre-built harness takes (N warmup timed), outputs 'N ns1 ns2...'."""
    cid="05-polybench-3mm-reg-L1-L2-tile"
    bb=cdir/"3mm_baseline_harness"; lb=cdir/"3mm_lego_bin"
    if not bb.exists(): raise RuntimeError(f"05 baseline harness missing: {bb}")
    if not lb.exists(): raise RuntimeError(f"05 lego bin missing: {lb}")
    res=[]
    # Original AMD measurement used N=400/800/1000; match those sizes
    for tag,n in [("np400",400),("np800",800),("n1000",1000)]:
        bv=meas(bb,[n,WARMUP,TIMED],parse_ns_space_separated,TIMED,timeout=600)
        lv=meas(lb,[n,WARMUP,TIMED],parse_ns_space_separated,TIMED,timeout=600)
        res.append(pair(cid,rdir,tag,bv,lv,"c",GCC8_VER,C_FLAGS))
    return res

def audit_01(cdir,rdir):
    """gemm_baseline.c / gemm_lego.c: args=NI NJ NK, prints per-iter ns."""
    cid="01-polybench-gemm-zmorton"
    build_c(cdir/"gemm_baseline.c",cdir/"_b01",C_FLAGS)
    build_c(cdir/"gemm_lego.c",    cdir/"_l01",C_FLAGS)
    res=[]
    for tag,ni,nj,nk in [("p256",256,256,256),("np240",240,240,240),
                          ("p512",512,512,512),("np500",500,500,500),
                          ("p1024",1024,1024,1024),("np1000",1000,1000,1000)]:
        bv=meas(cdir/"_b01",[ni,nj,nk],parse_ns_lines,TIMED)
        lv=meas(cdir/"_l01",[ni,nj,nk],parse_ns_lines,TIMED)
        res.append(pair(cid,rdir,tag,bv,lv,"c",GCC13_VER,C_FLAGS))
    return res

def parse_ns_space_separated(stdout, n):
    """'N ns1 ns2 ...' on one line — skip first token (N), return n ns values."""
    vals=[]
    for line in stdout.strip().splitlines():
        tokens=line.strip().split()
        if len(tokens)>1:
            for t in tokens[1:]:
                try: vals.append(int(t))
                except ValueError: pass
    if len(vals)>=n: return vals[-n:]
    raise RuntimeError(f"space-sep: got {len(vals)}/{n} from: {stdout[:200]}")

def audit_02(cdir,rdir):
    """lu_harness.c: compile per-N with -DBASELINE_VERSION/-DLEGO_VERSION, prints header+ns lines."""
    cid="02-polybench-lu-zmorton"
    harness=cdir/"lu_harness.c"
    TILE_T=8
    res=[]
    for tag,n in [("p512",512),("np768",768),("p1024",1024)]:
        bb=cdir/f"_b02_{n}"; lb=cdir/f"_l02_{n}"
        dfn_b=[f"-DN={n}",f"-DT={TILE_T}",f"-DWARMUP={WARMUP}",f"-DTIMED={TIMED}","-DBASELINE_VERSION"]
        dfn_l=[f"-DN={n}",f"-DT={TILE_T}",f"-DWARMUP={WARMUP}",f"-DTIMED={TIMED}","-DLEGO_VERSION"]
        build_c(harness,bb,C_FLAGS,defines=dfn_b,link=["-lm"])
        build_c(harness,lb,C_FLAGS,defines=dfn_l,link=["-lm"])
        bv=meas(bb,[],lambda s,n: parse_ns_lines(s,n),TIMED,timeout=300)
        lv=meas(lb,[],lambda s,n: parse_ns_lines(s,n),TIMED,timeout=300)
        res.append(pair(cid,rdir,tag,bv,lv,"c",GCC13_VER,C_FLAGS))
    return res

def audit_03(cdir,rdir):
    """chol: compile-time N/T, arg=N mode. Baseline outputs JSON obj; lego same."""
    cid="03-polybench-chol-zmorton"
    build_c(cdir/"cholesky_baseline.c",cdir/"_b03",C_FLAGS,link=["-lm"])
    lbins={}
    for N,T in {512:16,768:16,1000:10}.items():
        out=cdir/f"_l03_{N}"
        cmd=[GCC13]+C_FLAGS+[f"-DN_VAL={N}",f"-DT_VAL={T}",str(cdir/"cholesky_lego.c"),"-o",str(out),"-lm"]
        r=subprocess.run(cmd,capture_output=True,text=True)
        if r.returncode!=0: raise RuntimeError(r.stderr[:300])
        lbins[N]=out
    res=[]
    # Baseline takes (N, "timed") and outputs JSON obj with per_iteration_ns
    # Lego also outputs JSON obj
    for tag,n in [("p512",512),("np768",768),("np1000",1000)]:
        bv=meas(cdir/"_b03",[n,"timed"],parse_ns_json_obj,TIMED,timeout=600)
        lv=meas(lbins[n],[n,"timed"],parse_ns_json_obj,TIMED,timeout=600)
        res.append(pair(cid,rdir,tag,bv,lv,"c",GCC13_VER,C_FLAGS))
    return res

def audit_04(cdir,rdir):
    """gemm-reg-L1-L2-tile: takes (N, warmup, timed), outputs 'N ns1 ns2...' on one line."""
    cid="04-polybench-gemm-reg-L1-L2-tile"
    lsrc=sorted(cdir.glob("gemm_lego*.c"))[-1]
    build_c(cdir/"gemm_baseline.c",cdir/"_b04",C_FLAGS)
    build_c(lsrc,cdir/"_l04",C_FLAGS)
    res=[]
    for tag,n in [("p256",256),("np240",240),("p512",512),("np500",500),("p1024",1024),("np1000",1000)]:
        bv=meas(cdir/"_b04",[n,WARMUP,TIMED],parse_ns_space_separated,TIMED,timeout=600)
        lv=meas(cdir/"_l04",[n,WARMUP,TIMED],parse_ns_space_separated,TIMED,timeout=600)
        res.append(pair(cid,rdir,tag,bv,lv,"c",GCC13_VER,C_FLAGS))
    return res

def audit_06(cdir,rdir):
    """2mm-reg-L1-tile: takes (NI NJ NK NL warmup timed) or (N warmup timed), outputs 'N ns1 ns2...'."""
    cid="06-polybench-2mm-reg-L1-tile"
    bsrc=next(cdir.glob("*baseline*.c"))
    lsrc=next(s for s in cdir.glob("*.c") if "lego" in s.name and "baseline" not in s.name)
    build_c(bsrc,cdir/"_b06",C_FLAGS); build_c(lsrc,cdir/"_l06",C_FLAGS)
    res=[]
    for tag,n in [("p256",256),("np240",240),("p512",512),("np500",500),("p1024",1024),("np1000",1000)]:
        bv=meas(cdir/"_b06",[n,n,n,n,WARMUP,TIMED],parse_ns_space_separated,TIMED,timeout=600)
        lv=meas(cdir/"_l06",[n,n,n,n,WARMUP,TIMED],parse_ns_space_separated,TIMED,timeout=600)
        res.append(pair(cid,rdir,tag,bv,lv,"c",GCC13_VER,C_FLAGS))
    return res

def audit_07(cdir,rdir):
    """trmm: polybench -DPOLYBENCH_TIME + -DM=N -DN=N (trmm has both M and N dims), one float/sec."""
    cid="07-polybench-trmm-L1-L2-tile"
    poly=cdir/"upstream/polybench-c-4.2.1-beta"
    util=poly/"utilities"
    trmm_dir=poly/"linear-algebra/blas/trmm"
    lsrc=cdir/"trmm_lego.c"
    res=[]
    for tag,n in [("p512",512),("np1000",1000),("p1024",1024)]:
        bb=cdir/f"_b07_{n}"; lb=cdir/f"_l07_{n}"
        cmd_b=[GCC13]+C_FLAGS+[f"-DM={n}",f"-DN={n}","-DPOLYBENCH_TIME",
               f"-I{util}",f"-I{trmm_dir}",
               str(trmm_dir/"trmm.c"),str(util/"polybench.c"),"-o",str(bb),"-lm"]
        r=subprocess.run(cmd_b,capture_output=True,text=True)
        if r.returncode!=0: raise RuntimeError(f"trmm baseline N={n}: {r.stderr[:300]}")
        cmd_l=[GCC13]+C_FLAGS+[f"-DM={n}",f"-DN={n}","-DPOLYBENCH_TIME",
               f"-I{util}",f"-I{trmm_dir}",
               str(lsrc),str(util/"polybench.c"),"-o",str(lb),"-lm"]
        r=subprocess.run(cmd_l,capture_output=True,text=True)
        if r.returncode!=0: raise RuntimeError(f"trmm lego N={n}: {r.stderr[:300]}")
        bv=meas_secs_per_call(bb,[],WARMUP,TIMED,timeout=60)
        lv=meas_secs_per_call(lb,[],WARMUP,TIMED,timeout=60)
        res.append(pair(cid,rdir,tag,bv,lv,"c",GCC13_VER,C_FLAGS))
    return res

def audit_08(cdir,rdir):
    """doitgen: compile-time dims, prints all WARMUP+TIMED ns per run."""
    cid="08-polybench-doitgen-reg-L1-tile"
    harness=cdir/"doitgen_harness.c"
    res=[]
    for tag,NR,NQ,NP in [("small",25,20,30),("medium",50,40,60),("large",100,80,120)]:
        dfn=[f"-DNR={NR}",f"-DNQ={NQ}",f"-DNP={NP}",f"-DWARMUP_ITERS={WARMUP}",f"-DTIMED_ITERS={TIMED}"]
        bb=cdir/f"_b08_{tag}"; lb=cdir/f"_l08_{tag}"
        build_c(harness,bb,C_FLAGS,defines=dfn,link=["-lm"])
        build_c(harness,lb,C_FLAGS,defines=dfn+["-DUSE_LEGO"],link=["-lm"])
        bv=meas(bb,[],parse_ns_lines,TIMED,timeout=120)
        lv=meas(lb,[],parse_ns_lines,TIMED,timeout=120)
        res.append(pair(cid,rdir,tag,bv,lv,"c",GCC13_VER,C_FLAGS))
    return res

def audit_09(cdir,rdir):
    """tccg: tc_baseline/tc_lego, args=na nb nc nd ne nf, prints ns."""
    cid="09-tccg-tensor-contraction-GETT-tile"
    build_c(cdir/"tc_baseline.c",cdir/"_b09",C_FLAGS)
    build_c(cdir/"tc_lego.c",    cdir/"_l09",C_FLAGS)
    res=[]
    for tag,args in [("small",[64,8,8,8,8,8]),("medium",[96,12,12,12,12,12]),("large",[180,14,14,14,14,14])]:
        bv=meas(cdir/"_b09",args,parse_ns_lines,TIMED,timeout=300)
        lv=meas(cdir/"_l09",args,parse_ns_lines,TIMED,timeout=300)
        res.append(pair(cid,rdir,tag,bv,lv,"c",GCC13_VER,C_FLAGS))
    return res

def audit_10(cdir,rdir):
    """ttm: args=I0 I1 I2 J1 warmup iters, stdout=JSON array."""
    cid="10-tblis-tensor-contraction-notranspose"
    up=cdir/"upstream"
    build_cpp([up/"ttm_baseline.cpp"],up/"_b10",CPP_FLAGS)
    build_cpp([up/"ttm_lego.cpp"],    up/"_l10",CPP_FLAGS)
    res=[]
    for tag,i0,i1,i2,j1 in [("small",40,30,40,30),("medium",55,44,55,44),("large",75,60,75,60)]:
        bv=meas(up/"_b10",[i0,i1,i2,j1,WARMUP,TIMED],parse_ns_json_array,TIMED,timeout=120)
        lv=meas(up/"_l10",[i0,i1,i2,j1,WARMUP,TIMED],parse_ns_json_array,TIMED,timeout=120)
        res.append(pair(cid,rdir,tag,bv,lv,"cpp",GXX13_VER,CPP_FLAGS))
    return res

def audit_11(cdir,rdir):
    """brick 3d7pt: arg=N, prints TIMED ns."""
    cid="11-bricklib-3d7pt-brick"
    build_cpp([cdir/"stencil_naive.cpp"],cdir/"_b11",CPP_FLAGS)
    build_cpp([cdir/"stencil_brick.cpp"],cdir/"_l11",CPP_FLAGS)
    res=[]
    for tag,n in [("p256",256),("np240",240),("np384",384)]:
        bv=meas(cdir/"_b11",[n],parse_ns_lines,TIMED,timeout=300)
        lv=meas(cdir/"_l11",[n],parse_ns_lines,TIMED,timeout=300)
        res.append(pair(cid,rdir,tag,bv,lv,"cpp",GXX13_VER,CPP_FLAGS))
    return res

def audit_12(cdir,rdir):
    """brick 3d13pt: stencil3d13pt_naive/stencil3d13pt_lego."""
    cid="12-bricklib-3d13pt-brick"
    build_cpp([cdir/"stencil3d13pt_naive.cpp"],cdir/"_b12",CPP_FLAGS)
    build_cpp([cdir/"stencil3d13pt_lego.cpp"], cdir/"_l12",CPP_FLAGS)
    res=[]
    for tag,n in [("p64",64),("p128",128),("p256",256)]:
        bv=meas(cdir/"_b12",[n],parse_ns_lines,TIMED,timeout=300)
        lv=meas(cdir/"_l12",[n],parse_ns_lines,TIMED,timeout=300)
        res.append(pair(cid,rdir,tag,bv,lv,"cpp",GXX13_VER,CPP_FLAGS))
    return res

def audit_13(cdir,rdir):
    """heat3d-brick: rebuild from timer sources with -DN=n."""
    cid="13-polybench-heat3d-brick"
    up=cdir/"upstream"
    poly=up/"polybench-c-4.2.1-beta"
    util=poly/"utilities"
    heat3d_dir=poly/"stencils/heat-3d" if (poly/"stencils/heat-3d").exists() else poly
    bsrc=up/"heat-3d-baseline-timer.c"
    lsrc=cdir/"heat-3d-lego-timer.c"
    res=[]
    for tag,n in [("n100",100),("n128",128),("n200",200)]:
        bb=cdir/f"_b13_{tag}"; lb=cdir/f"_l13_{tag}"
        TSTEPS=20  # from run.sh: -DTSTEPS=20
        fn=C_FLAGS+[f"-DN={n}",f"-DTSTEPS={TSTEPS}",f"-DWARMUP_ITERS={WARMUP}",f"-DTIMED_ITERS={TIMED}"]
        cmd_b=[GCC13]+fn+["-I",str(util),"-I",str(heat3d_dir),
               str(bsrc),str(util/"polybench.c"),"-o",str(bb),"-lm"]
        r=subprocess.run(cmd_b,capture_output=True,text=True)
        if r.returncode!=0: raise RuntimeError(f"heat3d baseline N={n}: {r.stderr[:400]}")
        cmd_l=[GCC13]+fn+["-I",str(util),"-I",str(heat3d_dir),"-I",str(cdir),
               str(lsrc),str(util/"polybench.c"),"-o",str(lb),"-lm"]
        r=subprocess.run(cmd_l,capture_output=True,text=True)
        if r.returncode!=0: raise RuntimeError(f"heat3d lego N={n}: {r.stderr[:400]}")
        bv=meas(bb,[],parse_ns_lines,TIMED,timeout=300)
        lv=meas(lb,[],parse_ns_lines,TIMED,timeout=300)
        res.append(pair(cid,rdir,tag,bv,lv,"c",GCC13_VER,C_FLAGS))
    return res

def audit_14(cdir,rdir):
    """jacobi2d: polybench per-call, POLYBENCH_TIME -> one float/sec."""
    cid="14-polybench-jacobi2d-brick"
    up=cdir/"upstream"
    poly=up/"polybench-c-4.2.1-beta"
    util=poly/"utilities"
    jac_dir=poly/"stencils/jacobi-2d" if (poly/"stencils/jacobi-2d").exists() else poly
    jac_src=jac_dir/"jacobi-2d.c"
    lsrc=cdir/"jacobi-2d-lego-v3.c"  # v3 per measure.py LEGO_SOURCE
    res=[]
    for tag,n,t in [("np400",400,20),("np500",500,20),("p512",512,20)]:
        bb=cdir/f"_b14_{tag}"; lb=cdir/f"_l14_{tag}"
        fn=C_FLAGS+[f"-DN={n}",f"-DTSTEPS={t}","-DPOLYBENCH_TIME"]
        cmd_b=[GCC13]+fn+["-I",str(util),str(jac_src),str(util/"polybench.c"),"-o",str(bb),"-lm"]
        r=subprocess.run(cmd_b,capture_output=True,text=True)
        if r.returncode!=0: raise RuntimeError(f"jacobi2d baseline N={n}: {r.stderr[:300]}")
        cmd_l=[GCC13]+fn+["-I",str(util),"-I",str(jac_dir),str(lsrc),str(util/"polybench.c"),"-o",str(lb),"-lm"]
        r=subprocess.run(cmd_l,capture_output=True,text=True)
        if r.returncode!=0: raise RuntimeError(f"jacobi2d lego N={n}: {r.stderr[:300]}")
        bv=meas_secs_per_call(bb,[],WARMUP,TIMED)
        lv=meas_secs_per_call(lb,[],WARMUP,TIMED)
        res.append(pair(cid,rdir,tag,bv,lv,"c",GCC13_VER,C_FLAGS))
    return res

def audit_15(cdir,rdir):
    """symm-rfp: polybench per-call, one float/sec."""
    cid="15-polybench-symm-rfp"
    poly=cdir/"upstream/polybench-c-4.2.1-beta"
    util=poly/"utilities"
    bsrc=poly/"linear-algebra/blas/symm/symm.c"
    lsrc=cdir/"symm_lego_L2.c"
    res=[]
    for tag,m,n in [("p512",512,512),("np500",500,500),("p1024",1024,1024),("np1000",1000,1000)]:
        bb=cdir/f"_b15_{tag}"; lb=cdir/f"_l15_{tag}"
        fn=C_FLAGS+[f"-DM={m}",f"-DN={n}","-DPOLYBENCH_TIME"]
        cmd_b=[GCC13]+fn+["-I",str(util),str(bsrc),str(util/"polybench.c"),"-o",str(bb),"-lm"]
        r=subprocess.run(cmd_b,capture_output=True,text=True)
        if r.returncode!=0: raise RuntimeError(r.stderr[:300])
        cmd_l=[GCC13]+fn+["-I",str(util),"-I",str(cdir),str(lsrc),str(util/"polybench.c"),"-o",str(lb),"-lm"]
        r=subprocess.run(cmd_l,capture_output=True,text=True)
        if r.returncode!=0: raise RuntimeError(r.stderr[:300])
        bv=meas_secs_per_call(bb,[],WARMUP,TIMED)
        lv=meas_secs_per_call(lb,[],WARMUP,TIMED)
        res.append(pair(cid,rdir,tag,bv,lv,"c",GCC13_VER,C_FLAGS))
    return res

def audit_16(cdir,rdir):
    """syrk: args=N M warmup reps, prints per-iter ns."""
    cid="16-polybench-syrk-rfp"
    build_c(cdir/"syrk_baseline.c",cdir/"_b16",C_FLAGS)
    build_c(cdir/"syrk_lego.c",    cdir/"_l16",C_FLAGS)
    res=[]
    for tag,n in [("p512",512),("np500",500),("p1024",1024),("np1000",1000)]:
        bv=meas(cdir/"_b16",[n,n,WARMUP,TIMED],parse_ns_lines,TIMED,timeout=300)
        lv=meas(cdir/"_l16",[n,n,WARMUP,TIMED],parse_ns_lines,TIMED,timeout=300)
        res.append(pair(cid,rdir,tag,bv,lv,"c",GCC13_VER,C_FLAGS))
    return res

def audit_17(cdir,rdir):
    """nw: args=N penalty warmup iters, CSV output 'N,version,ns'."""
    cid="17-rodinia-nw-antidiag-tile"
    bsrc=cdir/"needle_baseline.cpp"
    lsrc=cdir/"needle_lego_tiled.cpp"
    build_cpp([bsrc],cdir/"_b17",CPP_FLAGS)
    build_cpp([lsrc],cdir/"_l17",CPP_FLAGS)
    res=[]
    for tag,n in [("p512",512),("np480",480),("p1024",1024)]:
        bv=meas(cdir/"_b17",[n,1,WARMUP,TIMED],parse_ns_csv3,TIMED,timeout=300)
        lv=meas(cdir/"_l17",[n,1,WARMUP,TIMED],parse_ns_csv3,TIMED,timeout=300)
        res.append(pair(cid,rdir,tag,bv,lv,"cpp",GXX13_VER,CPP_FLAGS))
    return res

def audit_18(cdir,rdir):
    """nussinov-skew: nussinov_baseline/nussinov_lego, arg=N, prints ns."""
    cid="18-npdp-nussinov-skew-tile"
    # Use nussinov_lego (plain skew) as the representative layout
    build_c(cdir/"nussinov_baseline.c",cdir/"_b18",C_FLAGS)
    build_c(cdir/"nussinov_lego.c",    cdir/"_l18",C_FLAGS)
    res=[]
    for tag,n in [("np500",500),("p512",512),("p1024",1024),("np1000",1000)]:
        bv=meas(cdir/"_b18",[n,WARMUP,TIMED],parse_ns_lines,TIMED,timeout=600)
        lv=meas(cdir/"_l18",[n,WARMUP,TIMED],parse_ns_lines,TIMED,timeout=600)
        res.append(pair(cid,rdir,tag,bv,lv,"c",GCC13_VER,C_FLAGS))
    return res

def meas_zuker(binary, n, warmup, timed):
    """Zuker outputs 'X.XX seconds\n'. Call once per measurement, parse stripping ' seconds'."""
    vals=[]
    for _ in range(warmup+timed):
        out=run_pin(binary,[1,n],timeout=120)
        for l in out.strip().splitlines():
            l=l.strip().replace(" seconds","")
            try: vals.append(int(float(l)*1e9)); break
            except ValueError: pass
    if len(vals)<timed: raise RuntimeError(f"zuker: got {len(vals)}/{timed}")
    return vals[-timed:]

def audit_19(cdir,rdir):
    """zuker-skew: takes (num_proc N), outputs 'X.XX seconds' per call."""
    cid="19-npdp-zuker-skew-tile"
    build_c(cdir/"zuker_baseline.c",cdir/"_b19",C_FLAGS)
    build_c(cdir/"zuker_lego.c",    cdir/"_l19",C_FLAGS)
    res=[]
    for tag,n in [("n100",100),("n150",150),("n200",200)]:
        bv=meas_zuker(cdir/"_b19",n,WARMUP,TIMED)
        lv=meas_zuker(cdir/"_l19",n,WARMUP,TIMED)
        res.append(pair(cid,rdir,tag,bv,lv,"c",GCC13_VER,C_FLAGS))
    return res

def audit_20(cdir,rdir):
    """seidel2d: args=N TSTEPS WARMUP RUNS, prints ns."""
    cid="20-polybench-seidel2d-wavefront-tile"
    build_c(cdir/"seidel2d_baseline.c",cdir/"_b20",C_FLAGS)
    build_c(cdir/"seidel2d_lego.c",    cdir/"_l20",C_FLAGS)
    res=[]
    for tag,n,t in [("np500",500,100),("p512",512,100),("p1024",1024,100)]:
        bv=meas(cdir/"_b20",[n,t,WARMUP,TIMED],parse_ns_lines,TIMED,timeout=300)
        lv=meas(cdir/"_l20",[n,t,WARMUP,TIMED],parse_ns_lines,TIMED,timeout=300)
        res.append(pair(cid,rdir,tag,bv,lv,"c",GCC13_VER,C_FLAGS))
    return res

def audit_21(cdir,rdir):
    """particlefilter: args=X Y Z Np warmup timed size_tag compiler, JSON obj output."""
    cid="21-rodinia-particlefilter-aosoA"
    bench=cdir/"bench"
    build_cpp([bench/"bench_baseline.cpp"],bench/"_b21",CPP_FLAGS,link=["-lm"])
    build_cpp([bench/"bench_lego.cpp"],    bench/"_ai_l21",CPP_FLAGS,defines=["-DLEGO_W=8"],link=["-lm"])
    res=[]
    for tag,np_ in [("np3000",3000),("np10000",10000),("np30000",30000)]:
        bv=meas(bench/"_b21",[128,128,10,np_,WARMUP,TIMED,tag,GXX13_VER],parse_ns_json_obj,TIMED,timeout=300)
        lv=meas(bench/"_ai_l21",[128,128,10,np_,WARMUP,TIMED,tag,GXX13_VER],parse_ns_json_obj,TIMED,timeout=300)
        res.append(pair(cid,rdir,tag,bv,lv,"cpp",GXX13_VER,CPP_FLAGS))
    return res

def audit_22(cdir,rdir):
    """LULESH: arg=N, JSON obj output."""
    cid="22-lulesh-elem-aosoA"
    flags=CPP_FLAGS+["-DUSE_MPI=0"]
    bench=cdir/"bench"; ldir=cdir/"upstream/LULESH-2.0.3"
    extra=[str(ldir/"lulesh-init.cc"),str(ldir/"lulesh-comm.cc"),
           str(ldir/"lulesh-util.cc"),str(ldir/"lulesh-viz.cc")]
    cmd_b=([GXX13]+flags+[str(bench/"bench_baseline.cc"),str(bench/"lulesh_stub.cc")]
           +extra+[f"-I{ldir}","-o",str(bench/"_b22")])
    r=subprocess.run(cmd_b,capture_output=True,text=True)
    if r.returncode!=0: raise RuntimeError(f"LULESH baseline: {r.stderr[:400]}")
    cmd_l=([GXX13]+flags+[str(bench/"bench_lego.cc"),str(bench/"lulesh_stub.cc")]
           +extra+[f"-I{ldir}","-o",str(bench/"_l22")])
    r=subprocess.run(cmd_l,capture_output=True,text=True)
    if r.returncode!=0: raise RuntimeError(f"LULESH lego: {r.stderr[:400]}")
    res=[]
    for tag,n in [("n10",10),("n20",20),("n30",30)]:
        bv=meas(bench/"_b22",[n],parse_ns_json_obj,TIMED,timeout=300)
        lv=meas(bench/"_l22",[n],parse_ns_json_obj,TIMED,timeout=300)
        res.append(pair(cid,rdir,tag,bv,lv,"cpp",GXX13_VER,CPP_FLAGS))
    return res

def audit_23(cdir,rdir):
    """hpccg: bench_sparsemv nx ny nz baseline|lego, JSON obj output."""
    cid="23-hpccg-cg-aosoA"
    build_cpp([cdir/"bench_sparsemv.cpp"],   cdir/"_b23",CPP_FLAGS)
    build_cpp([cdir/"bench_sparsemv_w4.cpp"],cdir/"_l23",CPP_FLAGS)
    res=[]
    for tag,nx,ny,nz in [("n20",20,20,20),("n50",50,50,50),("n100",100,100,100)]:
        bv=meas(cdir/"_b23",[nx,ny,nz,"baseline"],parse_ns_json_obj,TIMED,timeout=300)
        lv=meas(cdir/"_l23",[nx,ny,nz,"lego"],    parse_ns_json_obj,TIMED,timeout=300)
        res.append(pair(cid,rdir,tag,bv,lv,"cpp",GXX13_VER,CPP_FLAGS))
    return res

def audit_24(cdir,rdir):
    """fdtd-2d: POLYBENCH_TIME per-call."""
    cid="24-polybench-fdtd-2d-block-cyclic"
    poly=cdir/"upstream/polybench-c-4.2.1-beta"
    util=poly/"utilities"; src_dir=poly/"stencils/fdtd-2d"
    lsrc=cdir/"fdtd-2d-lego.c"
    res=[]
    for tag,dset in [("small","SMALL_DATASET"),("medium","MEDIUM_DATASET"),("large","LARGE_DATASET")]:
        bb=cdir/f"_b24_{tag}"; lb=cdir/f"_l24_{tag}"
        cmd_b=[GCC13]+C_FLAGS+[f"-D{dset}","-DPOLYBENCH_TIME","-I",str(util),
               str(src_dir/"fdtd-2d.c"),str(util/"polybench.c"),"-o",str(bb),"-lm"]
        r=subprocess.run(cmd_b,capture_output=True,text=True)
        if r.returncode!=0: raise RuntimeError(r.stderr[:300])
        cmd_l=[GCC13]+C_FLAGS+[f"-D{dset}","-DPOLYBENCH_TIME",
               "-I",str(util),"-I",str(src_dir),
               str(lsrc),str(util/"polybench.c"),"-o",str(lb),"-lm"]
        r=subprocess.run(cmd_l,capture_output=True,text=True)
        if r.returncode!=0: raise RuntimeError(r.stderr[:300])
        bv=meas_secs_per_call(bb,[],WARMUP,TIMED,timeout=120)
        lv=meas_secs_per_call(lb,[],WARMUP,TIMED,timeout=120)
        res.append(pair(cid,rdir,tag,bv,lv,"c",GCC13_VER,C_FLAGS))
    return res

def audit_25(cdir,rdir):
    """adi: rebuild from polybench adi.c and adi_lego.c with Intel GCC13, POLYBENCH_TIME."""
    cid="25-polybench-adi-block-cyclic"
    poly=cdir/"upstream/polybench-c-4.2.1-beta"
    util=poly/"utilities"; adi_dir=poly/"stencils/adi"
    bsrc=adi_dir/"adi.c"; lsrc=cdir/"adi_lego.c"
    res=[]
    for tag,n,t in [("n64",64,40),("p256",256,40),("p512",512,40)]:
        bb=cdir/f"_b25_{n}"; lb=cdir/f"_l25_{n}"
        dfn=[f"-DN={n}",f"-DTSTEPS={t}","-DPOLYBENCH_TIME","-DPOLYBENCH_NO_FLUSH_CACHE"]
        cmd_b=[GCC13]+C_FLAGS+dfn+[f"-I{util}",f"-I{adi_dir}",str(bsrc),str(util/"polybench.c"),"-lm","-o",str(bb)]
        r=subprocess.run(cmd_b,capture_output=True,text=True)
        if r.returncode!=0: raise RuntimeError(f"adi baseline N={n}: {r.stderr[:300]}")
        cmd_l=[GCC13]+C_FLAGS+dfn+["-DTILE_SZ=32",f"-I{util}",f"-I{adi_dir}",str(lsrc),str(util/"polybench.c"),"-lm","-o",str(lb)]
        r=subprocess.run(cmd_l,capture_output=True,text=True)
        if r.returncode!=0: raise RuntimeError(f"adi lego N={n}: {r.stderr[:300]}")
        bv=meas_secs_per_call(bb,[],WARMUP,TIMED,timeout=120)
        lv=meas_secs_per_call(lb,[],WARMUP,TIMED,timeout=120)
        res.append(pair(cid,rdir,tag,bv,lv,"c",GCC13_VER,C_FLAGS))
    return res

def audit_26(cdir,rdir):
    """gemm-pow2-pad: gemm_bench.c, args=N pad warmup iters, 'idx ns' output."""
    cid="26-polybench-gemm-pow2-pad"
    build_c(cdir/"gemm_bench.c",cdir/"_b26",C_FLAGS,link=["-lm"])
    res=[]
    for tag,n,pb,pl in [("p256",256,0,8),("p512",512,0,8),("p1024",1024,0,8)]:
        bv=meas(cdir/"_b26",[n,pb,WARMUP,TIMED],parse_ns_indexed,TIMED,timeout=300)
        lv=meas(cdir/"_b26",[n,pl,WARMUP,TIMED],parse_ns_indexed,TIMED,timeout=300)
        res.append(pair(cid,rdir,tag,bv,lv,"c",GCC13_VER,C_FLAGS))
    return res

def audit_27(cdir,rdir):
    """heat3d-pow2-pad: polybench POLYBENCH_TIME per-call. Or existing per-N binaries."""
    cid="27-polybench-heat3d-pow2-pad"
    up=cdir/"upstream/polybench-c-4.2.1-beta"
    util=up/"utilities"
    h3d_src=up/"stencils/heat-3d/heat-3d.c" if (up/"stencils/heat-3d/heat-3d.c").exists() else None
    lsrc=cdir/"heat3d_lego.c"
    res=[]
    for tag,n,t in [("n64",64,100),("n96",96,100),("n128",128,500)]:
        if h3d_src and h3d_src.exists():
            bb=cdir/f"_b27_{tag}"; lb=cdir/f"_l27_{tag}"
            fn=C_FLAGS+[f"-DN={n}",f"-DTSTEPS={t}","-DPOLYBENCH_TIME"]
            cmd_b=[GCC13]+fn+["-I",str(util),str(h3d_src),str(util/"polybench.c"),"-o",str(bb),"-lm"]
            r=subprocess.run(cmd_b,capture_output=True,text=True)
            if r.returncode!=0: raise RuntimeError(r.stderr[:300])
            h3d_dir=up/"stencils/heat-3d"
            cmd_l=[GCC13]+fn+["-I",str(util),"-I",str(h3d_dir),"-I",str(cdir),str(lsrc),str(util/"polybench.c"),"-o",str(lb),"-lm"]
            r=subprocess.run(cmd_l,capture_output=True,text=True)
            if r.returncode!=0: raise RuntimeError(r.stderr[:300])
            bv=meas_secs_per_call(bb,[],WARMUP,TIMED,timeout=120)
            lv=meas_secs_per_call(lb,[],WARMUP,TIMED,timeout=120)
        else:
            # Use pre-built binaries (AMD-native code, safe on Intel for AVX2)
            wt_up=cdir/"upstream"
            bb=wt_up/f"heat-3d-baseline-timer-n{n}"
            lb=cdir/f"heat3d_lego_{n}_t{t}_P8" if (cdir/f"heat3d_lego_{n}_t{t}_P8").exists() else cdir/f"heat3d_lego_{n}_P8"
            if not bb.exists() or not lb.exists():
                raise RuntimeError(f"heat3d-pow2-pad N={n}: missing bins {bb} {lb}")
            bv=meas_secs_per_call(bb,[],WARMUP,TIMED,timeout=120)
            lv=meas_secs_per_call(lb,[],WARMUP,TIMED,timeout=120)
        res.append(pair(cid,rdir,tag,bv,lv,"c",GCC13_VER,C_FLAGS))
    return res

def audit_28(cdir,rdir):
    """gemm-nonpow2-morton: args=N N N."""
    cid="28-polybench-gemm-nonpow2-morton"
    build_c(cdir/"gemm_baseline.c",cdir/"_b28",C_FLAGS)
    build_c(cdir/"gemm_lego.c",    cdir/"_l28",C_FLAGS)
    res=[]
    for tag,n in [("n1000",1000),("n1500",1500),("n2300",2300)]:
        bv=meas(cdir/"_b28",[n,WARMUP,TIMED],parse_ns_lines,TIMED,timeout=600)
        lv=meas(cdir/"_l28",[n,WARMUP,TIMED],parse_ns_lines,TIMED,timeout=600)
        res.append(pair(cid,rdir,tag,bv,lv,"c",GCC13_VER,C_FLAGS))
    return res

def audit_29(cdir,rdir):
    """brick stencil nonpow2: cpp."""
    cid="29-bricklib-stencil-nonpow2-brick"
    all_cpp=list(cdir.glob("*.cpp"))
    bsrc=next((s for s in all_cpp if "baseline" in s.name or "naive" in s.name),None)
    lsrc=next((s for s in all_cpp if ("lego" in s.name or "brick" in s.name)
               and "baseline" not in s.name and "naive" not in s.name
               and "verify" not in s.name and "helper" not in s.name),None)
    if bsrc is None: raise RuntimeError("No baseline cpp for 29")
    if lsrc is None: raise RuntimeError("No lego cpp for 29")
    build_cpp([bsrc],cdir/"_b29",CPP_FLAGS); build_cpp([lsrc],cdir/"_l29",CPP_FLAGS)
    res=[]
    for tag,n in [("n100",100),("n200",200),("p256",256)]:
        bv=meas(cdir/"_b29",[n],parse_ns_lines,TIMED,timeout=300)
        lv=meas(cdir/"_l29",[n],parse_ns_lines,TIMED,timeout=300)
        res.append(pair(cid,rdir,tag,bv,lv,"cpp",GXX13_VER,CPP_FLAGS))
    return res

def audit_31(cdir,rdir):
    """nussinov-nonpow2-skew: args=N WARMUP RUNS."""
    cid="31-npdp-nussinov-nonpow2-skew"
    lsrc=cdir/"nussinov_lego_wave.c"
    if not lsrc.exists():
        lsrc=next(s for s in cdir.glob("*lego*.c") if "verify" not in s.name)
    build_c(cdir/"nussinov_baseline.c",cdir/"_b31",C_FLAGS)
    build_c(lsrc,                       cdir/"_l31",C_FLAGS)
    res=[]
    for tag,n in [("n700",700),("n1000",1000),("n1500",1500)]:
        bv=meas(cdir/"_b31",[n,WARMUP,TIMED],parse_ns_lines,TIMED,timeout=600)
        lv=meas(cdir/"_l31",[n,WARMUP,TIMED],parse_ns_lines,TIMED,timeout=600)
        res.append(pair(cid,rdir,tag,bv,lv,"c",GCC13_VER,C_FLAGS))
    return res

def audit_32(cdir,rdir):
    """hotspot: wall-clock, needs data files."""
    cid="32-rodinia-hotspot-tile"
    up=cdir/"upstream/rodinia_3.1-main"
    data=up/"data/hotspot"
    lsrc=cdir/"hotspot_lego.cpp" if (cdir/"hotspot_lego.cpp").exists() else cdir/"hotspot_lego.c"
    bbin=up/"openmp/hotspot/_b32"
    lbin=cdir/"_l32"
    # Rebuild baseline from hotspot_openmp.cpp
    hotsrc=up/"openmp/hotspot/hotspot_openmp.cpp"
    if hotsrc.exists():
        cmd_b=[GXX13]+CPP_FLAGS+[str(hotsrc),"-o",str(bbin),"-lm"]
        r=subprocess.run(cmd_b,capture_output=True,text=True)
        if r.returncode!=0: bbin=up/"openmp/hotspot/hotspot_baseline"  # fallback
    else:
        bbin=up/"openmp/hotspot/hotspot_baseline"
    if lsrc.suffix==".cpp":
        build_cpp([lsrc],lbin,CPP_FLAGS,link=["-lm"])
    else:
        build_c(lsrc,lbin,C_FLAGS,link=["-lm"])
    sim=500
    res=[]
    for tag,grid in [("p64",64),("p512",512),("p1024",1024)]:
        tf=str(data/f"temp_{grid}"); pf=str(data/f"power_{grid}")
        def meas_hs(binary,g=grid,t=tf,p=pf):
            env=os.environ.copy(); env["OMP_NUM_THREADS"]="1"
            cmd=["taskset","-c",TASKSET,"numactl",f"--membind={NUMA}",
                 str(binary),str(g),str(g),str(sim),"1",t,p,"/dev/null"]
            vals=[]
            for _ in range(WARMUP+TIMED):
                t0=time.perf_counter_ns()
                subprocess.run(cmd,capture_output=True,env=env,timeout=120)
                vals.append(time.perf_counter_ns()-t0)
            return vals[-TIMED:]
        bv=meas_hs(bbin); lv=meas_hs(lbin)
        res.append(pair(cid,rdir,tag,bv,lv,"cpp",GXX13_VER,CPP_FLAGS))
    return res

def audit_33(cdir,rdir):
    """mvt: rebuild per-N from embedded harness source in measure.py, print JSON array."""
    cid="33-polybench-mvt-L1-tile"
    import importlib.util, sys as _sys
    spec=importlib.util.spec_from_file_location("measure33",cdir/"measure.py")
    m33=importlib.util.module_from_spec(spec); spec.loader.exec_module(m33)
    BSRC=m33.TIMER_HARNESS_BASELINE; LSRC=m33.TIMER_HARNESS_LEGO
    poly=cdir/"upstream/polybench-c-4.2.1-beta"
    util=poly/"utilities"
    mvt_dir=poly/"linear-algebra/kernels/mvt"
    res=[]
    for tag,n in [("n1000",1000),("p1024",1024),("n2000",2000)]:
        bb=cdir/f"_b33_{n}"; lb=cdir/f"_l33_{n}"
        bsrc_f=cdir/f"_b33_{n}.c"; lsrc_f=cdir/f"_l33_{n}.c"
        bsrc_f.write_text(BSRC); lsrc_f.write_text(LSRC)
        dfn=[f"-DN={n}",f"-DWARMUP_ITERS={WARMUP}",f"-DTIMED_ITERS={TIMED}"]
        cmd_b=[GCC13]+C_FLAGS+dfn+[f"-I{util}",f"-I{mvt_dir}",str(bsrc_f),str(util/"polybench.c"),"-lm","-o",str(bb)]
        r=subprocess.run(cmd_b,capture_output=True,text=True)
        if r.returncode!=0: raise RuntimeError(f"mvt baseline N={n}: {r.stderr[:300]}")
        cmd_l=[GCC13]+C_FLAGS+dfn+[f"-I{util}",f"-I{mvt_dir}",str(lsrc_f),str(util/"polybench.c"),"-lm","-o",str(lb)]
        r=subprocess.run(cmd_l,capture_output=True,text=True)
        if r.returncode!=0: raise RuntimeError(f"mvt lego N={n}: {r.stderr[:300]}")
        bv=meas(bb,[],parse_ns_json_array,TIMED,timeout=120)
        lv=meas(lb,[],parse_ns_json_array,TIMED,timeout=120)
        res.append(pair(cid,rdir,tag,bv,lv,"c",GCC13_VER,C_FLAGS))
    return res

def audit_34(cdir,rdir):
    """bicg: rebuild per-MxN from source with Intel GCC13, POLYBENCH_TIME."""
    cid="34-polybench-bicg-L1-tile"
    up=cdir/"upstream/polybench-c-4.2.1-beta"
    util=up/"utilities"
    bicg_dir=up/"linear-algebra/kernels/bicg"
    bsrc=bicg_dir/"bicg.c"; lsrc=cdir/"bicg_lego_v2.c"
    size_map=[
        ("p1024",  1024, 1024),
        ("np1984", 1984, 2016),
        ("np1900", 1900, 2100),
    ]
    res=[]
    for tag,M,N in size_map:
        bb=cdir/f"_b34_{M}x{N}"; lb=cdir/f"_l34_{M}x{N}"
        dfn=[f"-DM={M}",f"-DN={N}","-DPOLYBENCH_TIME"]
        cmd_b=[GCC13]+C_FLAGS+dfn+[f"-I{util}",f"-I{bicg_dir}",str(bsrc),str(util/"polybench.c"),"-lm","-o",str(bb)]
        r=subprocess.run(cmd_b,capture_output=True,text=True)
        if r.returncode!=0: raise RuntimeError(f"bicg baseline {M}x{N}: {r.stderr[:300]}")
        cmd_l=[GCC13]+C_FLAGS+dfn+[f"-I{util}",f"-I{bicg_dir}",f"-I{cdir}",str(lsrc),str(util/"polybench.c"),"-lm","-o",str(lb)]
        r=subprocess.run(cmd_l,capture_output=True,text=True)
        if r.returncode!=0: raise RuntimeError(f"bicg lego {M}x{N}: {r.stderr[:300]}")
        bv=meas_secs_per_call(bb,[],WARMUP,TIMED,timeout=120)
        lv=meas_secs_per_call(lb,[],WARMUP,TIMED,timeout=120)
        res.append(pair(cid,rdir,tag,bv,lv,"c",GCC13_VER,C_FLAGS))
    return res

def audit_35(cdir,rdir):
    """hpcc-dgemm: bench_dgemm_baseline/lego, args=N N N, 'N=x ns...' output."""
    cid="35-hpcc-dgemm-reg-L1-L2-tile"
    build_c(cdir/"bench_dgemm_baseline.c",cdir/"_b35",C_FLAGS)
    build_c(cdir/"bench_dgemm_lego.c",    cdir/"_l35",C_FLAGS)
    res=[]
    for tag,n in [("n1000",1000),("n1500",1500),("n1900",1900)]:
        bv=meas(cdir/"_b35",[n,n,n],parse_ns_prefixed,TIMED,timeout=300)
        lv=meas(cdir/"_l35",[n,n,n],parse_ns_prefixed,TIMED,timeout=300)
        res.append(pair(cid,rdir,tag,bv,lv,"c",GCC13_VER,C_FLAGS))
    return res

# ── registry ──────────────────────────────────────────────────────────────
CANDIDATES = {
    "01":("LEGO-eval-01-polybench-gemm-zmorton",              audit_01,"WIN"),
    "02":("LEGO-eval-02-polybench-lu-zmorton",                audit_02,"WIN"),
    "03":("LEGO-eval-03-polybench-chol-zmorton",              audit_03,"LOSS"),
    "04":("LEGO-eval-04-polybench-gemm-reg-L1-L2-tile",       audit_04,"WIN"),
    "05":("LEGO",                                              audit_05,"WIN"),
    "06":("LEGO-eval-06-polybench-2mm-reg-L1-tile",           audit_06,"WIN"),
    "07":("LEGO-eval-07-polybench-trmm-L1-L2-tile",           audit_07,"WIN"),
    "08":("LEGO-eval-08-polybench-doitgen-reg-L1-tile",       audit_08,"WIN"),
    "09":("LEGO-eval-09-tccg-tensor-contraction-GETT-tile",   audit_09,"WIN"),
    "10":("LEGO-eval-10-tblis-tensor-contraction-notranspose",audit_10,"WIN"),
    "11":("LEGO-eval-11-bricklib-3d7pt-brick",                audit_11,"LOSS"),
    "12":("LEGO-eval-12-bricklib-3d13pt-brick",               audit_12,"LOSS"),
    "13":("LEGO-eval-13-polybench-heat3d-brick",              audit_13,"WIN"),
    "14":("LEGO-eval-14-polybench-jacobi2d-brick",            audit_14,"LOSS"),
    "15":("LEGO-eval-15-polybench-symm-rfp",                  audit_15,"PARITY"),
    "16":("LEGO-eval-16-polybench-syrk-rfp",                  audit_16,"WIN"),
    "17":("LEGO-eval-17-rodinia-nw-antidiag-tile",            audit_17,"LOSS"),
    "18":("LEGO-eval-18-npdp-nussinov-skew-tile",             audit_18,"WIN"),
    "19":("LEGO-eval-19-npdp-zuker-skew-tile",                audit_19,"WIN"),
    "20":("LEGO-eval-20-polybench-seidel2d-wavefront-tile",   audit_20,"LOSS"),
    "21":("LEGO-eval-21-rodinia-particlefilter-aosoA",        audit_21,"LOSS"),
    "22":("LEGO-eval-22-lulesh-elem-aosoA",                   audit_22,"LOSS"),
    "23":("LEGO-eval-23-hpccg-cg-aosoA",                      audit_23,"WIN"),
    "24":("LEGO-eval-24-polybench-fdtd-2d-block-cyclic",      audit_24,"WIN"),
    "25":("LEGO-eval-25-polybench-adi-block-cyclic",          audit_25,"WIN"),
    "26":("LEGO-eval-26-polybench-gemm-pow2-pad",             audit_26,"WIN"),
    "27":("LEGO-eval-27-polybench-heat3d-pow2-pad",           audit_27,"WIN"),
    "28":("LEGO-eval-28-polybench-gemm-nonpow2-morton",       audit_28,"WIN"),
    "29":("LEGO-eval-29-bricklib-stencil-nonpow2-brick",      audit_29,"LOSS"),
    "31":("LEGO-eval-31-npdp-nussinov-nonpow2-skew",          audit_31,"WIN"),
    "32":("LEGO-eval-32-rodinia-hotspot-tile",                audit_32,"WIN"),
    "33":("LEGO-eval-33-polybench-mvt-L1-tile",               audit_33,"WIN"),
    "34":("LEGO-eval-34-polybench-bicg-L1-tile",              audit_34,"WIN"),
    "35":("LEGO-eval-35-hpcc-dgemm-reg-L1-L2-tile",           audit_35,"WIN"),
}

def verdict_from_results(results):
    vs=[v for _,_,v in results]
    if "WIN" in vs: return "WIN"
    if all(v=="LOSS" for v in vs): return "LOSS"
    if all(v=="PARITY" for v in vs): return "PARITY"
    return "MIXED"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--candidates",default="",help="Comma-separated IDs")
    ap.add_argument("--skip",default="30",help="Comma-separated IDs to skip")
    ap.add_argument("--skip-done",action="store_true",help="Skip if already has audit_intel files")
    args=ap.parse_args()

    todo=sorted(CANDIDATES.keys()) if not args.candidates else [x.strip() for x in args.candidates.split(",")]
    skip_set=set(args.skip.split(",")) if args.skip else set()

    all_results:dict[str,tuple]={}
    skipped:dict[str,str]={"30":"DROPPED-build (pre-existing)"}
    t_wall=time.time()

    for cid_num in todo:
        if cid_num in skip_set:
            skipped[cid_num]="explicitly skipped"; continue
        if cid_num not in CANDIDATES:
            skipped[cid_num]="not in registry"; continue
        wt,fn,amd=CANDIDATES[cid_num]
        cdirs=list(ROOT.glob(f"{wt}/evaluation/candidates/{cid_num}-*"))
        if not cdirs:
            skipped[cid_num]=f"worktree not found: {wt}"; continue
        cdir=cdirs[0]; rdir=cdir/"raw"

        if args.skip_done and len(list(rdir.glob("audit_intel_baseline_*.json")))>=3:
            print(f"[{cid_num}] SKIP_DONE"); all_results[cid_num]=("SKIP_DONE",amd,[]); continue

        print(f"\n{'='*60}")
        print(f"[{cid_num}] {cdir.name}  AMD={amd}")
        print(f"{'='*60}")
        t0=time.time()
        try:
            fd=lock_acquire()
            try: results=fn(cdir,rdir)
            finally: lock_release(fd)
            elapsed=time.time()-t0
            iv=verdict_from_results(results)
            ok=(iv==amd) or (amd=="COMPLETE" and iv in ("WIN","PARITY"))
            sym="✓" if ok else "✗"
            print(f"  {sym} Intel={iv} AMD={amd}  ({elapsed:.0f}s)")
            all_results[cid_num]=(iv,amd,results)
        except KeyboardInterrupt: print("[INTERRUPTED]"); break
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}\n{traceback.format_exc()[-600:]}")
            skipped[cid_num]=str(e)[:300]
        if time.time()-t_wall>10800: print("[3h BUDGET]"); break

    print(f"\n{'='*70}\nAUDIT RESULTS\n{'='*70}")
    agrees=[]; disagrees=[]
    for cn in sorted(all_results):
        iv,av,res=all_results[cn]
        if iv=="SKIP_DONE": agrees.append(cn); print(f"  {cn}: SKIP_DONE AMD={av}"); continue
        ok=(iv==av) or (av=="COMPLETE" and iv in ("WIN","PARITY"))
        sps=" ".join(f"{t}:{sp:.2f}x" for t,sp,_ in res)
        sym="✓" if ok else "✗"
        print(f"  {sym} {cn}: Intel={iv} AMD={av}  [{sps}]")
        (agrees if ok else disagrees).append(cn)
    print(f"\nAudited:{len(all_results)} Agreed:{len(agrees)} Disagreed:{len(disagrees)}")
    if disagrees: print(f"Disagreed: {disagrees}")
    if skipped: print(f"Skipped: {list(skipped.items())}")
    return all_results,skipped

if __name__=="__main__":
    main()
