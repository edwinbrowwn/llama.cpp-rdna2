#!/usr/bin/env python3
"""Development-only real-decode RCCL policy calibrator.

It benchmarks RCCL Auto versus the offline-certified v6 plugin policy in clean
llama-bench processes. It is intentionally external to production llama.cpp.
"""
from __future__ import annotations
import argparse, hashlib, json, os, pathlib, re, subprocess, sys, tempfile, time

FIXED_ENV = {
    "HSA_FORCE_FINE_GRAIN_PCIE": "1", "GGML_HIP_SAFE_STATE_IO": "1",
    "GGML_HIP_GFX1030_Q8_CACHE": "1", "GGML_HIP_GFX1030_GDN_SIBLING_FUSION": "1",
    "GGML_HIP_GFX1030_Q8_1_FUSION": "1", "GGML_HIP_GFX1030_NATIVE": "1",
    "GGML_HIP_GFX1030_ADD_RMS_NORM_FUSION": "1", "NCCL_P2P_DISABLE": "0",
    "NCCL_P2P_LEVEL": "PXB", "GGML_TP_SHARDED_OUTPUT": "1",
    "GGML_CUDA_ALLREDUCE": "nccl", "HSA_OVERRIDE_GFX_VERSION": "10.3.0",
    "HSA_NO_SCRATCH_RECLAIM": "1", "GGML_CUDA_P2P": "1", "GGML_HIP_GRAPHS": "1",
}
POLICY = {"name": "Ring/LL/3-hot", "algorithm": "Ring", "protocol": "LL", "channels": 3,
          "bytes": [20480], "plugin_mode": "force"}
CONFLICTING = ("NCCL_ALGO", "NCCL_PROTO", "NCCL_MIN_NCHANNELS", "NCCL_MAX_NCHANNELS", "NCCL_NTHREADS")

def run_capture(cmd, env=None):
    return subprocess.run(cmd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)

def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()

def command_text(cmd):
    r = run_capture(cmd)
    return r.stdout + r.stderr

def signature(args):
    rocminfo = command_text(["rocminfo"])
    smi = command_text(["rocm-smi", "--showtopo"])
    hip = command_text(["hipcc", "--version"])
    rccl = command_text(["strings", "/opt/rocm/core-7.14/lib/librccl.so"])
    archs = sorted(re.findall(r"^\s*Name:\s+(gfx\d+)", rocminfo, re.MULTILINE))
    v620 = len(re.findall(r"Marketing Name:\s+AMD Radeon Pro V620", rocminfo))
    rccl_match = re.search(r"RCCL version\s*:?\s*(\d+\.\d+\.\d+)", rccl)
    topology = smi.strip()
    return {
        "gpu_architectures": archs, "v620_marketing_count": v620, "gpu_count": len(archs),
        "topology_sha256": sha(topology), "rocm_sha256": sha(hip),
        "rccl_version": rccl_match.group(1) if rccl_match else "unknown",
        "build_commit": subprocess.check_output(["git", "-C", args.worktree, "rev-parse", "HEAD"], text=True).strip(),
        "model_size": pathlib.Path(args.model).stat().st_size,
        "tp_devices": ["ROCm0", "ROCm1", "ROCm2", "ROCm3"], "tensor_split": [1, 1, 1, 1],
        "hot_sizes_bytes": [20480, 993280],
    }

def eligible(sig):
    return (sig["gpu_count"] == 4 and sig["v620_marketing_count"] == 4 and
            sig["gpu_architectures"] == ["gfx1030"] * 4 and sig["rccl_version"] != "unknown")

def benchmark(args, name, candidate):
    env = os.environ.copy(); env.update(FIXED_ENV)
    for k in CONFLICTING: env.pop(k, None)
    if candidate:
        env["GGML_HIP_RCCL_TUNE"] = "force"
        env["NCCL_TUNER_PLUGIN"] = args.plugin
    else:
        env.pop("GGML_HIP_RCCL_TUNE", None); env.pop("NCCL_TUNER_PLUGIN", None)
    cmd = [args.binary, "-m", args.model, "-p", "0", "-n", str(args.tokens), "-b", "512", "-ub", "512",
           "-t", "12", "-ngl", "999", "-sm", "tensor", "-dev", "ROCm0/ROCm1/ROCm2/ROCm3",
           "-ts", "1/1/1/1", "-fa", "on", "-r", str(args.reps), "-o", "json"]
    r = run_capture(cmd, env)
    out = pathlib.Path(args.evidence) / f"{name}.stdout"
    err = pathlib.Path(args.evidence) / f"{name}.stderr"
    out.write_text(r.stdout); err.write_text(r.stderr)
    if r.returncode != 0: raise RuntimeError(f"{name} failed rc={r.returncode}: {r.stderr[-1000:]}")
    rows = json.loads(r.stdout)
    values = [float(v) for row in rows for v in row["samples_ts"]]
    if len(values) < 2: raise RuntimeError(f"{name} has insufficient samples")
    import statistics
    return {"name": name, "samples_ts": values, "steady_ts": statistics.mean(values[1:]), "returncode": r.returncode}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--mode", choices=("auto","off","force"), default="auto")
    p.add_argument("--worktree", required=True); p.add_argument("--model", required=True); p.add_argument("--binary", required=True)
    p.add_argument("--plugin", required=True); p.add_argument("--cache", required=True); p.add_argument("--evidence", required=True)
    p.add_argument("--tokens", type=int, default=128); p.add_argument("--reps", type=int, default=3)
    a=p.parse_args(); pathlib.Path(a.evidence).mkdir(parents=True, exist_ok=True)
    sig=signature(a); result={"mode":a.mode,"signature":sig,"eligible":eligible(sig),"policy":POLICY,"cache_used":False}
    if a.mode == "off" or not eligible(sig):
        result["selected"]="Auto"; result["reason"]="off or ineligible topology"
    elif a.mode == "force":
        result["selected"]=POLICY["name"]; result["reason"]="explicit force; no benchmark"
    else:
        cache_path=pathlib.Path(a.cache); cache=None
        if cache_path.exists():
            try: cache=json.loads(cache_path.read_text())
            except (OSError, json.JSONDecodeError): cache=None
        cache_ok=cache is not None and cache.get("signature") == sig and cache.get("validated") is True
        result["cache_used"]=cache_ok
        if cache_ok:
            auto=benchmark(a,"revalidate-auto",False); cand=benchmark(a,"revalidate-cached",True)
        else:
            auto=benchmark(a,"calibrate-auto",False); cand=benchmark(a,"calibrate-ringll3",True)
        gain=(cand["steady_ts"]/auto["steady_ts"]-1.0)*100.0
        result.update({"auto":auto,"candidate":cand,"gain_percent":gain})
        if gain >= 3.0:
            result["selected"]=POLICY["name"]; result["reason"]="candidate exceeds 3% measured threshold"
            cache_data={"schema":1,"validated":True,"signature":sig,"policy":POLICY,"last_validation":result}
            tmp=cache_path.with_suffix(".tmp"); tmp.write_text(json.dumps(cache_data,indent=2)+"\n"); os.replace(tmp,cache_path)
        else:
            result["selected"]="Auto"; result["reason"]="candidate below threshold or unstable"
    pathlib.Path(a.evidence,"result.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
    if result["selected"] == "Auto" and a.mode == "auto" and result.get("eligible") and result.get("gain_percent", 0) < 3.0:
        return 0
    return 0
if __name__ == "__main__": sys.exit(main())
