#!/usr/bin/env python3
"""Correlate scheduler microbatches/stages with rocprofv3 kernel dispatches."""
import argparse, bisect, collections, json, re, statistics
from pathlib import Path

MB_RE = re.compile(r"microbatch_begin .* microbatch=(\d+) seq=(-?\d+) tokens=(\d+) seqs=(\d+) graph_type=(\d+) reuse=(\d+)")
DISPATCH_RE = re.compile(r"dispatch_begin dispatch=(\d+) slot=(\d+) splits=(\d+) nodes=(\d+) ts_us=(\d+)")
BEGIN_RE = re.compile(r"stage_enqueue_begin dispatch=(\d+) slot=(\d+) split=(\d+) backend='([^']+)' nodes=(\d+) ts_us=(\d+)")
END_RE = re.compile(r"stage_enqueue_end dispatch=(\d+) slot=(\d+) split=(\d+) backend='([^']+)' ts_us=(\d+)")
SYNC_RE = re.compile(r"graph_reuse_full_sync .* microbatch=(\d+).* duration_us=(\d+)")

def union_ns(xs):
    merged=[]
    for a,b in sorted(xs):
        if not merged or a>merged[-1][1]: merged.append([a,b])
        else: merged[-1][1]=max(merged[-1][1],b)
    return sum(b-a for a,b in merged)

def category(name):
    n=name.lower()
    if "nccl" in n or "rccl" in n: return "collective"
    if "mul_mat_vec_q" in n: return "q8_mmvq"
    if "mul_mat_q" in n: return "q8_mmq"
    if "quantize_q8" in n: return "quantize"
    if "gated_delta_net" in n or "ssm_conv" in n: return "gdn"
    if "flash_attn" in n: return "flash_attention"
    if "rms_norm" in n or "l2_norm" in n: return "norm"
    if "get_rows" in n or "set_rows" in n: return "rows"
    if "cpy" in n or "copy" in n: return "copy_kernel"
    if "bin_bcast" in n or "unary" in n: return "elementwise"
    return "other"

def parse_log(path):
    current=None; dispatch_meta={}; pending=collections.defaultdict(collections.deque); stages=[]; sync={}
    for line in path.read_text(errors="replace").splitlines():
        m=MB_RE.search(line)
        if m:
            mid,seq,tok,seqs,gtype,reuse=map(int,m.groups());current={"microbatch":mid,"seq":seq,"tokens":tok,"seqs":seqs,"graph_type":gtype,"reuse":bool(reuse)};continue
        m=DISPATCH_RE.search(line)
        if m:
            d,slot,splits,nodes,ts=map(int,m.groups());meta=dict(current or {});meta.update({"dispatch":d,"slot":slot,"splits":splits,"nodes":nodes,"dispatch_host_start_ns":ts*1000});dispatch_meta[d]=meta;continue
        m=BEGIN_RE.search(line)
        if m:
            d,slot,split,backend,nodes,ts=m.groups();key=(int(d),int(split),backend);pending[key].append({"dispatch":int(d),"slot":int(slot),"split":int(split),"backend":backend,"nodes":int(nodes),"host_start_ns":int(ts)*1000});continue
        m=END_RE.search(line)
        if m:
            d,slot,split,backend,ts=m.groups();key=(int(d),int(split),backend)
            if pending[key]:
                x=pending[key].popleft();x["host_end_ns"]=int(ts)*1000;stages.append(x)
            continue
        m=SYNC_RE.search(line)
        if m: sync[int(m.group(1))]=int(m.group(2))
    for d,m in dispatch_meta.items():
        if m.get("microbatch") in sync:m["graph_reuse_full_sync_us"]=sync[m["microbatch"]]
    stages=[s for s in stages if s["backend"].startswith("Meta(")]
    return dispatch_meta,sorted(stages,key=lambda x:x["host_start_ns"])

def mean_stats(v):
    return {"count":len(v),"mean":statistics.mean(v) if v else 0,"median":statistics.median(v) if v else 0,"min":min(v) if v else 0,"max":max(v) if v else 0}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--trace",type=Path,required=True);ap.add_argument("--log",type=Path,required=True);ap.add_argument("--output",type=Path,required=True);ap.add_argument("--tokens",type=int,default=4);ap.add_argument("--seqs",type=int,default=4);a=ap.parse_args()
    dm,stages=parse_log(a.log);starts=[s["host_start_ns"] for s in stages]
    tool=json.load(open(a.trace))["rocprofiler-sdk-tool"][0]
    names={x["kernel_id"]:(x.get("demangled_kernel_name") or x.get("kernel_name") or "?") for x in tool["kernel_symbols"]}
    corr={}
    for r in tool["buffer_records"]["hip_api"]:
        i=bisect.bisect_right(starts,r["start_timestamp"])-1
        if i>=0 and r["start_timestamp"]<=stages[i]["host_end_ns"]:corr[r["correlation_id"]["internal"]]=i
    by_stage=collections.defaultdict(list)
    for k in tool["buffer_records"]["kernel_dispatch"]:
        i=corr.get(k["correlation_id"]["internal"])
        if i is not None:by_stage[i].append(k)
    selected={d for d,m in dm.items() if m.get("tokens")==a.tokens and m.get("seqs")==a.seqs and m.get("reuse")}
    rows=[];cats=collections.defaultdict(lambda:[0,0]);names_agg=collections.defaultdict(lambda:[0,0])
    for i,s in enumerate(stages):
        if s["dispatch"] not in selected:continue
        ks=by_stage[i]
        if not ks:continue
        intervals=[(k["start_timestamp"],k["end_timestamp"]) for k in ks]
        per_agent=collections.defaultdict(list)
        for k in ks:
            per_agent[k["dispatch_info"]["agent_id"]["handle"]].append((k["start_timestamp"],k["end_timestamp"]))
            name=names.get(k["dispatch_info"]["kernel_id"],"?");dur=k["end_timestamp"]-k["start_timestamp"]
            cats[(s["split"],category(name))][0]+=dur;cats[(s["split"],category(name))][1]+=1;names_agg[(s["split"],name)][0]+=dur;names_agg[(s["split"],name)][1]+=1
        rows.append({**s,"kernel_count":len(ks),"device_envelope_ns":max(b for _,b in intervals)-min(x for x,_ in intervals),"device_union_busy_ns":union_ns(intervals),"agent_busy_ns":{str(agent):union_ns(v) for agent,v in per_agent.items()}})
    splits=sorted({r["split"] for r in rows});out={"method":"HIP correlation IDs inside scheduler stage enqueue intervals; stable reused decode selected by microbatch tokens/seqs","trace":str(a.trace),"log":str(a.log),"selected_dispatches":sorted(selected),"selected_dispatch_count":len(selected),"assigned_stage_rows":len(rows),"stages":{},"dispatches":rows}
    for sp in splits:
        rs=[r for r in rows if r["split"]==sp]
        out["stages"][str(sp)]={"backend":rs[0]["backend"],"device_envelope_ms":mean_stats([r["device_envelope_ns"]/1e6 for r in rs]),"device_union_busy_ms":mean_stats([r["device_union_busy_ns"]/1e6 for r in rs]),"host_enqueue_ms":mean_stats([(r["host_end_ns"]-r["host_start_ns"])/1e6 for r in rs]),"kernel_count":mean_stats([r["kernel_count"] for r in rs]),"categories":[{"category":c,"total_ms":v[0]/1e6,"count":v[1],"mean_per_dispatch_ms":v[0]/1e6/len(rs)} for (s,c),v in sorted(cats.items(),key=lambda kv:kv[1][0],reverse=True) if s==sp],"top_kernels":[{"name":n,"total_ms":v[0]/1e6,"count":v[1],"mean_per_dispatch_ms":v[0]/1e6/len(rs)} for (s,n),v in sorted(names_agg.items(),key=lambda kv:kv[1][0],reverse=True) if s==sp][:25]}
    syncs=[dm[d].get("graph_reuse_full_sync_us",0)/1000 for d in selected if dm[d].get("graph_reuse_full_sync_us") is not None];out["graph_reuse_full_sync_ms"]=mean_stats(syncs)
    a.output.write_text(json.dumps(out,indent=2)+"\n")
    print("PARALLEL_CRITICAL_PATH_OK",json.dumps({"dispatches":len(selected),"stage_envelope_ms":{sp:out["stages"][sp]["device_envelope_ms"]["mean"] for sp in out["stages"]},"sync_ms":out["graph_reuse_full_sync_ms"]["mean"]},sort_keys=True))
if __name__=="__main__":main()
