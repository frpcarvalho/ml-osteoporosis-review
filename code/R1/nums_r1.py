import json, re, math, statistics as st
from collections import Counter, defaultdict
R1="/sessions/dazzling-brave-wright/mnt/ML_Osteoporosis_Review/Submission_Bone_MajorRevision/Revision_R1"
recs={}
for l in open(f"{R1}/reextraction_records.jsonl"):
    r=json.loads(l); recs[r["id"]]=r
NOT_TARGET="O_Wu_2021 O_Chang_2024 O_Kong_2023 O_Nicolaes_2023 O_UsateguiMartin_2025 O_ShangGuan_2025".split()
ELIG="U309 O_Hung U66 U851 U853 U1030 U248 U260 U1004".split()
OVER="U35 U326 U330 U691 U826 O_WeiT U120".split()
inc=[i for i in recs if i not in NOT_TARGET+ELIG]
syn=[i for i in inc if i not in OVER]
print("included",len(inc),"update",sum(i.startswith('U') for i in inc),"orig",sum(i.startswith('O') for i in inc),"| non-dup for synthesis",len(syn))
D=json.load(open('/tmp/ma/ds.json'))
pool=defaultdict(set); npool=defaultdict(set)
for x in D["rows"]: pool[x["contrast"]].add(x["id"])
for x in D["notpool"]: npool[x["contrast"]].add(x["id"])
anypool=set().union(*pool.values()); anyauc=anypool|set().union(*npool.values())
print("studies with poolable AUC (any contrast):",len(anypool),"| AUC but not poolable only:",len(anyauc-anypool),"| no AUC:",len(set(syn)-anyauc))
for c in pool: print(" ",c,"poolable",len(pool[c]),"notpool",len(npool[c]-pool[c]))
# SE sources for OP rows (study-level: any row)
src=Counter(); 
by=defaultdict(set)
for x in D["rows"]:
    by[(x["contrast"],x["id"])].add(x["se_src"])
for c in ["OP","LBMD"]:
    print(c,"SE source", Counter(sorted(v)[0] if len(v)==1 else "mixed" for (cc,i),v in by.items() if cc==c))
# primary contrast distribution per included study (all contrasts reported among primary estimates)
CM={"OP_vs_nonOP":"OP vs non-OP","OP_vs_rest_multiclass":"OP vs non-OP","OP_vs_nonOP_hip":"OP vs non-OP","OP_vs_nonOP_LS":"OP vs non-OP",
 "lowBMD_vs_normal":"Low BMD vs normal","LBMD_vs_normal":"Low BMD vs normal","LBMD_FN_vs_normal":"Low BMD vs normal","OP_vs_normal":"OP vs normal",
 "osteopenia_vs_normal":"Osteopenia vs normal/rest","osteopenia_vs_rest":"Osteopenia vs normal/rest","multiclass":"Multiclass","multiclass_mean":"Multiclass","multiclass_adjacent":"Multiclass",
 "OP_vs_osteopenia":"OP vs osteopenia"}
cc=Counter(); 
for i in inc:
    cs={CM.get(e["contrast"],e["contrast"]) for e in recs[i].get("estimates",[]) if e.get("primary") or True}
    for c in cs: cc[c]+=1
print("contrasts reported (any estimate) among included:",cc)
# reference standard
def ref(r):
    s=(r.get("ref",{}).get("std") or "").lower(); site=(r.get("ref",{}).get("site") or "").lower()
    if "qus" in s or "ultrasound" in s: return "QUS"
    if "qct" in s: return "QCT"
    if "dxa" in s or "dexa" in s:
        return "DXA (incl. peripheral site)" if re.search(r"radius|forearm",site+s) else "Central DXA"
    return "Not reported/other"
print("reference:",Counter(ref(recs[i]) for i in inc))
def site(r):
    s=(r.get("ref",{}).get("site") or "").lower()
    if re.search(r"lowest|min|any site|either|or |and|ls, fn|multiple|of sites",s) and not re.search(r"^hip$",s): 
        if re.search(r"femoral neck|hip|femur|fn|tf",s) and re.search(r"lumbar|spine|ls|l1",s): return "Multiple sites (lowest/any)"
    if re.search(r"femoral neck|hip|femur|\bfn\b|proximal femur|trochanter",s) and not re.search(r"lumbar|spine|l1|ls",s): return "Hip/femoral neck"
    if re.search(r"lumbar|spine|l1|vertebra|t12|t10|ls\b",s) and not re.search(r"femoral|hip|femur",s): return "Lumbar spine/vertebrae"
    if s.strip() in ("","nr","nr in retrieved text","nr in excerpt") or "nr" in s[:3]: return "Not reported"
    return "Multiple sites (lowest/any)"
sites=Counter(site(recs[i]) for i in inc); print("site:",sites)
json.dump({i:site(recs[i]) for i in inc},open('/tmp/ma/site.json','w'))
def modality(r):
    m=(r.get("modality") or "").lower()
    if re.search(r"\bct\b|tomograph|ldct|qct",m): return "CT"
    if re.search(r"radiograph|x-ray|xray|cxr|panoramic",m): return "Radiograph"
    if "mri" in m: return "MRI"
    if re.search(r"dxa image|ultrason|optical|radiofrequency|signal",m): return "Other imaging/signal"
    return "Clinical/laboratory"
print("modality:",Counter(modality(recs[i]) for i in inc))
print("opportunistic CT:",sum(bool(recs[i].get("opportunistic_ct")) for i in inc))
def vbest(r):
    rank={"external":5,"external_pretrained":5,"temporal":4,"internal_holdout":3,"CV":2,"cv":2,"unclear":1.5,"apparent":1,"apparent_bootstrap":1,"training":0}
    es=[e for e in r.get("estimates",[]) if e.get("primary")] or r.get("estimates",[])
    if not es: return "No AUC/estimate"
    s=max(es,key=lambda e:rank.get(e["set"],0))["set"]
    return {"external_pretrained":"external","cv":"CV","apparent_bootstrap":"apparent","training":"apparent","unclear":"internal_holdout"}.get(s,s)
vb=Counter(vbest(recs[i]) for i in inc); print("validation (best):",vb)
P={o["id"]:o for o in json.load(open('/tmp/pb/probast.json'))}
print("calibration (4.7=Y):",sum(P[i]["items"]["4.7"]=="Y" for i in inc if i in P),"of",sum(i in P for i in inc))
print("RoB:",Counter(P[i]["overall"] for i in inc if i in P),"Applic:",Counter(P[i]["appl"]["overall"] for i in inc if i in P))
print("DL best model:",sum(bool(re.search(r"\bdl\b|cnn|deep|densenet|resnet|efficientnet|vgg|transformer|yolo|dcnn",(recs[i].get('algo_family') or '')+' '+(recs[i].get('modality') or ''),re.I)) for i in inc))
win=[i for i in inc if recs[i].get("final_model_designated") is False and (recs[i].get("n_models") or 0)>=3]
print("winner-selection (no designated model, >=3 models):",len(win))
def ws(v):
    if v is True: return "yes"
    if v is False: return "no"
    if v in (None,"n/a"): return "n/a"
    v=str(v).lower()
    return "yes" if v.startswith(("yes","likely")) else ("partly" if v.startswith("partly") else ("no" if v.startswith("no") else "n/a"))
WS={i:ws(recs[i].get("within_site")) for i in inc}
json.dump(WS,open('/tmp/ma/ws.json','w'))
print("within-site:",Counter(WS[i] for i in inc if modality(recs[i])!="Clinical/laboratory"))
# sens/spec across included primary
ss=[(e["sens"],e["spec"]) for i in inc for e in recs[i].get("estimates",[]) if e.get("primary") and e.get("sens") is not None and e.get("spec") is not None]
print("sens/spec pairs:",len(ss),"studies",len({i for i in inc for e in recs[i].get("estimates",[]) if e.get("primary") and e.get("sens") is not None and e.get("spec") is not None}))
if ss: print(" sens median",st.median(a for a,b in ss),min(a for a,b in ss),max(a for a,b in ss)," spec median",st.median(b for a,b in ss),min(b for a,b in ss),max(b for a,b in ss))
# internal->external change within study
deltas=[]
for i in syn:
    es=[e for e in recs[i].get("estimates",[]) if e.get("auc")]
    for e1 in es:
        if e1["set"] in("external","temporal"):
            ints=[e for e in es if e["set"] in("internal_holdout","CV") and CM.get(e["contrast"])==CM.get(e1["contrast"])]
            if ints:
                e0=max(ints,key=lambda e:e["auc"]) if len(ints)>1 else ints[0]
                deltas.append((i,e0["auc"],e1["auc"],e1["set"])); break
print("internal->external pairs:",len(deltas))
d=[b-a for _,a,b,_ in deltas]
if d: print(" median change",st.median(d),"range",min(d),max(d),"decreased",sum(x<0 for x in d),"increased",sum(x>0 for x in d))
json.dump(deltas,open('/tmp/ma/deltas.json','w'))
