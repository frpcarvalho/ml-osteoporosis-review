import json, math, re
exec(open('/tmp/ma/build.py').read().split('for i,r in recs.items()')[0])
EXCL=set(EXCL_ELIG+EXCL_OVER+NOT_TARGET)
CMAP={"OP_vs_nonOP":"OP","OP_vs_rest_multiclass":"OP","OP_vs_nonOP_hip":"OP","OP_vs_nonOP_LS":"OP_LS",
 "lowBMD_vs_normal":"LBMD","LBMD_vs_normal":"LBMD","LBMD_FN_vs_normal":"LBMD","OP_vs_normal":"OPvN",
 "osteopenia_vs_normal":"OPEN","osteopenia_vs_rest":"OPEN","multiclass":"MULTI","multiclass_mean":"MULTI",
 "OP_vs_osteopenia":"OPvOPEN"}
SMAP={"cv":"CV","external_pretrained":"external","apparent_bootstrap":"apparent"}
RANK={"external":5,"temporal":4,"internal_holdout":3,"CV":2,"unclear":1.5,"apparent":1}
lg=lambda p: math.log(p/(1-p))
def se_of(e):
    a=min(max(e["auc"],0.5001),0.999)
    lo,hi=e.get("lo"),e.get("hi")
    if lo is not None and hi is not None and 0<lo<hi<1 and lo<=e["auc"]+1e-9<=hi+2e-3:
        return (lg(min(hi,0.9995))-lg(max(lo,0.5)))/(2*1.959964),"CI"
    p,q=e.get("npos"),e.get("nneg")
    if p and q:
        q1=a/(2-a); q2=2*a*a/(1+a)
        v=(a*(1-a)+(p-1)*(q1-a*a)+(q-1)*(q2-a*a))/(p*q)
        return math.sqrt(v)/(a*(1-a)),"HM_counts"
    return None,None
def hm(a,p,q):
    q1=a/(2-a); q2=2*a*a/(1+a)
    v=(a*(1-a)+(p-1)*(q1-a*a)+(q-1)*(q2-a*a))/(p*q); return math.sqrt(v)/(a*(1-a))
def cls(r):
    m=(r.get("modality") or "").lower(); al=(r.get("algo_family") or "").lower()
    img=bool(re.search(r"ct\b|ct |radiograph|x-ray|xray|mri|panoramic|image|dxa images|radiomic|hu\b|tomograph|cxr",m))
    dl=bool(re.search(r"\bdl\b|cnn|deep|densenet|resnet|efficientnet|vgg|transformer|yolo|dcnn|u-net|convolution",al+" "+m))
    std=(r.get("ref",{}).get("std") or "").lower(); site=(r.get("ref",{}).get("site") or "").lower()
    ref="QUS" if ("qus" in std or "ultrasound" in std) else ("QCT" if "qct" in std else ("DXA" if ("dxa" in std or "dexa" in std) else "NR"))
    central = ref=="DXA" and not re.search(r"radius|forearm|heel|calcan",site+" "+std)
    return dict(imaging=img, dl=dl, ref=ref, central_dxa=central, opp_ct=bool(r.get("opportunistic_ct")),
                quasi=r["id"] in {"U331","U326","U330","U337","U1015","U255","U879","U1021","U1260"})
rows=[]; notpool=[]
for i,r in recs.items():
    if i in EXCL: continue
    ests=[dict(e) for e in r.get("estimates",[]) if e.get("primary") and e.get("auc") is not None]
    for e in ests:
        e["c"]=CMAP.get(e["contrast"]); e["s"]=SMAP.get(e["set"],e["set"])
    ests=[e for e in ests if e["c"] and e["s"] in RANK]
    if any(e["c"]=="OP" for e in ests): ests=[e for e in ests if e["c"]!="OP_LS"]
    for c in set(e["c"] for e in ests):
        ce=[e for e in ests if e["c"]==c]; top=max(RANK[e["s"]] for e in ce)
        for e in [e for e in ce if RANK[e["s"]]==top]:
            se,how=se_of(e)
            row=dict(id=i,study=r["study"],contrast=c,set=e["s"],model=e["model"],auc=e["auc"],n=e.get("n"),npos=e.get("npos"),nneg=e.get("nneg"),
                     sens=e.get("sens"),spec=e.get("spec"),y=lg(min(max(e["auc"],0.5001),0.999)),se=se,se_src=how,**cls(r))
            (rows if se else notpool).append(row)
import statistics
prev={}
for c in set(x["contrast"] for x in rows):
    ps=[x["npos"]/(x["npos"]+x["nneg"]) for x in rows if x["contrast"]==c and x["npos"] and x["nneg"]]
    prev[c]=statistics.median(ps) if len(ps)>=3 else 0.3
print("median prevalence",prev)
DER={"U542":(42,0.431),"U842":(291,30/291),"U331":(1608,None),"U902":(234,None)}
still=[]
for x in notpool:
    if x["id"] in DER and not x["n"]:
        n,pp=DER[x["id"]]; x["n"]=n; x["n_note"]="derived from reported totals/split"
        if pp: x["npos"]=round(n*pp); x["nneg"]=n-x["npos"]
for x in notpool:
    if x["n"] and x["n"]>=10:
        p=max(2,round(x["n"]*prev.get(x["contrast"],0.3))); q=max(2,x["n"]-p)
        if x.get("npos") and x.get("nneg"): p,q=x["npos"],x["nneg"]
        x["se"]=hm(min(max(x["auc"],0.5001),0.999),p,q); x["se_src"]="HM_derived_counts" if x.get("npos") else "HM_imputed_prev"; rows.append(x)
    else: still.append(x)
notpool=still
print("poolable estimates",len(rows),"not poolable",len(notpool))
from collections import Counter
print(Counter((x["contrast"]) for x in rows)); print("studies OP poolable",len({x['id'] for x in rows if x['contrast']=='OP'}),"OP notpool",len({x['id'] for x in notpool if x['contrast']=='OP'}))
json.dump(dict(rows=rows,notpool=notpool),open('/tmp/ma/ds.json','w'))
