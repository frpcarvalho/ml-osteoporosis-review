import json, re, os, glob
R1="/sessions/dazzling-brave-wright/mnt/ML_Osteoporosis_Review/Submission_Bone_MajorRevision/Revision_R1"
recs={}
for l in open(f"{R1}/reextraction_records.jsonl"):
    r=json.loads(l); recs[r["id"]]=r
inv=json.load(open(f"{R1}/fulltext_inventory.json"))["update"]
NOT_TARGET="O_Wu_2021 O_Chang_2024 O_Kong_2023 O_Nicolaes_2023 O_UsateguiMartin_2025 O_ShangGuan_2025".split()
EXCL_SYN={"U309","O_Hung","U66","U851","U853","U1030","U248","U260","U35","U326","U330","U691","U826","O_WeiT","U120","U1004"}
# text features
F={}
pmc2id={u["loc"]:"U"+k for k,u in inv.items() if u["src"]=="pmc"}
for l in open("/tmp/pb/pmc_feat.txt"):
    p,*fs=l.split(); 
    if p in pmc2id: F[pmc2id[p]]=set(fs)-{"-"}
pat=dict(C=r"calibration (curve|plot|slope|intercept|belt|analysis|error|metric|performance|was assessed|was evaluated)|brier|hosmer|reliability diagram|expected calibration|well.calibrated",M=r"imputation|imputed|complete.case",Mm=r"missing (data|values)",
 S=r"smote|oversampl",H=r"case.control|healthy (controls|volunteers)",I=r"within (one|two|three|six|\d+) (days?|weeks?|months?|years?)|(on|within) the same day|(one|two|three|four|six|\d+) (days?|weeks?|months?) (of|before|after|apart)|same (visit|week|session|admission|examination)|time interval|interval between")
for f in glob.glob("/tmp/pb/txt/*.txt"):
    t=open(f,errors="ignore").read().lower(); t=re.sub(r"-\n","",t); t=re.sub(r"\s+"," ",t)
    k_=t.rfind(" references ");  t=t[:k_] if k_>len(t)*0.5 else t
    key=os.path.basename(f)[:-4]
    if key.startswith("U"): rid=key
    else:
        au=key.split("_")[1].lower(); cand=[i for i in recs if i.lower().startswith("o_"+au)]
        if key in ("O_WeiQ_2025","O_WeiT_2025"): cand=[key[:-5]]
        rid=cand[0] if cand else None
    if rid: F[rid]={k for k,p in pat.items() if re.search(p,t)}
def tier(r):
    rank={"external":5,"external_pretrained":5,"temporal":4,"internal_holdout":3,"CV":2,"cv":2,"unclear":1.5,"apparent":1,"apparent_bootstrap":1,"training":0}
    es=[e for e in r.get("estimates",[]) if e.get("primary")] or r.get("estimates",[])
    if not es: return None,None
    best=max(es,key=lambda e:rank.get(e["set"],0)); return best["set"],best
SPECIAL_POP=re.compile(r"rheumat|ckd|kidney|hiv|diabet|t2dm|sj[öo]gren|cystic|hyperparathy|spine surgery|surgery|hepatitis|prefrail|frail|copd|dialysis",re.I)
out=[]
for i,r in recs.items():
    if i in NOT_TARGET: continue
    f=F.get(i,set()); flag=(r.get("eligibility_flag") or "")+" "+(r.get("notes") or "")
    st,e=tier(r); std=(r.get("ref",{}).get("std") or "").lower(); site=(r.get("ref",{}).get("site") or "").lower()
    pop=(r.get("population") or "")+" "+r.get("study","")+" "+(r.get("modality") or "")
    A={}; why={}
    # D1
    cc = i in ("U68","U1222","U880")  # verified in full text: case-control with healthy/age-matched controls
    A["1.1"]="PN" if cc else ("PY" if re.search(r"kaggle|public",flag+std,re.I) else "Y"); why["1.1"]="case-control / healthy controls" if cc else ""
    sel=re.search(r"excluded before modell|HU-DXA mismatch|osteopenia excluded",flag,re.I)
    A["1.2"]="N" if sel else "PY"; why["1.2"]="selective exclusion/spectrum: "+sel.group(0) if sel else ""
    # D2
    inc=re.search(r"incorporation|radiomics on QCT images",flag,re.I) and i in ("U1137",)
    A["2.1"]="Y"; A["2.2"]="PY"; A["2.3"]="N" if inc else "Y"; why["2.3"]="predictor derived from reference-standard images" if inc else ""
    # D3
    if "qus" in std or "ultrasound" in std: A["3.1"]="PN"; why["3.1"]="QUS reference"
    elif re.search(r"radius|forearm",site+" "+std): A["3.1"]="PN"; why["3.1"]="peripheral DXA site in reference"
    elif "nr" in std[:3] or std.strip()=="" : A["3.1"]="NI"; why["3.1"]="reference standard not reported in extracted text"
    else: A["3.1"]="Y"
    A["3.2"]="Y" if A["3.1"]!="NI" else "NI"
    A["3.3"]="Y"; A["3.4"]="PY" if not re.search(r"Z-score|z<=|z<|mixes",flag+(r.get("ref",{}).get("threshold") or ""),re.I) else "PN"
    A["3.5"]="PY"
    img=bool(re.search(r"ct|radiograph|x-ray|mri|panoramic|image|tomograph",(r.get("modality") or ""),re.I))
    A["3.6"]="Y" if ("I" in f or not img) else "NI"; why["3.6"]="" if A["3.6"]=="Y" else "interval imaging-DXA not reported"
    # D4
    npos=e.get("npos") if e else None; n=e.get("n") if e else None
    if npos: A["4.1"]="Y" if npos>=100 else "N"
    elif n: A["4.1"]="PY" if n*0.35>=100 else "PN"
    else: A["4.1"]="NI"
    OV41={"U630":("PY","total n=12,108, ~50% low BMD: evaluation-set events >100"),"U863":("PY","total n=3,545 (1,675 low BMD): evaluation-set events >100"),
          "O_Li":("PY","female internal test n=3,796 (NHANES): events >100"),"U249":("PN","total n=237: evaluation-set events <100"),
          "U209":("PY","external AUC CI 0.814-0.843 implies large external set")}
    if i in OV41: A["4.1"]=OV41[i][0]
    why["4.1"]=OV41[i][1] if i in OV41 else f"events in evaluation set: {npos if npos else ('~'+str(round(n*0.35)) if n else 'NR')}"
    A["4.3"]="N" if sel else "PY"
    A["4.4"]=("Y" if "M" in f else "PN") if not img else ("Y" if "M" in f else "PY")
    leak=re.search(r"leakage|SMOTE applied to whole|feature selection apparently on full",flag,re.I)
    A["4.6"]="N" if leak else ("PY" if "S" not in f else "PY")
    why["4.6"]=leak.group(0) if leak else ""
    cal=("C" in f) or r.get("calibration") is True
    A["4.7"]="Y" if cal else "N"; why["4.7"]="" if cal else "calibration not assessed"
    win=(r.get("final_model_designated") is False and (r.get("n_models") or 0)>=3) or re.search(r"winner|selected on external",flag,re.I)
    if st in ("external","external_pretrained","temporal"): A["4.8"]="Y" if not re.search(r"selected on external|model chosen on external|selected using all datasets",flag,re.I) else "PN"
    elif st in ("internal_holdout","CV","cv","unclear"): A["4.8"]="PN" if win else "PY"
    elif st in ("apparent","apparent_bootstrap","training"): A["4.8"]="N"
    else: A["4.8"]="NI"
    why["4.8"]=f"best evaluation: {st}"+("; best-of-many model selection" if win else "")
    def dom(keys):
        v=[A[k] for k in keys]
        return "High" if any(x in("N","PN") for x in v) else ("Unclear" if "NI" in v else "Low")
    D={"D1":dom(["1.1","1.2"]),"D2":dom(["2.1","2.2","2.3"]),"D3":dom(["3.1","3.2","3.3","3.4","3.5","3.6"]),"D4":dom(["4.1","4.3","4.4","4.6","4.7","4.8"])}
    ov="High" if "High" in D.values() else ("Unclear" if "Unclear" in D.values() else "Low")
    ap1="High" if (SPECIAL_POP.search(pop) or re.search(r"fracture patients|spectrum",flag,re.I)) else "Low"; ap3="High" if A["3.1"]=="PN" else ("Unclear" if A["3.1"]=="NI" else "Low")
    apov="High" if "High" in (ap1,ap3) else ("Unclear" if ap3=="Unclear" else "Low")
    out.append(dict(id=i,study=r["study"],in_synthesis=i not in EXCL_SYN,items=A,why={k:v for k,v in why.items() if v},domains=D,overall=ov,appl=dict(D1=ap1,D2="Low",D3=ap3,overall=apov),textsrc=bool(f) or i in F))
json.dump(out,open("/tmp/pb/probast.json","w"),indent=1)
from collections import Counter
print(len(out),"no text features:",[o["id"] for o in out if o["id"] not in F])
for d in ["D1","D2","D3","D4"]: print(d,Counter(o["domains"][d] for o in out))
print("overall",Counter(o["overall"] for o in out)); print("appl",Counter(o["appl"]["overall"] for o in out))
print("4.7",Counter(o["items"]["4.7"] for o in out),"4.1",Counter(o["items"]["4.1"] for o in out),"4.8",Counter(o["items"]["4.8"] for o in out))
