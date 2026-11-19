exec(open('/tmp/ma/meta.py').read())
R=json.load(open('/tmp/ma/results.json'))
SITE=json.load(open('/tmp/ma/site.json')); WS=json.load(open('/tmp/ma/ws.json'))
recs={}
for l in open('/sessions/dazzling-brave-wright/mnt/ML_Osteoporosis_Review/Submission_Bone_MajorRevision/Revision_R1/reextraction_records.jsonl'):
    r=json.loads(l); recs[r["id"]]=r
OP=[x for x in D["rows"] if x["contrast"]=="OP"]
def P(lab,rs):
    a=agg(rs)
    if len(a)<3: print(lab,"k<3"); return
    o=fit([x["y"] for x in a],[x["se"]**2 for x in a]); R["OP "+lab]={k:o.get(k) for k in ["k","auc","lo","hi","pi_lo","pi_hi","tau2","I2"]}
    print(f"{lab:40s} k={o['k']} AUC {o['auc']:.3f} ({o['lo']:.3f}-{o['hi']:.3f}) PI {o.get('pi_lo',float('nan')):.3f}-{o.get('pi_hi',float('nan')):.3f} I2 {o['I2']:.1f}")
for s in ["Hip/femoral neck","Lumbar spine/vertebrae","Multiple sites (lowest/any)","Not reported"]:
    P("site "+s,[x for x in OP if SITE.get(x["id"])==s])
img=[x for x in OP if x["imaging"]]
P("imaging within-site",[x for x in img if WS.get(x["id"])=="yes"])
P("imaging between-site",[x for x in img if WS.get(x["id"])=="no"])
des=[x for x in OP if not (recs[x["id"]].get("final_model_designated") is False and (recs[x["id"]].get("n_models") or 0)>=3)]
P("no winner selection (designated/single model)",des)
P("winner selection",[x for x in OP if x not in des])
# meta-regression with log10 n (subset with n)
a=[x for x in agg(OP) if x.get("n")]
y=np.array([x["y"] for x in a]); v=np.array([x["se"]**2 for x in a]); k=len(y)
cv={"external_or_temporal":[1.0 if x["set"] in("external","temporal") else 0 for x in a],"imaging":[1.0 if x["imaging"] else 0 for x in a],
    "deep_learning":[1.0 if x["dl"] else 0 for x in a],"QCT_reference":[1.0 if x["ref"]=="QCT" else 0 for x in a],
    "opportunistic_CT":[1.0 if x["opp_ct"] else 0 for x in a],"log10_n":[math.log10(x["n"]) for x in a]}
X=np.column_stack([np.ones(k)]+[cv[c] for c in cv]); t0=tau2_reml(y,v); o=fit(y,v,X)
MR={"k":k,"R2":max(0,(t0-o["tau2"])/t0),"tau2_res":o["tau2"],"tau2_0":t0}
crit=stats.t.ppf(0.975,o["df"])
print("MR with log n: k",k,"R2",MR["R2"],"tau2res",o["tau2"])
for j,c in enumerate(["intercept"]+list(cv)):
    b=o["b"][j]; se=o["se"][j]; p=2*(1-stats.t.cdf(abs(b/se),o["df"])); MR[c]=dict(b=b,lo=b-crit*se,hi=b+crit*se,p=p)
    print(f"  {c:22s} b={b:+.3f} ({b-crit*se:+.3f} to {b+crit*se:+.3f}) p={p:.3f}")
R["OP meta-regression (with log n)"]=MR
# CI for the earlier meta-regression too
mr=R["OP meta-regression"]
json.dump(R,open('/tmp/ma/results.json','w'),indent=1)
