exec(open('/tmp/ma/meta.py').read())
import statistics
rows=D["rows"]; notpool=D["notpool"]
R={}
def summ(name,o): 
    R[name]={k:o.get(k) for k in ["k","auc","lo","hi","pi_lo","pi_hi","tau2","I2","Q","p_het"]}
    pi=f" PI {o['pi_lo']:.3f}-{o['pi_hi']:.3f}" if o.get('pi_lo') else ""
    print(f"{name:45s} k={o['k']:3d} AUC {o['auc']:.3f} ({o['lo']:.3f}-{o['hi']:.3f}){pi} I2={o['I2']:.1f} tau2={o['tau2']:.3f}")
def pool(rs,method="REML",hk=True): 
    a=agg(rs); return fit([x["y"] for x in a],[x["se"]**2 for x in a],method=method,hk=hk),a
for c in ["OP","LBMD","OPvN","OPEN","MULTI","OPvOPEN"]:
    rs=[x for x in rows if x["contrast"]==c]
    if len({x['id'] for x in rs})>=3: o,a=pool(rs); summ(f"{c} main (REML+HK)",o)
    else: print(c,"k<3:",[(x['id'],x['auc']) for x in rs])
OP=[x for x in rows if x["contrast"]=="OP"]
o,a=pool(OP,"DL",False); summ("OP DerSimonian-Laird (no HK)",o)
EXT=lambda x: x["set"] in ("external","temporal")
for lab,f in [("OP central DXA only",lambda x:x["central_dxa"]),("OP excl quasi-densitometric",lambda x:not x["quasi"]),
              ("OP external/temporal only",EXT),("OP internal/CV/apparent",lambda x:not EXT(x)),
              ("OP SE from CI/real counts only",lambda x:x["se_src"] in ("CI","HM_counts","HM_derived_counts")),
              ("OP imaging",lambda x:x["imaging"]),("OP clinical/tabular",lambda x:not x["imaging"]),
              ("OP deep learning",lambda x:x["dl"]),("OP conventional ML",lambda x:not x["dl"]),
              ("OP ref DXA",lambda x:x["ref"]=="DXA"),("OP ref QCT",lambda x:x["ref"]=="QCT"),("OP ref QUS/NR",lambda x:x["ref"] in("QUS","NR")),
              ("OP opportunistic CT",lambda x:x["opp_ct"]),("OP not opportunistic CT",lambda x:not x["opp_ct"])]:
    rs=[x for x in OP if f(x)]
    if len({x['id'] for x in rs})>=3: o,_=pool(rs); summ(lab,o)
    else: print(lab,"k<3")
# include non-poolable with conservative SE
ses=[x["se"] for x in agg(OP)]; q75=float(np.percentile(ses,75))
extra=[dict(x,se=q75,se_src="imputed_q75") for x in notpool if x["contrast"]=="OP"]
o,_=pool(OP+extra); summ("OP + non-poolable (SE=Q75 imputed)",o)
tl=three_level(OP); R["OP three-level"]=tl; print("OP three-level (all same-tier estimates)",tl)
# leave-one-out
a=agg(OP); loo=[fit([x["y"] for x in a if x is not z],[x["se"]**2 for x in a if x is not z])["auc"] for z in a]
R["OP LOO range"]=[min(loo),max(loo)]; print("LOO range",min(loo),max(loo))
json.dump(R,open('/tmp/ma/results.json','w'),indent=1)
