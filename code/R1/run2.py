exec(open('/tmp/ma/meta.py').read())
R=json.load(open('/tmp/ma/results.json'))
a=agg([x for x in D["rows"] if x["contrast"]=="OP"])
y=np.array([x["y"] for x in a]); v=np.array([x["se"]**2 for x in a]); k=len(y)
covs={"external_or_temporal":[1.0 if x["set"] in("external","temporal") else 0 for x in a],
      "imaging":[1.0 if x["imaging"] else 0 for x in a],"deep_learning":[1.0 if x["dl"] else 0 for x in a],
      "QCT_reference":[1.0 if x["ref"]=="QCT" else 0 for x in a],"opportunistic_CT":[1.0 if x["opp_ct"] else 0 for x in a]}
X=np.column_stack([np.ones(k)]+[covs[c] for c in covs])
t0=tau2_reml(y,v); o=fit(y,v,X); tr=o["tau2"]
print("meta-regression: R2 =",max(0,(t0-tr)/t0), "residual tau2",tr,"k",k)
MR={}
for j,c in enumerate(["intercept"]+list(covs)):
    b=o["b"][j]; se=o["se"][j]; tval=b/se; p=2*(1-stats.t.cdf(abs(tval),o["df"]))
    MR[c]=dict(b=b,se=se,p=p,OR_logit=math.exp(b)); print(f"{c:22s} b={b:+.3f} se={se:.3f} p={p:.4f}")
MR["R2"]=max(0,(t0-tr)/t0); MR["tau2_res"]=tr; MR["tau2_0"]=t0; R["OP meta-regression"]=MR
# collinearity check
print("corr imaging~DL",np.corrcoef(covs["imaging"],covs["deep_learning"])[0,1], "imaging~oppCT",np.corrcoef(covs["imaging"],covs["opportunistic_CT"])[0,1])
# Egger (all) on logit AUC
se=np.sqrt(v); Z=y/se; prec=1/se
eg=stats.linregress(prec,Z); te=eg.intercept/eg.intercept_stderr; pe=2*(1-stats.t.cdf(abs(te),k-2))
print("Egger intercept",eg.intercept,"p",pe); R["Egger"]=dict(intercept=eg.intercept,p=pe,k=k)
# Deeks-type: logit AUC vs 1/sqrt(ESS), weighted by ESS, subset with counts
sub=[x for x in a if x.get("npos") and x.get("nneg")]
ess=np.array([4*x["npos"]*x["nneg"]/(x["npos"]+x["nneg"]) for x in sub]); ys=np.array([x["y"] for x in sub])
Xd=np.column_stack([np.ones(len(sub)),1/np.sqrt(ess)]); W=np.diag(ess)
b=np.linalg.solve(Xd.T@W@Xd,Xd.T@W@ys); r=ys-Xd@b; s2=(r@W@r)/(len(sub)-2); cov=s2*np.linalg.inv(Xd.T@W@Xd)
td=b[1]/math.sqrt(cov[1,1]); pd=2*(1-stats.t.cdf(abs(td),len(sub)-2))
print("Deeks-type slope",b[1],"p",pd,"k",len(sub)); R["Deeks"]=dict(slope=float(b[1]),p=pd,k=len(sub))
# bivariate sens/spec subset (needs sens, spec, npos, nneg)
bv=[x for x in a if x.get("sens") is not None and x.get("spec") is not None and x.get("npos") and x.get("nneg")]
print("bivariate subset k=",len(bv))
if len(bv)>=5:
    def cell(p,n): x=round(p*n); return x,n-x
    Y=[];S=[]
    for x in bv:
        tp,fn=cell(x["sens"],x["npos"]); tn,fp=cell(x["spec"],x["nneg"])
        tp,fn,tn,fp=[c+0.5 for c in (tp,fn,tn,fp)]
        Y.append([math.log(tp/fn),math.log(tn/fp)]); S.append([1/tp+1/fn,1/tn+1/fp])
    Y=np.array(Y);S=np.array(S)
    def nll(p):
        m1,m2,l1,l2,z=p; s1,s2=math.exp(l1),math.exp(l2); rho=math.tanh(z); tot=0
        for yi,si in zip(Y,S):
            Sig=np.array([[s1**2+si[0],rho*s1*s2],[rho*s1*s2,s2**2+si[1]]]); d=yi-[m1,m2]
            tot+=0.5*(np.linalg.slogdet(Sig)[1]+d@np.linalg.solve(Sig,d))
        return tot
    res=optimize.minimize(nll,[1,1,0,0,0],method="Nelder-Mead",options=dict(maxiter=4000))
    m1,m2,l1,l2,z=res.x; H=None
    BV=dict(k=len(bv),sens=ilg(m1),spec=ilg(m2),tau_sens=math.exp(l1),tau_spec=math.exp(l2),rho=math.tanh(z))
    print("bivariate",BV); R["bivariate"]=BV
json.dump(R,open('/tmp/ma/results.json','w'),indent=1)
