import json, math, numpy as np
from scipy import stats, optimize
D=json.load(open('/tmp/ma/ds.json'))
ilg=lambda x: 1/(1+math.exp(-x))
def tau2_dl(y,v):
    w=1/v; mu=np.sum(w*y)/w.sum(); Q=np.sum(w*(y-mu)**2); k=len(y)
    return max(0,(Q-(k-1))/(w.sum()-np.sum(w**2)/w.sum())),Q
def tau2_reml(y,v,X=None):
    k=len(y); X=np.ones((k,1)) if X is None else X
    def nll(t):
        t=math.exp(t); W=np.diag(1/(v+t)); XtWX=X.T@W@X
        b=np.linalg.solve(XtWX,X.T@W@y); r=y-X@b
        return 0.5*(np.sum(np.log(v+t))+np.linalg.slogdet(XtWX)[1]+r@W@r)
    res=optimize.minimize_scalar(nll,bounds=(-15,5),method='bounded')
    t=math.exp(res.x); return 0.0 if t<1e-6 else t
def fit(y,v,X=None,method="REML",hk=True):
    y=np.asarray(y,float); v=np.asarray(v,float); k=len(y)
    X=np.ones((k,1)) if X is None else np.asarray(X,float); p=X.shape[1]
    t=tau2_reml(y,v,X) if method=="REML" else tau2_dl(y,v)[0]
    W=np.diag(1/(v+t)); XtWX=X.T@W@X; b=np.linalg.solve(XtWX,X.T@W@y); cov=np.linalg.inv(XtWX)
    r=y-X@b
    if hk:
        q=max(1.0, (r@W@r)/(k-p)) if False else (r@W@r)/(k-p)   # Hartung-Knapp (no truncation)
        cov=cov*q; crit=stats.t.ppf(0.975,k-p); df=k-p
    else: crit=1.959964; df=None
    se=np.sqrt(np.diag(cov))
    w=1/v; mu0=np.sum(w*y)/w.sum(); Q=float(np.sum(w*(y-mu0)**2))
    I2=max(0,(Q-(k-1))/Q)*100 if Q>0 else 0
    out=dict(k=k,b=b.tolist(),se=se.tolist(),tau2=t,Q=Q,I2=I2,crit=crit,df=df,cov=cov.tolist())
    if p==1:
        m=b[0]; out.update(auc=ilg(m),lo=ilg(m-crit*se[0]),hi=ilg(m+crit*se[0]))
        if k>2:
            pi=stats.t.ppf(0.975,k-2)*math.sqrt(t+se[0]**2); out.update(pi_lo=ilg(m-pi),pi_hi=ilg(m+pi))
        out["p_het"]=1-stats.chi2.cdf(Q,k-1)
    return out
def agg(rows):
    """one estimate per study: mean logit of same-tier estimates, var assuming r=0.5"""
    by={}
    for x in rows: by.setdefault(x["id"],[]).append(x)
    out=[]
    for i,xs in by.items():
        m=len(xs); y=sum(x["y"] for x in xs)/m
        ses=[x["se"] for x in xs]; var=sum((0.5 if a!=b else 1)*ses[a]*ses[b] for a in range(m) for b in range(m))/m**2
        z=dict(xs[0]); z.update(y=y,se=math.sqrt(var),m=m,auc=ilg(y)); out.append(z)
    return out
def three_level(rows):
    ids=sorted(set(x["id"] for x in rows)); y=np.array([x["y"] for x in rows]); v=np.array([x["se"]**2 for x in rows])
    Z=np.array([[1.0 if x["id"]==i else 0 for i in ids] for x in rows]); k=len(y); X=np.ones((k,1))
    def V(p): s2s,s2e=np.exp(p); return np.diag(v+s2e)+s2s*Z@Z.T
    def nll(p):
        Vi=np.linalg.inv(V(p)); XtVX=X.T@Vi@X; b=np.linalg.solve(XtVX,X.T@Vi@y); r=y-X@b
        return 0.5*(np.linalg.slogdet(V(p))[1]+np.linalg.slogdet(XtVX)[1]+r@Vi@r)
    res=optimize.minimize(nll,[-3,-3],method="Nelder-Mead")
    Vi=np.linalg.inv(V(res.x)); XtVX=X.T@Vi@X; b=float(np.linalg.solve(XtVX,X.T@Vi@y)); se=math.sqrt(1/XtVX[0,0])
    # cluster-robust (CR1) SE
    r=y-b; meat=0
    for i in ids:
        idx=[j for j,x in enumerate(rows) if x["id"]==i]; u=(Vi[np.ix_(idx,range(k))]@r).sum()  # approx
    rob=None
    crit=stats.t.ppf(0.975,len(ids)-1)
    return dict(k_est=k,k_studies=len(ids),auc=ilg(b),lo=ilg(b-crit*se),hi=ilg(b+crit*se),s2_study=float(np.exp(res.x[0])),s2_est=float(np.exp(res.x[1])))
