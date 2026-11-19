exec(open('/tmp/ma/meta.py').read())
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8})
R=json.load(open('/tmp/ma/results.json'))
inv=json.load(open('/sessions/dazzling-brave-wright/mnt/ML_Osteoporosis_Review/Submission_Bone_MajorRevision/Revision_R1/fulltext_inventory.json'))["update"]
def lab(x):
    if x["id"].startswith("U"):
        u=inv.get(x["id"][1:],{}); return f'{u.get("au","?").rstrip(".")} ({u.get("yr","")})'
    s=x["study"]; return s if "(" in s else s
a=agg([x for x in D["rows"] if x["contrast"]=="OP"])
for x in a: x["lab"]=lab(x)
# dedupe identical labels
from collections import Counter
cnt=Counter(x["lab"] for x in a); seen=Counter()
for x in a:
    if cnt[x["lab"]]>1: seen[x["lab"]]+=1; x["lab"]=x["lab"][:-1]+"" + chr(96+seen[x["lab"]])+")" if x["lab"].endswith(")") else x["lab"]
a.sort(key=lambda x:x["auc"])
o=fit([x["y"] for x in a],[x["se"]**2 for x in a])
n=len(a); fig,ax=plt.subplots(figsize=(7.2,0.16*n+1.6))
for i,x in enumerate(a):
    lo,hi=ilg(x["y"]-1.96*x["se"]),ilg(x["y"]+1.96*x["se"]); yy=n-i
    c="#1f4e78" if x["set"] in("external","temporal") else "#7f7f7f"
    ax.plot([lo,hi],[yy,yy],color=c,lw=0.8); ax.plot(x["auc"],yy,"s",color=c,ms=2.6)
    ax.text(0.405,yy,x["lab"],va="center",ha="left",fontsize=6)
    ax.text(1.012,yy,f'{x["auc"]:.3f} ({lo:.2f}–{hi:.2f})',va="center",fontsize=5.6)
yd=-0.5; m,l,h=o["auc"],o["lo"],o["hi"]
ax.fill([l,m,h,m],[yd,yd+0.45,yd,yd-0.45],color="#c00000")
ax.plot([o["pi_lo"],o["pi_hi"]],[yd-1.2,yd-1.2],color="#c00000",lw=1.2)
ax.text(0.405,yd,f'Pooled (REML, Hartung–Knapp), k={o["k"]}',fontsize=6.5,weight="bold",va="center")
ax.text(0.405,yd-1.2,"95% prediction interval",fontsize=6,va="center")
ax.text(1.012,yd,f'{m:.3f} ({l:.3f}–{h:.3f})',fontsize=6,weight="bold",va="center")
ax.text(1.012,yd-1.2,f'{o["pi_lo"]:.3f}–{o["pi_hi"]:.3f}',fontsize=6,va="center")
ax.axvline(m,color="#c00000",ls=":",lw=0.7); ax.set_xlim(0.40,1.0); ax.set_ylim(-2.2,n+1)
ax.set_yticks([]); ax.set_xlabel("AUC (95% CI)"); ax.set_xticks([0.6,0.7,0.8,0.9,1.0])
ax.set_title(f'Osteoporosis vs non-osteoporosis: I² = {o["I2"]:.1f}%, τ² = {o["tau2"]:.3f}',fontsize=8)
ax.text(0.40,n+0.6,"Blue: external/temporal validation; grey: internal",fontsize=6)
for s in ["top","right","left"]: ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig("Fig2_Forest_R1.png",dpi=300); plt.close()
# subgroup figure
order=[("Overall","OP main (REML+HK)"),("Validation: external/temporal","OP external/temporal only"),("Validation: internal/CV","OP internal/CV/apparent"),
 ("Data: imaging","OP imaging"),("Data: clinical/tabular","OP clinical/tabular"),("Algorithm: deep learning","OP deep learning"),("Algorithm: conventional ML","OP conventional ML"),
 ("Reference: DXA","OP ref DXA"),("Reference: QCT","OP ref QCT"),("Opportunistic CT: yes","OP opportunistic CT"),("Opportunistic CT: no","OP not opportunistic CT"),("Risk of bias: high","OP RoB High RoB"),("Risk of bias: low/unclear","OP RoB Low/unclear RoB")]
fig,ax=plt.subplots(figsize=(7.2,4.2)); n=len(order)
for i,(l,key) in enumerate(order):
    r=R[key]; yy=n-i
    ax.plot([r["pi_lo"],r["pi_hi"]],[yy,yy],color="#bdbdbd",lw=3,solid_capstyle="butt")
    ax.plot([r["lo"],r["hi"]],[yy,yy],color="#1f4e78",lw=1.4); ax.plot(r["auc"],yy,"D",color="#1f4e78" if i else "#c00000",ms=4)
    ax.text(1.005,yy,f'{r["auc"]:.3f} ({r["lo"]:.3f}–{r["hi"]:.3f})  k={r["k"]}  I²={r["I2"]:.0f}%',va="center",fontsize=6.3)
ax.set_xlim(0.35,1.0); ax.set_yticks([n-i for i in range(n)]); ax.set_yticklabels([l for l,_ in order],fontsize=7); ax.tick_params(axis="y",length=0); ax.get_yticklabels()[0].set_weight("bold"); ax.set_xlabel("AUC; dark line = 95% CI, grey bar = 95% prediction interval")
for s in ["top","right","left"]: ax.spines[s].set_visible(False)
plt.tight_layout(); plt.subplots_adjust(left=0.26,right=0.70); plt.savefig("Fig3_Subgroups_R1.png",dpi=300); plt.close()
# funnel
y=np.array([x["y"] for x in a]); se=np.array([x["se"] for x in a]); mu=math.log(o["auc"]/(1-o["auc"]))
fig,ax=plt.subplots(figsize=(5,4))
ax.scatter(y,se,s=12,color="#1f4e78",alpha=0.8); ss=np.linspace(0,se.max()*1.05,50)
ax.plot(mu-1.96*ss,ss,"k--",lw=0.7); ax.plot(mu+1.96*ss,ss,"k--",lw=0.7); ax.axvline(mu,color="#c00000",lw=0.8)
ax.invert_yaxis(); ax.set_xlabel("logit(AUC)"); ax.set_ylabel("Standard error")
eg=R["Egger"]; dk=R["Deeks"]
ax.set_title(f'Egger regression p = {eg["p"]:.3f} (k={eg["k"]}); Deeks-type test p = {dk["p"]:.2f} (k={dk["k"]})',fontsize=7.5)
plt.tight_layout(); plt.savefig("Fig4_Funnel_R1.png",dpi=300); plt.close()
print("figs ok")
