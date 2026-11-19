import json, math, csv
R1="/sessions/dazzling-brave-wright/mnt/ML_Osteoporosis_Review/Submission_Bone_MajorRevision/Revision_R1"
recs={}
for l in open(f"{R1}/reextraction_records.jsonl"):
    r=json.loads(l); recs[r["id"]]=r
EXCL_ELIG="U309 O_Hung U66 U851 U853 U1030 U248 U260".split()
EXCL_OVER="U35 U326 U330 U691 U826 O_WeiT U120".split()
NOT_TARGET="O_Wu_2021 O_Chang_2024 O_Kong_2023 O_Nicolaes_2023 O_UsateguiMartin_2025 O_ShangGuan_2025 U1004".split()
for i,r in recs.items():
    f=r.get("eligibility_flag") or ""
    if "EXCLUDE" in f and i not in EXCL_ELIG+EXCL_OVER+NOT_TARGET:
        print("UNHANDLED EXCLUDE FLAG:",i,f[:90])
print(len(recs), "excluded:",len(EXCL_ELIG)+len(EXCL_OVER)+len(NOT_TARGET))
