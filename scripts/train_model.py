
from pathlib import Path
from collections import defaultdict, deque
from datetime import datetime, timedelta
import csv, math, random
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/"data"/"raw"
INITIAL_ELO=1500.; K=28.; ALPHA=.12

def num(v):
    try:
        x=float(v); return x if np.isfinite(x) else None
    except: return None

records=[]
for p in sorted(RAW.glob("*.csv")):
    with p.open(newline="",encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try: d=datetime.strptime(str(r.get("tourney_date","")),"%Y%m%d")
            except: continue
            if not r.get("winner_id") or not r.get("loser_id"): continue
            s=(r.get("surface") or "").title()
            if s not in {"Hard","Clay","Grass"}: continue
            score=(r.get("score") or "").upper()
            completed=not any(x in score for x in ["RET","W/O","DEF"])
            records.append((d,int(float(r.get("match_num") or 0)),r,completed))
records.sort(key=lambda x:(x[0],x[1]))
if not records: raise RuntimeError("No ATP data was downloaded.")

elo=defaultdict(lambda:INITIAL_ELO); selo=defaultdict(lambda:INITIAL_ELO)
stats=defaultdict(lambda:{"serve":None,"return":None}); sstats=defaultdict(lambda:{"serve":None,"return":None})
hist=defaultdict(lambda:deque(maxlen=60)); trail=defaultdict(lambda:deque(maxlen=10))
names={}; last_seen={}; ranks={}; ages={}; rows=[]; rng=random.Random(123)

def ewm(old,new): return new if old is None else ALPHA*new+(1-ALPHA)*old
def state(pid,surface,date):
    st,ss=stats[pid],sstats[(pid,surface)]
    h=[m for m in hist[pid] if m["date"]<date]
    l5,l10=h[-5:],h[-10:]; surf=[m for m in h if m["surface"]==surface][-10:]
    overall=elo[pid]
    return {
      "overall_elo":overall,"surface_elo":selo[(pid,surface)],
      "serve":ss["serve"] if ss["serve"] is not None else (st["serve"] if st["serve"] is not None else .635),
      "return_rating":ss["return"] if ss["return"] is not None else (st["return"] if st["return"] is not None else .365),
      "win5":sum(m["win"] for m in l5)/len(l5) if l5 else .5,
      "win10":sum(m["win"] for m in l10)/len(l10) if l10 else .5,
      "surface_win10":sum(m["win"] for m in surf)/len(surf) if surf else .5,
      "opp_elo10":float(np.mean([m["opp"] for m in l10])) if l10 else 1500.,
      "recent_perf10":float(np.mean([m["perf"] for m in l10])) if l10 else 0.,
      "matches7":sum((date-m["date"]).days<=7 for m in h),
      "matches14":sum((date-m["date"]).days<=14 for m in h),
      "rest_days":min(max((date-last_seen[pid]).days if pid in last_seen else 30,0),60),
      "elo_change10":overall-trail[pid][0] if trail[pid] else 0.
    }

for date,_,r,completed in records:
    w,l=str(r["winner_id"]),str(r["loser_id"]); s=(r.get("surface") or "").title()
    names[w]=r.get("winner_name",""); names[l]=r.get("loser_name","")
    wf,lf=state(w,s,date),state(l,s,date)
    wr=num(r.get("winner_rank")) or 500.; lr=num(r.get("loser_rank")) or 500.
    wa=num(r.get("winner_age")); la=num(r.get("loser_age"))
    d=[wf["overall_elo"]-lf["overall_elo"],wf["surface_elo"]-lf["surface_elo"],
       wf["serve"]-lf["serve"],wf["return_rating"]-lf["return_rating"],
       math.log(lr)-math.log(wr),wf["win5"]-lf["win5"],wf["win10"]-lf["win10"],
       wf["surface_win10"]-lf["surface_win10"],wf["opp_elo10"]-lf["opp_elo10"],
       wf["recent_perf10"]-lf["recent_perf10"],wf["matches7"]-lf["matches7"],
       wf["matches14"]-lf["matches14"],wf["rest_days"]-lf["rest_days"],
       wf["elo_change10"]-lf["elo_change10"],(wa-la) if wa is not None and la is not None else 0.]
    flip=rng.random()<.5
    if completed: rows.append({"year":date.year,"y":0 if flip else 1,"x":[-z for z in d] if flip else d})
    pw=1/(1+10**((elo[l]-elo[w])/400)); prew,prel=elo[w],elo[l]
    delta=K*(1-pw); elo[w]+=delta; elo[l]-=delta
    ps=1/(1+10**((selo[(l,s)]-selo[(w,s)])/400)); sd=K*(1-ps); selo[(w,s)]+=sd; selo[(l,s)]-=sd
    vals=[num(r.get(k)) for k in ["w_svpt","l_svpt","w_1stWon","w_2ndWon","l_1stWon","l_2ndWon"]]
    if all(v is not None for v in vals) and vals[0]>0 and vals[1]>0:
        wsv,lsv,w1,w2,l1,l2=vals; wsp=(w1+w2)/wsv; lsp=(l1+l2)/lsv
        for pid,sp,rp in [(w,wsp,1-lsp),(l,lsp,1-wsp)]:
            stats[pid]["serve"]=ewm(stats[pid]["serve"],sp); stats[pid]["return"]=ewm(stats[pid]["return"],rp)
            sstats[(pid,s)]["serve"]=ewm(sstats[(pid,s)]["serve"],sp); sstats[(pid,s)]["return"]=ewm(sstats[(pid,s)]["return"],rp)
    hist[w].append({"date":date,"surface":s,"win":1,"opp":prel,"perf":1-pw})
    hist[l].append({"date":date,"surface":s,"win":0,"opp":prew,"perf":-(1-pw)})
    trail[w].append(prew); trail[l].append(prel); last_seen[w]=date; last_seen[l]=date
    ranks[w]=wr; ranks[l]=lr; ages[w]=wa; ages[l]=la

features=["overall_elo_diff","surface_elo_diff","serve_diff","return_diff","log_rank_advantage",
"win5_diff","win10_diff","surface_win10_diff","opp_elo10_diff","recent_perf10_diff",
"matches7_diff","matches14_diff","rest_days_diff","elo_change10_diff","age_diff"]
train=[r for r in rows if r["year"]<=2024]; test=[r for r in rows if r["year"]==2025]
pipe=Pipeline([("scale",StandardScaler()),("model",LogisticRegression(C=.7,max_iter=5000))])
pipe.fit(np.array([r["x"] for r in train]),np.array([r["y"] for r in train]))
pt=pipe.predict_proba(np.array([r["x"] for r in test]))[:,1]
metrics={"training_matches":len(train),"test_matches_2025":len(test),
"accuracy_2025":float(accuracy_score([r["y"] for r in test],pt>=.5)),
"log_loss_2025":float(log_loss([r["y"] for r in test],pt)),
"brier_2025":float(brier_score_loss([r["y"] for r in test],pt)),
"latest_data_date":max(x[0] for x in records).strftime("%Y-%m-%d")}
pipe.fit(np.array([r["x"] for r in rows]),np.array([r["y"] for r in rows]))
joblib.dump({"pipeline":pipe,"features":features,"metrics":metrics},ROOT/"model"/"model.joblib")
latest=max(x[0] for x in records)
out=[]
for pid,name in names.items():
    if pid not in last_seen or (latest-last_seen[pid]).days>730: continue
    for s in ["Hard","Clay","Grass"]:
        z=state(pid,s,latest+timedelta(days=1))
        out.append({"player_id":pid,"player":name,"surface":s,**z,"rank":int(ranks.get(pid,500)),
                    "age":float(ages.get(pid) or 0),"last_match":last_seen[pid].strftime("%Y-%m-%d")})
(ROOT/"data"/"generated").mkdir(parents=True, exist_ok=True)
pd.DataFrame(out).sort_values(["player","surface"]).to_csv(ROOT/"data"/"generated"/"player_state.csv.gz",index=False,compression="gzip")
print(metrics)
