from __future__ import annotations

import os, sys, json, math, time, argparse, traceback, hashlib
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

TARGET = "outcome_log_excess_reduction"
EXPECTED_N = 1022
EXPECTED_PATIENTS = 458
EXPECTED_CONSECUTIVE = 564


def log(x):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {x}", flush=True)


def sha256(p: Path):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()


def read_any(p: Path):
    # pandas handles .csv.gz automatically when compression='infer'.
    if p.name.lower().endswith((".csv", ".csv.gz")):
        errs=[]
        for enc in (None, "utf-8-sig", "cp949", "latin1"):
            try:
                kw=dict(low_memory=False, compression="infer")
                if enc: kw["encoding"] = enc
                return pd.read_csv(p, **kw)
            except Exception as e:
                errs.append(repr(e))
        raise RuntimeError(f"CSV_READ_FAILED path={p} errors={errs[-2:]}")
    if p.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(p)
    if p.suffix.lower() == ".pkl":
        obj = pd.read_pickle(p)
        if isinstance(obj, pd.DataFrame): return obj.copy()
        raise RuntimeError(f"PKL_NOT_DATAFRAME path={p} type={type(obj)}")
    raise RuntimeError(f"UNSUPPORTED_FILE={p}")


def col_exact_or_ci(df, names):
    cmap={str(c).lower():str(c) for c in df.columns}
    for n in names:
        if n.lower() in cmap: return cmap[n.lower()]
    return None


def safe_semantic(df, include_tokens, reject=("next","future","outcome","target","tplus1","post")):
    hits=[]
    for c in df.columns:
        lc=str(c).lower().replace("-","_")
        if any(r in lc for r in reject):
            continue
        if all(any(t in lc for t in token_group) for token_group in include_tokens):
            hits.append(str(c))
    return hits[0] if len(hits)==1 else None


def resolve_source(root: Path):
    # Exact known lineage first; no project-wide guessing.
    pats=[
        "T1D_ADDS_V33_HASHIMOTO_PHARMACODYNAMIC_RESPONSE_*/03_ALL_TRANSITIONS.csv",
        "T1D_ADDS_V33_HASHIMOTO_PHARMACODYNAMIC_RESPONSE_*/03_ALL_TRANSITIONS.csv.gz",
        "T1D_ADDS_V29_10_15_HASHIMOTO_CANONICAL_TRUTH_AUDIT_*/06_CANONICAL_1022_TRANSITIONS.csv",
    ]
    cand=[]
    for pat in pats: cand += list(root.glob(pat))
    valid=[]
    for p in sorted(set(cand), key=lambda x:x.stat().st_mtime, reverse=True):
        try:
            d=read_any(p)
            pc=col_exact_or_ci(d,["patient_id","__patient_id__"])
            tc=col_exact_or_ci(d,[TARGET])
            if pc and tc and len(d)==EXPECTED_N and d[pc].astype(str).nunique()==EXPECTED_PATIENTS and pd.to_numeric(d[tc],errors="coerce").notna().sum()==EXPECTED_N:
                valid.append((p,d,pc,tc))
                log(f"SOURCE_VALID={p} rows={len(d)} patients={d[pc].astype(str).nunique()}")
        except Exception as e:
            log(f"SOURCE_REJECT={p} reason={repr(e)}")
    if not valid:
        raise RuntimeError("EXACT_SOURCE_NOT_FOUND. Expected V33 03_ALL_TRANSITIONS.csv or V29 canonical 1022 file")
    # Prefer V33 03_ALL_TRANSITIONS because it contains transition-level clinical covariates.
    valid.sort(key=lambda x: ("03_ALL_TRANSITIONS" in x[0].name, x[0].stat().st_mtime), reverse=True)
    return valid[0]


def metrics(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float)
    m=np.isfinite(y)&np.isfinite(p); y=y[m]; p=p[m]
    if len(y)<3: return dict(n=len(y),r2=np.nan,rmse=np.nan,mae=np.nan,pearson=np.nan,spearman=np.nan)
    return dict(n=len(y), r2=r2_score(y,p), rmse=mean_squared_error(y,p)**0.5,
                mae=mean_absolute_error(y,p), pearson=stats.pearsonr(y,p).statistic,
                spearman=stats.spearmanr(y,p).statistic)


def align_oof_to_source(src, oof, pc, tc):
    opc=col_exact_or_ci(oof,["patient_id","__patient_id__"])
    obs=col_exact_or_ci(oof,["observed",TARGET,"y_true","target"])
    pred=col_exact_or_ci(oof,["predicted","oof_pred","oof_prediction","y_pred","yhat"])
    if not opc or not pred:
        return None, None, None
    ysrc=pd.to_numeric(src[tc],errors="coerce").to_numpy(float)

    def ok(d):
        if len(d)!=len(src): return False
        if not np.array_equal(src[pc].astype(str).to_numpy(), d[opc].astype(str).to_numpy()): return False
        if obs:
            yo=pd.to_numeric(d[obs],errors="coerce").to_numpy(float)
            if not np.isclose(ysrc,yo,rtol=1e-9,atol=1e-10,equal_nan=True).all(): return False
        return True

    d=oof.reset_index(drop=True)
    if ok(d): return d,pred,obs

    pair=col_exact_or_ci(d,["pair_index","row_index","transition_index"])
    if pair:
        q=d.sort_values(pair).reset_index(drop=True)
        if ok(q): return q,pred,obs

    # Last exact-safe alignment: patient + within-patient occurrence order, then verify observed target.
    if obs:
        a=src[[pc,tc]].copy(); a["__occ__"]=a.groupby(pc,sort=False).cumcount()
        b=d[[opc,obs,pred]].copy(); b["__occ__"]=b.groupby(opc,sort=False).cumcount()
        b=b.rename(columns={opc:pc})
        m=a.merge(b,on=[pc,"__occ__"],how="left",validate="one_to_one",sort=False)
        if len(m)==len(src) and m[pred].notna().all():
            yo=pd.to_numeric(m[obs],errors="coerce").to_numpy(float)
            if np.isclose(ysrc,yo,rtol=1e-9,atol=1e-10,equal_nan=True).all():
                return m,pred,obs
    return None,None,None


def resolve_oof(root:Path,src,pc,tc):
    pats=[
        "T1D_ADDS_V34_12_LOCKED_PRIMARY_TARGET_PATIENT_GROUPED_OOF_*/07_FINAL_1022_AVERAGED_OOF.csv.gz",
        "T1D_ADDS_V34_12_LOCKED_PRIMARY_TARGET_PATIENT_GROUPED_OOF_*/07_FINAL_1022_AVERAGED_OOF.csv",
        "T1D_ADDS_V34_30_TARGETED_SCIPYFREE_EXACT_RECOVERY_*/00_EXACT_RECOVERED_1022.csv",
        "T1D_ADDS_V34_31_ONE_SHOT_FINAL_LONGITUDINAL_CDSS_*/10_CDSS_VALIDATION_OUTPUT_1022.csv",
    ]
    cand=[]
    for pat in pats: cand += list(root.glob(pat))
    valid=[]
    y=pd.to_numeric(src[tc],errors="coerce").to_numpy(float)
    for p in sorted(set(cand),key=lambda x:x.stat().st_mtime,reverse=True):
        try:
            d=read_any(p)
            a,pred,obs=align_oof_to_source(src,d,pc,tc)
            if a is None: continue
            pr=pd.to_numeric(a[pred],errors="coerce").to_numpy(float)
            if np.isfinite(pr).sum()!=EXPECTED_N: continue
            met=metrics(y,pr)
            log(f"OOF_VALID={p} pred={pred} r={met['pearson']:.4f} RMSE={met['rmse']:.4f} MAE={met['mae']:.4f}")
            # Fingerprint guard: broad enough for minor locked-version differences.
            if 0.45<=met['pearson']<=0.75 and 0.55<=met['rmse']<=1.0 and 0.35<=met['mae']<=0.75:
                score=(100 if "07_FINAL_1022_AVERAGED_OOF" in p.name else 0) - abs(met['pearson']-0.616)*50 - abs(met['rmse']-0.756)*20 - abs(met['mae']-0.510)*20
                valid.append((score,p,a,pred,met))
        except Exception as e:
            log(f"OOF_REJECT={p} reason={repr(e)}")
    if not valid:
        raise RuntimeError("EXACT_LOCKED_OOF_NOT_FOUND_OR_FINGERPRINT_MISMATCH. Expected 07_FINAL_1022_AVERAGED_OOF.csv.gz")
    valid.sort(key=lambda x:(x[0],x[1].stat().st_mtime),reverse=True)
    return valid[0][1],valid[0][2],valid[0][3],valid[0][4]


def resolve_clinical_cols(src):
    # Current/baseline only. Explicitly exclude next/outcome/future columns to prevent leakage.
    tsh=col_exact_or_ci(src,["baseline_tsh","current_tsh","tsh_current","tsh"])
    if tsh is None:
        cand=[c for c in src.columns if "tsh" in str(c).lower() and not any(z in str(c).lower() for z in ["next","outcome","future","target","delta"])]
        if len(cand)==1: tsh=cand[0]
    ft4=col_exact_or_ci(src,["baseline_ft4","current_ft4","ft4_current","ft4","free_t4","baseline_free_t4"])
    if ft4 is None:
        cand=[c for c in src.columns if any(z in str(c).lower() for z in ["ft4","free_t4","free thyroxine"]) and not any(z in str(c).lower() for z in ["next","outcome","future","target","delta"])]
        if len(cand)==1: ft4=cand[0]
    dose=col_exact_or_ci(src,["prior_dose_numeric","baseline_lt4_dose","current_lt4_dose","lt4_dose","levothyroxine_dose","thyroxine_dose"])
    if dose is None:
        cand=[c for c in src.columns if ("dose" in str(c).lower() and any(z in str(c).lower() for z in ["lt4","levothyrox","thyrox","prior_dose"])) and not any(z in str(c).lower() for z in ["next","outcome","future"])]
        if len(cand)==1: dose=cand[0]
    ordinal=col_exact_or_ci(src,["outcome_lab_index","transition_ordinal","visit_ordinal","transition_number","ordinal"])
    return tsh,ft4,dose,ordinal


def validate_covariate(df, col, label, min_nonmissing=10, min_unique=2):
    """Return col only when it has usable observed information; otherwise return None.

    This specifically prevents an all-missing placeholder column (e.g. baseline_ft4)
    from entering SimpleImputer/StandardScaler and changing the intended benchmark.
    """
    if col is None:
        log(f"CLINICAL_COVERAGE {label}=NOT_FOUND")
        return None
    s = pd.to_numeric(df[col], errors="coerce")
    n = int(s.notna().sum())
    u = int(s.dropna().nunique())
    cov = n / max(len(df), 1)
    log(f"CLINICAL_COVERAGE {label} col={col} nonmissing={n}/{len(df)} coverage={cov:.3f} unique={u}")
    if n < min_nonmissing or u < min_unique:
        log(f"CLINICAL_COLUMN_EXCLUDED {label} col={col} reason=INSUFFICIENT_OBSERVED_INFORMATION")
        return None
    return col


def model_pipe(cols,df):
    # Defensive second gate: never send an all-missing column to sklearn imputers.
    usable=[]
    for c in cols:
        if c not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            if pd.to_numeric(df[c],errors="coerce").notna().sum() > 0:
                usable.append(c)
        elif df[c].notna().sum() > 0:
            usable.append(c)
    if not usable:
        raise RuntimeError(f"NO_USABLE_FEATURES_FOR_BASELINE requested={cols}")
    nums=[c for c in usable if pd.api.types.is_numeric_dtype(df[c])]
    cats=[c for c in usable if c not in nums]
    trans=[]
    if nums:
        trans.append(("num",Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler())]),nums))
    if cats:
        try: oh=OneHotEncoder(handle_unknown="ignore",sparse_output=False)
        except TypeError: oh=OneHotEncoder(handle_unknown="ignore",sparse=False)
        trans.append(("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",oh)]),cats))
    return Pipeline([("pre",ColumnTransformer(trans,remainder="drop")),("reg",LinearRegression())])


def grouped_oof(df,ycol,pc,features):
    y=pd.to_numeric(df[ycol],errors="coerce").to_numpy(float)
    g=df[pc].astype(str).to_numpy(); out=np.full(len(df),np.nan); fold=np.full(len(df),-1,int)
    gkf=GroupKFold(n_splits=5)
    for k,(tr,te) in enumerate(gkf.split(df,y,g),1):
        if set(g[tr]) & set(g[te]): raise RuntimeError(f"PATIENT_LEAKAGE_FOLD_{k}")
        m=model_pipe(features,df); m.fit(df.iloc[tr][features],y[tr]); out[te]=m.predict(df.iloc[te][features]); fold[te]=k
        log(f"BASELINE_OOF fold={k}/5 train={len(tr)} test={len(te)} features={features}")
    return out,fold


def derive_previous_target(src,pc,target_col,ordinal):
    out=pd.Series(np.nan,index=src.index,dtype=float)
    if ordinal is None: return out,"UNAVAILABLE_NO_ORDINAL",0
    w=pd.DataFrame({"idx":np.arange(len(src)),"p":src[pc].astype(str),"ord":pd.to_numeric(src[ordinal],errors="coerce"),"y":pd.to_numeric(src[target_col],errors="coerce")})
    if w["ord"].isna().any(): return out,f"UNAVAILABLE_NONNUMERIC_{ordinal}",0
    w=w.sort_values(["p","ord","idx"])
    w["py"]=w.groupby("p",sort=False)["y"].shift(1); w["po"]=w.groupby("p",sort=False)["ord"].shift(1)
    w["cur"]=np.where((w["ord"]-w["po"])==1,w["py"],np.nan)
    out.iloc[w["idx"].to_numpy()]=w["cur"].to_numpy()
    return out,f"previous_{TARGET}_when_{ordinal}_increments_by_1",int(out.notna().sum())


def _cluster_boot_chunk(seed0,nrep,groups,y,b,m):
    rng=np.random.default_rng(seed0); uniq=np.unique(groups); cache={u:np.where(groups==u)[0] for u in uniq}
    arr=np.empty((nrep,5),float)
    for i in range(nrep):
        draw=rng.choice(uniq,size=len(uniq),replace=True); idx=np.concatenate([cache[u] for u in draw])
        mb=metrics(y[idx],b[idx]); mm=metrics(y[idx],m[idx])
        arr[i]=[mm['r2']-mb['r2'], mb['rmse']-mm['rmse'], mb['mae']-mm['mae'], mm['pearson']-mb['pearson'], mm['spearman']-mb['spearman']]
    return arr


def paired_boot(groups,y,b,m,nboot,workers,seed,label):
    chunks=max(workers*4,8); sizes=[nboot//chunks]*chunks
    for i in range(nboot%chunks): sizes[i]+=1
    jobs=[(seed+100003*i,s) for i,s in enumerate(sizes) if s]
    res=[]; done=0; t0=time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs=[ex.submit(_cluster_boot_chunk,se,n,groups,y,b,m) for se,n in jobs]
        for f in as_completed(futs):
            a=f.result(); res.append(a); done+=len(a)
            rate=done/max(time.time()-t0,1e-6); eta=(nboot-done)/max(rate,1e-9)
            log(f"BOOT_PROGRESS {label} {done}/{nboot} {100*done/nboot:.1f}% rate={rate:.1f}/s ETA={eta/60:.1f}m")
    return np.vstack(res)


def summarize_inc(base_name,y,b,m,groups,nboot,workers,seed):
    mb=metrics(y,b); mm=metrics(y,m)
    arr=paired_boot(groups,y,b,m,nboot,workers,seed,base_name)
    names=["delta_R2_ML_minus_baseline","delta_RMSE_baseline_minus_ML","delta_MAE_baseline_minus_ML","delta_Pearson_ML_minus_baseline","delta_Spearman_ML_minus_baseline"]
    point=[mm['r2']-mb['r2'],mb['rmse']-mm['rmse'],mb['mae']-mm['mae'],mm['pearson']-mb['pearson'],mm['spearman']-mb['spearman']]
    rows=[]
    for j,nm in enumerate(names):
        v=arr[:,j]; lo,hi=np.nanpercentile(v,[2.5,97.5]); p=2*min(np.mean(v<=0),np.mean(v>=0)); p=min(1.0,float(p))
        rows.append(dict(baseline=base_name,metric=nm,estimate=point[j],ci_low=lo,ci_high=hi,p_two_sided=p,n=len(y),patient_n=pd.Series(groups).nunique(),bootstrap_n=nboot))
    return mb,mm,pd.DataFrame(rows)


def _corr_boot_chunk(seed0,nrep,groups,x,y):
    rng=np.random.default_rng(seed0); uniq=np.unique(groups); cache={u:np.where(groups==u)[0] for u in uniq}; a=np.empty((nrep,2),float)
    for i in range(nrep):
        draw=rng.choice(uniq,size=len(uniq),replace=True); idx=np.concatenate([cache[u] for u in draw]); xx=x[idx]; yy=y[idx]
        a[i,0]=stats.pearsonr(xx,yy).statistic if np.nanstd(xx)>0 and np.nanstd(yy)>0 else np.nan
        a[i,1]=stats.spearmanr(xx,yy).statistic if np.nanstd(xx)>0 and np.nanstd(yy)>0 else np.nan
    return a


def corr_boot(df,pc,xcol,ycol,nboot,workers,seed,label):
    q=df[[pc,xcol,ycol]].dropna().copy(); g=q[pc].astype(str).to_numpy(); x=pd.to_numeric(q[xcol],errors="coerce").to_numpy(float); y=pd.to_numeric(q[ycol],errors="coerce").to_numpy(float)
    m=np.isfinite(x)&np.isfinite(y); g,x,y=g[m],x[m],y[m]
    point=[stats.pearsonr(x,y).statistic,stats.spearmanr(x,y).statistic]
    chunks=max(workers*4,8); sizes=[nboot//chunks]*chunks
    for i in range(nboot%chunks): sizes[i]+=1
    res=[]; done=0;t0=time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs=[ex.submit(_corr_boot_chunk,seed+100019*i,s,g,x,y) for i,s in enumerate(sizes) if s]
        for f in as_completed(futs):
            a=f.result();res.append(a);done+=len(a);rate=done/max(time.time()-t0,1e-6);eta=(nboot-done)/max(rate,1e-9)
            log(f"TRAJ_BOOT_PROGRESS {label} {done}/{nboot} {100*done/nboot:.1f}% rate={rate:.1f}/s ETA={eta/60:.1f}m")
    a=np.vstack(res); rows=[]
    for j,nm in enumerate(["Pearson_r","Spearman_rho"]):
        lo,hi=np.nanpercentile(a[:,j],[2.5,97.5]); rows.append(dict(analysis=label,metric=nm,estimate=point[j],ci_low=lo,ci_high=hi,n=len(x),patient_n=pd.Series(g).nunique(),bootstrap_n=nboot))
    return pd.DataFrame(rows)


def residualize(y,x):
    y=np.asarray(y,float);x=np.asarray(x,float); X=np.c_[np.ones(len(x)),x]; beta=np.linalg.lstsq(X,y,rcond=None)[0]; return y-X@beta


def cluster_ols(y,X,groups,names,label):
    import statsmodels.api as sm
    X=sm.add_constant(np.asarray(X,float),has_constant="add"); mod=sm.OLS(np.asarray(y,float),X).fit(cov_type="cluster",cov_kwds={"groups":np.asarray(groups)})
    terms=["Intercept"]+names; rows=[]
    ci=mod.conf_int()
    for i,t in enumerate(terms): rows.append(dict(analysis=label,term=t,beta=mod.params[i],se=mod.bse[i],ci_low=ci[i,0],ci_high=ci[i,1],p=mod.pvalues[i],n=len(y),patient_n=pd.Series(groups).nunique()))
    return pd.DataFrame(rows)


def mixedlm(y,pred,current,groups):
    import statsmodels.api as sm
    X=sm.add_constant(pd.DataFrame({"predicted_next_response":pred,"current_response":current}),has_constant="add")
    fit=sm.MixedLM(endog=np.asarray(y,float),exog=X,groups=np.asarray(groups)).fit(reml=False,method="lbfgs",disp=False)
    ci=fit.conf_int(); rows=[]
    for t in X.columns:
        rows.append(dict(analysis="MixedLM observed_next ~ predicted_next + current_response + (1|patient)",term=t,beta=fit.params[t],se=fit.bse[t],ci_low=ci.loc[t,0],ci_high=ci.loc[t,1],p=fit.pvalues[t],n=len(y),patient_n=pd.Series(groups).nunique()))
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",required=True); ap.add_argument("--bootstrap",type=int,default=20000); ap.add_argument("--workers",type=int,default=20); ap.add_argument("--seed",type=int,default=42); args=ap.parse_args()
    root=Path(args.root).resolve();
    if not root.exists(): raise RuntimeError(f"ROOT_NOT_FOUND={root}")
    out=root/f"T1D_ADDS_V35_4_EXACT_LOCKED_BENCHMARK_{datetime.now().strftime('%Y%m%d_%H%M%S')}"; out.mkdir(parents=True,exist_ok=False)
    log(f"ROOT={root}");log(f"OUT={out}");log(f"CPU_LOGICAL={os.cpu_count()} WORKERS={args.workers} BOOTSTRAP={args.bootstrap}")
    try:
        src_path,src,pc,target_col=resolve_source(root); src=src.reset_index(drop=True)
        oof_path,oof,pred_col,oof_met=resolve_oof(root,src,pc,target_col)
        ml=pd.to_numeric(oof[pred_col],errors="coerce").to_numpy(float); y=pd.to_numeric(src[target_col],errors="coerce").to_numpy(float); groups=src[pc].astype(str).to_numpy()
        log(f"LOCKED_SOURCE={src_path}");log(f"LOCKED_OOF={oof_path}");log(f"LOCKED_OOF_METRICS={json.dumps(oof_met)}")

        tsh_raw,ft4_raw,dose_raw,ordinal=resolve_clinical_cols(src)
        log(f"CLINICAL_COLUMNS_RAW TSH={tsh_raw} FT4={ft4_raw} LT4_DOSE={dose_raw} ORDINAL={ordinal}")
        tsh=validate_covariate(src,tsh_raw,"TSH")
        ft4=validate_covariate(src,ft4_raw,"FT4")
        dose=validate_covariate(src,dose_raw,"LT4_DOSE")
        log(f"CLINICAL_COLUMNS_USABLE TSH={tsh} FT4={ft4} LT4_DOSE={dose} ORDINAL={ordinal}")
        if tsh is None: raise RuntimeError("SAFE_CURRENT_TSH_NOT_USABLE_IN_EXACT_TRANSITION_SOURCE")

        baselines={}; fold=None
        baselines["TSH_only_linear"],fold=grouped_oof(src,target_col,pc,[tsh])
        clinical=[c for c in [tsh,ft4,dose] if c]
        if len(clinical)>=2:
            if ft4 and dose:
                name="TSH_FT4_LT4_linear"
            elif dose:
                name="TSH_LT4_linear"
            else:
                name="TSH_FT4_linear"
            log(f"ROUTINE_BENCHMARK={name} features={clinical}")
            baselines[name],_=grouped_oof(src,target_col,pc,clinical)
        else:
            log("ROUTINE_MULTIVARIABLE_BENCHMARK_SKIPPED: no usable FT4/LT4 covariate beyond TSH")

        prev,prevprov,prevn=derive_previous_target(src,pc,target_col,ordinal); log(f"PERSISTENCE_CURRENT_RESPONSE={prevprov} N={prevn}")
        if prevn:
            baselines["Persistence_Yt_equals_previous_transition_response"]=prev.to_numpy(float)
            if prevn!=EXPECTED_CONSECUTIVE: log(f"WARNING_CONSECUTIVE_N expected={EXPECTED_CONSECUTIVE} observed={prevn}")

        predout=pd.DataFrame({"row_index":np.arange(len(src)),"patient_id":groups,"observed_next_response":y,"final_locked_ml_oof":ml,"baseline_fold":fold,"previous_transition_response":prev})
        for k,v in baselines.items(): predout[k]=v
        predout.to_csv(out/"01_BENCHMARK_OOF_PREDICTIONS.csv",index=False)

        metric_rows=[{"model":"Final_locked_ML","subset":"all_1022","patient_n":EXPECTED_PATIENTS,**metrics(y,ml)}]; incs=[]
        for j,(name,b) in enumerate(baselines.items(),1):
            b=np.asarray(b,float); mask=np.isfinite(y)&np.isfinite(ml)&np.isfinite(b); yy,mm,bb,gg=y[mask],ml[mask],b[mask],groups[mask]
            mb,mmm,inc=summarize_inc(name,yy,bb,mm,gg,args.bootstrap,args.workers,args.seed+100000*j)
            metric_rows.append({"model":name,"subset":f"matched_{name}","patient_n":pd.Series(gg).nunique(),**mb})
            metric_rows.append({"model":"Final_locked_ML","subset":f"matched_{name}","patient_n":pd.Series(gg).nunique(),**mmm})
            incs.append(inc)
        pd.DataFrame(metric_rows).to_csv(out/"02_BENCHMARK_METRICS.csv",index=False)
        incdf=pd.concat(incs,ignore_index=True) if incs else pd.DataFrame(); incdf.to_csv(out/"03_PAIRED_INCREMENTAL_METRICS_20K_BOOTSTRAP.csv",index=False)

        # Shared-baseline decoupling uses only true consecutive transitions with a real previous response.
        traj=pd.DataFrame({"patient_id":groups,"y_next":y,"pred_next":ml,"y_current":prev})
        con=traj.dropna(subset=["y_current"]).copy(); con["obs_delta"]=con.y_next-con.y_current; con["pred_delta"]=con.pred_next-con.y_current
        con["obs_innovation"]=residualize(con.y_next,con.y_current); con["pred_innovation"]=residualize(con.pred_next,con.y_current)
        cparts=[
            corr_boot(con,"patient_id","pred_next","y_next",args.bootstrap,args.workers,args.seed+700001,"Direct next-visit predicted vs observed within consecutive-pair subset"),
            corr_boot(con,"patient_id","pred_delta","obs_delta",args.bootstrap,args.workers,args.seed+700002,"Raw delta (shared baseline; descriptive only)"),
            corr_boot(con,"patient_id","pred_innovation","obs_innovation",args.bootstrap,args.workers,args.seed+700003,"Innovation correlation after removing current-response effect"),
        ]
        traj_results=pd.concat(cparts,ignore_index=True);traj_results.to_csv(out/"04_TRAJECTORY_DECOUPLING_BOOTSTRAP.csv",index=False)
        con.to_csv(out/"05_TRAJECTORY_CONSECUTIVE_564.csv",index=False)

        regs=[cluster_ols(con.y_next,np.c_[con.pred_next,con.y_current],con.patient_id,["predicted_next_response","current_response"],"Cluster-robust cross-lag: observed_next ~ predicted_next + current_response")]
        try: regs.append(mixedlm(con.y_next.to_numpy(float),con.pred_next.to_numpy(float),con.y_current.to_numpy(float),con.patient_id.to_numpy()))
        except Exception as e: log(f"MIXEDLM_WARNING={repr(e)}")
        regdf=pd.concat(regs,ignore_index=True,sort=False); regdf.to_csv(out/"06_CROSSLAG_AND_MIXED_EFFECTS.csv",index=False)

        manifest={"source":str(src_path),"source_sha256":sha256(src_path),"oof":str(oof_path),"oof_sha256":sha256(oof_path),"rows":len(src),"patients":src[pc].astype(str).nunique(),"target":target_col,"patient_col":pc,"oof_prediction_col":pred_col,"tsh_col":tsh,"ft4_col":ft4,"lt4_dose_col":dose,"ordinal_col":ordinal,"consecutive_n":int(len(con)),"bootstrap":args.bootstrap,"workers":args.workers,"seed":args.seed,"policies":{"no_synthetic_dates":True,"no_automatic_time_transform":True,"no_first_observed_as_onset":True,"no_model_output_as_truth":True}}
        (out/"07_MANIFEST.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
        (out/"08_SUMMARY.txt").write_text("\n".join(["V35.4 EXACT LOCKED CLINICAL BENCHMARK + TRAJECTORY DECOUPLING","="*100,json.dumps(manifest,ensure_ascii=False,indent=2),"\nBENCHMARK METRICS\n"+pd.DataFrame(metric_rows).to_string(index=False),"\nINCREMENTAL\n"+incdf.to_string(index=False),"\nTRAJECTORY\n"+traj_results.to_string(index=False),"\nREGRESSION\n"+regdf.to_string(index=False)]),encoding="utf-8")
        log("FINAL_STATUS=PASS_V35_4");log(f"OUTPUT={out}")
    except Exception as e:
        (out/"FATAL_ERROR.txt").write_text(traceback.format_exc(),encoding="utf-8")
        log(f"FINAL_STATUS=FAIL_V35_4 error={repr(e)}"); traceback.print_exc(); raise

if __name__=="__main__":
    main()
