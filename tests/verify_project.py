"""Checks tied to leakage, persistence, metrics, and usable notebook outputs."""
from pathlib import Path
import sys
import json
import compileall
import nbformat
import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.classification import load_data,RAW_FEATURES,TARGET,configurations,metrics,CV,automatic_probability
from src.clustering import load_data as cluster_data,predict_new,scores,FEATURES
from src.validation import validate_batch

assert compileall.compile_dir(str(ROOT/'src'),quiet=1)
assert compileall.compile_dir(str(ROOT/'app'),quiet=1)
raw,df,train,test=load_data()
assert (len(raw),len(df),len(train),len(test))==(2500,2500,2000,500)
assert not df.duplicated().any()
assert set(train.index).isdisjoint(test.index)
assert len(train.merge(test,on=list(df.columns),how='inner'))==0
for a,b in CV.split(train[RAW_FEATURES],train[TARGET]):
    assert set(train.iloc[a].index).isdisjoint(train.iloc[b].index)
    assert set(train.iloc[a].index).isdisjoint(test.index)
allruns=pd.read_csv(ROOT/'results/all_classification_runs.csv')
assert len(allruns)==15 and allruns.Slug.nunique()==10
for _,row in allruns.iterrows():
    tag='after' if row.Stage=='After tuning' else 'before'
    model=joblib.load(ROOT/'models'/f'{row.Slug}_{tag}.joblib')
    actual=metrics(model,test[RAW_FEATURES],test[TARGET])
    for key,value in actual.items():
        assert np.isclose(value,row[key],atol=1e-12),(row.Slug,key,value,row[key])
    assert 'balance' not in model.named_steps
    custom=test[RAW_FEATURES].iloc[:2].copy()
    custom.loc[custom.index[0],['price_gbp','mileage_miles']]=np.nan
    custom.loc[custom.index[1],['brand','fuel_type']]=['UnseenBrand Example','UnseenFuel']
    prob=automatic_probability(model,custom)
    assert np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all()
    assert len(model.predict(custom))==2
selected=json.loads((ROOT/'results/selected_model.json').read_text())
assert np.isclose(selected['cv_macro_f1'],allruns.CV_macro_F1.max())
print('PASS classification: split isolation, 15 saved model metrics, fold identities, no resampling, unseen/missing inputs, CV-only selection')

_,cdf,ctrain,ctest=cluster_data()
assert (len(cdf),len(ctrain),len(ctest))==(4009,3207,802)
assert set(ctrain.index).isdisjoint(ctest.index)
for kind in ['kmeans','agglomerative']:
    table=pd.read_csv(ROOT/'results'/kind/'comparison.csv')
    for tag in ['before','after']:
        bundle=joblib.load(ROOT/'models'/f'{kind}_{tag}.joblib')
        X=bundle['preprocessor'].transform(ctrain)
        labels=bundle['model'].labels_
        assert len(labels)==len(ctrain)
        row=table.iloc[0 if tag=='before' else 1]
        for key,value in scores(X,labels).items():assert np.isclose(value,row[key],atol=1e-10)
        pred=predict_new(bundle,ctest)
        for key,value in scores(bundle['preprocessor'].transform(ctest),pred).items():
            assert np.isclose(value,row['Holdout_'+key],atol=1e-10)
        assert len(pred)==802
    fold=pd.read_csv(ROOT/'results'/kind/'cv_fold_results.csv')
    assert set(fold.fold)=={1,2,3,4,5}
    assert len(fold)==(25 if kind=='kmeans' else 75)
    profiles=pd.read_csv(ROOT/'results'/kind/'cluster_profiles.csv')
    assert profiles.cars.sum()==3207
print('PASS clustering: disjoint split, saved labels, all before/after metrics, holdout assignments, five-fold searches, profile totals')

notebooks=list((ROOT/'notebooks').glob('*.ipynb'))
assert len(notebooks)==8
images=0
for path in notebooks:
    nb=nbformat.read(path,as_version=4);nbformat.validate(nb)
    counts=[]
    for cell in nb.cells:
        if cell.cell_type!='code':continue
        assert cell.execution_count is not None,path.name
        counts.append(cell.execution_count)
        assert all(o.output_type!='error' for o in cell.outputs),path.name
        images+=sum('image/png' in o.get('data',{}) for o in cell.outputs)
    assert counts==list(range(1,len(counts)+1)),path.name
assert images>=30,images
print('PASS notebooks: eight valid files, all code cells executed in order, no error outputs,',images,'embedded plots')


valid=test[RAW_FEATURES].head(5).copy()
assert len(validate_batch(valid))==5
bad_inputs=[valid.drop(columns=['brand']),valid.iloc[:0],
            valid.assign(year=2099),valid.assign(price_gbp=-1),
            valid.assign(mileage_miles='not a number'),valid.assign(price_gbp=np.inf),
            pd.concat([valid]*2001,ignore_index=True)]
for invalid in bad_inputs:
    try:validate_batch(invalid)
    except ValueError:pass
    else:raise AssertionError('Invalid batch accepted')
model=joblib.load(ROOT/'models'/selected['model_file'])
batch=validate_batch(valid)
pred=model.predict(batch[RAW_FEATURES])
single=np.array([model.predict(batch.iloc[[i]][RAW_FEATURES])[0] for i in range(len(batch))])
assert np.array_equal(pred,single)
print('PASS batch GUI backend: valid CSV rows, missing columns, empty/oversized CSV, bad numeric values, invalid ranges, batch/single prediction agreement')

# Explicit after-tuning table and correct source dataset must accompany the UI.
after=pd.read_csv(ROOT/'results/after_tuning_comparison.csv')
assert len(after)==5 and (after.Stage=='After tuning').all()
assert set(after.Slug)==set(configurations())
assert df.transmission.value_counts().to_dict()=={'Automatic':1250,'Manual':1250}
assert train.transmission.value_counts().to_dict()=={'Manual':1000,'Automatic':1000}
assert test.transmission.value_counts().to_dict()=={'Manual':250,'Automatic':250}
assert (ROOT/'app/assets/torquex_gt.svg').exists()
print('PASS updated dataset, balanced split, after-tuning table and TORQUEX asset')
print('ALL PROJECT CHECKS PASSED')
