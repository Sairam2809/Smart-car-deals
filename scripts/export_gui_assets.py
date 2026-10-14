"""Precompute small GUI metadata so opening TORQUEX does not import training code."""
from pathlib import Path
import json
import hashlib
import pandas as pd
from sklearn.model_selection import train_test_split
ROOT=Path(__file__).resolve().parents[1]
df=pd.read_csv(ROOT/'data/classification_cars.csv')
train,test=train_test_split(df,test_size=.2,stratify=df.transmission,random_state=42)
allruns=pd.read_csv(ROOT/'results/all_classification_runs.csv')
selected=json.loads((ROOT/'results/selected_model.json').read_text())
winner=allruns[(allruns.Slug==selected['slug'])&(allruns.Stage==selected['stage'])].iloc[0]
choices={'brands':sorted(train.brand.unique().tolist()),'fuels':sorted(train.fuel_type.unique().tolist()),
         'models_by_brand':{b:sorted(g.model.unique().tolist()) for b,g in train.groupby('brand')},
         'year_min':int(train.year.min()),'year_max':int(train.year.max())}
summary={'title':'TORQUEX','classification_rows':len(df),'classification_train':len(train),
         'classification_test':len(test),'class_counts':df.transmission.value_counts().to_dict(),
         'classification_currency':'GBP','classification_distance':'miles','oversampling':False,
         'clustering_rows':len(pd.read_csv(ROOT/'data/used_cars.csv')),
         'selection':selected,'selected_test_metrics':winner.to_dict(),
         'after_tuning':allruns[allruns.Stage=='After tuning'].sort_values('CV_macro_F1',ascending=False).to_dict('records'),
         'models':allruns[['Slug','Model','Part']].drop_duplicates().to_dict('records'),
         'inputs':choices}
(ROOT/'results/dashboard.json').write_text(json.dumps(summary,indent=2))
train.drop(columns='transmission').head(3).to_csv(ROOT/'data/prediction_template.csv',index=False)
pd.DataFrame({'source_row':df.index,'split':['train' if i in train.index else 'test' for i in df.index]}).to_csv(ROOT/'data/classification_split.csv',index=False)
manifest={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
          for p in [ROOT/'data/classification_cars.csv',ROOT/'data/used_cars.csv']}
(ROOT/'data/source_manifest.json').write_text(json.dumps(manifest,indent=2))
print('Exported small GUI metadata, input template, split and source fingerprints.')
