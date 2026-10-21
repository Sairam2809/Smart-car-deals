"""Create a balanced sample from real source listings; never oversample or synthesize."""
from pathlib import Path
import hashlib
import json
import urllib.request
import zipfile
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]/'reproduced_data'
ROOT.mkdir(exist_ok=True)
SOURCE='https://www.kaggle.com/api/v1/datasets/download/adityadesai13/used-car-dataset-ford-and-mercedes'
ARCHIVE=ROOT/'uk_source.zip'
if not ARCHIVE.exists():
    ARCHIVE.write_bytes(urllib.request.urlopen(SOURCE,timeout=90).read())
brands={'audi.csv':'Audi','bmw.csv':'BMW','ford.csv':'Ford','hyundi.csv':'Hyundai',
        'merc.csv':'Mercedes-Benz','skoda.csv':'Skoda','toyota.csv':'Toyota',
        'vauxhall.csv':'Vauxhall','vw.csv':'Volkswagen'}
columns=['brand','model','year','price','mileage','fuelType','tax','mpg','engineSize','transmission']
rename={'price':'price_gbp','mileage':'mileage_miles','fuelType':'fuel_type',
        'tax':'tax_gbp','mpg':'mpg_uk','engineSize':'engine_size_l'}
frames=[]
with zipfile.ZipFile(ARCHIVE) as z:
    for filename,brand in brands.items():
        d=pd.read_csv(z.open(filename)).rename(columns={'tax(£)':'tax'})
        d['brand']=brand
        d['source_file']=filename
        d['source_data_row']=range(1,len(d)+1)
        for c in ['model','transmission','fuelType']:d[c]=d[c].str.strip()
        frames.append(d)
raw=pd.concat(frames,ignore_index=True)
binary=raw[raw.transmission.isin(['Manual','Automatic'])].copy()
valid=binary.dropna(subset=columns)
valid=valid[valid.year.between(1970,2026)&(valid.price>0)&(valid.mileage>=0)
            &(valid.tax>=0)&(valid.mpg>0)&(valid.engineSize>0)].copy()
exact_removed=int(valid.duplicated(subset=columns).sum())
dedup=valid.drop_duplicates(subset=columns)
feature_cols=[c for c in columns if c!='transmission']
# Remove all conflicting feature-identical groups, rather than silently choose a label.
conflict=dedup.duplicated(subset=feature_cols,keep=False)
eligible=dedup[~conflict].copy()
sample=pd.concat([eligible[eligible.transmission.eq(label)].sample(n=1250,replace=False,random_state=42)
                  for label in ['Manual','Automatic']]).sample(frac=1,random_state=42).reset_index(drop=True)
output=sample[columns].rename(columns=rename)
assert len(output)==2500 and output.transmission.value_counts().to_dict()=={'Manual':1250,'Automatic':1250}
assert not output.duplicated().any()
assert not output.drop(columns='transmission').duplicated().any()
assert not sample[['source_file','source_data_row']].duplicated().any()
assert output.notna().all().all()
# Confirm every selected value matches its normalized source row, not only its ID.
source_by_key=raw.set_index(['source_file','source_data_row'])
for _,row in sample.iterrows():
    original=source_by_key.loc[(row.source_file,row.source_data_row)]
    assert all(original[c]==row[c] for c in columns)
output.to_csv(ROOT/'classification_cars.csv',index=False)
provenance=sample[['source_file','source_data_row']].copy()
provenance.insert(0,'output_data_row',range(1,2501))
provenance.to_csv(ROOT/'classification_provenance.csv',index=False)
audit={'source_url':SOURCE,'source_page':'https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes',
       'source_archive_sha256':hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
       'source_manufacturer_files':list(brands),'source_rows':len(raw),'binary_rows':len(binary),
       'valid_binary_rows':len(valid),'exact_duplicate_rows_removed':exact_removed,
       'conflicting_feature_identical_rows_removed':int(conflict.sum()),
       'eligible_counts':eligible.transmission.value_counts().to_dict(),
       'output_counts':output.transmission.value_counts().to_dict(),'output_shape':list(output.shape),
       'output_duplicates_excluding_metadata':int(output.duplicated().sum()),
       'output_duplicate_feature_rows_excluding_target':int(output.drop(columns='transmission').duplicated().sum()),
       'missing_values':int(output.isna().sum().sum()),'seed':42,'sampling_with_replacement':False,
       'synthetic_rows':0,'copied_extra_rows':0,'every_row_verified_against_source':True,
       'csv_sha256':hashlib.sha256((ROOT/'classification_cars.csv').read_bytes()).hexdigest()}
(ROOT/'classification_source_audit.json').write_text(json.dumps(audit,indent=2))
print(json.dumps(audit,indent=2))
print(pd.crosstab(output.brand,output.transmission).to_string())
