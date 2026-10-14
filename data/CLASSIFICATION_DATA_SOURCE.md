# Balanced real used-car listings: 2,500 records

Use `classification_cars.csv` as the dataset.

| Transmission | Distinct records |
|---|---:|
| Manual | 1,250 |
| Automatic | 1,250 |
| Total | 2,500 |

The records are sampled WITHOUT replacement from the publicly released **100,000 UK Used Car
Data set** by Aditya Desai. The publisher describes the source as scraped used-car listings.
No synthetic feature values were created. No extra copies were made to balance the classes.
Every delivered feature value was checked against its source row after whitespace normalization.

Source: https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes

Publisher metadata lists CC0 / Public Domain. Retrieved 2026-10-11 (India date).
These are historical listings, not a current live market feed.

## What "unique" means here

There are zero duplicate rows across the actual car attributes, with no artificial ID used to
make repeated records appear unique. There are also zero duplicate predictor rows when the
transmission target is excluded. Source row references are unique.

However, the source does not include VINs or registration identifiers. We cannot certify that
every record refers to a different physical vehicle: the same vehicle might have been advertised
again with changed price or mileage. Describe this as **2,500 distinct real listing records**,
not 2,500 VIN-verified vehicles.

## Columns

| Column | Meaning |
|---|---|
| brand | Manufacturer, taken from the source manufacturer's file |
| model | Model name, with surrounding whitespace removed |
| year | Model/registration year reported in the source |
| price_gbp | Advertised price in British pounds, not INR |
| mileage_miles | Recorded distance in miles, not kilometres |
| fuel_type | Source fuel category |
| tax_gbp | Source vehicle tax amount in pounds |
| mpg_uk | Source-reported miles-per-gallon figure for these UK listings; no unit conversion applied |
| engine_size_l | Source engine displacement in litres |
| transmission | Classification target: Manual or Automatic |

## Preparation and checks

1. Combined nine manufacturer files: Audi, BMW, Ford, Hyundai, Mercedes-Benz, Skoda, Toyota,
   Vauxhall and Volkswagen: 99,187 source rows.
2. Did NOT append focus/cclass or their unclean files, which overlap the manufacturer files.
3. Kept only explicitly labelled Manual and Automatic rows. Semi-Auto and Other were excluded,
   not relabelled as Automatic.
4. Normalized whitespace and the Hyundai tax header. Added brand from the source filename.
5. Kept complete records with year 1970–2026, positive price/mpg/engine size and non-negative
   mileage/tax. Invalid/missing values were excluded, not invented or imputed.
6. Removed 1,089 exact duplicate rows from this eligible binary subset and two feature-identical
   rows with conflicting transmission labels.
7. The resulting pool contained 55,345 Manual and 19,792 Automatic records.
8. Randomly selected 1,250 from each class without replacement; shuffled with seed 42.
9. Verified class counts, no missing values, no duplicated feature rows, unique source references
   and exact membership of every delivered row in the normalized source data.

`classification_source_audit.json` records checks and SHA-256 file hashes. `classification_provenance.csv` maps every output data row
to the source manufacturer file and its one-based data-row number (header excluded).
Provenance is for auditing only; do not add those row numbers as model features.

## Using it for classification

```python
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/classification_cars.csv')
X = df.drop(columns='transmission')
y = df['transmission']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

This gives 2,000 training rows (1,000 per class) and 500 test rows (250 per class).
No oversampling is needed for this balanced sample. Fit encoding, imputation if later required,
scaling and hyperparameter search on training data/folds only. Keep the same split for every model.
Calling `drop_duplicates()` on this CSV will not undo the balance.

The sample has deliberately balanced class proportions; it is not representative of the natural
UK market prevalence. Test metrics on it describe this sampled population, not all real-world cars.

## Difference from the earlier project

This is a NEW UK dataset, not extra Car Dekho observations. It has different columns, currency and
distance units. There are no owner or seller_type columns. The TORQUEX notebooks and GUI have been updated and retrained for this input schema. The original
Car Dekho project needs these preprocessing/input-schema changes before using it. Do not simply rename this file
to `car_details.csv` or feed it to the previously fitted models. Rerun all ten classifiers on the new
common split for a valid comparison. The previous clustering dataset can remain separate.

## Reproduce the sample

`scripts/prepare_classification_dataset.py` downloads the public source archive when missing, performs the documented
selection and writes the CSV, provenance and audit to `reproduced_data/`. Requires Python and pandas. The preparation
run used pandas 2.2.3. Kaggle source updates may change the rows; compare the archive SHA-256
with `classification_source_audit.json` to confirm the same source version. The TORQUEX project trains and evaluates its models separately on the supplied CSV.
