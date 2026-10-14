"""Exercise the actual Streamlit script, navigation and prediction form submissions."""
from pathlib import Path
import sys
import tempfile
import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def ok(at,label):
    assert not at.exception,[(e.message,e.stack_trace) for e in at.exception]
    assert not at.error,[e.value for e in at.error]
    print('PASS',label)

at=AppTest.from_file(str(ROOT/'app/app.py'),default_timeout=90).run()
ok(at,'Overview')
at.sidebar.radio[0].set_value('Predict transmission').run()
ok(at,'Prediction page')
at.button[0].click().run()
ok(at,'Single classification form')
assert any('Predicted transmission:' in s.value for s in at.success)
for name in ['Random Forest','AdaBoost','Gradient Boosting','Bagging','MLP','Logistic Regression','KNN','Gaussian Naive Bayes','Decision Tree','SVC']:
    at.selectbox[0].set_value(name).run()
    at.button[0].click().run()
    ok(at,'Predict '+name)
at.sidebar.radio[0].set_value('Model comparisons').run()
ok(at,'Ten-model comparison')
at.radio(key='comparison_view').set_value('After tuning').run()
ok(at,'After-tuning-only comparison')
at.radio(key='comparison_view').set_value('All ten models').run()
ok(at,'All ten new-dataset models')
at.radio(key='comparison_view').set_value('Model diagnostics').run()
ok(at,'Part B tuning comparison')
for slug in ['random_forest','adaboost','gradient_boosting','bagging','mlp']:
    at.selectbox(key='diagnostic_model').set_value(slug).run();ok(at,'Diagnostics '+slug)
at.sidebar.radio[0].set_value('Car clusters').run()
at.radio(key='cluster_view').set_value('Assign a car').run()
for kind in ['kmeans','agglomerative']:
    at.selectbox[0].set_value(kind).run()
    at.button[0].click().run();ok(at,'Cluster assignment '+kind)
    assert any('Assigned segment:' in s.value for s in at.success)
at.radio(key='cluster_view').set_value('Segments').run()
at.multiselect[0].set_value([]).run();ok(at,'Empty cluster filter')
at.radio(key='cluster_view').set_value('After tuning').run();ok(at,'Clustering after tuning')
at.radio(key='cluster_view').set_value('Before vs after').run();ok(at,'Clustering before/after')
at.sidebar.radio[0].set_value('Data garage').run();ok(at,'Classification audit')
at.radio(key='audit_dataset').set_value('Clustering · original').run();ok(at,'Clustering audit')
at.sidebar.radio[0].set_value('Review guide').run();ok(at,'Project guide')
print('ALL GUI CHECKS PASSED')
