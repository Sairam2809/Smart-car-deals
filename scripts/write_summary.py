"""Refresh the README's numeric summary from executed results."""
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def markdown(frame):
    lines=['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    for row in frame.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{x:.4f}' if isinstance(x,float) else str(x) for x in row)+' |')
    return '\n'.join(lines)

allruns=pd.read_csv(ROOT/'results/all_classification_runs.csv')
b=allruns[allruns.Part=='B']
rows=[]
for name,g in b.groupby('Model',sort=False):
    before=g[g.Stage=='Before tuning'].iloc[0];after=g[g.Stage=='After tuning'].iloc[0]
    rows.append({'Model':name,'Macro F1 before':before.F1_macro,'Macro F1 after':after.F1_macro,
                 'Test delta':after.F1_macro-before.F1_macro,
                 'CV macro F1 before':before.CV_macro_F1,'CV macro F1 after':after.CV_macro_F1})
cluster=pd.concat([pd.read_csv(ROOT/f'results/{k}/comparison.csv') for k in ['kmeans','agglomerative']],ignore_index=True)
cluster.to_csv(ROOT/'results/clustering_comparison.csv',index=False)
text='''## Executed result summary

These tables are generated directly from the executed notebooks. Test deltas are after minus before;
they are not a promise that tuning improves every model. Fold-selected settings may match the baseline.

### Part B classification

'''+markdown(pd.DataFrame(rows))+'''

### Clustering

Training partition metrics are below. Silhouette and Calinski–Harabasz are higher-is-better;
Davies–Bouldin is lower-is-better. See the full CSV for separately labelled holdout and CV results.

'''+markdown(cluster[['Model','Stage','k','Silhouette','Davies_Bouldin','Calinski_Harabasz']])+'''

The complete ten-model comparison is in `results/ten_model_comparison.csv` and notebook 13.
These computed results still need your team's own explanation and discussion.
'''
readme=ROOT/'README.md'
base=readme.read_text(encoding='utf-8').split('<!-- GENERATED_RESULTS -->')[0].rstrip()
readme.write_text(base+'\n\n<!-- GENERATED_RESULTS -->\n'+text,encoding='utf-8')
print('Updated README and clustering comparison CSV.')
