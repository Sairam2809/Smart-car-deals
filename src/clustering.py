"""Unsupervised numeric car segmentation, fold-safe selection and diagnostics."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import pairwise_distances_argmin
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

ROOT=Path(__file__).resolve().parents[1]
SEED=42
FEATURES=['car_age','log_price','log_mileage']

def clean_data(raw):
    df=raw.drop_duplicates().copy()
    df['price_usd']=pd.to_numeric(df['price'].astype('string').str.replace(r'[^0-9.]','',regex=True),errors='coerce')
    df['mileage_miles']=pd.to_numeric(df['milage'].astype('string').str.replace(r'[^0-9.]','',regex=True),errors='coerce')
    df['model_year']=pd.to_numeric(df['model_year'],errors='coerce')
    df.loc[~df.model_year.between(1900,2026),'model_year']=np.nan
    df.loc[df.price_usd<=0,'price_usd']=np.nan
    df.loc[df.mileage_miles<0,'mileage_miles']=np.nan
    df['car_age']=2026-df.model_year
    return df

class ClusterFeatures(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None):return self
    def transform(self,X):
        df=X.copy()
        return pd.DataFrame({'car_age':df['car_age'],
            'log_price':np.log1p(df['price_usd']),
            'log_mileage':np.log1p(df['mileage_miles'])},index=df.index)

def preprocessor():
    return Pipeline([('features',ClusterFeatures()),('impute',SimpleImputer(strategy='median')),
                     ('scale',StandardScaler())])

def load_data():
    raw=pd.read_csv(ROOT/'data/used_cars.csv')
    df=clean_data(raw)
    train,test=train_test_split(df,test_size=.2,random_state=SEED)
    return raw,df,train,test

def make_model(kind,k=3,linkage_name='ward'):
    if kind=='kmeans':return KMeans(n_clusters=int(k),n_init=20,random_state=SEED)
    return AgglomerativeClustering(n_clusters=int(k),linkage=linkage_name,metric='euclidean')

def centroids(X,labels):
    return np.vstack([X[labels==i].mean(axis=0) for i in range(int(labels.max())+1)])

def validation_labels(kind,model,X_train,labels,X_valid):
    if kind=='kmeans':return model.predict(X_valid)
    # AgglomerativeClustering has no predict(). This explicit proxy is only for new-row validation.
    return pairwise_distances_argmin(X_valid,centroids(X_train,labels))

def scores(X,labels):
    n=len(np.unique(labels))
    if n<2 or n>=len(labels):
        return {'Silhouette':np.nan,'Davies_Bouldin':np.nan,'Calinski_Harabasz':np.nan}
    # Fixed sampling limits quadratic silhouette work. DB/CH use all supplied rows.
    return {'Silhouette':silhouette_score(X,labels,sample_size=min(2000,len(X)),random_state=SEED),
            'Davies_Bouldin':davies_bouldin_score(X,labels),
            'Calinski_Harabasz':calinski_harabasz_score(X,labels)}

def tune(kind,train):
    candidates=[(k,'ward') for k in range(2,7)] if kind=='kmeans' else [
        (k,l) for k in range(2,7) for l in ['ward','complete','average']]
    folds=list(KFold(5,shuffle=True,random_state=SEED).split(train))
    prepared=[]
    for a,b in folds:
        prep=preprocessor()
        prepared.append((prep.fit_transform(train.iloc[a]),prep.transform(train.iloc[b])))
    rows=[]
    for k,l in candidates:
        for fold,(Xa,Xb) in enumerate(prepared,1):
            m=make_model(kind,k,l);labels=m.fit_predict(Xa)
            val=validation_labels(kind,m,Xa,labels,Xb)
            rows.append({'k':k,'linkage':l,'fold':fold,**scores(Xb,val),
                         'represented_clusters':len(np.unique(val))})
    details=pd.DataFrame(rows)
    summary=details.groupby(['k','linkage']).agg(
        CV_silhouette=('Silhouette','mean'),CV_silhouette_std=('Silhouette','std'),
        CV_Davies_Bouldin=('Davies_Bouldin','mean'),CV_Calinski_Harabasz=('Calinski_Harabasz','mean'),
        valid_folds=('Silhouette','count'),min_represented_clusters=('represented_clusters','min')).reset_index()
    # Reject settings whose validation assignments collapse requested clusters.
    summary['eligible']=(summary.valid_folds==5)&(summary.min_represented_clusters==summary.k)
    valid=summary[summary.eligible].sort_values('CV_silhouette',ascending=False)
    if valid.empty:raise ValueError('No clustering candidate has non-collapsed validation clusters.')
    best=valid.iloc[0]
    return summary,details,{'k':int(best.k),'linkage':str(best.linkage)}

def fit_compare(kind,train,test,summary,best):
    out=ROOT/'results'/kind;out.mkdir(exist_ok=True,parents=True)
    prep=preprocessor();X=prep.fit_transform(train);Xt=prep.transform(test)
    rows=[];fitted={}
    for stage,pars in [('Before tuning',{'k':3,'linkage':'ward'}),('After tuning',best)]:
        model=make_model(kind,pars['k'],pars['linkage']);labels=model.fit_predict(X)
        test_labels=validation_labels(kind,model,X,labels,Xt)
        cv=summary[(summary.k==pars['k'])&(summary.linkage==pars['linkage'])].iloc[0]
        row={'Model':'K-Means' if kind=='kmeans' else 'Agglomerative','Stage':stage,
             'k':pars['k'],'linkage':pars['linkage'] if kind!='kmeans' else 'not applicable',
             **scores(X,labels),'CV_silhouette':cv.CV_silhouette,
             'CV_silhouette_std':cv.CV_silhouette_std,
             **{'Holdout_'+key:value for key,value in scores(Xt,test_labels).items()}}
        rows.append(row)
        bundle={'preprocessor':prep,'model':model,'centroids':centroids(X,labels),
                'kind':kind,'features':FEATURES,'parameters':pars}
        tag='before' if stage=='Before tuning' else 'after'
        joblib.dump(bundle,ROOT/'models'/f'{kind}_{tag}.joblib',compress=3)
        fitted[stage]=(model,labels,bundle)
    table=pd.DataFrame(rows);table.to_csv(out/'comparison.csv',index=False)
    summary.to_csv(out/'search_results.csv',index=False)
    labels=fitted['After tuning'][1]
    pca=PCA(2,random_state=SEED).fit(X);coords=pca.transform(X)
    labeled=train.copy();labeled['cluster']=labels;labeled['PC1']=coords[:,0];labeled['PC2']=coords[:,1]
    labeled.to_csv(out/'clustered_training_cars.csv',index=False)
    test_out=test.copy();test_out['cluster']=validation_labels(kind,fitted['After tuning'][0],X,labels,Xt)
    test_out.to_csv(out/'clustered_holdout_cars.csv',index=False)
    profiles=labeled.groupby('cluster').agg(cars=('price_usd','size'),median_price_usd=('price_usd','median'),
        median_mileage_miles=('mileage_miles','median'),median_age=('car_age','median'))
    profiles.to_csv(out/'cluster_profiles.csv')
    for col in ['brand','fuel_type','transmission']:
        pd.crosstab(labeled.cluster,labeled[col],normalize='index').to_csv(out/f'{col}_profile.csv')
    (out/'metadata.json').write_text(json.dumps({'best_parameters':best,
        'selection_metric':'5-fold validation silhouette on training pool only',
        'holdout_assignment':'native KMeans.predict' if kind=='kmeans' else 'nearest training-cluster centroid proxy, not native hierarchical prediction',
        'pca_variance_ratio':pca.explained_variance_ratio_.tolist(),
        'silhouette_sample_limit':2000,'features':FEATURES},indent=2))
    return table,fitted,prep,X,Xt,pca,profiles

def plot_clusters(kind,fitted,X,pca):
    out=ROOT/'results'/kind
    coords=pca.transform(X);fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,(stage,(_,labels,_)) in zip(axes,fitted.items()):
        for c in np.unique(labels):
            ax.scatter(coords[labels==c,0],coords[labels==c,1],s=9,alpha=.5,label=f'Cluster {c}')
        ax.set(title=stage,xlabel=f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',ylabel=f'PC2 ({pca.explained_variance_ratio_[1]:.1%})');ax.legend(fontsize=8)
    fig.suptitle('Training clusters projected onto the same PCA axes');fig.tight_layout()
    fig.savefig(out/'pca_comparison.png',dpi=130);plt.show()
    idx=np.random.default_rng(SEED).choice(len(X),min(1200,len(X)),replace=False)
    embedding=TSNE(n_components=2,perplexity=30,init='pca',learning_rate='auto',random_state=SEED).fit_transform(X[idx])
    labels=fitted['After tuning'][1][idx]
    fig,ax=plt.subplots(figsize=(7,4))
    for c in np.unique(labels):
        ax.scatter(embedding[labels==c,0],embedding[labels==c,1],s=10,alpha=.6,label=f'Cluster {c}')
    ax.set(title='Tuned clusters: t-SNE of 1,200 sampled training cars',xlabel='t-SNE 1',ylabel='t-SNE 2');ax.legend()
    fig.tight_layout();fig.savefig(out/'tsne.png',dpi=130);plt.show()
    if kind=='kmeans':
        ks=list(range(1,9));inertia=[make_model(kind,k).fit(X).inertia_ for k in ks]
        pd.DataFrame({'k':ks,'inertia':inertia}).to_csv(out/'elbow.csv',index=False)
        fig,ax=plt.subplots(figsize=(7,4));ax.plot(ks,inertia,'o-')
        ax.set(title='Elbow diagnostic (selection also uses CV silhouette)',xlabel='Number of clusters',ylabel='Training inertia')
        fig.tight_layout();fig.savefig(out/'elbow.png',dpi=130);plt.show()
    else:
        idx=np.random.default_rng(SEED).choice(len(X),min(600,len(X)),replace=False)
        method=fitted['After tuning'][2]['parameters']['linkage']
        z=linkage(X[idx],method=method)
        fig,ax=plt.subplots(figsize=(10,4));dendrogram(z,truncate_mode='lastp',p=25,ax=ax)
        ax.set(title=f'{method.title()} dendrogram: 600 sampled training cars, truncated',xlabel='Merged groups (leaf counts in parentheses)',ylabel='Linkage distance')
        fig.tight_layout();fig.savefig(out/'dendrogram.png',dpi=130);plt.show()

def predict_new(bundle,frame):
    X=bundle['preprocessor'].transform(frame)
    if bundle['kind']=='kmeans':return bundle['model'].predict(X)
    return pairwise_distances_argmin(X,bundle['centroids'])
