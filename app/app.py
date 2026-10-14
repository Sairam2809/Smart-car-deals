"""TORQUEX: lightweight landing screen, cached assets and prediction on demand."""
from pathlib import Path
import sys
import json
import base64
import html
import streamlit as st

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.schema import RAW_FEATURES,PART_B

st.set_page_config(page_title='TORQUEX | Automotive Intelligence',page_icon='🏎️',layout='wide')

@st.cache_data(show_spinner=False)
def text_asset(path,mtime):return (ROOT/path).read_text(encoding='utf-8')

def text(path):return text_asset(path,(ROOT/path).stat().st_mtime_ns)

@st.cache_data(show_spinner=False)
def json_asset(path,mtime):return json.loads((ROOT/path).read_text(encoding='utf-8'))

def data_json(path):return json_asset(path,(ROOT/path).stat().st_mtime_ns)

@st.cache_data(show_spinner=False)
def csv_asset(path,mtime):
    import pandas as pd
    return pd.read_csv(ROOT/path)

def table(path):return csv_asset(path,(ROOT/path).stat().st_mtime_ns)

@st.cache_resource(show_spinner='Opening the selected model…',max_entries=8)
def model_asset(filename,mtime):
    import joblib
    return joblib.load(ROOT/'models'/filename)

def model(filename):
    return model_asset(filename,(ROOT/'models'/filename).stat().st_mtime_ns)

def probability(fitted,X):
    return fitted.predict_proba(X)[:,list(fitted.classes_).index('Automatic')]

def heading(kicker,title,description):
    st.markdown(f'<div class="tx-tag">{kicker}</div><div class="tx-page-title">{title}</div><div class="tx-page-desc">{description}</div>',unsafe_allow_html=True)

def section(title):st.markdown(f'<div class="tx-section">{title}</div>',unsafe_allow_html=True)

def download(frame,filename,label='Download results'):
    st.download_button(label,frame.to_csv(index=False).encode(),file_name=filename,mime='text/csv',on_click='ignore')

def figure_bar(frame,x,y,color=None,barmode=None,title=''):
    import plotly.express as px
    fig=px.bar(frame,x=x,y=y,color=color,barmode=barmode,
               color_discrete_sequence=['#f03745','#91a8c8','#f4b76d'],title=title)
    fig.update_layout(template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
                      margin=dict(l=5,r=5,t=45,b=5),legend_title_text='',font_color='#bdc1ce')
    fig.update_yaxes(range=[0,1],gridcolor='#292d37')
    st.plotly_chart(fig,width='stretch')

st.markdown('<style>'+text('app/assets/style.css')+'</style>',unsafe_allow_html=True)
with st.sidebar:
    st.markdown('<div class="tx-logo">TORQUE<span>X</span></div><div class="tx-logo-sub">AUTOMOTIVE INTELLIGENCE</div>',unsafe_allow_html=True)
    st.markdown('<div class="tx-tag">CONTROL PANEL</div>',unsafe_allow_html=True)
    page=st.radio('Navigation',['Dashboard','Predict transmission','Model comparisons','Car clusters','Data garage','Review guide'],key='navigation',label_visibility='collapsed')
    st.markdown('<div class="tx-side-footer"><span class="tx-status"><i class="tx-dot"></i> REVIEW 02</span><br><br>REAL LISTINGS. MEASURED RESULTS.<br>Classification + clustering<br>23CSE301 / 2026–27</div>',unsafe_allow_html=True)

st.markdown(f'<div class="tx-topline"><span>TORQUEX <strong> / {html.escape(page.upper())}</strong></span><span>ML CAPSTONE · REVIEW 02</span></div>',unsafe_allow_html=True)
if not (ROOT/'results/dashboard.json').exists():
    st.error('Prepared results are missing. Run python scripts/execute_inprocess.py from the TORQUEX folder, then refresh.')
    st.stop()
summary=data_json('results/dashboard.json')
selected=summary['selection']

if page=='Dashboard':
    encoded=base64.b64encode(text('app/assets/torquex_gt.svg').encode()).decode()
    st.markdown(f'''<div class="tx-hero"><div><div class="tx-tag">AUTOMOTIVE MACHINE LEARNING</div>
<h1>INTELLIGENCE.<br>IN <em>TOP GEAR.</em></h1>
<p>A closer look under the hood. Predict a car’s transmission, compare tuned models, and discover what connects similar cars.</p>
<div class="tx-rules"><div><b>2,500</b>REAL LISTINGS</div><div><b>50 : 50</b>CLASS BALANCE</div><div><b>00</b>SYNTHETIC ROWS</div></div></div>
<img class="tx-car" alt="TORQUEX red GT concept car" src="data:image/svg+xml;base64,{encoded}"></div>''',unsafe_allow_html=True)
    m=summary['selected_test_metrics']
    cards=[('Classifiers','10','models','Five Part A + five Part B'),('Classification holdout','500','cars','250 Automatic / 250 Manual'),
           ('Selected model macro F1',f'{m["F1_macro"]:.3f}','','Measured on the held-out sample'),('Clustering dataset','4,009','cars','Original used-car listings')]
    st.markdown('<div class="tx-kpis">'+''.join(f'<div class="tx-kpi"><div class="label">{a}</div><div class="value">{b}<span class="unit">{c}</span></div><div class="note">{d}</div></div>' for a,b,c,d in cards)+'</div>',unsafe_allow_html=True)
    left,right=st.columns([1.15,1],gap='large')
    with left:
        section('AFTER TUNING / PART B')
        rows=''.join(f'<div class="tx-track"><span>{html.escape(r["Model"])}</span><div class="tx-rail"><div class="tx-fill" style="width:{100*r["F1_macro"]:.2f}%"></div></div><strong>{r["F1_macro"]:.3f}</strong></div>' for r in summary['after_tuning'])
        st.markdown('<div class="tx-card"><div class="tx-tag">HELD-OUT MACRO F1</div>'+rows+'</div>',unsafe_allow_html=True)
        st.caption('All five tuned Part B models. Ordering follows training CV; bars show held-out results.')
    with right:
        section('SELECTED FOR PREDICTION')
        st.markdown(f'<div class="tx-card"><div class="tx-tag">FIVE-FOLD CV SELECTION</div><h3>{html.escape(selected["name"])}</h3><p>{html.escape(selected["stage"])} · CV macro F1 {selected["cv_macro_f1"]:.3f}<br>Parameters and model selection use training folds. Final test results are reserved for evaluation.</p></div>',unsafe_allow_html=True)
        st.markdown('<div class="tx-card"><div class="tx-tag">TWO GARAGES. TWO QUESTIONS.</div><p><b>Transmission:</b> balanced UK listings · GBP / miles<br><b>Segments:</b> original used-car listings · USD / miles<br><br>Use the control panel to start a prediction or explore the evidence.</p></div>',unsafe_allow_html=True)

elif page=='Predict transmission':
    heading('CLASSIFICATION / VEHICLE CONFIGURATOR','Know your transmission.','Configure a real-world car profile. All ten classifiers use the new UK dataset: GBP prices and distance in miles.')
    options={r['Model']:r['Slug'] for r in summary['models']}
    a,b=st.columns([2,1])
    name=a.selectbox('Prediction model',list(options),index=list(options).index(selected['name']),key='predict_model')
    slug=options[name];stages=['Before tuning','After tuning'] if slug in PART_B else ['Baseline']
    default=stages.index(selected['stage']) if slug==selected['slug'] and selected['stage'] in stages else len(stages)-1
    stage=b.selectbox('Model version',stages,index=default,key=f'predict_stage_{slug}')
    filename=f'{slug}_{"after" if stage=="After tuning" else "before"}.joblib'
    mode=st.radio('Input mode',['One car','Batch CSV'],horizontal=True,key='predict_mode')
    if mode=='One car':
        import pandas as pd
        choices=summary['inputs']
        brand=st.selectbox('Manufacturer',choices['brands'],index=choices['brands'].index('Ford'),key='brand')
        with st.form('vehicle_form'):
            a,b,c=st.columns(3)
            car_model=a.selectbox('Model',choices['models_by_brand'][brand],key=f'car_model_{brand}')
            year=b.number_input('Model year',min_value=1970,max_value=2026,value=2018,step=1)
            fuel=c.selectbox('Fuel type',choices['fuels'],index=choices['fuels'].index('Petrol') if 'Petrol' in choices['fuels'] else 0)
            price=a.number_input('Advertised price (GBP)',min_value=1,value=16000,step=500)
            miles=b.number_input('Mileage (miles)',min_value=0,value=35000,step=1000)
            engine=c.number_input('Engine displacement (litres)',min_value=.1,max_value=10.,value=1.5,step=.1)
            tax=a.number_input('Vehicle tax (GBP)',min_value=0,value=150,step=10)
            mpg=b.number_input('Fuel economy (source MPG)',min_value=.1,value=55.,step=1.)
            compare=c.checkbox('Compare before and after',value=slug in PART_B,disabled=slug not in PART_B)
            submit=st.form_submit_button('RUN TRANSMISSION CHECK',type='primary')
        if submit:
            X=pd.DataFrame([dict(brand=brand,model=car_model,year=year,price_gbp=price,mileage_miles=miles,
                                fuel_type=fuel,tax_gbp=tax,mpg_uk=mpg,engine_size_l=engine)])
            with st.spinner('Running the selected car profile…'):
                fitted=model(filename);label=fitted.predict(X)[0];p=float(probability(fitted,X)[0])
            st.success(f'Predicted transmission: {label}')
            a,b=st.columns(2);a.metric('Automatic model probability',f'{p:.1%}');b.metric('Manual model probability',f'{1-p:.1%}')
            if compare and slug in PART_B:
                records=[]
                for suffix,title in [('before','Before tuning'),('after','After tuning')]:
                    fitted=model(f'{slug}_{suffix}.joblib')
                    records.append({'Version':title,'Prediction':fitted.predict(X)[0],'Automatic probability':float(probability(fitted,X)[0])})
                st.dataframe(pd.DataFrame(records),hide_index=True,width='stretch')
            if not choices['year_min']<=year<=choices['year_max']:
                st.warning(f'Training years range from {choices["year_min"]} to {choices["year_max"]}; this input is outside that range.')
        st.caption('Probabilities are uncalibrated model outputs. Results describe this balanced UK sample, not the natural frequency of transmissions in the market.')
    else:
        st.write('Use the template schema. You can upload up to 10,000 rows; the target column is optional and ignored for predictions.')
        st.download_button('DOWNLOAD INPUT TEMPLATE',(ROOT/'data/prediction_template.csv').read_bytes(),file_name='TORQUEX_input_template.csv',mime='text/csv',on_click='ignore')
        with st.form('batch_form'):
            upload=st.file_uploader('Car profiles CSV',type=['csv'],key='batch_upload')
            run=st.form_submit_button('RUN BATCH PREDICTION',type='primary')
        if run and upload is None:st.info('Choose a CSV file before running the prediction.')
        if upload is not None and run:
            import pandas as pd
            from src.validation import validate_batch
            try:
                frame=validate_batch(pd.read_csv(upload));fitted=model(filename)
                output=frame.copy();output['predicted_transmission']=fitted.predict(frame[RAW_FEATURES])
                output['probability_automatic']=probability(fitted,frame[RAW_FEATURES])
                st.success(f'Predicted {len(output):,} car profiles.')
                st.dataframe(output.head(100),hide_index=True,width='stretch')
                st.caption('Preview shows at most 100 rows. The download contains every prediction.')
                download(output,'TORQUEX_predictions.csv','DOWNLOAD ALL PREDICTIONS')
            except (ValueError,KeyError,pd.errors.ParserError,UnicodeDecodeError) as exc:st.error(str(exc))

elif page=='Model comparisons':
    heading('PERFORMANCE / TEST BENCH','Every change, measured.','Baseline, tuned and after-tuning comparisons from the same 500-row held-out split. Automatic is the positive class for ROC and precision–recall.')
    mode=st.radio('Comparison view',['Before vs after','After tuning','All ten models','Model diagnostics'],horizontal=True,key='comparison_view')
    metric=st.selectbox('Score',['F1_macro','F1_weighted','Accuracy','Precision_weighted','Recall_weighted','ROC_AUC','Automatic_recall','Automatic_precision','Balanced_accuracy','CV_macro_F1'],key='comparison_metric')
    if mode=='Before vs after':
        frame=table('results/part_b_before_after.csv')
        figure_bar(frame,'Model',metric,'Stage','group','Five Part B models: before and after tuning')
        st.dataframe(frame,hide_index=True,width='stretch')
        section('TUNING DELTAS / AFTER MINUS BEFORE')
        st.dataframe(table('results/tuning_deltas.csv'),hide_index=True,width='stretch')
        download(frame,'TORQUEX_before_after.csv')
    elif mode=='After tuning':
        frame=table('results/after_tuning_comparison.csv')
        figure_bar(frame,'Model',metric,title='All five Part B models after tuning')
        st.caption('Every row is the tuned variant, including any whose best parameters match the baseline.')
        st.dataframe(frame,hide_index=True,width='stretch');download(frame,'TORQUEX_after_tuning.csv')
    elif mode=='All ten models':
        frame=table('results/ten_model_comparison.csv')
        figure_bar(frame,'Model',metric,'Part',title='Ten-model consolidated comparison')
        st.caption('Part A baselines were rerun on this new dataset. Part B uses each model’s CV-selected version. These are not the original Review 1 scores.')
        st.dataframe(frame,hide_index=True,width='stretch');download(frame,'TORQUEX_ten_models.csv')
    else:
        slug=st.selectbox('Part B model',list(PART_B),format_func=PART_B.get,key='diagnostic_model')
        st.dataframe(table(f'results/{slug}/comparison.csv'),hide_index=True,width='stretch')
        chart=st.selectbox('Diagnostic',['Confusion matrices','ROC and precision–recall','Class reports','Search results','Model details'],key='diagnostic_chart')
        if chart in ['Confusion matrices','ROC and precision–recall']:
            filename='confusion_matrices.png' if chart=='Confusion matrices' else 'roc_pr.png'
            st.image(str(ROOT/f'results/{slug}/{filename}'),width='stretch')
        elif chart=='Class reports':
            a,b=st.columns(2)
            a.write('Before tuning');a.dataframe(table(f'results/{slug}/before_class_report.csv'),hide_index=True)
            b.write('After tuning');b.dataframe(table(f'results/{slug}/after_class_report.csv'),hide_index=True)
        elif chart=='Search results':st.dataframe(table(f'results/{slug}/search_results.csv'),hide_index=True,width='stretch')
        else:
            metadata=data_json(f'results/{slug}/metadata.json');st.json(metadata['best_parameters'])
            st.write('Training warnings:',metadata['warnings'] or 'None')
            for filename in ['feature_importance.png','training_loss.png']:
                if (ROOT/f'results/{slug}/{filename}').exists():st.image(str(ROOT/f'results/{slug}/{filename}'),width='stretch')

elif page=='Car clusters':
    heading('CLUSTERING / VEHICLE SEGMENTS','Find a car’s company.','The original 4,009-row used-car dataset, grouped by age, price and mileage. Prices here are USD; distances are miles.')
    kind=st.selectbox('Clustering model',['kmeans','agglomerative'],format_func=lambda x:'K-Means' if x=='kmeans' else 'Agglomerative hierarchical',key='cluster_model')
    mode=st.radio('Cluster view',['Segments','Before vs after','After tuning','Diagnostics','Assign a car'],horizontal=True,key='cluster_view')
    if mode=='Segments':
        import plotly.express as px
        frame=table(f'results/{kind}/clustered_training_cars.csv');frame['Segment']='Cluster '+frame.cluster.astype(str)
        profiles=table(f'results/{kind}/cluster_profiles.csv')
        choices=sorted(frame.Segment.unique());chosen=st.multiselect('Visible segments',choices,default=choices,key=f'segments_{kind}')
        subset=frame[frame.Segment.isin(chosen)]
        if subset.empty:st.info('Choose a segment to show its cars.')
        else:
            fig=px.scatter(subset,x='PC1',y='PC2',color='Segment',hover_data=['brand','model','price_usd','mileage_miles','car_age'],
                           color_discrete_sequence=['#ff5765','#7fa7e3','#f4bf73','#9ec9a8','#c8a7ef','#c9ced7'],opacity=.7)
            fig.update_traces(marker_size=5)
            fig.update_layout(template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',margin=dict(l=5,r=5,t=10,b=5),height=440)
            st.plotly_chart(fig,width='stretch')
        st.caption('Clusters were fitted on standardized features; PCA is only a display projection. Cluster IDs are not rankings.')
        st.dataframe(profiles,hide_index=True,width='stretch')
        with st.expander('Preview selected listings'):
            st.dataframe(subset[['brand','model','price_usd','mileage_miles','car_age','cluster']].head(100),hide_index=True,width='stretch')
        download(subset.drop(columns='Segment'),'TORQUEX_clustered_cars.csv','DOWNLOAD SELECTED LISTINGS')
    elif mode in ['Before vs after','After tuning']:
        frame=table('results/clustering_comparison.csv')
        if mode=='After tuning':frame=frame[frame.Stage=='After tuning']
        metric=st.selectbox('Cluster metric',['Silhouette','Davies_Bouldin','Calinski_Harabasz','CV_silhouette','Holdout_Silhouette'],key='cluster_metric')
        import plotly.express as px
        fig=px.bar(frame,x='Model',y=metric,color='Stage',barmode='group',color_discrete_sequence=['#e53949','#a0b4cf'])
        fig.update_layout(template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig,width='stretch');st.dataframe(frame,hide_index=True,width='stretch')
        st.caption('Higher silhouette / Calinski–Harabasz is better; lower Davies–Bouldin is better. Holdout metrics are separate from fitted-partition metrics.')
        st.info('Hierarchical CV and holdout assignments use a nearest-training-centroid proxy because the original algorithm has no native predict method.')
        download(frame,'TORQUEX_clustering_comparison.csv')
    elif mode=='Diagnostics':
        choices={'PCA before and after':'pca_comparison.png','t-SNE':'tsne.png',
                 ('Elbow curve' if kind=='kmeans' else 'Dendrogram'):('elbow.png' if kind=='kmeans' else 'dendrogram.png')}
        chart=st.selectbox('Cluster diagnostic',list(choices)+['Five-fold tuning search'],key=f'cluster_diagnostic_{kind}')
        if chart=='Five-fold tuning search':st.dataframe(table(f'results/{kind}/search_results.csv'),hide_index=True,width='stretch')
        else:st.image(str(ROOT/f'results/{kind}/{choices[chart]}'),width='stretch')
    else:
        if kind=='agglomerative':st.info('This assigns the new car to its nearest learned cluster centre; it does not rebuild the hierarchy.')
        with st.form('cluster_form'):
            a,b,c=st.columns(3)
            year=a.number_input('Model year',1970,2026,2018,step=1)
            price=b.number_input('Price (USD)',min_value=1,value=25000,step=500)
            miles=c.number_input('Mileage (miles)',min_value=0,value=50000,step=1000)
            submit=st.form_submit_button('FIND SIMILAR SEGMENT',type='primary')
        if submit:
            import pandas as pd
            from src.clustering import predict_new
            bundle=model(f'{kind}_after.joblib')
            label=int(predict_new(bundle,pd.DataFrame([{'car_age':2026-year,'price_usd':price,'mileage_miles':miles}]))[0])
            st.success(f'Assigned segment: Cluster {label}')
            profiles=table(f'results/{kind}/cluster_profiles.csv')
            st.dataframe(profiles[profiles.cluster==label],hide_index=True,width='stretch')
            cars=table(f'results/{kind}/clustered_training_cars.csv')
            st.write('Example listings in this segment')
            st.dataframe(cars[cars.cluster==label][['brand','model','price_usd','mileage_miles','car_age']].head(10),hide_index=True,width='stretch')

elif page=='Data garage':
    heading('DATA / PROVENANCE','Real records. Clear origins.','Inspect the two datasets, their units, source checks and split sizes.')
    dataset=st.radio('Dataset',['Classification · UK','Clustering · original'],horizontal=True,key='audit_dataset')
    if dataset.startswith('Classification'):
        raw=table('data/classification_cars.csv')
        a,b,c,d=st.columns(4);a.metric('Distinct records','2,500');b.metric('Manual / Automatic','1,250 / 1,250');c.metric('Training / test','2,000 / 500');d.metric('Duplicate rows',int(raw.duplicated().sum()))
        st.info('No oversampling and no synthetic records. These are distinct listings, not VIN-verified unique physical vehicles. Price: GBP. Mileage: miles.')
        with st.expander('Source and selection checks'):st.json(data_json('data/classification_source_audit.json'))
    else:
        raw=table('data/used_cars.csv')
        a,b,c=st.columns(3);a.metric('Original rows','4,009');b.metric('Training','3,207');c.metric('Holdout','802')
        st.info('Same clustering dataset as the previous package. Three numeric inputs: age, log price, log mileage. Brand and transmission are used for post-fit interpretation.')
    st.dataframe(raw.isna().sum().to_frame('Missing source values'),width='stretch')
    st.dataframe(raw.head(100),hide_index=True,width='stretch')
    download(raw,'TORQUEX_classification.csv' if dataset.startswith('Classification') else 'TORQUEX_clustering.csv','DOWNLOAD DATASET')

else:
    heading('REVIEW / HANDOVER','Ready for the walkthrough.','Use this route to explain the question, data preparation, model training, tuning and evidence.')
    st.markdown('''
**1. Classification:** Eight delivered notebooks include the five Part B models, two clustering models,
and a consolidated comparison notebook. All ten classifiers use the new balanced 2,500-row UK dataset.
Part A baselines are rerun on the new split. They are not copied Review 1 results.

**2. Tuning evidence:** Before/after tables, score deltas, all-tuned comparisons, required weighted
metrics, confusion matrices and ROC-AUC. ROC and precision–recall curves treat Automatic as positive.
Training folds select parameters and the final model; test results are not used to choose them.

**3. Clustering:** Original dataset, K-Means and hierarchical models, five-fold selection, three
required metrics, elbow, dendrogram, PCA and t-SNE. Hierarchical new-car prediction uses an explicitly
labelled centroid approximation.

**4. Faster UI:** The landing page reads small summaries. No model is loaded until a prediction is
requested. Model objects are cached, forms submit in one action, and only the chosen view is rendered.
No training, network download or parameter search runs inside the GUI.

**5. Your team’s work:** Add your own interpretation to the notebook prompts, explain the metrics,
and maintain meaningful GitHub commits. AI scaffolding is disclosed in README as the guidelines require.
The supplied criteria expect the same dataset across reviews: document the dataset change and the
Part A rerun with your instructor. The package cannot establish that academic approval.

**Scope:** Review 2 classification and clustering. The earlier regression track was not supplied.
The app runs locally; public deployment and an external GitHub history are not created by this ZIP.
''')
    st.download_button('DOWNLOAD GUIDELINE CHECKLIST',text('GUIDELINE_CHECKLIST.md'),file_name='TORQUEX_guideline_checklist.md')
    st.caption('Rebuild with python scripts/execute_inprocess.py, or use execute_notebooks.py with a local Jupyter kernel. Restart the app after retraining.')

st.markdown('<div class="tx-footer">TORQUEX / AUTOMOTIVE INTELLIGENCE &nbsp; · &nbsp; REAL DATA. REPRODUCIBLE RESULTS.</div>',unsafe_allow_html=True)
