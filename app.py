from collections import namedtuple
import altair as alt
import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from constant import *
from constant import compute_CD
from st_aggrid import AgGrid
import plotly.graph_objects as go
from statistical_test import graph_ranks
import plotly.express as px
import tempfile

import os

from barchart import plotly_bar_charts_3d
from runtime_plot import create_plot_varylen, create_plot_varynum
from anomaly_plot import plot_anomaly_figure
from tlb_plot import plot_fixed_w_3dbar, plot_fixed_a_3dbar

# from Friedman_Nemenyi_test import *

LINE_WIDTH = 5
MARKER_SIZE = 13
FONT_SIZE = 20
SMALL_FONT_SIZE = 13

THEME_COLOR = '#f8b1a1'

st.set_page_config(layout="wide") 
@st.cache_data
def load_data():
    characteristics_df = pd.read_csv('data/characteristics.csv')
    characteristics_df.reset_index()
    characteristics_df.drop(columns=characteristics_df.columns[0], axis=1, inplace=True)
    characteristics_df.columns = ['Name', 'NumOfTrainingSamples', 'NumOfTestingSample', 'NumOfSamples', 'SeqLength', 'NumOfClasses', 'Type']
    characteristics_df = characteristics_df[['Name', 'NumOfSamples', 'SeqLength', 'NumOfClasses', 'Type']]

    results = pd.read_csv('./data/bop_results.csv')
    results = results.replace('hist_euclidean','Euclid')
    results = results.replace('boss','BOSS')
    results = results.replace('cosine_similarity','Cosine')
    results = results.replace('kl','KL-Div')

    symbol_results = pd.read_csv('./data/symbol_results.csv')
    symbol_results['metric'] = 'symbolic-l1'
    symbol_results = symbol_results[symbol_results['method'].isin(['sax','sfa','spartan_pca_allocation'])]

    results = pd.concat([results,symbol_results],ignore_index=True)
    results = results.replace('sax','SAX')
    results = results.replace('sfa','SFA')
    results = results.replace('spartan_pca_allocation','SPARTAN')

    results['method_metric'] = results['method'] + '+' + results['metric']

    results = pd.pivot(results,index='dataset',columns='method_metric',values='acc')
    results= results.reset_index()

    onenn_results = pd.read_csv('./data/a4_w12_all_methods.csv')
    onenn_results = onenn_results.rename(columns={'dataset_name':'dataset','classifier_name':'method','accuracy':'acc'})

    onenn_results['method_metric'] = onenn_results['method'] + '+' + 'symbolic-l1'

    onenn_results = pd.pivot(onenn_results,index='dataset',columns = 'method_metric',values='acc')
    onenn_results= onenn_results.reset_index()

    clustering_results = pd.read_csv('./data/clustering/clustering_3methods.csv')
    clustering_results = clustering_results.rename(columns={'dataset_name':'dataset','classifier_name':'method','accuracy':'acc'})

    clustering_results['method_metric'] = clustering_results['method'] + '+' + 'symbolic-l1'

    clustering_results = pd.pivot(clustering_results,index='dataset',columns = 'method_metric',values='acc')
    clustering_results= clustering_results.reset_index()

    acc_results = pd.read_csv('./data/symbol_results.csv')
    runtime_results = pd.read_csv('./data/runtime_results_randomized.csv')

    acc_results = acc_results[acc_results['method'] != 'spartan_pca']

    acc_results = acc_results.replace('sax','SAX')
    acc_results = acc_results.replace('sfa','SFA')
    # acc_results = acc_results.replace('spartan_pca','SPARTAN')
    acc_results = acc_results.replace('spartan_pca_allocation','SPARTAN')
    acc_results = acc_results.set_index('method')

    scaling_ranks = acc_results.groupby(by=['dataset'])['acc'].rank(ascending=False)
    acc_results['rank'] = scaling_ranks
    acc_results = acc_results.reset_index()

    runtime_results = runtime_results.replace('PAA','SAX')
    runtime_results = runtime_results.replace('DFT','SFA')
    runtime_results = runtime_results.replace('PCA','SPARTAN')
    runtime_results = runtime_results.replace('PCA+Allocation','SPARTAN')
    runtime_results = runtime_results.replace('PCA+Allocation_randomized','SPARTAN+speedup')

    spartan_results = acc_results[acc_results['method'] == 'SPARTAN']
    spartan_results['method'] = 'SPARTAN+speedup'

    acc_results = pd.concat([acc_results,spartan_results])
    acc_results['rank'] = acc_results.groupby(by=['dataset'])['acc'].rank(ascending=False)
    acc_results = acc_results.reset_index()

    runtime_results = pd.merge(acc_results,runtime_results,how='left',on=['method','dataset'])
    runtime_results['train_time'] = runtime_results['train_time']*1000
    runtime_results['pred_time'] = runtime_results['pred_time']*1000
    runtime_results['total_time'] = runtime_results['train_time'] + runtime_results['pred_time']

    # tlb_file = './data/tlb/'

    # tlb_files = os.listdir(tlb_file)
    # tlb_dfs = {}
    # tlbs_all = pd.DataFrame()
    # for file in tlb_files:
    #     dset = file.split('_')[0]
    #     csv_file = tlb_file + file
    #     tlb_dfs[dset] = pd.read_csv(csv_file)
    #     tlb_dfs[dset].replace(['sax','sfa','spartan'],['SAX','SFA','SPARTAN'],inplace=True)
    #     tlb_dfs[dset]['dataset'] = dset
    # tlbs_all = pd.concat(list(tlb_dfs.values()))

    tlbs_all = pd.read_csv("data/tlb_update/tlb_results_update.csv", index_col=False)
    # tlbs_all = tlbs_all[~((tlbs_all["method"] == "sfa") & (tlbs_all["param"] == "var"))]

    # tlbs_all['method'] = tlbs_all.apply(
    # lambda row: 'SPARTAN_woDAA' if row['param'] == 'woDAA' and row['method'] == 'spartan' 
    # else 'SPARTAN' if row['method'] == 'spartan' 
    # else row['method'].upper(), axis=1
    # )

    # tlbs_all = tlbs_all.drop(columns=['param'])


    # print(tlbs_all.head())
    # tlbs_all.to_csv("data/tlb_update/tlb_results_update.csv")


    return characteristics_df, results,onenn_results,tlbs_all,symbol_results,acc_results,runtime_results,acc_results,tlbs_all, clustering_results


characteristics_df, results,onenn_results,tlb_dfs,symbol_results,acc_results,runtime_results,acc_results,tlbs_all, clustering_results = load_data()


def generate_dataframe(df, datasets, methods_family, metrics):
    df = df.loc[df['dataset'].isin(datasets)][[method_g + '+' + metric for metric in metrics for method_g in methods_family]]
    df = df[df.mean().sort_values().index]
    df.insert(0, 'datasets', datasets)
    return df

def plot_boxplot(df,metrics_list,datasets,method_family,key='bop'):
    fig = go.Figure()
    for i, cols in enumerate(df.columns[1:]):
        fig.add_trace(go.Box(y=df[cols], name=cols,
                                marker=dict(
                                    opacity=1,
                                    color='rgb(8,81,156)',
                                    outliercolor='rgba(219, 64, 82, 0.6)',
                                    line=dict(
                                        outliercolor='rgba(219, 64, 82, 0.6)',
                                        outlierwidth=2)),
                                line_color='rgb(8,81,156)'
                            ))
    fig.update_layout(showlegend=False, 
                        width=1290, 
                        height=600, 
                        template="plotly_white", 
                        font=dict(
                                size=39,
                                color="black"))
    
    fig.update_xaxes(tickfont_size=16)
    fig.update_yaxes(tickfont_size=16)
    #fig.update_xaxes(tickfont_size=15, ticks="outside", ticklen=20, tickwidth=2)
    st.plotly_chart(fig, key=f"Box-{key}")


    st.markdown('# Classification Accuracy Per Dataset')
    cols_list = []
    for i, col in enumerate(df.columns):
        if i > 0:
            cols_list.append(col)
        else:
            cols_list.append(col)

    df.columns = cols_list
    AgGrid(df,key=f"table-{key}",reload_data=True,fit_columns_on_grid_load=True)
def plot_stat_plot(df, datasets,stat_methods_family,metrics,classification_type='1nn'):
    # container_method = st.container()
    # stat_methods_family = container_method.multiselect('Select a group of methods', sorted(methods_family), key='selector_stat_methods')
    
    # df = df.loc[df['dataset'].isin(datasets)][[method_g + '+' + metric for metric in metrics for method_g in stat_methods_family]]
    # df.insert(0, 'dataset', datasets)

    # [method_g + '+' + metric for metric in metrics for method_g in stat_methods_family]
    container_stat_test = st.container()
    stat_test = container_stat_test.selectbox('Select Statistical Test',stat_test_options,index=0,key='stat_test_select_' + classification_type)
    significance = container_stat_test.selectbox('Select Significance Level',significance_optons,index=0,key='significance_level_select_' + classification_type)

    print(df.head())
    if len(datasets) > 0:
        if len(stat_methods_family) > 1 and len(stat_methods_family) < 13:
            def stat_plots(df_toplot):
                def cd_diagram_process(df, rank_ascending=False):
                    df = df.rank(ascending=rank_ascending, axis=1)
                    return df

                df_toplot.drop(columns=df_toplot.columns[0], axis=1, inplace=True)

                rank_ri_df  = cd_diagram_process(df_toplot)
                rank_df = rank_ri_df.mean().sort_values()

                names = []
                for method in rank_df.index.values:
                    names.append(method)

                avranks =  rank_df.values

                # Convert wide format to long format
                # df_long = df.reset_index()
                df_new = df.copy()
                df_new["dataset_name"] = df.index
                df_long = df_new.melt(id_vars=['dataset_name'], var_name='classifier_name', value_name='accuracy')

                # # Rename 'datasets' to 'dataset' for consistency
                # df_long.rename(columns={'datasets': 'dataset_name'}, inplace=True)

                cd = compute_CD(avranks, 128, alpha=significance,test=stat_test)
                fig = graph_ranks(avranks, names, cd=cd, width=9, textspace=1.25)
                # fig = draw_cd_diagram(df_perf=df_long, name=None, alpha=float(significance), height=3, width=9,fontsize=15, textspace=1, space_between_names=0.6, linewidth=2)
                st.pyplot(fig)
                rank_df = rank_df.reset_index()
                rank_df.columns = ['Method Name', 'Average Rank']
                st.table(rank_df)

            stat_plots(df)
def plot_tlb_stat_plot(df,datasets,stat_methods_family):

    container_stat_test = st.container()
    stat_test = container_stat_test.selectbox('Select Statistical Test',stat_test_options,index=0,key='stat_test_select_tlb')
    significance = container_stat_test.selectbox('Select Significance Level',significance_optons,index=0,key='significance_level_select_tlb')
    

    if len(datasets) > 0:
        if len(stat_methods_family) > 1 and len(stat_methods_family) < 13:
            def stat_plots(df_toplot):
                def cd_diagram_process(df, rank_ascending=False):
                    df = df.rank(ascending=rank_ascending, axis=1)
                    return df

                df_toplot.drop(columns=df_toplot.columns[0], axis=1, inplace=True)

                rank_ri_df  = cd_diagram_process(df_toplot)
                rank_df = rank_ri_df.mean().sort_values()

                names = []
                for method in rank_df.index.values:
                    names.append(method)

                avranks =  rank_df.values
                cd = compute_CD(avranks, 128, alpha=significance,test=stat_test)
                fig = graph_ranks(avranks, names, cd=cd, width=9, textspace=1.25)

                # df_new = df.copy()
                # df_new["dataset_name"] = df.index
                # df_long = df_new.melt(id_vars=['dataset_name'], var_name='classifier_name', value_name='accuracy')

                # fig = draw_cd_diagram(df_perf=df_long, name=None, alpha=float(significance), height=3, width=9,fontsize=15, textspace=1, space_between_names=0.6, linewidth=2)
                st.pyplot(fig)
                # Save the figure to a temporary file
             
                rank_df = rank_df.reset_index()
                rank_df.columns = ['Method Name', 'Average Rank']
                st.table(rank_df)

            stat_plots(df)


st.markdown(
    """
    <style>
        /* Change sidebar background color */
        [data-testid="stSidebar"] {
            background-color: #d2e7f6  !important;  /* Light peach  #f8b1a1 f3d0c8 */
        }

        [data-testid="stSidebar"] h1 {
            font-size: 30px,
            color: green !important;  /* Dark Gray #3D3D3D #aca29f */
            font-weight: bold !important;
        }

        /* Change the background color of the multiselect choice buttons */
        div[data-testid="stMultiSelect"] > div {
            background-color: white;  /* Change to desired color */
            color: green;  /* Change text color */
            border-radius: 8px;  /* Add rounded corners */
            padding: 5px 10px;  /* Adjust padding */
        }

        /* Change the close (X) button color inside multiselect */
        div[data-testid="stMultiSelect"] span {
            color: white;
        }

        /* Change hover effect */
        div[data-testid="stMultiSelect"] > div:hover {
            background-color: #f8b1a1;  /* Darker shade for hover */
        }

        span[data-baseweb="tag"] {
            background-color: #f8b1a1 !important; /* #f3d0c8 */
        }

        button[data-baseweb="tab"] {
            font-size: 30px !important;  /* Adjust this size as needed */
            font-weight: bold !important;
        }
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            font-size:1.25rem;
            }

    </style>
    """,
    unsafe_allow_html=True
)

# st.markdown("""
# <style>
#     [data-testid=stSidebar] {
#         background-color: DarkSlateGray;
#     }
# </style>
# """, unsafe_allow_html=True)


with st.sidebar:

    st.markdown('# SAIL: Symbolic Representation Explorer')
     
    # container_metric = st.container()
    # all_metric = st.checkbox('Select all',key='all_metrics')
    # if all_metric: metrics = container_metric.multiselect('Select metric',list_measures,list_measures)
    # else: metrics = container_metric.multiselect('Select metric',list_measures)
    
    container_dataset = st.container()  
    all_cluster = st.checkbox("Select all", key='all_clusters',value=True)
    if all_cluster: cluster_size = container_dataset.multiselect('Select number of classes', sorted(list(list_num_clusters)), sorted(list(list_num_clusters)))
    else: cluster_size = container_dataset.multiselect('Select number of classes', sorted(list(list_num_clusters)))

    container_dataset = st.container()  
    all_length = st.checkbox("Select all", key='all_lengths',value=True)
    if all_length: length_size = container_dataset.multiselect('Select sequence length size', sorted(list(list_seq_length)), sorted(list(list_seq_length)))
    else: length_size = container_dataset.multiselect('Select length sequence size', sorted(list(list_seq_length)))

    container_dataset = st.container()  
    all_type = st.checkbox("Select all", key='all_types',value=True)
    if all_type: types = container_dataset.multiselect('Select sequence type', sorted(list(list_type)), sorted(list(list_type)))
    else: types = container_dataset.multiselect('Select sequence type', sorted(list(list_type)))
   
    container_dataset = st.container()  
    all_dataset = st.checkbox("Select all", key='all_dataset',value=True)
    if all_dataset: datasets = container_dataset.multiselect('Select datasets', sorted(find_datasets(cluster_size, length_size, types)), sorted(find_datasets(cluster_size, length_size, types)))
    else: datasets = container_dataset.multiselect('Select datasets', sorted(find_datasets(cluster_size, length_size, types)))

    # container_method = st.container()
    # all_method = st.checkbox("Select all",key='all_method')
    # if all_method: methods_family = container_method.multiselect('Select a group of methods', methods, methods, key='selector_methods_all')
    # else: methods_family = container_method.multiselect('Select a group of methods',methods, key='selector_methods')

# tab_desc, tab_acc, tab_time, tab_stats, tab_analysis, tab_misconceptions, tab_ablation, tab_dataset, tab_method = st.tabs(["Description", "Evaluation", "Runtime", "Statistical Tests", "Comparative Analysis", "Misconceptions", "DNN Ablation Analysis", "Datasets", "Methods"]) 
tab_desc, tab_dataset,tab_methods,tab_1nn_classification,tab_classification_accuracy,tab_clustering,tab_tlb,tab_ad, tab_runtime,tab_references = st.tabs(["Overview", "Datasets",'Methods',"1NN-Classification Accuracy","BOP Classification Accuracy","Clustering","Tightness of Lower Bound", "Anomaly Detection", "Runtime Analysis","References"]) 


# with tab_desc:
#     st.markdown('# SAIL: A Voyage to Symbolic Representation Solutions for Time-Series Analysis')
#     st.markdown(description_intro1)
#     background = Image.open('./data/sail_demo_pipeline.png')
#     col1, col2, col3 = st.columns([1.2, 5, 0.2])
#     col2.image(background, width=900, caption='Overview of the SAIL representation method.')
#     st.markdown(description_intro2)
#     # st.markdown(description_intro2)
#     # background = Image.open('./data/taxonomy.png')
#     # col1, col2, col3 = st.columns([1.2, 5, 0.2])
#     # col2.image(background, width=900, caption='Taxonomy of time-series clustering methods in Odyssey.')

with tab_desc:
    st.markdown('# SAIL: A Voyage to Symbolic Representation Solutions for Time-Series Analysis')
    st.markdown(description_intro1)

    # Load image
    background = Image.open('data/demo_spartan_pipeline_update.png')

    # Center align the image and add a caption
    col1, col2, col3 = st.columns([2, 5, 2])  # Adjust column widths for centering
    with col2:
        st.image(background, use_container_width=True)
        st.markdown("<p style='text-align: center; font-style: italic;'>Overview of the SAIL representation method.</p>", unsafe_allow_html=True)

    st.markdown(description_intro2)




with tab_dataset:
    st.markdown('# Dataset Description')
    st.markdown(text_description_dataset)
    AgGrid(characteristics_df)

# with tab_methods:
#     st.markdown('# Method Description')
#     baseline = Image.open('./data/baseline_methods_v3.png')
#     col1, col2, col3 = st.columns([1.2, 5, 0.2])
#     col2.image(baseline, width=900, caption='Summary of basline methods SAX, SFA, and variants.')
#     st.markdown(text_description_models,unsafe_allow_html=True)
#     spartan_pipeline = Image.open('./data/spartan_pipeline.png')
#     col2.image(spartan_pipeline,width=900, caption='Overview of SPARTAN representation method.')

with tab_methods:
    st.markdown('# Method Description')

    # Load and display the baseline methods image
    baseline = Image.open('./data/baseline_methods_v3.png')
    col1, col2, col3 = st.columns([1, 5, 1])  # Center alignment
    with col2:
        st.image(baseline, use_container_width=True)
        st.markdown("<p style='text-align: center; font-style: italic;'>Summary of baseline methods SAX, variants, and SFA.</p>", unsafe_allow_html=True)

    # Load and display the SPARTAN pipeline image
    spartan_pipeline = Image.open('./data/spartan_pipeline_v3.png')
    col1, col2, col3 = st.columns([1, 5, 1])  # Center alignment
    with col2:
        st.image(spartan_pipeline, use_container_width=True)
        st.markdown("<p style='text-align: center; font-style: italic;'>Overview of SPARTAN representation method.</p>", unsafe_allow_html=True)

    # Description text
    # st.markdown(text_description_models, unsafe_allow_html=True)
    st.markdown(get_text_description_models(), unsafe_allow_html=True)

    

with tab_classification_accuracy:

    st.markdown('# Bag-Of-Patterns Classification Accuracy Results')
    st.markdown(text_bop_classification_description)

    tab_bop_boxplot,tab_bop_pairwise,tab_bop_stats = st.tabs(['Boxplot','Pairwise','Statistical Tests'])
    
    with tab_bop_boxplot:
        st.markdown(text_boxplot_explanation)

        container_accuracy_method = st.container()
        all_method = st.checkbox("Select all",key='all_method',value=True)
        if all_method: methods_family = container_accuracy_method.multiselect('Select a group of methods', methods, methods, key='selector_methods_all')
        else: methods_family = container_accuracy_method.multiselect('Select a group of methods',methods, key='selector_methods')

        container_accuracy_metric = st.container()
        all_metric = st.checkbox('Select all',key='all_metrics',value=True)
        if all_metric: metrics = container_accuracy_metric.multiselect('Select metric',list_measures,list_measures)
        else: metrics = container_accuracy_metric.multiselect('Select metric',list_measures)

        box_df = generate_dataframe(results,datasets,methods_family,metrics)
        plot_boxplot(box_df,metrics,datasets,methods_family, key='bop_classification')
    with tab_bop_pairwise:
        st.markdown(text_pairwise_comparison)
        option1 = st.selectbox('Method 1',tuple(methods),index=0)
        metric1 = st.selectbox('Metric 1',bop_metrics_list,index=0)
        # methods_family = methods_family[1:] + methods_family[:1]
        option2 = st.selectbox('Method 2',tuple(methods),index=2)
        metric2 = st.selectbox('Metric 2',bop_metrics_list,index=0)

        method_metric_1 = option1 + '+' + metric1
        method_metric_2 = option2 + '+' + metric2

        if len(methods_family) > 0 and len(datasets) > 0:
            fig = go.Figure()
            trace1 = fig.add_scattergl(x=box_df[method_metric_1], y=box_df[method_metric_2], mode='markers', name='(' + method_metric_1 + '+'+ method_metric_2 +')',  text=datasets,
                                    textposition="bottom center",
                                    marker = dict(size=10,
                                                opacity=.7,
                                                color='red',
                                                line = dict(width=1, color = '#1f77b4')
                                                ))
            fig.add_trace(go.Scatter(
                                x=[min(min(box_df[method_metric_1])+1e-4, min(box_df[method_metric_2])+1e-4), max(max(box_df[method_metric_1])+1e-4, max(box_df[method_metric_2])+1e-4)],
                                y=[min(min(box_df[method_metric_1])+1e-4, min(box_df[method_metric_2])+1e-4), max(max(box_df[method_metric_1])+1e-4, max(box_df[method_metric_2])+1e-4)],
                                name="X=Y"
                            ))
            trace2 = fig.add_histogram(x=box_df[method_metric_1], name='x density', marker=dict(color='#1f77b4', opacity=0.7),
                                yaxis='y2'
                                )
            trace3 = fig.add_histogram(y=box_df[method_metric_2], name='y density', marker=dict(color='#1f77b4', opacity=0.7), 
                                xaxis='x2'
                                )
            fig.layout = dict(xaxis=dict(domain=[0, 0.85], showgrid=False, zeroline=False),
                            yaxis=dict(domain=[0, 0.85], showgrid=False, zeroline=False),
                            xaxis_title=option1, yaxis_title=option2,
                            showlegend=False,
                            margin=dict(t=50),
                            hovermode='closest',
                            bargap=0,
                            xaxis2=dict(domain=[0.85, 1], showgrid=False, zeroline=False),
                            yaxis2=dict(domain=[0.85, 1], showgrid=False, zeroline=False),
                            height=600,
                            )

            def do_zoom(layout, xaxis_range, yaxis_range):
                inds = ((xaxis_range[0] <= box_df[method_metric_1]) & (box_df[method_metric_1] <= xaxis_range[1]) &
                        (yaxis_range[0] <= box_df[method_metric_2]) & (box_df[method_metric_2] <= yaxis_range[1]))

                with fig.batch_update():
                    trace2.x = box_df[method_metric_1][inds]
                    trace3.y = box_df[method_metric_2][inds]
                
            fig.layout.on_change(do_zoom, 'xaxis.range', 'yaxis.range')
            fig.update_xaxes(tickfont_size=16)
            fig.update_yaxes(tickfont_size=16)
            st.plotly_chart(fig)
    with tab_bop_stats:
        st.markdown(text_cd_diagram_explanation)
        metric_options = bop_metrics_list
        cd_df = results
        methods_list = methods

        container_cd = st.container()
        all_cd_metrics = st.checkbox('Select all',key='all_cd_metrics_bop',value=False)
        if all_cd_metrics: cd_metric = container_cd.multiselect('Select metric',metric_options,['Euclid'],key='selector_cd_metrics_all_bop')
        else: cd_metric = container_cd.multiselect('Select metric',metric_options,['Euclid'],key='selector_cd_metrics_bop')

        container_cd_accuracy_method = st.container()
        all_cd_method = st.checkbox("Select all",key='all_cd_method_bop',value=False)
        if all_cd_method: cd_methods_family = container_cd_accuracy_method.multiselect('Select a group of methods', methods_list, methods_list, key='selector_cd_methods_all_bop')
        else: cd_methods_family = container_cd_accuracy_method.multiselect('Select a group of methods',methods_list,methods_list, key='selector_cd_methods_bop')

        cd_df_subset = generate_dataframe(cd_df,datasets,cd_methods_family,cd_metric)
        plot_stat_plot(cd_df_subset,datasets,cd_methods_family,cd_metric,'bop')

with tab_1nn_classification:
    st.markdown('# 1-Nearest Neighbor Classification Accuracy Results')
    st.markdown(text_1nn_classification_description)
    st.latex(test_symbolic_l1_distance)
    st.markdown(text_1nn_classification_description_2)

    tab_1nn_boxplot,tab_1nn_pairwise,tab_1nn_stats = st.tabs(['Boxplot','Pairwise','Statistical Tests'])

    onenn_methods_list_default = onenn_methods_list.copy()
    if 'SAX_VFD' in onenn_methods_list:
        onenn_methods_list_default.remove('SAX_VFD')
            

    with tab_1nn_boxplot:

        st.markdown(text_boxplot_explanation)
        
        container_1nn_accuracy_method = st.container()
        all_onenn_method = st.checkbox("Select all",key='all_onenn_method',value=False)
        
        if all_onenn_method: onenn_methods_family = container_1nn_accuracy_method.multiselect('Select a group of methods', onenn_methods_list, onenn_methods_list_default, key='selector_onenn_methods_all')
        else: onenn_methods_family = container_1nn_accuracy_method.multiselect('Select a group of methods',onenn_methods_list, onenn_methods_list_default, key='selector_onenn_methods')

        container_1nn_accuracy_metric = st.container()
        all_onenn_metric = st.checkbox('Select all',key='all_onenn_metrics',value=False)
        if all_onenn_metric: onenn_metrics = container_1nn_accuracy_metric.multiselect('Select metric',onenn_metrics_list,onenn_metrics_list)
        else: onenn_metrics = container_1nn_accuracy_metric.multiselect('Select metric',onenn_metrics_list,onenn_metrics_list)

        onenn_box_df = generate_dataframe(onenn_results,datasets,onenn_methods_family,onenn_metrics)
        plot_boxplot(onenn_box_df,onenn_metrics,datasets,onenn_methods_family,key='onenn_classification')
    
    with tab_1nn_pairwise:
        st.markdown(text_pairwise_comparison)
        option1 = st.selectbox('Method 1',onenn_methods_list, index=0)
        # methods_family = methods_family[1:] + methods_family[:1]
        option2 = st.selectbox('Method 2',onenn_methods_list, index=2)

        method_metric_1 = option1 + '+' + 'symbolic-l1'
        method_metric_2 = option2 + '+' + 'symbolic-l1'

        if len(methods_family) > 0 and len(datasets) > 0:
            fig = go.Figure()
            trace1 = fig.add_scattergl(x=onenn_box_df[method_metric_1], y=onenn_box_df[method_metric_2], mode='markers', name='(' + method_metric_1 + '+'+ method_metric_2 +')',  text=datasets,
                                    textposition="bottom center",
                                    marker = dict(size=10,
                                                opacity=.7,
                                                color='red',
                                                line = dict(width=1, color = '#1f77b4')
                                                ))
            fig.add_trace(go.Scatter(
                                x=[min(min(onenn_box_df[method_metric_1])+1e-4, min(onenn_box_df[method_metric_2])+1e-4), max(max(onenn_box_df[method_metric_1])+1e-4, max(onenn_box_df[method_metric_2])+1e-4)],
                                y=[min(min(onenn_box_df[method_metric_1])+1e-4, min(onenn_box_df[method_metric_2])+1e-4), max(max(onenn_box_df[method_metric_1])+1e-4, max(onenn_box_df[method_metric_2])+1e-4)],
                                name="X=Y"
                            ))
            trace2 = fig.add_histogram(x=onenn_box_df[method_metric_1], name='x density', marker=dict(color='#1f77b4', opacity=0.7),
                                yaxis='y2'
                                )
            trace3 = fig.add_histogram(y=onenn_box_df[method_metric_2], name='y density', marker=dict(color='#1f77b4', opacity=0.7), 
                                xaxis='x2'
                                )
            fig.layout = dict(xaxis=dict(domain=[0, 0.85], showgrid=False, zeroline=False),
                            yaxis=dict(domain=[0, 0.85], showgrid=False, zeroline=False),
                            xaxis_title=option1, yaxis_title=option2,
                            showlegend=False,
                            margin=dict(t=50),
                            hovermode='closest',
                            bargap=0,
                            xaxis2=dict(domain=[0.85, 1], showgrid=False, zeroline=False),
                            yaxis2=dict(domain=[0.85, 1], showgrid=False, zeroline=False),
                            height=600,
                            )

            def do_zoom(layout, xaxis_range, yaxis_range):
                inds = ((xaxis_range[0] <= box_df[method_metric_1]) & (box_df[method_metric_1] <= xaxis_range[1]) &
                        (yaxis_range[0] <= box_df[method_metric_2]) & (box_df[method_metric_2] <= yaxis_range[1]))

                with fig.batch_update():
                    trace2.x = onenn_box_df[method_metric_1][inds]
                    trace3.y = onenn_box_df[method_metric_2][inds]
                
            fig.layout.on_change(do_zoom, 'xaxis.range', 'yaxis.range')
            fig.update_xaxes(tickfont_size=16)
            fig.update_yaxes(tickfont_size=16)
            st.plotly_chart(fig)
    
    with tab_1nn_stats:
        st.markdown(text_cd_diagram_explanation)
        metric_options = onenn_metrics_list
        cd_df = onenn_results
        methods_list = onenn_methods_list

        container_cd = st.container()
        all_cd_metrics = st.checkbox('Select all',key='all_cd_metrics_1nn',value=False)
        if all_cd_metrics: cd_metric = container_cd.multiselect('Select metric',metric_options,metric_options,key='selector_cd_metrics_all_1nn')
        else: cd_metric = container_cd.multiselect('Select metric',metric_options,metric_options,key='selector_cd_metrics_1nn')

        methods_list_default = methods_list.copy()
        if 'SAX_VFD' in methods_list:
            methods_list_default.remove('SAX_VFD')

        container_cd_accuracy_method = st.container()
        all_cd_method = st.checkbox("Select all",key='all_cd_method_1nn',value=False)
        if all_cd_method: cd_methods_family = container_cd_accuracy_method.multiselect('Select a group of methods', methods_list, methods_list_default, key='selector_cd_methods_all_1nn')
        else: cd_methods_family = container_cd_accuracy_method.multiselect('Select a group of methods',methods_list, methods_list_default, key='selector_cd_methods_1nn')

        cd_df_subset = generate_dataframe(cd_df,datasets,cd_methods_family,cd_metric)
        plot_stat_plot(cd_df_subset,datasets,cd_methods_family,cd_metric,'1nn')

with tab_clustering:
    st.markdown('# Time-series Clustering Results')
    st.markdown(text_clustering_description)
    # st.latex(test_symbolic_l1_distance)
    # st.markdown(text_1nn_classification_description_2)

    # tab_clustering_boxplot,tab_clustering_pairwise,tab_clustering_stats = st.tabs(['Boxplot','Pairwise','Statistical Tests'])
    tab_clustering_stats, tab_clustering_pairwise,tab_clustering_boxplot = st.tabs(['Statistical Tests', 'Pairwise', 'Boxplot'])
    with tab_clustering_boxplot:

        st.markdown(text_boxplot_explanation)
        
        container_clustering_accuracy_method = st.container()
        all_clustering_method = st.checkbox("Select all",key='all_clustering_method',value=True)
        if all_clustering_method: 
            clustering_methods_family = container_clustering_accuracy_method.multiselect('Select a group of methods',
                                                                                               clustering_methods_list,
                                                                                               clustering_methods_list,
                                                                                               key='selector_clustering_methods_all')
        else: clustering_methods_family = container_clustering_accuracy_method.multiselect('Select a group of methods',
                                                                               clustering_methods_list,
                                                                               key='selector_clustering_methods')

        container_clustering_accuracy_metric = st.container()
        all_clustering_metric = st.checkbox('Select all',key='all_clustering_metrics',value=True)
        if all_clustering_metric: 
            clustering_metrics = container_clustering_accuracy_metric.multiselect('Select metric',
                                                                                  clustering_metrics_list,
                                                                                  clustering_metrics_list,
                                                                                  key='selector_clustering_metrics_all')
        else: 
            clustering_metrics = container_clustering_accuracy_metric.multiselect('Select metric',clustering_metrics_list)

        clustering_datasets = datasets.copy()
        if 'Crop' in datasets:
            clustering_datasets.remove('Crop')
        if 'ElectricDevices' in datasets:
            clustering_datasets.remove('ElectricDevices')

        clustering_box_df = generate_dataframe(clustering_results,
                                          clustering_datasets,
                                          clustering_methods_family,
                                          clustering_metrics)
        plot_boxplot(clustering_box_df,
                     clustering_metrics,
                     clustering_datasets,
                     clustering_methods_family,
                     key='clustering-box')

    with tab_clustering_pairwise:
        st.markdown(text_pairwise_comparison)
        option1 = st.selectbox('Clustering Method 1',clustering_methods_list,index=0)
        # methods_family = methods_family[1:] + methods_family[:1]
        option2 = st.selectbox('Clustering Method 2',clustering_methods_list,index=2)

        method_metric_1 = option1 + '+' + 'symbolic-l1'
        method_metric_2 = option2 + '+' + 'symbolic-l1'

        if len(methods_family) > 0 and len(datasets) > 0:
            fig = go.Figure()
            trace1 = fig.add_scattergl(x=clustering_box_df[method_metric_1], 
                                       y=clustering_box_df[method_metric_2], 
                                       mode='markers', 
                                       name='(' + method_metric_1 + '+'+ method_metric_2 +')',  
                                       text=datasets,
                                       textposition="bottom center",
                                       marker = dict(size=10,
                                                opacity=.7,
                                                color='red',
                                                line = dict(width=1, color = '#1f77b4')
                                                ))
            fig.add_trace(go.Scatter(
                                x=[min(min(clustering_box_df[method_metric_1])+1e-4, min(clustering_box_df[method_metric_2])+1e-4), max(max(clustering_box_df[method_metric_1])+1e-4, max(clustering_box_df[method_metric_2])+1e-4)],
                                y=[min(min(clustering_box_df[method_metric_1])+1e-4, min(clustering_box_df[method_metric_2])+1e-4), max(max(clustering_box_df[method_metric_1])+1e-4, max(clustering_box_df[method_metric_2])+1e-4)],
                                name="X=Y"
                            ))
            trace2 = fig.add_histogram(x=clustering_box_df[method_metric_1], name='x density', marker=dict(color='#1f77b4', opacity=0.7),
                                yaxis='y2'
                                )
            trace3 = fig.add_histogram(y=clustering_box_df[method_metric_2], name='y density', marker=dict(color='#1f77b4', opacity=0.7), 
                                xaxis='x2'
                                )
            fig.layout = dict(xaxis=dict(domain=[0, 0.85], showgrid=False, zeroline=False),
                            yaxis=dict(domain=[0, 0.85], showgrid=False, zeroline=False),
                            xaxis_title=option1, yaxis_title=option2,
                            showlegend=False,
                            margin=dict(t=50),
                            hovermode='closest',
                            bargap=0,
                            xaxis2=dict(domain=[0.85, 1], showgrid=False, zeroline=False),
                            yaxis2=dict(domain=[0.85, 1], showgrid=False, zeroline=False),
                            height=600,
                            )

            def do_zoom(layout, xaxis_range, yaxis_range):
                inds = ((xaxis_range[0] <= box_df[method_metric_1]) & (box_df[method_metric_1] <= xaxis_range[1]) &
                        (yaxis_range[0] <= box_df[method_metric_2]) & (box_df[method_metric_2] <= yaxis_range[1]))

                with fig.batch_update():
                    trace2.x = clustering_box_df[method_metric_1][inds]
                    trace3.y = clustering_box_df[method_metric_2][inds]
                
            fig.layout.on_change(do_zoom, 'xaxis.range', 'yaxis.range')
            fig.update_xaxes(tickfont_size=16)
            fig.update_yaxes(tickfont_size=16)
            st.plotly_chart(fig, key='clustering_pairwise')
    with tab_clustering_stats:
        st.markdown(text_cd_diagram_explanation)
        clustering_metric_options = clustering_metrics_list
        cd_clustering_df = clustering_results
        clustering_methods_list = clustering_methods_list

        container_clustering_cd = st.container()
        all_cd_clustering_metrics = st.checkbox('Select all',key='all_cd_metrics_clustering',value=True)
        if all_cd_clustering_metrics: 
            cd_clustering_metric = container_clustering_cd.multiselect('Select metric',
                                                                       clustering_metric_options,
                                                                       clustering_metric_options,
                                                                       key='selector_cd_metrics_all_clustering')
        else: 
            cd_clustering_metric = container_clustering_cd.multiselect('Select metric',
                                                                       clustering_metric_options,
                                                                       key='selector_cd_metrics_clustering')

        container_cd_clustering_accuracy_method = st.container()
        all_cd_clustering_method = st.checkbox("Select all",key='all_cd_method_clustering',value=True)
        if all_cd_clustering_method: 
            cd_clustering_methods_family = container_cd_clustering_accuracy_method.multiselect('Select a group of methods', 
                                                                                               clustering_methods_list, clustering_methods_list, key='selector_cd_methods_all_clustering')
        else: 
            cd_clustering_methods_family = container_cd_clustering_accuracy_method.multiselect('Select a group of methods',
                                                                                               clustering_methods_list,
                                                                                                key='selector_cd_methods_clustering')
        clustering_datasets = datasets.copy()
        if 'Crop' in datasets:
            clustering_datasets.remove('Crop')
        if 'ElectricDevices' in datasets:
            clustering_datasets.remove('ElectricDevices')
        cd_df_clustering_subset = generate_dataframe(cd_clustering_df,clustering_datasets,cd_clustering_methods_family,cd_clustering_metric)
        plot_stat_plot(cd_df_clustering_subset,clustering_datasets,cd_clustering_methods_family,cd_clustering_metric,'clustering-cd')


with tab_tlb:
    st.markdown('# Tightness of Lower Bound Results')
    st.markdown(text_tlb_description)

    tab_tlb_barplots, tab_tlb_statplots = st.tabs(['Comparison Plots','Statistical Tests'])

    with tab_tlb_barplots:
        container_tlb = st.container()
        # all_metric = st.checkbox('Select all',key='all_tlbs')
        # if all_metric: tlb_dataset = container_tlb.multiselect('Select dataset',sorted(find_datasets(cluster_size, length_size, types)), sorted(find_datasets(cluster_size, length_size, types)))
        tlb_dataset = container_tlb.selectbox('Select dataset',sorted(find_datasets(cluster_size, length_size, types)),index=28)
        # tlb_results = tlb_dfs[tlb_dataset]
        tlb_results = tlb_dfs[tlb_dfs['dataset']==tlb_dataset]
        my_cmap = sns.color_palette("YlGnBu", as_cmap=True)

        tlb_col1, tlb_col2 = st.columns(2)

        with tlb_col1:
            # fig = plt.figure(figsize=(5,5))
            # ax = fig.add_subplot(projection='3d')
            # dx=dy=1

            # fixed_w = st.slider('Fix Word Length',4,10,4)
            # print("tlb result shape: ", tlb_results.shape)
            # fixed_w_results = tlb_results[tlb_results['w'] == fixed_w]
            # fixed_w_results = fixed_w_results.pivot(index='method',columns='a',values='tlb')
            
            # xpos = np.arange(fixed_w_results.shape[0])
            # ypos = np.arange(fixed_w_results.shape[1])

            # print("fix w result shape: ", fixed_w_results.shape)
            # print("fix w: ", fixed_w_results)

            # ax.set_xticks(xpos + dx/2)
            # ax.set_yticks(ypos + dy/2)

            # # create meshgrid 
            # # print xpos before and after this block if not clear
            # xpos, ypos = np.meshgrid(xpos, ypos)
            # xpos = xpos.flatten()
            # ypos = ypos.flatten()
            # # print(xpos)
            # # print(ypos)

            # # the bars starts from 0 attitude
            # zpos=np.zeros(fixed_w_results.shape).flatten()

            # dz = fixed_w_results.values.ravel(order='F')

            # colors = [0,50,100]*fixed_w_results.shape[1]
            # colors = plt.cm.YlGnBu(colors)

            # ax.bar3d(xpos,ypos,zpos,dx,dy,dz,color=colors,shade=True)

            # ax.invert_xaxis()
            # ax.yaxis.set_ticklabels(fixed_w_results.columns)
            # ax.xaxis.set_ticklabels(fixed_w_results.index)
            # ax.set_ylabel('Alphabet Size',labelpad=15)
            # ax.set_zlabel('TLB',labelpad=15)
            # ax.set_zlim(0,0.7)
            # st.markdown('### TLB per Alphabet Size w/ Fixed Word Length')
            # st.pyplot(fig)
            fixed_w = st.slider('Fix Word Length',4,10,4)
            fig = plot_fixed_w_3dbar(tlb_results, fixed_w)
            st.pyplot(fig,use_container_width=True)
            # st.plotly_chart(fig, use_container_width=True)
        with tlb_col2:
            fixed_a = st.slider('Fix Alphabet Size',4,10,4)
            fig = plot_fixed_a_3dbar(tlb_results, fixed_a)
            st.pyplot(fig,use_container_width=True)
            # fig = plt.figure(figsize=(4,4))

            # fixed_a = st.slider('Fix Alphabet Size',3,10,4)
            # fixed_a_results = tlb_results[tlb_results['a'] == fixed_a] 
            # fixed_a_results = fixed_a_results.pivot(index='method',columns='w',values='tlb')

            # fig = plt.figure(figsize=(5,5))
            # ax =fig.add_subplot(projection='3d')

            # xpos = np.arange(fixed_a_results.shape[0])
            # ypos = np.arange(fixed_a_results.shape[1])

            # ax.set_xticks(xpos + dx/2)
            # ax.set_yticks(ypos + dy/2)

            # # create meshgrid 
            # # print xpos before and after this block if not clear
            # xpos, ypos = np.meshgrid(xpos, ypos)
            # xpos = xpos.flatten()
            # ypos = ypos.flatten()

            # # the bars starts from 0 attitude
            # zpos=np.zeros(fixed_a_results.shape).flatten()

            # dz = fixed_a_results.values.ravel(order='F')

            # colors = [0,50,100]*fixed_a_results.shape[1]
            # colors = plt.cm.YlGnBu(colors)

            # ax.bar3d(xpos,ypos,zpos,dx,dy,dz,color=colors,shade=True)

            # ax.invert_xaxis()
            # ax.yaxis.set_ticklabels(fixed_a_results.columns)
            # ax.xaxis.set_ticklabels(fixed_a_results.index)
            # ax.set_ylabel('Word Length',labelpad=15)
            # ax.set_zlabel('TLB',labelpad=15)
            # ax.set_zlim(0,0.5)


            # st.markdown('### TLB per Word Length w/ Fixed Alphabet Size')
            # st.pyplot(fig)
    with tab_tlb_statplots:
        tlbs_subset = tlbs_all[tlbs_all['dataset'].isin(datasets)]

        container_tlb_method = st.container()
        all_tlb_method = st.checkbox("Select all",key='all_tlb_method',value=True)
        if all_tlb_method: tlb_methods_family = container_tlb_method.multiselect('Select a group of methods', methods, methods, key='selector_tlb_methods_all')
        else: tlb_methods_family = container_tlb_method.multiselect('Select a group of methods',methods, key='selector_tlb_methods')

        fixed_w = st.slider('Word Length',4,10,4,key='tlb_stat_w_sliders')
        fixed_a = st.slider('Alphabet Size',4,10,4,key='tlb_stat_a_sliders')

        tlbs_subset = tlbs_all[(tlbs_all['dataset'].isin(datasets)) & (tlbs_all['w'] == fixed_w) & (tlbs_all['a'] == fixed_a) & (tlbs_all['method'].isin(tlb_methods_family))]

        # tlbs_subset['method'] = tlbs_subset['method'].replace({
        #         'SAX': 'SAX + MINDIST', 
        #         'SFA': 'SFA + SFA-MINDIST',
        #         'SPARTAN': 'SPARTAN + PCA-MINDIST'
        #     })

        tlbs_subset = pd.pivot(tlbs_subset,index='dataset',columns='method',values='tlb').reset_index()

        plot_tlb_stat_plot(tlbs_subset,datasets,tlb_methods_family)


    # fig = go.Figure()

    # fig.add_trace(plotly_bar_charts_3d(tlb_x,tlb_y,tlb_z,color='x+y', x_title='Alphabet Size', y_title='Word Length',z_title= 'TLB'))

    # fig = plotly_bar_charts_3d(tlb_x,tlb_y,tlb_z,color='x+y', x_title='Alphabet Size', y_title='Word Length',z_title= 'TLB')

with tab_ad:

    st.markdown('# Case Study: Time-series Anomaly Detection')
    st.markdown(text_ad_description)

    ts_file_name = "074_UCR_Anomaly_DISTORTEDqtdbSel1005V_4000_12400_12800.out"
    ad_method = st.selectbox('Symbolic Method for Anomaly Detection',
                             AD_methods_list,
                             index=0)
    
    if ad_method not in AD_methods_list:
        ad_method = 'SPARTAN'

    anomaly_df = pd.read_csv(f"data/AD/{ts_file_name}-{ad_method}-prediction.csv")
    plot_anomaly_figure(anomaly_df)

with tab_runtime:
    st.markdown('# Accuracy-to-Runtime Analysis')
    st.markdown(text_runtime_description)

    # add runtime analysis (UCR)
    st.markdown('## Analysis #1: 128 UCR datasets')
    background = Image.open('data/runtime_analysis_ucr.png')

    # Center align the image and add a caption
    col1, col2, col3 = st.columns([2, 5, 2])  # Adjust column widths for centering
    with col2:
        st.image(background, use_container_width=True)
        st.markdown("<p style='text-align: center; font-style: italic;'>SPARTAN family versus top baseline methods on 128 UCR datasets.</p>", unsafe_allow_html=True)

    # add runtime analysis (large scale datasets)
    st.markdown('## Analysis #2: Large-scale datasets')
    runtime_result_option = st.selectbox('Select metric for runtime tradeoff comparison',runtime_options)

    if runtime_result_option == 'Comapring with varying time-series length':

        fig_accuracy, fig_runtime = create_plot_varylen(df_accuracy_varylen, 
                                                        df_runtime_varylen, 
                                                        LINE_WIDTH = LINE_WIDTH, 
                                                        MARKER_SIZE = MARKER_SIZE, 
                                                        FONT_SIZE = FONT_SIZE, 
                                                        SMALL_FONT_SIZE = SMALL_FONT_SIZE)
    
    else:
        fig_accuracy, fig_runtime = create_plot_varynum(df_accuracy_varynum, 
                                                        df_runtime_varynum, 
                                                        LINE_WIDTH = LINE_WIDTH, 
                                                        MARKER_SIZE = MARKER_SIZE, 
                                                        FONT_SIZE = FONT_SIZE, 
                                                        SMALL_FONT_SIZE = SMALL_FONT_SIZE)
    

    # # Streamlit App Layout
    # st.title("Time-Series Performance Comparison")

    # Create Two Columns
    col1, col2 = st.columns([1, 1])  # Two equal-sized columns

    # Display Plots in Respective Columns
    with col1:
        st.plotly_chart(fig_accuracy, use_container_width=True)

    with col2:
        st.plotly_chart(fig_runtime, use_container_width=True)

    # with runtime_col1:

    #     st.markdown('## Total Runtime vs. Accuracy Comparison (Train+Inference)')
    #     runtime_subset_mean.rename(columns={'train_time':'Training Runtime (ms/sample)','pred_time':'Inference Runtime (ms/sample)','total_time':'Total Runtime (ms/sample)','acc':'Mean Accuracy','rank':'Mean Rank'},inplace=True)
    #     fig = px.scatter(runtime_subset_mean,x='Total Runtime (ms/sample)',y=runtime_result_option,color='method',log_x=True)
    #     fig.update_traces(marker_size=10)
    #     if runtime_result_option == 'Mean Rank':
    #         fig.update_layout(
    #             yaxis = dict(autorange="reversed")
    #         )
    #     fig.update_layout( 
    #                     width=600, 
    #                     height=600, 
    #                     template="plotly_white", 
    #                     font=dict(
    #                             size=39,
    #                             color="black"))
    #     st.plotly_chart(fig)
    # with runtime_col2:
    #     st.markdown('## Training Runtime vs. Accuracy Comparison')
    #     fig = px.scatter(runtime_subset_mean,x='Training Runtime (ms/sample)',y=runtime_result_option,color='method',log_x=True)
    #     fig.update_traces(marker_size=10)
    #     if runtime_result_option == 'Mean Rank':
    #         fig.update_layout(
    #             yaxis = dict(autorange="reversed")
    #         )
    #     fig.update_layout( 
    #                     width=600, 
    #                     height=600, 
    #                     template="plotly_white", 
    #                     font=dict(
    #                             size=39,
    #                             color="black"))
    

    #     st.plotly_chart(fig)
    # with runtime_col3:
    #     st.markdown('## Inference Runtime vs. Accuracy Comparison')
    #     fig = px.scatter(runtime_subset_mean,x='Inference Runtime (ms/sample)',y=runtime_result_option,color='method',log_x=True)
    #     fig.update_traces(marker_size=10)
    #     if runtime_result_option == 'Mean Rank':
    #         fig.update_layout(
    #             yaxis = dict(autorange="reversed")
    #         )
    #     fig.update_layout( 
    #                     width=600, 
    #                     height=600, 
    #                     template="plotly_white", 
    #                     font=dict(
    #                             size=39,
    #                             color="black"))
    #     st.plotly_chart(fig)

with tab_references:
    st.markdown('# References')
    st.markdown(references)