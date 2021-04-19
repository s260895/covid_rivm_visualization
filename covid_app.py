import streamlit as st
# from compute2d import compute_2d_histogram
import numpy as np
import pandas as pd
import altair as at
from copy import copy
import plotly.graph_objects as go
# from paracoords import create_paracoords
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import plotly
import plotly.graph_objs as go

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_validate, ShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_validate, ShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import pandas as pd
# preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_validate, ShuffleSplit

# models
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor 
from sklearn.neural_network import MLPRegressor


#pipelines
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import make_column_transformer

# import pickle5 as pickle
import itertools
from joblib import dump,load

#set configurations and title
st.set_page_config(page_title="COVID-19 Infection Rate Visualizer",
    page_icon=":health:",
    layout='wide')

st.title('COVID-19 RIVM Infection Data Interactive Predictive Analytics Module')
# st.write(".")

#load data
#load data
cvd = pd.read_csv('COVID-19_aantallen_gemeente_cumulatief.csv',sep=";",header=0).dropna()[['Date_of_report','Province','Total_reported','Deceased']].groupby(['Date_of_report','Province']).sum()
cvd.reset_index(inplace=True)
features = ['Date_of_report', 'Province', 'Total_reported', 'Deceased']
numeric_columns = list()
label = ['Total_reported','Deceased',]
# split_data = [elem.split() for elem in auto_mpg[0]] 
# for elem in split_data:
#     for e in  elem:
#         if e == '?':
#             e = np.nan
# df = pd.DataFrame([[0,0,0,0,0,0,0,0]],columns=features)
# for elem in split_data:
#     df = df.append(pd.DataFrame([[float(e) if e!='?' else np.nan for e in elem]],columns=features))

# df,_ = train_test_split(df,train_size=0.05)

# # X = df[numeric_columns]
# y_1 = df[label[0]]
# y_2 = df[label[1]]

provinces = cvd['Province'].unique()


# def get_mlp_model(x=X.values,y=y.values.ravel(),mlp_alpha=0.1,mlp_activation='relu'):
#     return Pipeline([('scaler', StandardScaler()), ('pca',PCA(n_components=1)), ('regressor', MLPRegressor(hidden_layer_sizes=(100,50,10,),alpha=mlp_alpha,activation=mlp_activation_slider))]).fit(x,y)

# def get_svr_model(x=X.values,y=y.values.ravel(),svr_kernel='linear',svr_c=0.1):
#     return Pipeline([('scaler', StandardScaler()), ('pca',PCA(n_components=1)), ('regressor', LinearSVR(C=svr_c))]).fit(x,y)

# def get_linear_model(x=X.values,y=y.values.ravel()):
#     return Pipeline([('scaler', StandardScaler()), ('pca',PCA(n_components=1)), ('regressor', LinearRegression())]).fit(x,y)

# def get_rf_model(x=X.values,y=y.values.ravel(),rf_n_estimators=100):
#     return Pipeline([('scaler', StandardScaler()), ('pca',PCA(n_components=1)), ('regressor', RandomForestRegressor(n_estimators=rf_n_estimators) )]).fit(x,y)

# def get_candidate_models(x=X.values,y=y.values.ravel(),mlp_alpha=0.3,mlp_activation='relu',svr_kernel='linear',svr_c=0.1):
    
#     # mlp_pipe = Pipeline([('scaler', StandardScaler()), ('pca',PCA(n_components=1)), ('regressor', MLPRegressor(epsilon=1e-7,hidden_layer_sizes=(100,50,10),alpha=mlp_alpha,activation=mlp_activation))]).fit(x,y)
#     # svr_pipe = Pipeline([('scaler', StandardScaler()), ('pca',PCA(n_components=1)), ('regressor', SVR(kernel=svr_kernel,C=svr_c,max_iter=200))]).fit(x,y)
#     linear_pipe = Pipeline([('scaler', StandardScaler()), ('pca',PCA(n_components=1)), ('regressor', LinearRegression())]).fit(x,y)
#     regressors = {
#         # 'MLP':mlp_pipe,
#             #   'SVR': svr_pipe ,
#                'Linear': linear_pipe
#              }
#     return regressors
def regression_plot(df=cvd,xlabel='Days',logy=False,province='Drenthe',regressors=None):
    
    xlabel = 'Days'
    # province = provinces[0]
    df = df[df['Province'] == province]

    ylabel = 'Province: ' + str(province)

    # .plot()

    fig, ax = plt.subplots()    
    # x_pca = None   
    # if regressors is not None:
        # for regressor in regressors.items():
        #regressor = regressor[1].fit(x.reshape(-1,1),y)
    # x_std = regressors.named_steps['scaler'].transform(x)
    # x_pca = regressors.named_steps['pca'].transform(x_std)
    # max_x = np.max(x_pca)
    # min_x = np.min(x_pca)

    ax.plot(df[['Total_reported']] ,label='Total Cases Reported',linewidth=3)
    ax.plot(df[['Deceased']] ,label='Total Deceased',linewidth=3)
    # ax.plot(x_pca,y, linewidth=0, marker='v', label='Data points',color='m')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale('log')
    ax.legend(facecolor='white')

    return fig

#create 2d correlation
# value_columns_white = copy(white)
# white2d_bins = pd.concat([compute_2d_histogram(var1, var2, white) for var1 in value_columns_white for var2 in value_columns_white])

# value_columns_red = copy(red)
# red2d_bins = pd.concat([compute_2d_histogram(var1, var2, white) for var1 in value_columns_red for var2 in value_columns_red])

#create selectors
# st.write("Choose your Parameters:")

# model_selection_slider = st.sidebar.select_slider('Model Type',['MLP','Linear','RF'],'Linear')

log_scaling_slider = st.sidebar.radio('Y-Axis Log Scale',[True,False],False)

province_slider = st.sidebar.select_slider('Province',list(provinces),'Drenthe')


# mlp_activation_slider = st.sidebar.select_slider('MLP Activation Function',['identity','logistic','tanh','relu'],'relu')

# rf_n_estimator_slider = st.sidebar.select_slider('RF No of Estimators',[100,200,500],100)
# mlp_hidden_layer_size_text = st.text_input("MLP Hidden Layer Size", (100,))

# regressors = get_candidate_models(x=X.values,y=y.values.ravel(),mlp_alpha=mlp_alpha_slider,mlp_activation=mlp_activation_slider,svr_kernel=svr_kernel_slider,svr_c=svr_c_slider)
# "The dataset contains attributes about each car and a mileage per gallon value. The plot below shows how three different models fit the given data. Play around with the sliders to see how performance and error scores of the models is affected by different combinations of model building parameters."

fig = regression_plot(df=cvd,logy=log_scaling_slider,province =province_slider,regressors=None)
st.write(fig)

# if model_selection_slider == 'MLP':
#     regressor = get_mlp_model(x=X.values,y=y.values.ravel(),mlp_alpha=mlp_alpha_slider,mlp_activation=mlp_activation_slider)
#     fig = regression_plot(x=X.values,y=y.values,logy=log_scaling_slider,regressors=regressor)
#     st.write(fig)
# if model_selection_slider == 'Linear':
#     regressor = get_linear_model(x=X.values,y=y.values.ravel())
#     fig = regression_plot(x=X.values,y=y.values,logy=log_scaling_slider,regressors=regressor)
#     st.write(fig)
# if model_selection_slider == 'RF':
#     regressor = get_rf_model(x=X.values,y=y.values.ravel(),rf_n_estimators=rf_n_estimator_slider)
#     fig = regression_plot(x=X.values,y=y.values,logy=log_scaling_slider,regressors=regressor)
#     st.write(fig)


# st.plotly_chart(fig, use_container_width=True)

st.markdown("*-Created by Sumeet Zankar*")