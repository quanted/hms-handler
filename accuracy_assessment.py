import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import re
import os
import logging
from statsmodels.api import OLS,add_constant
import matplotlib.pyplot as plt
from random import shuffle
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LassoCV,LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from multiprocessing import Process,Queue
from mylogger import myLogger
from data_analysis import get_comid_data
from mp_helper import MpHelper

class RunoffCompare:
    def __init__(self,):
        self.proc_count=8
        if not os.path.exists('results'):
            os.mkdir('results')
        self.modeldict={
            'sources':{'observed':'nldas','modeled':'cn'}, #[observed,modeled]
            'filter':'none',#'nonzero'
            'train_share':0.80,
            'split_order':'chronological',#'random'
            'max_poly_deg':5,
            'regularize':[None,'lasso']# 'l1' or 'lasso', 'l2' or 'ridge'
            
        }
        #self.logger=logging.getLogger(__name__)
        self.comidlist=pd.read_csv('catchments-list-cleaned.csv')['comid'].to_list()
    
    def build_comidlist_runoff_df(self):
        try:
            self.bigdf=pd.read_pickle('bigdf.pkl')
            print('dataframe has been loaded from disk')
            return
        except:
            print(f'building the dataframe')
        comid_dflist=[get_comid_data(comid) for comid in self.comidlist]
        self.comid_dflist=comid_dflist
        bigdf=pd.concat(comid_dflist,keys=self.comidlist)
        indexlist=bigdf.index.to_list()
        row_count=len(indexlist)
        tup_list=[(indexlist[i][0],bigdf.loc[:,'date'].iloc[i]) for i in range(row_count)]
        new_m_idx=pd.MultiIndex.from_tuples(tup_list,names=['comid','date'])
        bigdf.index=new_m_idx
        bigdf.drop(label='date',axis=1,inplace=True)
        self.bigdf=bigdf
        bigdf.to_pickle('bigdf.pkl')
        
    def build_train_test_big_df(self,):
        try: self.bigdf
        except: self.build_comidlist_runoff_df()
        split_order=self.modeldict['split_order']
        train_share=self.modeldict['train_share']
        datelist=self.bigdf.index.unique(level='date').sort_values()
        if split_order=='random':
            shuffle(datelist)
        date_count=len(datelist)
        split_idx=int(date_count*train_share)
        train_dates=datelist[:split_idx]
        test_dates=datelist[split_idx:]
        self.bigdf_train=self.bigdf.loc[(slice(None),train_dates),:]
        self.bigdf_test=self.bigdf.loc[(slice(None),test_dates),:]
        
    def runCorrectionModeler(self):
        try: self.bigdf_train,self.bigdf_test
        except: self.build_train_test_big_df()
        
        
        
        

    
    
        
        
        
        
        
class DataTool:
    def __init__(self):
        #self.logger=logging.getLogger(__name__)
        self.c_stats=pd.read_csv('catchment-stats-data.csv')
        self.flat_c_stats_df=None
        self.modeldict={
            'data_sources':['nldas'], #'gldas'
            'accuracy_metrics':['pearsons'], #'pearsons'
            'exclusions':['monthly','yearly','nonzero','25p'], # 'nonzero','monthly',''
            'geog_covs':['province'], #(a list with just 1) geographic covariates
        }
        self.geog_names=['division','province','section'] #descending size order
        self.expand_geog_names=False #True will append larger geogs names in hierarchy to smaller ones. e.g., to see which province a section is, etc.
        self.physio_path='ecoregions/physio.dbf'
        if not os.path.exists(self.physio_path):
            print(f'cannot locate {self.physio_path}, the physiographic boundary shapefile. download it and unzip it in a folder called "ecoregions".')
        self.states_path='geo_data/states/cb_2017_us_state_500k.dbf'
        
    def describe_data(self):
        print('first 5 rows:', self.c_stats.head())
        print('pandas describe:',self.c_stats.describe())
    
    def plot_acc_compare(self):
        self.run_acc_compare()
        self.eco=gpd.read_file(self.physio_path)
        self.eco.columns=[col.lower() for col in self.eco.columns.to_list()]
        geog=self.modeldict['geog_covs'][0]
        eco_geog=self.eco.dissolve(by=geog)
        eco_geog.index=[idx.lower() for idx in eco_geog.index.to_list()]
        
        params=self.model_result.params
        params.index=[f"{re.split('_',idx)[1].lower()}" for idx in params.index]
        params.index.name=geog
        params.name='coefficient'
        self.params=params
        
        self.param_gdf=eco_geog.join(params)
        
        fig=plt.figure(dpi=300,figsize=[12,6])
        fig.suptitle(f'modeldict:{self.modeldict}')
        ax=fig.add_subplot(1,1,1)
        
        #ax.set_title('top positive features')
        self.param_gdf.plot(column='coefficient',ax=ax,cmap='plasma',legend=True,)
        #self.param_gdf.boundary.plot(ax=ax,edgecolor='w',linewidth=1)
        self.add_states(ax)
        #ax.legend()
        
        """params=self.model_result.params.reset_index()
        params.columns=[geog,'coefficient']
        params.loc[:,'regressor']=params['regressor'].apply(lambda x:re.split('_',x)[1])
        self.params=params
        self.eco."""
        
    def add_states(self,ax):
        try: self.eco_clip_states
        except:
            states=gpd.read_file(self.states_path)
            eco_d=self.eco.copy()
            eco_d['dissolve_field']=1
            eco_d.dissolve(by='dissolve_field')
            self.eco_clip_states=gpd.clip(states,eco_d)
        self.eco_clip_states.boundary.plot(linewidth=1,ax=ax,color=None,edgecolor='w')
        
    def run_acc_compare(self,print_summary=False):
        #if regressiondict is None:
        #    regressiondict=self.modeldict['regressiondict']
        self.set_flat_c_stats_df()
        data_df=self.flat_c_stats_df
        data_df.dropna(inplace=True,axis=0)
        y_df=data_df.loc[:,'accuracy']
        X_df=data_df.drop(labels='accuracy',axis=1,inplace=False)
        #print('y_df',y_df)
        #print('X_df',X_df)
        X_dtypes_=dict(X_df.dtypes)
        obj_vars=[var for var,dtype in X_dtypes_.items() if dtype=='object']
        #float_idx=[i for i in range(X_df.shape[1]) if i not in obj_idx]
        #self.model=regressiondict['pipeline'](cat_idx=obj_idx,float_idx=float_idx)  
        X_float_df=self.floatify_df(X_df,obj_vars)
        #X_float_df=add_constant(X_float_df)
        self.X_float_df=X_float_df;self.y_df=y_df
        self.model=OLS(y_df,X_float_df)
        self.model_result=self.model.fit()
        if print_summary:
            print('OLS results for modeldict:')
            print(self.modeldict)
            print(self.model_result.summary())
        
    def floatify_df(self,df,obj_vars):
        cols=df.columns.to_list()
        for col in cols:
            ser=df.loc[:,col]
            unq=ser.unique()
            if len(unq)<2:
                df.drop(labels=col,axis=1,inplace=True)
            else:
                if col in obj_vars:
                    for dummy in unq:#[:-1]:
                        arr=np.zeros(ser.shape)
                        arr[ser==dummy]=1
                        dummy_name=f'{col}_{dummy}'
                        df.loc[:,dummy_name]=arr
                    df.drop(labels=col,axis=1,inplace=True)
        return df
                    
        
        
    
    def set_flat_c_stats_df(self):
        geog_cols=self.modeldict['geog_covs']
        
        c_stats=self.c_stats
        if self.expand_geog_names:
            c_stats=self.set_geog_names(c_stats,geog_cols)
        filter_names=self.modeldict_filter_filters(c_stats.columns.to_list())
        self.filter_names=filter_names
        df_index=[]
        data={'accuracy':[],'filter':[],'source':[],'metric':[],**{var:[] for var in geog_cols}}
        comids=c_stats.loc[:,'comid']
        comid_count=len(comids)
        for f_i,fil in enumerate(filter_names):
            name_parts=re.split('_',fil)
            #print(name_parts)
            fil_name='_'.join(name_parts[1:-1]) 
            #if len(fil_name)==0:fil_name='none'
            src=name_parts[0]
            met=name_parts[-1]
            #print(src,met)
            #ydata.append(c_stats.loc[:,fil])
            data['accuracy'].extend(c_stats.loc[:,fil])
            df_index.extend([(comid,f_i)for comid in comids])
            
            for key in geog_cols:
                data[key].extend(c_stats.loc[:,key].to_list())
            data['filter'].extend([fil_name for _ in range(comid_count)]) 
            data['source'].extend([src for _ in range(comid_count)]) 
            data['metric'].extend([met for _ in range(comid_count)]) 
        #data=self.set_geog_names(data,geog_cols)
        self.data=data
        self.flat_c_stats_df=pd.DataFrame(data,index=df_index)
        
    def set_geog_names(self,df,geog_cols):
        for col in geog_cols[::-1]:
            g_loc=self.geog_names.index(col)
            parent_geogs=df.loc[:,self.geog_names[:g_loc+1]]
            df.loc[:,col]=self.str_sum(parent_geogs)
            
        return df
    
    def str_sum(self,df):
        cols=df.columns.to_list()
        ser=df.loc[:,cols.pop(-1)]
        for col in cols:
            ser=ser+f'__{col}-'+df.loc[:,col]
        return ser
            
            
    def modeldict_filter_filters(self,var_names_to_filter):
        """
        starting with a list of variable names with distinct parts 
        separated by '_', keeps/removes based on inclusions/exclusions 
        in self.modeldict
        """
        exclusions=dict.fromkeys(self.modeldict['exclusions'])#dict for fast search
        model_metrics=dict.fromkeys(self.modeldict['accuracy_metrics'])
        sources=dict.fromkeys(self.modeldict['data_sources'])
        
        var_name_splits=[re.split('_',var) for var in var_names_to_filter]
        for v_i,var_name in enumerate(var_name_splits):
            if len(var_name)==2:
                var_name_splits[v_i]=[var_name[0],'none',var_name[1]]
                
        keep_idx=[i for i,name_parts in enumerate(var_name_splits) 
                  if name_parts[-1] in model_metrics 
                  and name_parts[0] in sources 
                  and not '_'.join(name_parts[1:-1]) in exclusions]
        return [var_names_to_filter[i] for i in keep_idx]
    
    

                                  