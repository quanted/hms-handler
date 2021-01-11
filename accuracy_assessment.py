
import numpy as np
import pandas as pd
import re
from vb_estimators import L1Lars
import logging
from statsmodels.api import OLS,add_constant

class DataTool:
    def __init__(self):
        self.logger=logging.getLogger(__name__)
        self.c_stats=pd.read_csv('catchment-stats-data.csv')
        self.flat_c_stats_df=None
        self.modeldict={
            'data_sources':['nldas'], #'gldas'
            'accuracy_metrics':['pearsons'], #'pearsons'
            'exclusions':['monthly','yearly','none','25p'], # 'nonzero','monthly',''
            'geog_covs':['section'], #geographic covariates
            'regressiondict':{
                'interaction':False,
                'pipeline':L1Lars,
            }
        }
        self.geog_names=['division','province','section'] #descending size order
        
    def describe_data(self):
        print('first 5 rows:', self.c_stats.head())
        print('pandas describe:',self.c_stats.describe())
    
    def run_acc_compare(self,regressiondict=None):
        if regressiondict is None:
            regressiondict=self.modeldict['regressiondict']
        if self.flat_c_stats_df is None:
            self.set_flat_c_stats_df()
        data_df=self.flat_c_stats_df
        data_df.dropna(inplace=True,axis=0)
        y_df=data_df.loc[:,'acc']
        X_df=data_df.drop(labels='acc',axis=1,inplace=False)
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
        c_stats=self.set_geog_names(c_stats,geog_cols)
        filter_names=self.modeldict_filter_filters(c_stats.columns.to_list())
        self.filter_names=filter_names
        #filter_name_splits=[re.split('_',fil) for fil in filter_names]
        df_index=[]
        #ydata=[]
        data={'acc':[],'filter':[],'source':[],'metric':[],**{var:[] for var in geog_cols}}
        comids=c_stats.loc[:,'comid']
        comid_count=len(comids)
        for f_i,fil in enumerate(filter_names):
            name_parts=re.split('_',fil)
            print(name_parts)
            fil_name='_'.join(name_parts[1:-1]) 
            #if len(fil_name)==0:fil_name='none'
            src=name_parts[0]
            met=name_parts[-1]
            print(src,met)
            #ydata.append(c_stats.loc[:,fil])
            data['acc'].extend(c_stats.loc[:,fil])
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
        for col in geog_cols[:0:-1]: #backwards but not item 0
            g_loc=self.geog_names.index(col)
            df.loc[:,col]=self.str_sum(df.loc[:,self.geog_names[:g_loc+1]])
        return df
    
    def str_sum(self,df):
        cols=df.columns.to_list()
        ser=df.loc[:,cols.pop(-1)]
        for col in cols:
            ser=ser+'_'+df.loc[:,col]
        return ser
            
            
    def modeldict_filter_filters(self,filter_names):
        exclusions=dict.fromkeys(self.modeldict['exclusions'])#dict for fast search
        model_metrics=dict.fromkeys(self.modeldict['accuracy_metrics'])
        sources=dict.fromkeys(self.modeldict['data_sources'])
        
        filter_name_splits=[re.split('_',fil) for fil in filter_names]
        for f_i,fil_name in enumerate(filter_name_splits):
            if len(fil_name)==2:
                filter_name_splits[f_i]=[fil_name[0],'none',fil_name[1]]
                
        keep_idx=[i for i,name_parts in enumerate(filter_name_splits) 
                  if name_parts[-1] in model_metrics 
                  and name_parts[0] in sources 
                  and not '_'.join(name_parts[1:-1]) in exclusions]
        return [filter_names[i] for i in keep_idx]
    
    
        
        