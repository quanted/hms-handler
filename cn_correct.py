import pickle,json
import numpy as np
import pandas as pd
import geopandas as gpd
import re
import os
import logging
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm,Normalize
from random import shuffle,seed
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LassoCV,LinearRegression,LassoLarsCV,ElasticNetCV
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.model_selection import RepeatedKFold,GridSearchCV,LeaveOneGroupOut
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor,StackingRegressor
from sklearn.base import BaseEstimator, TransformerMixin,RegressorMixin
from scipy.stats import pearsonr
from multiprocessing import Process,Queue
from mylogger import myLogger
from data_analysis import get_comid_data
from mp_helper import MpHelper,MpWrapper
from collections import Counter
from time import time
from copy import deepcopy



class SeriesCompare:
    def __init__(self,y,yhat):
        self.err=y-yhat
        sum_sq_err=np.sum(self.err**2)
        self.mse=np.mean(sum_sq_err)
        self.ymean=np.mean(y)
        #print('ymean',self.ymean)
        sum_sq_err_at_mean=np.sum((y-self.ymean)**2)
        self.nse=1.0-(sum_sq_err/sum_sq_err_at_mean)
        self.pearsons,self.pearsons_pval=pearsonr(yhat,y)
 


class ToFortranOrder(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        if type(X) is pd.DataFrame or type(X) is pd.Series:
            return np.asfortranarray(X.to_numpy())
        else:
            return np.asfortranarray(X)
    

class DropConst(BaseEstimator,TransformerMixin):
    def __init__(self):
        self.logger=logging.getLogger()
        pass
    def fit(self,X,y=None):
        if type(X) is np.ndarray:
            X_df=pd.DataFrame(X)
        else:
            X_df=X
        self.unique_=X_df.apply(pd.Series.nunique)
        return self
    def transform(self,X):
        if type(X) is pd.DataFrame:
            xd= X.loc[:,self.unique_>1]
        else:
            xd= X[:,self.unique_>1]
        if xd.shape[1]==0:
            xd=pd.DataFrame(np.ones((X.shape[0],1)))
        return xd



class PipeWrapper:
    def __init__(self,data_dict,model_spec_tup):
        #self.data_hash=joblib.hash(data_dict)
        self.model_spec_tup=model_spec_tup
        self.modeldict=self.model_spec_tup[2]
        x=data_dict['x']
        y=data_dict['y']
        if self.modeldict['cross_validate']:
            self.train_comids=dict.fromkeys(x.index.get_level_values('comid').unique())#dict for faster search
        
        myLogger.__init__(self,'PipeWrapper.log')
        self.modeled_runoff_col=self.modeldict['sources']['modeled']
        self.data_filter=self.modeldict['filter']
        if self.data_filter == 'none':
            self.model=PipelineModel(self.model_spec_tup)
            self.model.fit(x,y)
        elif self.data_filter == 'nonzero':
            modeled_runoff=x.loc[:,self.modeled_runoff_col]
            zero_idx=modeled_runoff==0
            self.model={}
            xz=x[zero_idx];yz=y[zero_idx]
            #self.logger.error(f'no cols in xz. x:{x}')
            self.model['zero']=PipelineModel(('lin-reg',{'max_poly_deg':1,'fit_intercept':False},self.modeldict))
            self.model['zero'].fit(xz,yz)
            if x[~zero_idx].shape[0]>0:
                self.model['nonzero']=PipelineModel(self.model_spec_tup)
            else:
                self.model['nonzero']=NullModel()
            self.model['nonzero'].fit(x[~zero_idx],y[~zero_idx])
        else:assert False,f'self.data_filter:{self.data_filter} not developed'
            
            
    def predict(self,x):
        if self.modeldict['cross_validate']:
            x=self.remove_train_comids(x)
            if type(x) is str:
                return x
        if self.data_filter == 'none':
            return pd.DataFrame(self.model.predict(x),columns=[self.modeled_runoff_col],index=x.index)
        elif self.data_filter == 'nonzero':
            yhat_df=pd.DataFrame([np.nan]*x.shape[0],columns=[self.modeled_runoff_col],index=x.index)
            modeled_runoff=x.loc[:,self.modeled_runoff_col]
            zero_idx=modeled_runoff==0
            x_zero=x[zero_idx]
            if x_zero.shape[0]>0:
                yhat_df[zero_idx]=self.model['zero'].predict(x_zero)[:,None]
            x_nonzero=x[~zero_idx]
            if x_nonzero.shape[0]>0:
                nonzero_yhat=self.model['nonzero'].predict(x_nonzero)
                yhat_df[~zero_idx]=nonzero_yhat[:,None]
            return yhat_df
        
    def remove_train_comids(self,x):
        #assumes multi-index: (comid,date)
        full_comid_list=x.index.get_level_values('comid').unique().to_list()
        if any([comid in self.train_comids for comid in full_comid_list]):
            keep_comid_list=[comid for comid in full_comid_list if not comid in self.train_comids]
            if len(keep_comid_list)==0:
                self.logger.warning(f'making prediction and len(full_comid_list):{len(full_comid_list)} and len(keep_comid_list): {len(keep_comid_list)}')
                return 'none'
            self.logger.info(f'making prediction and len(full_comid_list):{len(full_comid_list)} and len(keep_comid_list): {len(keep_comid_list)}. {keep_comid_list}, x: {x}')
            xclean=x.loc[(keep_comid_list,slice(None)),:]
            return xclean
        else:
            return x
    
    def get_prediction_and_stats(self,xtest,ytest):
        yhat_test=self.predict(xtest)
        if type(yhat_test) is str and yhat_test=='none':
            return 'none','none'
        if self.modeldict['cross_validate']:
            ytest=self.remove_train_comids(ytest)
        assert all([xtest.index[i]==ytest.index[i] for i in range(xtest.shape[0])])
        test_stats=SeriesCompare(ytest.to_numpy(),yhat_test.to_numpy()[:,0])
        
        return yhat_test,test_stats
 
class NullModel(BaseEstimator,RegressorMixin):
    def __init__(self):
        pass
    def fit(self,x,y,w):
        pass
    def predict(self,x):
        if len(x.shape)>1:
            return np.mean(x,axis=1)
        return x
    
    
class MakePolyX(BaseEstimator,TransformerMixin):
    def __init__(self,degree=2,col_name=None,interact=True,no_constant=True):
        self.degree=degree
        self.col_name=col_name
        self.interact=interact
        self.no_constant=no_constant
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        if self.col_name is None:
            x=X.iloc[:,0]
            self.col_name=x.name
        else:
            x=X.loc[:,self.col_name]
        xlist=[x]
        for i in range(2,self.degree+1):
            xlist.append(x**i)
            xlist[-1].name=f'{self.col_name}^{i}'
        X_=X.drop(self.col_name,inplace=False,axis=1)
        if self.interact:
            i_list=[]
            for col1 in X_.columns:
                for ser in xlist:
                    i_list.append(X_.loc[:,col1]*ser)
                    i_list[-1].name=f'{col1}_X_{ser.name}'
            if self.no_constant:
                xlist=i_list
            else:
                xlist.extend(i_list)
        xlist.append(X_)
        return pd.concat(xlist,axis=1,)
    

class PipelineModel(BaseEstimator,RegressorMixin,myLogger):
    def __init__(self,model_spec_tup):
        self.model_spec_tup=model_spec_tup
        myLogger.__init__(self,)
        
    def fit(self,x,y):    
        model_name,specs,modeldict=self.model_spec_tup
        model_col_name=modeldict['sources']['modeled']
        if 'inner_cv' in specs:
            inner_cv=specs['inner_cv']
            n_repeats=inner_cv['n_repeats'];n_splits=inner_cv['n_splits']
            n_jobs=inner_cv['n_jobs']
        else:
            n_repeats=3;n_splits=10;n_jobs=1
        
        
        if model_name.lower() =='lin-reg':
            deg=specs['max_poly_deg']
            if deg>1:
                pipe=make_pipeline(
                MakePolyX(degree=2,col_name=model_col_name,interact=True,no_constant=True),
                StandardScaler(),
                DropConst(),       
                LinearRegression(fit_intercept=specs['fit_intercept'],normalize=False))
                param_grid={'makepolyx__degree':np.arange(1,deg+1)}
                cv=RepeatedKFold(random_state=0,n_splits=n_splits,n_repeats=n_repeats)
                self.pipe_=GridSearchCV(pipe,param_grid=param_grid,cv=cv,n_jobs=n_jobs)
            else:
                self.pipe_=make_pipeline(
                StandardScaler(),
                DropConst(),       
                LinearRegression(fit_intercept=specs['fit_intercept']))
            #self.pipe_.fit(x,y)
        elif model_name.lower() in ['l1','lasso','lassocv']:
            deg=specs['max_poly_deg']
            if 'n_alphas' in specs:
                n_alphas=specs['n_alphas']
            else: n_alphas=100
            cv=RepeatedKFold(random_state=0,n_splits=n_splits,n_repeats=n_repeats)
            lasso_kwargs=dict(random_state=0,fit_intercept=specs['fit_intercept'],cv=cv,n_alphas=n_alphas)
            self.pipe_=make_pipeline(
                MakePolyX(degree=deg,col_name=model_col_name,interact=True,no_constant=False),
                StandardScaler(),
                #PolynomialFeatures(include_bias=False,interaction_only=True),
                DropConst(),
                ToFortranOrder(),
                LassoCV(**lasso_kwargs,n_jobs=n_jobs,normalize=False))
            #self.pipe_.fit(x.astype(np.float32),y.astype(np.float32))
        elif model_name.lower() in ['lassolars','lassolarscv']:
            deg=specs['max_poly_deg']
            cv=RepeatedKFold(random_state=0,n_splits=n_splits,n_repeats=n_repeats)
            lasso_kwargs=dict(fit_intercept=specs['fit_intercept'],cv=cv)
            self.pipe_=make_pipeline(
                MakePolyX(degree=deg,col_name=model_col_name,interact=True,no_constant=False),
                StandardScaler(),
                #PolynomialFeatures(include_bias=False,interaction_only=True),
                DropConst(),
                ToFortranOrder(),
                LassoLarsCV(**lasso_kwargs,n_jobs=n_jobs,normalize=False))
        elif model_name.lower() in ['elastic net','elastic-net','elasticnet']:
            deg=specs['max_poly_deg']
            if 'l1_ratio' in specs:
                l1_ratio=specs['l1_ratio']
            else:
                l1_ratio=list(1-np.logspace(-2,-.03,5))
            cv=RepeatedKFold(random_state=0,n_splits=n_splits,n_repeats=n_repeats)
            enet_kwargs=dict(fit_intercept=specs['fit_intercept'],cv=cv,l1_ratio=l1_ratio)
            self.pipe_=make_pipeline(
                MakePolyX(degree=deg,col_name=model_col_name,interact=True,no_constant=False),
                StandardScaler(),
                #PolynomialFeatures(include_bias=False,interaction_only=True),
                DropConst(),
                ToFortranOrder(),
                ElasticNetCV(**enet_kwargs,n_jobs=n_jobs,normalize=False))
        elif model_name.lower()=='gbr':
            if 'kwargs' in specs:
                kwargs=specs['kwargs']
            else:kwargs={}
            reg=GradientBoostingRegressor(random_state=0,**kwargs)
            if 'param_grid' in specs:
                cv=RepeatedKFold(random_state=0,n_splits=n_splits,n_repeats=n_repeats)
                param_grid=specs['param_grid']
                self.pipe_=GridSearchCV(reg,param_grid=param_grid,cv=cv,n_jobs=n_jobs)
            else:
                self.pipe_=reg 
        elif model_name.lower() in ['stackingregressor','stacking-regressor']:
            stack_specs=specs['stack_specs']
            pipelist=[]
            for spec in stack_specs:
                m_name=list(spec.keys())[0]
                m_spec=spec[m_name]
                new_modeldict=deepcopy(modeldict)
                new_modeldict['model_specs']=spec
                new_model_spec_tup=(m_name,m_spec,new_modeldict)
                pipelist.append((m_name,PipelineModel(new_model_spec_tup)))
            self.pipe_=StackingRegressor(pipelist,n_jobs=n_jobs)
                
        else:
            assert False,'model_name not recognized'
        try:
            self.pipe_.fit(x,y)
            #self.pipe_.fit(x.astype(np.float32),y.astype(np.float32))
        except:
            self.logger.exception(f'fit error. x.shape:{x.shape}, y.shape:{y.shape}, x: {x}, y: {y}')
            assert False, 'fit error!!!'
        #self.logger.info(f'fit complete, starting yhat_train prediction')
        #self.yhat_train=pd.DataFrame(self.pipe_.predict(x),index=x.index,columns=['yhat'])
        #self.logger.info(f'yhat_train prediction complete.')
    
        return self
            
    def predict(self,x):
        return self.pipe_.predict(x)
    
    def set_test_stats(self,xtest,ytest):
        self.yhat_test=self.predict(xtest)
        self.poly_test_stats=SeriesCompare(ytest,self.yhat_test)
        #dself.uncorrected_test_stats=SeriesCompare(ytest,xtest)
        


            
  
 

class ComidData(myLogger):            
    def __init__(self,comid,modeldict):
        myLogger.__init__(self,'catchment_data.log')
        self.comid=comid
        self.sources=modeldict['sources']
        self.modeldict=modeldict
        self.runoff_df=self.make_runoff_df(comid,multi_index=True) 
        #self.modeldict=modeldict  
        
        #sources=self.modeldict['sources']
        obs_src=self.sources['observed']
        mod_src=self.sources['modeled']
        self.runoff_model_data_df=self.runoff_df.loc[:,[obs_src,mod_src]]
        n=self.runoff_model_data_df.shape[0]
        split_idx=int(self.modeldict['train_share']*n)
        nonzero=self.runoff_model_data_df.loc[:,mod_src].iloc[:split_idx]>0
        self.nonzero_count=nonzero.shape[0] 
        self.test_results={} # will contain {m_name:{test_stats:...,yhat_test:...}}
        
        #data_filter=self.modeldict['filter']
        #self.obs_df,self.mod_df=self.filter_data(obs_df,mod_df,data_filter)
        
        #self.mod_train,self.obs_train,self.mod_test,self.obs_test=self.set_train_test(self.obs_df,self.mod_df)
    def make_runoff_df(self,comid,multi_index=False):
        df=get_comid_data(comid)
        date=df['date']
        if multi_index:
            midx_tups=[(comid,date_i) for date_i in date.to_list()]
            df.index=pd.MultiIndex.from_tuples(midx_tups,names=['comid','date'])
        else:
            df.index=date
        df.drop(columns='date',inplace=True)
        
        return df     
    
    def set_train_test(self,):
        df=self.runoff_model_data_df
        train_share=self.modeldict['train_share']
        n=df.shape[0]
        split_idx=int(train_share*n)
        y_df=df.loc[:,self.sources['observed']]
        x_df=df.drop(self.sources['observed'],axis=1,inplace=False)
    
        self.x_train=x_df.iloc[:split_idx]
        self.y_train=y_df.iloc[:split_idx]
        self.x_test=x_df.iloc[split_idx:]
        self.y_test=y_df.iloc[split_idx:]
 

            
class Runner(myLogger):
    def __init__(self,X,y,m_name,specs,modeldict,train_comids=None,return_save_string=True):
        self.X=X;self.y=y
        self.m_name=m_name
        self.specs=specs
        self.modeldict=modeldict
        self.results_folder=modeldict['results_folder']
        self.train_comids=train_comids
        self.return_save_string=return_save_string
        
    def run(self):
        myLogger.__init__(self)
        self.model=self.runmodel()
        
    def runmodel(self):
        try:
            self.logger.info(f'starting {self.m_name}')
            t0=time()
            if not self.train_comids is None:
                self.logger.info(f'Before applying boolean, m_name:{self.m_name} has self.X.shape:{self.X.shape}')
                self.X=self.X.loc[(self.train_comids,slice(None)),:]
                self.y=self.y.loc[(self.train_comids,slice(None))]
                self.logger.info(f'after applying boolean, m_name:{self.m_name} has self.X.shape:{self.X.shape}')
            else:
                self.logger.info(f'm_name:{self.m_name} has train_comids:{self.train_comids}')
            data_dict={'x':self.X,'y':self.y}
            args=[data_dict,(self.m_name,self.specs,self.modeldict)]
            name=os.path.join(self.results_folder,f'pipe-{joblib.hash(args)}.pkl')
            if os.path.exists(name):
                try:
                    with open(name,'rb') as f:
                        model=pickle.load(f)
                    #self.model_results[m_name]=model
                    self.logger.info(f'succesful load from disk for {self.m_name} from {name}')
                    if self.return_save_string:
                        return name
                    else: return model
                except:
                    self.logger.exception(f'error loading {name} for {self.m_name}, redoing.')
            model=PipeWrapper(*args)
            #self.model_results[m_name]=model
            with open(name,'wb') as f:
                pickle.dump(model,f)
            t1=time()
            self.logger.info(f'{self.m_name} took {(t1-t0)/60} minutes to complete')
            print(f'{self.m_name} took {(t1-t0)/60} minutes to complete')
            if self.return_save_string:
                return name
            else: return model   
        except:
            self.logger.exception(f'error ')
            assert False,'Halt'
        
        


class DataCollection(myLogger):
    def __init__(self,comidlist,modeldict,comid_geog_dict):
        myLogger.__init__(self,'data_collection.log')
        self.comidlist=comidlist
        self.modeldict=modeldict
        self.results_folder=modeldict['results_folder']
        self.comid_geog_dict=comid_geog_dict
        self.geog_names=['division','province','section']
        self.physio_path='ecoregions/physio.dbf'
        self.states_path='geo_data/states/cb_2017_us_state_500k.dbf'
        self.failed_comid_dict={}
        self.onehot=OneHotEncoder(sparse=False)
        
    def build(self):
        self.collectComidData()
        self.addGeogCols()
        self.setComidTrainTest()
        self.assembleTrainDFs()
        
    def collectComidData(self):
        self.comid_data_object_dict={}
        
        name=os.path.join(self.results_folder,f'comiddata-{joblib.hash(self.comidlist)}.pkl')
        if os.path.exists(name):
            try:
                with open(name,'rb') as f:
                    self.comid_data_object_dict,self.failed_comid_dict=pickle.load(f)
                return
            except:
                print('load failed, building comiddata')
            
        for comid in self.comidlist:
            comid_data_obj=ComidData(comid,self.modeldict)
            if comid_data_obj.nonzero_count>32:
                self.comid_data_object_dict[comid]=comid_data_obj
            else:
                self.failed_comid_dict[comid]=f'runoff_model_data_df too small with shape:{comid_data_obj.runoff_model_data_df.shape}'
        if len(self.failed_comid_dict)>0:
            self.logger.info(f'failed comids:{self.failed_comid_dict}')
        savetup=(self.comid_data_object_dict,self.failed_comid_dict)
        with open(name,'wb') as f:
            pickle.dump(savetup,f)
        
    
    def addGeogCols(self,):
        model_geog=self.modeldict['model_geog']
        if not type(model_geog) is list:
            model_geog=[model_geog]
        
        for comid,obj in self.comid_data_object_dict.items():
            geog_dict=self.comid_geog_dict[comid]
            for col_name in model_geog:
                val=geog_dict[col_name]
            
                try:
                    obj.runoff_model_data_df.loc[:,col_name]=val
                except:
                    print(f'comid:{comid},col_name:{col_name},val:{val}.')
                    assert False,f'comid:{comid},col_name:{col_name},val:{val}.'
    
    def setComidTrainTest(self):
        for comid,obj in self.comid_data_object_dict.items():
            obj.set_train_test()

    def assembleTrainDFs(self):
        self.comid_modeling_objects=[]
        for comid,obj in self.comid_data_object_dict.items():
            if not obj.x_train.isnull().any(axis=None):
                self.comid_modeling_objects.append(obj)
            else:
                null_cols=obj.x_train.isnull().any(axis=0)
                self.failed_comid_dict[comid]=f'null values encountered in the following columns: {null_cols}'
        big_x_train_list,big_y_train_list=zip(*[(obj.x_train,obj.y_train) for obj in self.comid_modeling_objects])
        self.big_x_train_raw=pd.concat(big_x_train_list)#.reset_index(drop=True)) # drop comid and date 
        self.big_x_train=self.makeDummies(self.big_x_train_raw,fit=True)
        
        self.big_y_train=pd.concat(big_y_train_list)#.reset_index(drop=True))   
        for obj in self.comid_modeling_objects:
            obj.x_test_float=self.makeDummies(obj.x_test,fit=False)
    
    def runModel(self): #singular!
        model_scale=self.modeldict['model_scale']
        if model_scale=='conus':
            proc_count=1
        else:
            proc_count=6
        X=self.big_x_train
        y=self.big_y_train
        self.model_results={}
        cv=self.modeldict['cross_validate']
        geog=self.modeldict['model_geog']
        reps=cv['n_reps']
        strat=cv['strategy']
        assert strat=='leave_one_member_out',f'{strat} not developed'
        group_indicator=self.big_x_train_raw.loc[:,[geog]] #_raw is data before dummies created
        if cv:
            self.logger.info(f'making comid_train_list for cv')
            comid_train_list=self.build_comid_train_list(reps)                                                     
            #bool_idx_list=[bool_idx for bool_idx in GroupDFSplitter(reps,num_groups=5).get_df_split(group_indicator)]
            self.logger.info(f'comid_train_list_list complete')
        for m_name,specs in self.modeldict['model_specs'].items():
            single_modeldict=self.modeldict
            single_modeldict['model_specs']={m_name:specs}
            if not cv:
                self.model_results[m_name]=[self.Runner(X,y,m_name,specs,single_modeldict).run().model]
            else:
                self.model_results[m_name]=[]
                
                args_list=[];kwargs_list=[]
                for train_comids in comid_train_list:
                    #self.logger.info(f'cv run_{i} starting')
                    #self.logger.info(f'building args_list for {specs}')
                    ##args=[X[bool_idx],y[bool_idx],m_name,specs,single_modeldict]
                    args=[X,y,m_name,specs,single_modeldict]
                    args_list.append(args)
                    kwargs={'train_comids':train_comids}
                    kwargs_list.append(kwargs)
                    #model=self.run_it(X[bool_idx],y[bool_idx],m_name,specs,single_modeldict).run()
                    #self.logger.info(f'cv run_{i} complete')
                    #self.model_results[m_name].append(model)
                self.logger.info(f'starting multiproc Runners for {m_name}')
                results=MpHelper().runAsMultiProc(Runner,args_list,kwargs_list=kwargs_list,proc_count=proc_count)
                self.model_results[m_name].extend([result.model for result in results])
                self.logger.info(f'Runners complete for {m_name}')
     
    def build_comid_train_list(self,reps,num_groups=5):
        comid_train_list=[]
        full_comid_list=[]
        geog_comids_dict={}
        model_geog=self.modeldict['model_geog']
        for comid,geogs in self.comid_geog_dict.items():
            geog=geogs[model_geog]
            if not geog in geog_comids_dict:
                geog_comids_dict[geog]=[]
            geog_comids_dict[geog].append(comid)
            full_comid_list.append(comid)
        for r in range(reps):
            seed(r)
            [shuffle(comids) for comids in geog_comids_dict.values()]
            for g in range(num_groups):
                drop_comids={}
                for comids in geog_comids_dict.values():
                    try:drop_comids[comids[g]]=None
                    except IndexError: self.logger.info(f'index error for idx:{g}')
                    except:
                        self.logger.exception('error')
                        assert False,'halt'
                comid_train_list.append([c for c in full_comid_list if not c in drop_comids])
        return comid_train_list
                
                                                                 
            
                                                                 
    def runTestData(self):
        for m_name,model_list in self.model_results.items():
            self.logger.info(f'building test data/stats for {m_name}')
            if not self.modeldict['cross_validate']: 
                if not type(model_list) is list:model_list=[model_list]
            for model in model_list:
                if type(model) is str:
                    with open(model,'rb') as f:
                        model=pickle.load(f)
                for obj in self.comid_modeling_objects:
                    if not m_name in obj.test_results:
                        obj.test_results[m_name]=[]
                    if not self.modeldict['cross_validate'] or not obj.comid in model.train_comids:
                        yhat_test,test_stats=model.get_prediction_and_stats(obj.x_test_float,obj.y_test)
                        if type(test_stats) is str:
                                self.logger.warning(f'test data from comid:{obj.comid}: {(yhat_test,test_stats)}')
                        else:
                            self.logger.info(f'test data shape and stats from comid:{obj.comid}: {(yhat_test.shape,test_stats)}')
                        obj.test_results[m_name].append({'test_stats':test_stats,'yhat_test':yhat_test})
            for obj in self.comid_modeling_objects:
                uncorr_yhat=obj.x_test.loc[:,self.modeldict['sources']['modeled']]
                obj.test_results['uncorrected']=[{'test_stats':SeriesCompare(obj.y_test.values,uncorr_yhat.values),'yhat_test':uncorr_yhat}]
            
   
                
                
    def makeDummies(self,df,fit=False):
        if fit:
            self.obj_cols=[]
            self.num_cols=[]
            for col,dtype in df.dtypes.items():
                if dtype=='object':
                    self.obj_cols.append(col)
                else:
                    self.num_cols.append(col)
                    
            self.onehot.fit(df.loc[:,self.obj_cols])
            self.encoded_cols=self.onehot.get_feature_names(self.obj_cols)    
        dum_data=self.onehot.transform(df.loc[:,self.obj_cols])
        dum_df=pd.DataFrame(dum_data,index=df.index,columns=self.encoded_cols)
        return pd.concat([df.loc[:,self.num_cols],dum_df],axis=1)

    

    
        

class CompareCorrect(myLogger):
    def __init__(self,model_specs=None,modeldict=None):
        myLogger.__init__(self,'comparecorrect.log')
        self.dc_list=[]
        if not os.path.exists('results'):
            os.mkdir('results')
        if modeldict is None:
            self.modeldict={
                'cross_validate':{'n_reps':3,'strategy':'leave_one_member_out'},#False,#
                'model_geog':'section',
                'sources':{'observed':'nldas','modeled':'cn'}, #[observed,modeled]
                'filter':'nonzero',#'none',#'nonzero'
                'train_share':0.50,
                'split_order':'chronological',#'random'
                'model_scale':'division',#'division',#'comid'
                'model_specs':{
                    #no intercept b/c no dummy drop
                    'lasso':{'max_poly_deg':3,'fit_intercept':False,'inner_cv':{'n_repeats':3,'n_splits':10,'n_jobs':1}},
                    'gbr':{
                        'kwargs':{},#these pass through to sklearn's gbr
                        #'n_estimators':10000,
                        #'subsample':1,
                        #'max_depth':3
                    },
                    'lin-reg':{'max_poly_deg':3,'fit_intercept':False,'inner_cv':{'n_repeats':3,'n_splits':10,'n_jobs':1}},
                }
            } 
        else:
            self.modeldict=modeldict
        if not model_specs is None:
            self.modeldict['model_specs']=model_specs
        self.results_folder=os.path.join('results',f'{joblib.hash(self.modeldict)}')
        self.modeldict['results_folder']=self.results_folder
        if not os.path.exists(self.results_folder):
            os.mkdir(self.results_folder)
        modeldictpath=os.path.join(self.results_folder,'modeldict.json')
        with open(modeldictpath,'w') as f:
            json.dump(self.modeldict,f)
        printdir=os.path.join(self.results_folder,'print')
        if not os.path.exists(printdir):
            os.mkdir(printdir)
        
        #self.logger=logging.getLogger(__name__)
        clist_df=pd.read_csv('catchments-list-cleaned.csv')
        self.comid_physio=clist_df.drop('comid',axis=1,inplace=False)
        self.comid_physio.index=clist_df.loc[:,'comid']
        raw_comidlist=clist_df['comid'].to_list()
        self.comidlist=[key for key,val in Counter(raw_comidlist).items() if val==1]#[0:20]
        
        #keep order, but remove duplicates
        
        self.geog_names=['division','province','section'] #descending size order
        self.expand_geog_names=False #True will append larger geogs names in hierarchy to smaller ones. e.g., to see which province a section is, etc.
        self.physio_path='ecoregions/physio.dbf'
        if not os.path.exists(self.physio_path):
            print(f'cannot locate {self.physio_path}, the physiographic boundary shapefile. download it and unzip it in a folder called "ecoregions".')
        self.states_path='geo_data/states/cb_2017_us_state_500k.dbf'
        self.comid_geog_dict=self.makeComidGeogDict()
    
    
    def runBigModel(self,):
        model_scale=self.modeldict['model_scale']
        if model_scale.lower()=='conus':
            dc=self.runDataCollection(self.comidlist,self.modeldict)
            #dc.plotGeoTestData()
        elif model_scale.lower() in self.geog_names:
            geog_comidlist_dict={}
            for comid,cg_dict in self.comid_geog_dict.items():
                g=cg_dict[model_scale]
                if not g in geog_comidlist_dict:
                    geog_comidlist_dict[g]=[]
                geog_comidlist_dict[g].append(comid)
            for geog,comidlist in geog_comidlist_dict.items():
                modeldict=self.modeldict.copy()
                modeldict['model_scale']=(model_scale,geog)
                self.dc_list.append(self.runDataCollection(comidlist,modeldict))
    
    
    def setEcoGeog(self):
        geog=self.modeldict['model_geog']
        bigger_geog=self.geog_names[self.geog_names.index(geog)-1]
        try: self.eco
        except:
            self.eco=gpd.read_file(self.physio_path)
            self.eco.columns=[col.lower() for col in self.eco.columns.to_list()]
            self.eco.loc[:,geog].fillna(self.eco.loc[:,bigger_geog],inplace=True)
            eco_geog=self.eco.dissolve(by=geog)
            self.eco_geog=eco_geog
            
    def setDCResultsDict(self):
        data_dict={}
        geog=self.modeldict['model_geog']
        for dc in self.dc_list:
            m_names= list(dc.model_results.keys())+['uncorrected'] #adding b/c in test_results, but not model_results
            for m_name in m_names:
                if not m_name in data_dict:
                    data_dict[m_name]={'nse':[],'pearson':[],geog:[]}
                    
                for obj in dc.comid_modeling_objects:
                    for result_dict in obj.test_results[m_name]:
                        data_dict[m_name]['nse'].append(result_dict['test_stats'].nse)
                        data_dict[m_name]['pearson'].append(result_dict['test_stats'].nse)
                        data_dict[m_name][geog].append(self.comid_geog_dict[obj.comid][geog])
        self.dc_results_dict=data_dict
    
    def plotGeoTestData(self,plot_negative=True):
        try: self.eco_geog
        except: self.setEcoGeog()
        try: self.dc_results_dict
        except: self.setDCResultsDict()
        geog=self.modeldict['model_geog']
        for m_name,m_data_dict in self.dc_results_dict.items():            
            mean_acc_df=pd.DataFrame(m_data_dict).groupby(geog).mean()
            geog_acc_df=self.eco_geog.merge(mean_acc_df,on=geog)
            for metric in ['nse','pearson']:
                plt.rcParams['axes.facecolor'] = 'lightgrey'
                fig=plt.figure(dpi=300,figsize=[9,6])
                fig.patch.set_facecolor('w')
                fig.suptitle(f'modeldict:{self.modeldict}')
                ax=fig.add_subplot(1,1,1)
                ax.set_title(f'{m_name}_{metric}')
                self.eco_geog.plot(color='darkgrey',ax=ax)
                #pos_geog_acc_df=geog_acc_df[geog_acc_df.loc[:,metric]>0]
                #pos_geog_acc_df.plot(column=metric,ax=ax,cmap='plasma',legend=True,)
                if plot_negative:
                    norm = TwoSlopeNorm(vmin=-1,vcenter=0, vmax=1)
                    cmap='RdBu'#'brg'##'plasma'
                    cbar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
                    #self.geog_acc_df=geog_acc_df
                    geog_acc_df.plot(column=metric,ax=ax, cmap=cmap, norm=norm,legend=False,)
                    fig.colorbar(cbar, ax=ax)
                else:
                    cmap='plasma'
                    norm=Normalize(vmin=0,vmax=1)
                    cbar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
                    pos_geog_acc_df=geog_acc_df[geog_acc_df.loc[:,metric]>0]
                    pos_geog_acc_df.plot(column=metric,ax=ax,cmap='plasma',norm=norm,legend=False,)
                    fig.colorbar(cbar,ax=ax)
                self.add_states(ax)
                fig_name=f'{self.modeldict["model_scale"]}_{m_name}_{metric}.png'
                if not plot_negative:
                    fig_name='pos-score_'+fig_name
                if type(self.modeldict['cross_validate']) is dict:
                    fig_name='cv_'+fig_name
                
                
                fig.savefig(os.path.join(self.results_folder,'print',fig_name))
                
            
            
            
    def runDataCollection(self,comidlist,modeldict):
        args=[[comid,modeldict] for comid in comidlist]
        save_hash=joblib.hash(args)
        results_folder=modeldict['results_folder']
        save_path=os.path.join(results_folder,f'data-collection_{save_hash}')
        loaded=False
        if os.path.exists(save_path):
            try:
                with open(save_path,'rb') as f:
                    dc=pickle.load(f)
                return dc
            except:
                self.logger.exception(f'error loading data collection from {save_path} running data collection steps')
        else:
            print('running big model')
        if len(comidlist)<len(self.comidlist):
            comid_geog_dict={comid:geog for comid,geog in self.comid_geog_dict.items() if comid in comidlist}
        else:
            comid_geog_dict=self.comid_geog_dict
        dc=DataCollection(comidlist,modeldict,self.comid_geog_dict)
        dc.build()
        dc.runModel()
        dc.runTestData()
        with open(save_path,'wb') as f:
            pickle.dump(dc,f)
        return dc
    
    
        
    def makeComidGeogDict(self,geogs=None):
        if geogs is None:geogs=self.geog_names
        comid_geog_dict={}
        
        for comid in self.comidlist:
            comid_geog_dict[comid]={}
            for geog in geogs:
                if geog in self.geog_names: 
                    g_name=self.comid_physio.loc[comid,geog]
                    if pd.isnull(g_name):
                        bigger_geog=self.geog_names[self.geog_names.index(geog)-1]
                        g_name=self.comid_physio.loc[comid,bigger_geog]
                    comid_geog_dict[comid][geog]=g_name
                elif type(geog) is str and geog=='streamcat':
                    assert False,'not developed'
        return comid_geog_dict
    
    
        
        
    def add_states(self,ax):
        try: self.eco_clip_states
        except:
            states=gpd.read_file(self.states_path)
            eco_d=self.eco.copy()
            eco_d['dissolve_field']=1
            eco_d.dissolve(by='dissolve_field')
            self.eco_clip_states=gpd.clip(states,eco_d)
        self.eco_clip_states.boundary.plot(linewidth=1,ax=ax,color=None,edgecolor='k')                
    
class MultiCorrectionTool(myLogger):
    ###currently supports picking from multiple model_specs for each geog in the model conditioning level (model_geog)
    ###future work may pick the best from among other items in modeldict, e.g., model_scale
    def __init__(self,modeldict=None,model_specs=None):
        myLogger.__init__(self,'multi-correction-tool.log')
        if modeldict is None:
            self.modeldict={
                'cross_validate':{'n_reps':3,'strategy':'leave_one_member_out'},#False,#
                'model_geog':'section',
                'sources':{'observed':'nldas','modeled':'cn'}, 
                'filter':'nonzero',#'none',
                'train_share':0.50,
                'split_order':'chronological',#'random'
                'model_scale':'division',#'division',#'comid'
                'model_specs':{#these are usually set in the run_multi_correct.py or multi_correct.ipynb files that run MultiCorrectionTool
                    'lasso':{'max_poly_deg':3,'fit_intercept':False,'inner_cv':{'n_repeats':3,'n_splits':10,'n_jobs':1}},
                    'gbr':{'kwargs':{},},
                    'lin-reg':{'max_poly_deg':3,'fit_intercept':False,'inner_cv':{'n_repeats':3,'n_splits':10,'n_jobs':1}},
                }} 
        else:
            self.modeldict=modeldict
        if not model_specs is None:
            self.modeldict['model_specs']=model_specs
        self.m_names=list(self.modeldict['model_specs'].keys())
        assert type(self.modeldict['cross_validate']) is dict,f'expecting a dict for cross_validate but got:{self.modeldict["cross_validate"]}'
        
        
    def runCorrections(self,):
        self.corrections=[]
        model_specs=self.modeldict['model_specs']  
        new_model_dict=self.modeldict.copy()
        for m_spec_name,m_spec in model_specs.items():
            new_model_dict['model_specs']={m_spec_name:m_spec}
            self.corrections.append(CompareCorrect(modeldict=new_model_dict))
        for cc in self.corrections:
            cc.runBigModel()
            cc.plotGeoTestData(plot_negative=False)
            cc.plotGeoTestData(plot_negative=True) 
            
    def sellectCorrection(self,metric='nse'):
        try:self.corrections
        except:self.runCorrections()
        comid_geog_dict=self.corrections[0].comid_geog_dict
        geogs=[]
        for comid,geog in comid_geog_dict.items():
            if type(geog) is list:
                geogs.extend(geog)
            else:geogs.append(geog)
        geogs=dict.fromkeys(geogs)
        self.geog_model_select_dict={g:{'cc_idx':None,'m_name':None,metric:-np.inf} for g in geogs}
        for cc_idx,cc in enumerate(self.corrections):
            try:cc.dc_results_dict
            except:cc.setDCResultsDict()
            geog=cc.modeldict['model_geog']
            for m_name,m_data_dict in cc.dc_results_dict.items():  
                if m_name=='uncorrected' and cc_idx>0:continue #avoid repeated assessment...
                mean_acc_df=pd.DataFrame(m_data_dict).groupby(geog).mean()
                for row_ser in mean_acc_df.iterrows():
                    g=row_ser[geog]
                    this_score=row_ser[metric]
                    best_score=self.geog_model_select_dict[g][metric]
                    
                    if this_score>best_score:
                        self.geog_model_select_dict[g]={metric:this_score,'cc_idx':cc_idx,'m_name':m_name}
                    
     
                
                    
                
        
        
        
        
   
        
if __name__=="__main__":
    assert False,'pickle requires running CompareCorrect.runModelCorrection from another python file. Try run_multi_correct.py or multi_correct.ipynb'
