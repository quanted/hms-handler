import pickle,json
import numpy as np
import pandas as pd
import geopandas as gpd
import re
import os
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm,Normalize
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid.parasite_axes import SubplotHost
import mapclassify
from random import shuffle,seed
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LassoCV,Lasso,LinearRegression,LassoLarsCV,ElasticNetCV,ElasticNet,RidgeCV,Ridge
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
from traceback import format_exc



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
        neg_bool=yhat<0
        self.neg_count=(neg_bool).sum()
        self.neg_sum=yhat[neg_bool].sum()
 


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
            self.model['zero']=PipelineModel(('lin_reg',{'max_poly_deg':1,'fit_intercept':False},self.modeldict))
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
            return yhat_df.abs()
        
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
 
class ZeroModel(BaseEstimator,RegressorMixin):
    def __init__(self):
        pass
    def fit(self,x,y,w=None):
        pass
    def predict(self,x):
        n=x.shape[0]
        return np.zeros([n,])
    
    
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
            for col1 in X_.columns:#these are the dummies
                for ser in xlist:
                    i_list.append(X_.loc[:,col1]*ser) #interacting each dummy with each polynomial term
                    i_list[-1].name=f'{col1}_X_{ser.name}'
            if self.no_constant:
                xlist=i_list
            else:
                xlist.extend(i_list) #add the uninteracted-with-dummies polynomial back in.
        xlist.append(X_)#include categories without interaction.
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
        cv=RepeatedKFold(random_state=0,n_splits=n_splits,n_repeats=n_repeats)
        pipe_name=re.split('\-',model_name)[0]
        ##################################
        if pipe_name.lower() in ['lin_reg','linreg']:
            deg=specs['max_poly_deg']
            if deg>1:
                pipe=make_pipeline(
                MakePolyX(degree=deg,col_name=model_col_name,interact=True,no_constant=True),
                DropConst(),        
                #StandardScaler(),
                LinearRegression(fit_intercept=specs['fit_intercept'],normalize=False,))
                if 'poly_search' in specs and specs['poly_search']:
                    param_grid={'makepolyx__degree':np.arange(1,deg+1)}
                    self.pipe_=GridSearchCV(pipe,param_grid=param_grid,cv=cv,n_jobs=n_jobs)
                else: self.pipe_=pipe
            else:
                self.pipe_=make_pipeline(
                MakePolyX(degree=1,col_name=model_col_name,interact=True,no_constant=True),
                DropConst(), 
                #StandardScaler(),
                LinearRegression(fit_intercept=specs['fit_intercept']))
            #self.pipe_.fit(x,y)
        ##################################
        elif pipe_name.lower() in ['l1','lasso','lassocv']:
            deg=specs['max_poly_deg']
            if 'n_alphas' in specs:
                n_alphas=specs['n_alphas']
            else: n_alphas=100
            alphas=list(np.logspace(-5,1.4,n_alphas))
            if 'poly_search' in specs and specs['poly_search']:
                lasso_kwargs=dict(warm_start=True,fit_intercept=specs['fit_intercept'],selection='random',random_state=0)
                
                reg=Lasso(**lasso_kwargs,normalize=False)
            else:
                
                lasso_kwargs=dict(random_state=0,fit_intercept=specs['fit_intercept'],cv=cv,n_alphas=n_alphas)
            if 'kwargs' in specs:
                for key,val in specs['kwargs'].items():
                    lasso_kwargs[key]=val
            reg=LassoCV(**lasso_kwargs,n_jobs=n_jobs,normalize=False)
            innerpipe=make_pipeline(
                MakePolyX(degree=deg,col_name=model_col_name,interact=True,no_constant=True),
                DropConst(),
                StandardScaler(),
                ToFortranOrder(),
                reg
            )
            param_grid={'makepolyx__degree':np.arange(1,deg+1),
                       'lasso__alpha':alphas
                       }
            if 'poly_search' in specs and specs['poly_search']:
                self.pipe_=GridSearchCV(innerpipe,param_grid=param_grid,cv=cv,n_jobs=n_jobs)
            else: self.pipe_=innerpipe
            #self.pipe_.fit(x.astype(np.float32),y.astype(np.float32))
        ##################################
        elif pipe_name.lower() in ['l2','ridge','ridgecv']:
            deg=specs['max_poly_deg']
            if 'n_alphas' in specs:
                n_alphas=specs['n_alphas']
            else: n_alphas=100
            if 'alphas' in specs:
                alphas=specs['alphas']
            else:
                if 'n_alphas' in specs:
                    n_alphas=specs['n_alphas']
                else: n_alphas=100
                alphas=list(np.logspace(-5,1.4,n_alphas))
            if 'poly_search' in specs and specs['poly_search']:
                ridge_kwargs=dict(random_state=0,fit_intercept=specs['fit_intercept'])
                reg=Ridge(**ridge_kwargs,normalize=False)
            else:
                ridge_kwargs=dict(fit_intercept=specs['fit_intercept'],cv=cv,alphas=alphas)
                reg=RidgeCV(**ridge_kwargs,normalize=False)
            
            innerpipe=make_pipeline(
                MakePolyX(degree=deg,col_name=model_col_name,interact=True,no_constant=True),
                DropConst(),
                StandardScaler(),
                #ToFortranOrder(),#not for ridge b/c lin algebra solution
                reg
            )
            param_grid={'makepolyx__degree':np.arange(1,deg+1),
                       'ridge__alpha':alphas
                       }
            if 'poly_search' in specs and specs['poly_search']:
                self.pipe_=GridSearchCV(innerpipe,param_grid=param_grid,cv=cv,n_jobs=n_jobs)
            else:
                self.pipe_=innerpipe
            #self.pipe_.fit(x.astype(np.float32),y.astype(np.float32))
        ##################################
        elif pipe_name.lower() in ['lassolars','lassolarscv']:
            assert False, 'needs a refresh'
            deg=specs['max_poly_deg']
            lasso_kwargs=dict(fit_intercept=specs['fit_intercept'],cv=cv)
            self.pipe_=make_pipeline(
                MakePolyX(degree=deg,col_name=model_col_name,interact=True,no_constant=True),
                DropConst(),
                StandardScaler(),
                ToFortranOrder(),
                LassoLarsCV(**lasso_kwargs,n_jobs=n_jobs,normalize=False))
        ##################################
        elif pipe_name.lower() in ['elastic net','elastic_net','elasticnet']:
            deg=specs['max_poly_deg']
            if 'l1_ratio' in specs:
                l1_ratio=specs['l1_ratio']
            else:
                l1_ratio=list(1-np.logspace(-2,-.03,5))
            '''if 'alpha' in specs:
                alphas=specs['alpha']
                
            else:'''
            if 'n_alpha' in specs:
                n_alpha=specs['n_alpha']
            else:n_alpha=10
                
            
            if 'poly_search' in specs and specs['poly_search']:
                enet_kwargs=dict(fit_intercept=specs['fit_intercept'],warm_start=True,selection='random',random_state=0)
                if 'kwargs' in specs:
                    for key,val in specs['kwargs'].items():
                        enet_kwargs[key]=val
                reg=ElasticNet(**enet_kwargs,normalize=False)
                alphas=list(np.logspace(-5,1.4,n_alpha))
                param_grid={'makepolyx__degree':np.arange(1,deg+1),
                       'elasticnet__l1_ratio':l1_ratio,
                       'elasticnet__alpha':alphas,
                       }
            else:
                enet_kwargs=dict(fit_intercept=specs['fit_intercept'],cv=cv,l1_ratio=l1_ratio,n_alphas=n_alpha)
                if 'kwargs' in specs:
                    for key,val in specs['kwargs'].items():
                        enet_kwargs[key]=val
                reg=ElasticNetCV(**enet_kwargs,n_jobs=n_jobs,normalize=False,)
            innerpipe=make_pipeline(
                MakePolyX(degree=deg,col_name=model_col_name,interact=True,no_constant=True),
                DropConst(),
                StandardScaler(),
                ToFortranOrder(),
                reg)
            if 'poly_search' in specs and specs['poly_search']:
                self.pipe_=GridSearchCV(innerpipe,param_grid=param_grid,cv=cv,n_jobs=n_jobs)
            else:
                self.pipe_=innerpipe
        ##################################
        elif pipe_name.lower()=='gbr':
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
        ##################################
        elif pipe_name.lower() in ['stackingregressor','stacking_regressor']:
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
        ##################################        
        else:
            assert False,f'pipe_name:{pipe_name} not recognized. model_name:{model_name}'
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
        self.runoff_n=n
        self.test_results={} # will contain {m_name:{test_stats:...,yhat_test:...}}
        self.val_results={}
        
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
        val_share=self.modeldict['val_share']
        test_share=1-val_share-train_share
        assert test_share>0,f'problem with splits. test not post. test_share:{test_share},val_share:{val_share},test_share:{test_share}'
        n=df.shape[0]
        train_split_idx=int(train_share*n)
        test_split_idx=int((train_share+test_share)*n)
        y_df=df.loc[:,self.sources['observed']]
        x_df=df.drop(self.sources['observed'],axis=1,inplace=False)
    
        self.x_train=x_df.iloc[:train_split_idx]
        self.y_train=y_df.iloc[:train_split_idx]
        self.x_test=x_df.iloc[train_split_idx:test_split_idx]
        self.y_test=y_df.iloc[train_split_idx:test_split_idx]
        self.x_val=x_df.iloc[test_split_idx:]
        self.y_val=y_df.iloc[test_split_idx:]
 

            
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
                #self.logger.info(f'Before applying boolean, m_name:{self.m_name} has self.X.shape:{self.X.shape}')
                self.X=self.X.loc[(self.train_comids,slice(None)),:]
                self.y=self.y.loc[(self.train_comids,slice(None))]
                #self.logger.info(f'after applying boolean, m_name:{self.m_name} has self.X.shape:{self.X.shape}')
            else:
                pass
                #self.logger.info(f'm_name:{self.m_name} has train_comids:{self.train_comids}')
            data_dict={'x':self.X,'y':self.y}
            args=[data_dict,(self.m_name,self.specs,self.modeldict)]
            name=os.path.join(self.results_folder,f'pipe-{joblib.hash(args)}.pkl')
            if os.path.exists(name):
                if self.return_save_string:
                    return name
                try:
                    with open(name,'rb') as f:
                        model=pickle.load(f)
                    #self.model_results[m_name]=model
                    self.logger.info(f'succesful load from disk for {self.m_name} from {name}')
                    return model
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
            if comid_data_obj.runoff_n>100:
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
        for obj in self.comid_modeling_objects:
            obj.x_val_float=self.makeDummies(obj.x_val,fit=False)
    
    def runModel(self): #singular!
        model_scale=self.modeldict['model_scale']
        if model_scale=='conus':
            proc_count=1
        else:
            proc_count=4
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
                
    def addFlatZeroModels(self):     
        new_model_results={}
        for m_name,m_list in self.model_results.items():
            new_m_name=m_name+'-flat0'
            new_model_results[new_m_name]=[]
            for m in m_list: 
                path=m
                new_path=m[:-4]+'-flat0.pkl'
                if not os.path.exists(new_path):
                    self.logger.info(f'making flat0 model form m_name:{m_name}')
                    with open(path,'rb') as f:
                        m=pickle.load(f)
                    #new_m=deepcopy(m)
                    m.model['zero']=ZeroModel()
                    with open(new_path,'wb') as f:
                        pickle.dump(m,f)
                    new_model_results[new_m_name].append(new_path)
                        
        self.model_results={**self.model_results,**new_model_results}
                
     
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
            
                                                                 
    def runTestData(self):
        m_name_list=[]
        for m_name,model_list in self.model_results.items():
            m_name_list.append(m_name)
            self.logger.info(f'building test data/stats for {m_name}')
            if not self.modeldict['cross_validate']: 
                if not type(model_list) is list:model_list=[model_list]
        data_dict={}
        for obj in self.comid_modeling_objects:
            data_dict[obj.comid]={
                'x_test':obj.x_test_float,'y_test':obj.y_test,
                'x_val':obj.x_val_float,'y_val':obj.y_val}
            for m_name in m_name_list: obj.test_results[m_name]=[]
            for m_name in m_name_list: obj.val_results[m_name]=[]    
        args_list=[]
        for m_name,model_list in self.model_results.items():
            args_list.append((m_name,model_list,data_dict,self.modeldict))
        proc_count=min(6,len(args_list))
        outlist=MpHelper().runAsMultiProc(TestRunner,args_list,proc_count=proc_count)
        for obj in self.comid_modeling_objects:
            for testrunner in outlist:
                m_name=testrunner.m_name
                comid_results_list=testrunner.results_dict[obj.comid]
                for c_result_dict in comid_results_list:
                    obj.test_results[m_name].append(
                        {'yhat_test':c_result_dict['yhat_test'],
                         'test_stats':c_result_dict['test_stats']})
                    obj.val_results[m_name].append(
                        {'yhat_val':c_result_dict['yhat_val'],
                         'val_stats':c_result_dict['val_stats']})
                
        for obj in self.comid_modeling_objects:
            uncorr_yhat=obj.x_test.loc[:,self.modeldict['sources']['modeled']]
            obj.test_results['uncorrected']=[{
                'test_stats':SeriesCompare(obj.y_test.values,uncorr_yhat.values),
                'yhat_test':uncorr_yhat}]
            uncorr_yhat=obj.x_val.loc[:,self.modeldict['sources']['modeled']]
            obj.val_results['uncorrected']=[{
                'val_stats':SeriesCompare(obj.y_val.to_numpy(),uncorr_yhat.to_numpy()),
                'yhat_val':uncorr_yhat}]
            
            
   


class TestRunner(myLogger):
    def __init__(self,m_name,model_list,data_dict,modeldict):
        self.m_name=m_name
        self.model_list=model_list
        self.data_dict=data_dict
        self.modeldict=modeldict
        
        
    def run(self):
        myLogger.__init__(self)
        results_dict={comid:[] for comid in self.data_dict.keys()}
        for model in self.model_list:
            if type(model) is str:
                with open(model,'rb') as f:
                    model=pickle.load(f)
            for comid,c_data in self.data_dict.items(): 
                if not self.modeldict['cross_validate'] or not comid in model.train_comids: #if cross_validate
                    x_test=c_data['x_test']
                    y_test=c_data['y_test']
                    #    , then skip comid if it was in training data!
                    yhat_test,test_stats=model.get_prediction_and_stats(x_test,y_test)
                    if type(test_stats) is str:
                            self.logger.Error(f'test data from comid:{comid}: {(yhat_test,test_stats)}')
                    else:pass#self.logger.info(f'test data shape and stats from comid:{comid}: {(yhat_test.shape,test_stats)}')
                    #######repeat for val data
                    x_val=c_data['x_val']
                    y_val=c_data['y_val']
                    #    , then skip comid if it was in training data!
                    yhat_val,val_stats=model.get_prediction_and_stats(x_val,y_val)
                    if type(val_stats) is str:
                            self.logger.Error(f'val data from comid:{comid}: {(yhat_val,val_stats)}')
                    else:pass#self.logger.info(f'val data shape and stats from comid:{comid}: {(yhat_val.shape,val_stats)}')
                    
                    results_dict[comid].append({
                        'yhat_test':yhat_test,'test_stats':test_stats,
                        'yhat_val':yhat_val,'val_stats':val_stats})
        self.results_dict=results_dict

    
        

class CompareCorrect(myLogger):
    def __init__(self,model_specs=None,modeldict=None):
        myLogger.__init__(self,'comparecorrect.log')
        self.dc_list=[]
        if not os.path.exists('results'):
            os.mkdir('results')
        if modeldict is None:
            max_deg=5
            self.modeldict={
                'cross_validate':{'n_reps':3,'strategy':'leave_one_member_out'},#False,#
                'model_geog':'section',
                'sources':{'observed':'nldas','modeled':'cn'}, #[observed,modeled]
                'filter':'nonzero',#'none',#'nonzero'
                'train_share':0.50,
                'val_share':0.25,#test_share is 1-train_share-val_share. validation is after model selection
                'split_order':'chronological',#'random'
                'model_scale':'division',#'division',#'comid'
                'model_specs':{f'lin_reg-{i}':{
                    'max_poly_deg':i,
                    'poly_search':False,
                    'fit_intercept':False
                    } for i in range(1,max_deg+1)}
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
        self.physio_path='ecoregions/physio.dbf'
        if not os.path.exists(self.physio_path):
            print(f'cannot locate {self.physio_path}, the physiographic boundary shapefile. download it and unzip it in a folder called "ecoregions".')
        self.states_path='geo_data/states/cb_2017_us_state_500k.dbf'
        self.comid_geog_dict=self.makeComidGeogDict()
        
    def run(self):
        self.runBigModel()
    
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
            
    def setDCTestDict(self):
        data_dict={}
        geog=self.modeldict['model_geog']
        for dc in self.dc_list:
            m_names= list(dc.model_results.keys())+['uncorrected'] #adding b/c in test_results, but not model_results
            for m_name in m_names:
                if not m_name in data_dict:
                    data_dict[m_name]={'nse':[],'pearson':[],geog:[]}#,'neg_count':[],'neg_sum'=[]
                    
                for obj in dc.comid_modeling_objects:
                    for result_dict in obj.test_results[m_name]:
                        data_dict[m_name]['nse'].append(result_dict['test_stats'].nse)
                        data_dict[m_name]['pearson'].append(result_dict['test_stats'].pearsons)
                        #data_dict[m_name]['neg_count'].append(result_dict['test_stats'].neg_count)
                        #data_dict[m_name]['neg_sum'].append(result_dict['test_stats'].neg_sum)
                        data_dict[m_name][geog].append(self.comid_geog_dict[obj.comid][geog])
        self.dc_test_dict=data_dict
        
        for result_dict in obj.val_results[m_name]:
                        data_dict[m_name]['nse'].append(result_dict['val_stats'].nse)
                        data_dict[m_name]['pearson'].append(result_dict['val_stats'].pearsons)
                        #data_dict[m_name]['neg_count'].append(result_dict['val_stats'].neg_count)
                        #data_dict[m_name]['neg_sum'].append(result_dict['val_stats'].neg_sum)
                        data_dict[m_name][geog].append(self.comid_geog_dict[obj.comid][geog])
        self.dc_val_dict=data_dict
    
    def getFancyMName(self,mname):
        name_part_list=re.split('-',mname)
        if name_part_list[0].lower()=='lin_reg':
            poly_deg=name_part_list[1]
            if int(poly_deg)<7:
                ordinal_dict={'1':'First','2':'Second','3':'Third','4':'Fourth','5':'Fifth',
                             '6':'Sixth'}
                poly_ordinal=ordinal_dict[poly_deg]
            else:
                poly_ordinal=f'poly_deg{th}'
            fancy_name=f'{poly_ordinal} Order OLS Regression'
        else:
            fancy_name=mname
            
        return fancy_name
    
    def plotGeoTestData(self,plot_negative=True,use_val_data=True):
        try: self.eco_geog
        except: self.setEcoGeog()
        try: self.dc_test_dict
        except: self.setDCTestDict()
        geog=self.modeldict['model_geog']
        if use_val_data:
            dc_data_dict=self.dc_val_dict
        else:
            dc_data_dict=self.dc_test_dict
        for m_name,m_data_dict in dc_data_dict.items():            
            mean_acc_df=pd.DataFrame(m_data_dict).groupby(geog).mean()
            geog_acc_df=self.eco_geog.merge(mean_acc_df,on=geog,how='left')
            plt.rcParams['hatch.linewidth'] = 0.1
            plt.rcParams['axes.facecolor'] = 'lightgrey'
            fig=plt.figure(dpi=300,figsize=[9,3.7])
            fig.patch.set_facecolor('w')
            fancy_m_name=self.getFancyMName(m_name)
            if use_val_data:
                fig.suptitle(f'Validation Scores for {fancy_m_name}')
            else:
                fig.suptitle(f'Test Scores for {fancy_m_name}')
            for i,metric in enumerate(['nse','pearson']):
                
                
                ax=fig.add_subplot(1,2,i+1)
                ax.set_title(f'{metric.upper()}')
                #e=self.eco_geog.plot(color='darkgrey',ax=ax,hatch='/////',zorder=0)
                #pos_geog_acc_df=geog_acc_df[geog_acc_df.loc[:,metric]>0]
                #pos_geog_acc_df.plot(column=metric,ax=ax,cmap='plasma',legend=True,)
                if plot_negative:
                    norm = TwoSlopeNorm(vmin=-1,vcenter=0, vmax=1)
                    cmap='RdBu'#'brg'##'plasma'
                    cbar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
                    #self.geog_acc_df=geog_acc_df
                    g=geog_acc_df.plot(column=metric,ax=ax, cmap=cmap,zorder=1, norm=norm,legend=True,missing_kwds={
                    "color": "lightgrey",
                    #"edgecolor": "red",
                    "hatch": "xxxxxxxxx",
                    "label": "Missing values",},legend_kwds={'orientation': "horizontal"})
                    #fig.colorbar(cbar, ax=ax)
                else:
                    cmap='plasma'
                    norm=Normalize(vmin=0,vmax=1)
                    cbar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
                    neg_geog_acc_df=geog_acc_df[(geog_acc_df.loc[:,metric]<=0)]
                    ng=neg_geog_acc_df.plot(ax=ax,color='lightgrey',hatch='oooooo',zorder=2,label=f'non-positive {metric.upper()}',legend=True,legend_kwds={'orientation': "horizontal"})
                    pos_geog_acc_df=geog_acc_df[~(geog_acc_df.loc[:,metric]<=0)] #includes nans
                    pg=pos_geog_acc_df.plot(column=metric,ax=ax,cmap='plasma',zorder=1,norm=norm,label=f'{metric}',legend=True,missing_kwds={
                    "color": "lightgrey",
                    #"edgecolor": "red",
                    "hatch": "xxxxxxxxx",
                    "label": "Missing values",},legend_kwds={'orientation': "horizontal"})
                    #fig.colorbar(cbar,ax=ax)
                self.add_states(ax)
                ax.set_xticklabels([])
                ax.set_xticks([])
                ax.set_yticklabels([])
                ax.set_yticks([])
                if i==0:
                    handles, labels = ax.get_legend_handles_labels()
                    handles=[
                        *handles,
                        mpatches.Patch(
                            hatch='xxxxxxxxx',facecolor='lightgrey',
                            label='<100 CN event days'),
                        mpatches.Patch(
                            hatch='oooooo',facecolor='lightgrey', 
                            label='negative score'),
                    ]
                    ax.legend(handles=handles,fontsize=6,bbox_to_anchor=(0.3,0.15),frameon=False)
            fig_name=f'{self.modeldict["model_scale"]}_{m_name}.tif'
            if not plot_negative:
                fig_name='pos-score_'+fig_name
            if type(self.modeldict['cross_validate']) is dict:
                fig_name='cv_'+fig_name
            fig.tight_layout()
            plt.show()
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
            print('running data collection')
        if len(comidlist)<len(self.comidlist):
            comid_geog_dict={comid:geog for comid,geog in self.comid_geog_dict.items() if comid in comidlist}
        else:
            comid_geog_dict=self.comid_geog_dict
        dc=DataCollection(comidlist,modeldict,self.comid_geog_dict)
        dc.build()
        dc.runModel()
        dc.addFlatZeroModels()
        dc.runTestData()
        with open(save_path,'wb') as f:
            pickle.dump(dc,f)
        return dc
    
    
        
    def makeComidGeogDict(self):
        geogs=self.geog_names
        comid_geog_dict={}
        
        for comid in self.comidlist:
            comid_geog_dict[comid]={}
            for geog in geogs:
                g=self.comid_physio.loc[comid,geog]
                if pd.isnull(g):
                    bigger_geog=self.geog_names[self.geog_names.index(geog)-1]
                    g=self.comid_physio.loc[comid,bigger_geog]
                comid_geog_dict[comid][geog]=g
        return comid_geog_dict
    
    
        
        
    def add_states(self,ax):
        try: self.eco_clip_states
        except:
            states=gpd.read_file(self.states_path)
            eco_d=self.eco.copy()
            eco_d['dissolve_field']=1
            eco_d.dissolve(by='dissolve_field')
            self.eco_clip_states=gpd.clip(states,eco_d)
        self.eco_clip_states.boundary.plot(linewidth=0.3,ax=ax,color=None,edgecolor='k')                
    
class MultiCorrectionTool(myLogger):
    ###currently supports picking from multiple model_specs (e.g., lin-reg, gbr,...) for each geog in the model conditioning level (model_geog)
    ###future work may pick the best from among other items in modeldict, e.g., model_scale
    def __init__(self,modeldict=None,model_specs=None,plot=False):
        myLogger.__init__(self,'multi-correction-tool.log')
        self.plot=plot
        self.selection_metric='nse'
        self.physio_path='ecoregions/physio.dbf'
        self.states_path='geo_data/states/cb_2017_us_state_500k.dbf'
        self.geog_names=['division','province','section']
        if modeldict is None:
            if model_specs is None:
                assert False,'provide a list of models'
            self.modeldict={
                'cross_validate':{'n_reps':3,'strategy':'leave_one_member_out'},#False,#
                'model_geog':'section',
                'sources':{'observed':'nldas','modeled':'cn'}, 
                'filter':'nonzero',#'none',
                'train_share':0.50,
                'val_share':0.25,#test_share is 1-train_share-val_share. validation is after model selection
                'split_order':'chronological',#'random'
                'model_scale':'division',#'division',#'comid'
                'model_specs':None}
        else:
            self.modeldict=modeldict
        
        self.model_specs=model_specs
        assert type(self.modeldict['cross_validate']) is dict,f'expecting a dict for cross_validate but got:{self.modeldict["cross_validate"]}'
        self.model_geog=self.modeldict['model_geog']
        self.model_scale=self.modeldict['model_scale']
        self.hash_id=joblib.hash((self.modeldict,self.model_specs))
        self.mct_results_folder=os.path.join('multi_correction_results',f'mct-{self.hash_id}')
        if not os.path.exists(self.mct_results_folder):
            os.makedirs(self.mct_results_folder)
        
    def runCorrections(self,load=True):
        name=os.path.join(self.mct_results_folder,f'corrections-{self.hash_id}.pkl')
        if os.path.exists(name) and load:
            try:
                with open(name,'rb') as f:
                    self.corrections=pickle.load(f)
                    return
            except:
                self.logger.exception(f'error loading from {name}')
        #self.corrections=[]
        model_dict_list=[]
        for m_spec_dict in self.model_specs:
            new_model_dict=self.modeldict.copy()
            new_model_dict['model_specs']=m_spec_dict
            model_dict_list.append(new_model_dict)
        kwargs_list=[{'modeldict':md} for md in model_dict_list]
        args_list=[[] for _ in range(len(model_dict_list))]
        self.corrections=MpHelper().runAsMultiProc(CompareCorrect,args_list,kwargs_list=kwargs_list,proc_count=1)
        if self.plot:
            for cc in self.corrections:
                #cc.runBigModel()
                print(cc.modeldict['results_folder'])
                cc.plotGeoTestData(plot_negative=False)
                cc.plotGeoTestData(plot_negative=True) 
        with open(name,'wb') as f:
            pickle.dump(self.corrections,f)
            
            
    def buildCorrectionResultsDF(self):
        name=os.path.join(self.mct_results_folder,f'correction-results-df-{self.hash_id}.pkl')
        if os.path.exists(name):
            try:
                with open(name,'rb') as f:
                    (self.correction_results_df,
                     self.correction_results_df_V,
                     self.comid_geog_dict)=pickle.load(f)
                return
            except:
                print(traceback_exc())
                print('rebuilding correctionresultsDF')
        try:self.corrections
        except:self.runCorrections()
        metric=self.selection_metric
        comid_geog_dict=self.corrections[0].comid_geog_dict
        self.comid_geog_dict=comid_geog_dict
        geogs=[]
        for comid,geogdict in comid_geog_dict.items():
            geogs.append(geogdict[self.model_geog])
        geogs=dict.fromkeys(geogs)
        mean_cv_acc_df_list=[]
        mean_cv_acc_df_list_V=[]
        for cc_idx,cc in enumerate(self.corrections):
            try:cc.dc_test_dict
            except:cc.setDCTestDict()
            for m_name,m_data_dict in cc.dc_test_dict.items():  
                if m_name=='uncorrected' and cc_idx>0:continue #avoid repeated assessment...
                #average over comids in each model_geog
                mean_acc_df=pd.DataFrame(m_data_dict).groupby(self.model_geog).mean()#.reset_index(name=self.model_geog)
                #mean_acc_df.loc[:,'estimator']=m_name
                tups=[(m_name,cc_idx,g) for g in mean_acc_df.index]#g for model_geog value
                midx=pd.MultiIndex.from_tuples(tups,names=['estimator','collection_idx',self.model_geog])
                mean_acc_df.index=midx
                #mean_acc_df.loc[:,'collection_idx']=cc_idx
                mean_cv_acc_df_list.append(mean_acc_df)
            ############VALIDATE############    
            for m_name,m_data_dict in cc.dc_val_dict.items():  
                if m_name=='uncorrected' and cc_idx>0:continue #avoid repeated assessment...
                #average over comids in each model_geog
                mean_acc_df=pd.DataFrame(m_data_dict).groupby(self.model_geog).mean()#.reset_index(name=self.model_geog)
                #mean_acc_df.loc[:,'estimator']=m_name
                tups=[(m_name,cc_idx,g) for g in mean_acc_df.index]#g for model_geog value
                midx=pd.MultiIndex.from_tuples(tups,names=['estimator','collection_idx',self.model_geog])
                mean_acc_df.index=midx
                #mean_acc_df.loc[:,'collection_idx']=cc_idx
                mean_cv_acc_df_list_V.append(mean_acc_df)
                
                
        self.correction_results_df=pd.concat(mean_cv_acc_df_list,axis=0)
        self.correction_results_df_V=pd.concat(mean_cv_acc_df_list_V,axis=0)
        idx=self.correction_results_df.index
        idx_V=self.correction_results_df_V.index
        idx_n=len(idx)
        assert idx.equal_levels(idx_V),f'validate version of self.correction_results_df has different index !!!'
        try:
            with open(name,'wb') as f:
                pickle.dump((self.correction_results_df,self.correction_results_df_V,self.comid_geog_dict),f)
        except:
            print(traceback_exc())
            print('error saving correctionresultsdf')             
    
            
       
    def selectCorrections(self):
        try: self.correction_results_df
        except: self.buildCorrectionResultsDF()
        metric=self.selection_metric
        model_geog=self.model_geog
        comid_geog_dict=self.comid_geog_dict
        best_idx=self.correction_results_df.groupby(model_geog)[metric].idxmax()#get index of best metric in each group
        best_model_df=self.correction_results_df.loc[best_idx]
        best_model_df_V=self.correction_results_df_V.loc[best_idx] #select value from val that is best in test
        self.best_model_df=best_model_df
        self.best_model_df_V=best_model_df_V
        
        geogs=[]
        for comid,geogdict in comid_geog_dict.items():
            geogs.append(geogdict[self.model_geog])
        geogs=dict.fromkeys(geogs)
        self.geog_model_select_dict={}#{g:{'cc_idx':None,'m_name':'None',metric:-np.inf} for g in geogs}
        for midx_tup,row_ser in best_model_df.iterrows():
            self.geog_model_select_dict[midx_tup[2]]={
                metric:row_ser[metric],'cc_idx':midx_tup[1],'estimator':midx_tup[0]}
        self.geog_model_select_dict_V={}#{g:{'cc_idx':None,'m_name':'None',metric:-np.inf} for g in geogs}
        for midx_tup,row_ser in best_model_df_V.iterrows():
            self.geog_model_select_dict_V[midx_tup[2]]={
                metric:row_ser[metric],'cc_idx':midx_tup[1],'estimator':midx_tup[0]}
        
        
    def setCorrectionSelectionAccuracy(self,):
        try: self.geog_model_select_dict
        except: self.selectCorrections()
        geogs=[];accuracies=[]
        accuracies_V=[]
        for geog,select_dict in self.geog_model_select_dict.items():
            geogs.append(geog)
            accuracies.append(select_dict[self.selection_metric])
            select_dict_V=self.geog_model_select_dict_V[geog]
            accuracies_V.append(select_dict_V[self.selection_metric])
            self.correction_selection_accuracy=pd.DataFrame({
                self.selection_metric:accuracies,self.model_geog:geogs})
            self.correction_selection_accuracy_V=pd.DataFrame({
                self.selection_metric:accuracies_V,self.model_geog:geogs})
        
        
        
        
    def setSortOrder(self,):
        try:self.correction_selection_accuracy
        except:setCorrectionSelectionAccuracy()
        model_geogs=self.correction_selection_accuracy.loc[:,self.model_geog]
        midx=self.expandGeogsToMultiIndex(model_geogs)
        full_g_scores=self.correction_selection_accuracy.loc[:,self.selection_metric].copy()
        full_g_scores.index=midx
        scale_sort=full_g_scores.groupby(self.model_scale,).mean().sort_values(ascending=False).index.to_list()
        double_sort=full_g_scores.sort_values(ascending=False).loc[scale_sort,:]
        self.double_sort=double_sort
        self.double_sort_index=double_sort.index
        self.double_sort_model_geog_index=double_sort.index.get_level_values(self.model_geog)
        self.div_id_dict={div:i for i,div in enumerate(self.double_sort_index.get_level_values('division').unique())}
        
        model_geogs=self.correction_selection_accuracy_V.loc[:,self.model_geog]
        midx=self.expandGeogsToMultiIndex(model_geogs)
        full_g_scores=self.correction_selection_accuracy_V.loc[:,self.selection_metric].copy()
        full_g_scores.index=midx
        scale_sort=full_g_scores.groupby(self.model_scale,).mean().sort_values(ascending=False).index.to_list()
        double_sort=full_g_scores.sort_values(ascending=False).loc[scale_sort,:]
        self.double_sort_V=double_sort
        self.double_sort_index_V=double_sort.index
        self.double_sort_model_geog_index_V=double_sort.index.get_level_values(self.model_geog)
        self.div_id_dict_V={div:i for i,div in enumerate(self.double_sort_index_V.get_level_values('division').unique())}
        
        
    
    def plotCorrectionRunoffComparison(self,sort=False,use_val_data=True,split_zero=True,time_range=None):
        s_metric=self.selection_metric
        if use_val_data:
            div_top_idx=self.double_sort_index_V.to_series().groupby(
                self.model_scale).head(1).index.get_level_values(self.model_geog)
        else:
            div_top_idx=self.double_sort_index.to_series().groupby(
                self.model_scale).head(1).index.get_level_values(self.model_geog)
        scale_best_modelg=div_top_idx.to_list()
        if use_val_data:
            val_str='-V'
        else:
            val_str=''
        name=os.path.join(self.mct_results_folder,f'best-model-geog-runoff{val_str}.pkl')
        loaded=False
        if os.path.exists(name):
            try:
                with open(name,'rb') as f:
                    best_modelg_runoff_dict=pickle.load(f)
                loaded=True
            except:pass
        
        if not loaded:
            try: self.corrections
            except: self.runCorrections()
            try: self.geog_model_select_dict
            except: self.selectCorrections()
            try: self.double_sort_index
            except: self.setSortOrder()
            
            best_modelg_runoff_dict={mg:{} for mg in scale_best_modelg}
            best_comid_runoff_dict={mg:{} for mg in scale_best_modelg}
            best_modelg_comidlist_dict=self.buildComidListDict(scale_best_modelg)
            for mg in scale_best_modelg:
                dc_list=self.corrections[self.geog_model_select_dict[mg]['cc_idx']].dc_list#cc_idx invarant to _V or not
                m_name=self.geog_model_select_dict[mg]['estimator']
                for dc in dc_list:
                    #cv_models=dc.model_results['m_name']

                    for obj in dc.comid_modeling_objects:
                        o_mg=dc.comid_geog_dict[obj.comid][self.model_geog]
                        if not o_mg==mg:# in scale_best_modelg:
                            continue
                        try:
                            if use_val_data:
                                best_comid_runoff_dict[mg][obj.comid]={
                                    'uncorrected':obj.x_val.loc[:,self.modeldict['sources']['modeled']],
                                    self.modeldict['sources']['observed']:obj.y_val,
                                    'corrected':pd.concat([
                                        result_dict['yhat_val'] for result_dict in
                                        obj.val_results[m_name]],axis=1).mean(axis=1)
                                    } 
                            else:
                            
                                best_comid_runoff_dict[mg][obj.comid]={
                                    'uncorrected':obj.x_test.loc[:,self.modeldict['sources']['modeled']],
                                    self.modeldict['sources']['observed']:obj.y_test,
                                    'corrected':pd.concat([
                                        result_dict['yhat_test'] for result_dict in
                                        obj.test_results[m_name]],axis=1).mean(axis=1)
                                    } #concatenating cv yhats for each comid 
                        except ValueError: 
                            print(f'ValueError for comid:{obj.comid}')
                        except:
                            print(f'comid:{obj.comid}',format_exc())
            #average over comids in each model_geog
            for mg in best_comid_runoff_dict:
                mg_dict={'corrected':[],'uncorrected':[],self.modeldict['sources']['observed']:[]}
                mg_df_dict={}
                for comid,runoff_dict in best_comid_runoff_dict[mg].items():
                    for g,val in runoff_dict.items():
                        mg_dict[g].append(val)
                for key,val in mg_dict.items():
                    mg_df_dict[key]=pd.concat(val,axis=0).mean(axis=0,level='date')
                best_modelg_runoff_dict[mg]=mg_df_dict
            with open(name,'wb') as f:
                pickle.dump(best_modelg_runoff_dict,f)
               
           
        base_name='runoff-comparison'
        event_dict={}
        if split_zero:
            best_modelg_runoff_dict_nonzero={};best_modelg_runoff_dict_zero={}
        for g,g_dict in best_modelg_runoff_dict.items():
            non_z_bool=g_dict['uncorrected']>0
            event_start_stop=pd.Series(
                data=non_z_bool.iloc[1:].to_numpy(dtype=int)-
                non_z_bool.iloc[:-1].to_numpy(dtype=int),
                index=non_z_bool.index.to_list()[1:])
            #non_z_bool.diff().astype('bool')#.to_numpy() # XOR logic in diff
            #print('event_start_stop',event_start_stop)
            stops=event_start_stop==-1
            start_dates=event_start_stop[event_start_stop==1].index.to_list()
            stop_dates=event_start_stop[event_start_stop==-1].index.to_list()
            #event_bool.iloc[0]=False #instead of nan
            #print('event_bool',event_bool)
            #event_bool.index=non_z_bool.index
            startstop_dict={'start':start_dates,'stop':stop_dates,'stops':stops}
            #print('startstop_dict:',startstop_dict)
            #events=#get_level_values('date').to_list()#
            #print('events',events)
            event_dict[g]=startstop_dict
            if split_zero:
                best_modelg_runoff_dict_nonzero[g]={}
                best_modelg_runoff_dict_zero[g]={}
                for r_name,r_ser in g_dict.items():
                    try:
                        
                    
                        best_modelg_runoff_dict_nonzero[g][r_name]=r_ser[non_z_bool]
                        #r_ser.iloc[1:][non_z_bool.iloc[1:]]|stops]
                        best_modelg_runoff_dict_zero[g][r_name]=r_ser[~non_z_bool]
                        
                    except:
                        print(f'error for r_name:{r_name}, r_ser:{r_ser}','traceback:')
                        print(traceback_exc())
        if split_zero:     
            self.makeRunoffPlot(
                best_modelg_runoff_dict_zero,base_name+'-zero',
                scale_best_modelg,sort,use_val_data,time_range,event_dict)
            self.makeRunoffPlot(
                best_modelg_runoff_dict_nonzero,base_name+'-nonzero',
                scale_best_modelg,sort,use_val_data,time_range,event_dict)
        else:
            self.makeRunoffPlot(
                best_modelg_runoff_dict,base_name,
                scale_best_modelg,sort,use_val_data,time_range,event_dict)
        
    def makeRunoffPlot(self,best_modelg_runoff_dict,name,scale_best_modelg,sort,use_val_data,time_range,event_dict):
        if type(time_range) is str:
            if time_range=='last year':
                time_range=slice(-365,None)
            else:assert False,f'unexpected time_range:{time_range}'
        else:
            assert time_range is None or type(time_range) in [list,slice,np.ndarray],f'unexpected type for time_range:{type(time_range)}'
        fig=plt.figure(dpi=200,figsize=[10,11])#16,12
        if use_val_data:
            labeltext='Validation Data'
            if not time_range is None:
                if type(time_range) is str:
                    labeltext+=' {time_range}'
                #else: assert False, f'unexpected type for time_range: {type(time_range)}'
            
            fig.text(0.5, 0.95,labeltext , ha='center', va='center')
        elif time_range is None:
            fig.text(0.5, 0.95, 'Test Data', ha='center', va='center')
        fig.text(0.95, 0.5, 'Natural Logarithm of 1 + Runoff', ha='center', va='center', rotation='vertical',fontsize=12)
        fig.text(0.5,0.05,'Time (days)',ha='center',va='center',fontsize=12)
        
        
        if sort or name[-4:]=='zero':
            fig.subplots_adjust(hspace=0.2)
        else:
            fig.subplots_adjust(hspace=None)
        fig.patch.set_facecolor('w')
        if sort:
            sort_str=f" Sorted by {self.modeldict['sources']['observed'].upper()}"
        else:
            sort_str=''
        if name[-4:]=='zero':
            if name[-7:]=='nonzero':
                split_zero='nonzero'
                fig.suptitle(f'Runoff{sort_str} For Uncorrected NonZero Runoff Days',fontsize=12)
            else:
                split_zero='zero'
                fig.suptitle(f'Runoff{sort_str} For Uncorrected Zero Runoff Days',fontsize=12)
        else:
            split_zero=False
            fig.suptitle(f'Daily Runoff{sort_str}',fontsize=12)
        colors = plt.get_cmap('tab10')(np.arange(10))
        linestyles=['-', '--', '-.', ':']
        d_n=len(best_modelg_runoff_dict)
        ax_list=[]
        #back_ax=ax = fig.add_subplot(111) 
        
        for i,(mg,runoffdict) in enumerate(best_modelg_runoff_dict.items()):
            if i==0:
                ax=fig.add_subplot(d_n,1,i+1)

            else:
                ax=fig.add_subplot(d_n,1,i+1)#,sharex=ax_list[0])
            
            ax_list.append(ax)
            if sort:
                sort_key=self.modeldict['sources']['observed']
                if time_range:
                    np_sort_idx=runoffdict[sort_key][time_range].sort_index(inplace=False).to_numpy().argsort()
                else:
                    np_sort_idx=runoffdict[sort_key].sort_index(inplace=False).to_numpy().argsort()
            
            df_max=0
            for k,(key,df) in enumerate(runoffdict.items()):
                this_max=np.log(1+df.max())
                if this_max>df_max:df_max=this_max
                df.sort_index(inplace=True)
                x=df.index.to_numpy().ravel()
                y=np.log(df.to_numpy().ravel()+1)
                if time_range:
                    x=x[time_range]
                    y=y[time_range]
                if sort:
                    x=np.arange(x.shape[0])
                    y=y[np_sort_idx]
                else:
                    if split_zero:
                        x=np.arange(x.shape[0])#x.astype(str) #to prevent automatic time gap filling and interpolation
                #self.xlist.append(x)
                #self.ylist.append(y)
                #ax.grid('on', linestyle='--',alpha=0.7,color='w')
                if key==self.modeldict['sources']['observed']:
                    """ax_list[-1].plot(
                        x,y,color=colors[k],
                        alpha=1,
                        linewidth=2,zorder=0)"""
                    ax_list[-1].plot(
                        x,y,color=colors[k],
                        alpha=.5,
                        linewidth=3,zorder=0)
                    ax_list[-1].scatter(
                        x,y,color=colors[k],
                        alpha=.7,s=2,zorder=2)
                    ax_list[-1].plot(#just for the legend
                        [],[],'o-',label=key,color=colors[k],
                        alpha=.5,linewidth=3,zorder=0)
                elif key=='uncorrected':#self.modeldict['sources']['modeled']:
                    ax_list[-1].plot(
                        x,y,color=colors[k],
                        alpha=.8,linestyle=linestyles[k],
                        linewidth=.4,zorder=3)
                    ax_list[-1].plot(#just for the legend
                        [],[],'o-',label=key,color=colors[k],
                        alpha=.8,linestyle=linestyles[k],
                        linewidth=.4,zorder=3)
                    ax_list[-1].scatter(
                        x,y,color=colors[k],
                        alpha=.7,s=1.3,zorder=3)
                else:
                    ax_list[-1].plot(
                        x,y,color=colors[k],
                        alpha=.9,linestyle=linestyles[k],
                        linewidth=1,zorder=1)
                    ax_list[-1].plot(#just for the legend
                        [],[],'*-',label=key,color=colors[k],
                        alpha=1,linestyle=linestyles[k],
                        linewidth=1.5,zorder=1)
                    ax_list[-1].scatter(
                        x,y,marker='*',color=colors[k],
                        alpha=.8,s=2,zorder=3)
                    """if split_zero=='nonzero':
                        df_=df.copy()
                        if time_range:
                            df_=df_.loc[time_range,:]
                        stops=event_dict[mg]['stops']
                        df_.loc[stops.index,'stops']=0
                        df_.reset_index(inplace=True,drop=False)
                        stops_idx=df_[df_.loc[:,'stops']==0].index.to_numpy()"""
                   
                    
                #ser.plot(ax=ax_list[-1],color=colors[k],label=key)
                """if not sort and k==len(runoffdict)-1:#just do once...
                    #dict.fromkeys(df.index.get_level_values('date').to_list())
                    start_dates=[e-pd.Timedelta(hours=12) for e in event_dict[mg]['start'] if e in x]
                    stop_dates=[e+pd.Timedelta(hours=12) for e in event_dict[mg]['stop'] if e in x]
                    #print(f'adding {len(event_dates)} vlines. df_max:{df_max}')
                    if not split_zero=='zero':
                        ax_list[-1].vlines(
                            np.array(start_dates),0,df_max,
                            color='r',linewidth=.2,alpha=0.5,
                            #linestyle='dotted',
                            zorder=0,label='CN Event Starts')
                    if not split_zero=='nonzero':
                        ax_list[-1].vlines(
                            np.array(stop_dates),0,df_max,
                            color='k',linewidth=.2,alpha=0.5,
                            #linestyle='dotted',
                            zorder=0,label='CN Event Ends')"""
        for i,ax in enumerate(ax_list):
            
            ax.tick_params(direction="in")
            ax.set_ylabel(f'{scale_best_modelg[i]}',rotation=60, fontsize=8, labelpad=35)
            if not sort:
                if split_zero:
                    if i==0:ax.legend(ncol=len(runoffdict)+1,bbox_to_anchor=(0.5, 1.1))
                else:
                    if i==0:ax.legend(ncol=len(runoffdict)+2,bbox_to_anchor=(0.5, 1.1))
            else:
                if i==0:ax.legend(ncol=len(runoffdict),bbox_to_anchor=(0.5, 1.1))
            if True:#i==0 or sort or split_zero:
                #ax.set_xlabel('X LABEL')    
                ax.xaxis.set_label_position('top') 
                ax.xaxis.tick_top()
                for tick in ax.xaxis.get_major_ticks():
                    tick.label.set_fontsize(7) 
                    tick.set_pad(-10)
            else:
                #pass
                
                ax.set_xticklabels([])
                #ax.set_xticks([])
        

        if sort:
            name+='_sorted'
        if use_val_data:
            name+='_V'
        if time_range:
            
            name+='_T'

        fig.savefig(os.path.join(self.mct_results_folder,name+'.tif'))
            
                        
                                 
                                 
    def buildComidListDict(self,modelg_list):
        modelg_comidlist_dict={mg:[] for mg in modelg_list}
        for c,gdict in self.comid_geog_dict.items():
            try: modelg_comidlist_dict[gdict[self.model_geog]].append(c)
            except KeyError:pass
            except:
                print(traceback_exc())
                assert False,'unexpected error'
        return modelg_comidlist_dict
        
        
            
    
    def plotCorrectionResultLines(self,use_val_data=True):
        s_metric=self.selection_metric
        try: self.double_sort
        except: self.setSortOrder()
        try: self.correction_results_df
        except: self.buildCorrectionResultsDF()
        try: self.correction_selection_accuracy
        except: self.setCorrectionSelectionAccuracy()
        if use_val_data:
            sort_idx=self.double_sort_model_geog_index_V
            g_scales=self.double_sort_index_V.get_level_values(self.model_scale).to_list()
        else:
            sort_idx=self.double_sort_model_geog_index
            g_scales=self.double_sort_index.get_level_values(self.model_scale).to_list()
        n=sort_idx.shape[0]
            
        
        
        #g_model=self.double_sort_index.get_level_values(self.model_geog)
        #scale_xticks=pd.Series(g_scales,name=self.model_scale).reset_index().groupby(self.model_scale).count().cumsum().iloc[:,0].to_list()
        scaledf=pd.Series(g_scales,name=self.model_scale)
        scale_xticks=scaledf.reset_index().groupby(self.model_scale).count().reindex(list(dict.fromkeys(g_scales))).cumsum().iloc[:,0].to_list()#index of upper bounds separating each group
        
        self.g_scales=g_scales
        self.scale_xticks=scale_xticks
        st0=[0]+scale_xticks
        scale_xticks_between=[0.5*(st0[i]+st0[i-1]) for i in range(1,len(st0))]
        #scale_xticks=[t/n for t in scale_xticks]
        scale_items=list(dict.fromkeys(g_scales)) #unique items, preserved order
        scale_items=[str(i) for i in range(len(scale_items))]
        g_ID=[f'{g_scales[i]}-{i}' for i in range(n)] 
        #g_ID=list(range(n))
        metrics=self.correction_results_df.columns.to_list()#invariant to _V
        m_names=self.correction_results_df.index.get_level_values('estimator').unique()

        #best_per_m=self.correction_results_df.loc[self.correction_results_df.groupby([self.model_geog,'estimator'])[s_metric].idxmax()] #if m_name is repeated, pick only the best for each section
        """m_ser_dict={}
        for m in m_names:
            ser=best_per_m.loc[(m,slice(None),slice(None)),:]
            ser.index=ser.index.get_level_values('section')
            m_ser_dict[m]=ser.reindex(sort_idx) #sorted """
        
        
        s_name_grouped_correction_results_df=self.correction_results_df.copy()#invariant to _V b/c multiIndex equality...
        midx=s_name_grouped_correction_results_df.index
        s_name_midx_tups=[]
        for tup in midx.to_list():
            s_name=re.split('-',tup[0])[0]
            s_name_midx_tups.append((s_name,*tup))
        new_midx=pd.MultiIndex.from_tuples(s_name_midx_tups,names=['base est',*list(midx.names)])
        s_name_grouped_correction_results_df.index=new_midx
        if use_val_data:
            s_name_grouped_correction_results_df_V=self.correction_results_df_V.copy()
            s_name_grouped_correction_results_df_V.index=new_midx
        #best_est_geog_df=s_name_grouped_correction_results_df.sort_values(
        #   s_metric,ascending=False).drop_duplicates(['base est',self.model_geog])
        best_base_est_idx=s_name_grouped_correction_results_df.groupby(
                ['base est',self.model_geog]
                )[s_metric].idxmax()
        if use_val_data:
            best_est_geog_df=s_name_grouped_correction_results_df_V.loc[best_base_est_idx]
        else:
            best_est_geog_df=s_name_grouped_correction_results_df.loc[best_base_est_idx]
        
        m_ser_dict={}
        short_m_names=new_midx.get_level_values('base est').unique().to_list()
        
        for m in short_m_names:
            ser=best_est_geog_df.loc[(m,slice(None),slice(None),slice(None)),:]
            ser.index=ser.index.get_level_values(self.model_geog)
            m_ser_dict[m]=ser.reindex(sort_idx) #sorted 
            
        plt.rcParams['hatch.linewidth'] = 0.1
        plt.rcParams['axes.facecolor'] = 'lightgrey'
        fig=plt.figure(dpi=300,figsize=[9,8])
        fig.patch.set_facecolor('w')
        if use_val_data:
            fig.text(0.5, 1.02, 'Validation Data', ha='center', va='center')
        else:
            fig.text(0.5, 1.02, 'Test Data', ha='center', va='center')
           
        fig.suptitle(f'Out Of Sample Average Validation Scores')
        colors = plt.get_cmap('tab10')(np.arange(10))
        for i,metric in enumerate(metrics):
            #ax=fig.add_subplot(2,1,i+1)
            ax=SubplotHost(fig, f'21{i+1}');fig.add_subplot(ax)
            ax.set_title(f'{metric.upper()}')
            ax.vlines(scale_xticks,-1,1,color='w',alpha=0.5)
            for ii,(m_name,ser) in enumerate(m_ser_dict.items()):
                val_arr=ser[metric]
                val_arr[val_arr<-1]=-1
                vals=val_arr.to_list()
                ax.scatter(g_ID,vals,color=colors[ii],alpha=0.6,label='_'+m_name,s=1.5)
                ax.plot(g_ID,vals,'o-',color=colors[ii],alpha=0.7,label=m_name,linewidth=1.3)
            if i==0:ax.legend()
            if i==0:
                ax.set_ylim(bottom=-1,top=1)
            else:
                if min(vals)<0:
                    ax.set_ylim(bottom=-1,top=1)
                else:
                    ax.set_ylim(bottom=0,top=1)
            ax.set_xticks(scale_xticks)
            
            ax.xaxis.set_major_formatter(ticker.NullFormatter())
            ax.xaxis.set_minor_locator(ticker.FixedLocator(scale_xticks_between))
            ax.xaxis.set_minor_formatter(ticker.FixedFormatter(scale_items))
            ax.xaxis.set_tick_params(which='minor',color='r',length=10)
            
            #ax.xaxis.set_tick_params(rotation=45,)
            
            if i>0:ax.xaxis.set_label_text('Sections Grouped by Divison ID')
            # labelcolor="r",)
            #for label in ax.xaxis.get_ticklabels():
            #    label.set_rotation(40)
        fig.tight_layout()
        plt.show()
        if use_val_data:
            val_str='_V'
        else:val_str=''
        fig.savefig(os.path.join(self.mct_results_folder,f'correction_results_lines{self.hash_id}{val_str}.tif'))
        
        
    
    def expandGeogsToMultiIndex(self,geog_list):
        comid_geog_dict=self.comid_geog_dict
        g_names=self.geog_names
        g_pos=g_names.index(self.model_geog)
        higher_geog_levels=g_names[:g_pos+1]
        higher_geog_dict={}
        for comid,gdict in comid_geog_dict.items():
            g=gdict[self.model_geog]
            if not g in higher_geog_dict:
                higher_geog_dict[g]=list(gdict.values())[:g_pos+1]#include g too
        geog_tup_list=[]
        for g in geog_list:
            geog_tup_list.append((*higher_geog_dict[g],))
        return pd.MultiIndex.from_tuples(geog_tup_list,names=higher_geog_levels)
            
    
    def saveCorrectionSelectionTable(self,expand_geog=True,use_val_data=True):
        assert self.geog_model_select_dict,'run the corrections and selection first'
        try: self.double_sort_model_geog_index
        except:self.setSortOrder()
        
        geogs=[];selections=[];accuracies=[]
        if use_val_data:
            val_str='-V'
            double_sort_mg_index=self.double_sort_model_geog_index_V
            mg_select_dict=self.geog_model_select_dict_V
        else:
            val_str=''
            double_sort_mg_index=self.double_sort_model_geog_index
            mg_select_dict=self.geog_model_select_dict
        
        for geog in double_sort_mg_index.to_list():
            select_dict=mg_select_dict[geog]
            geogs.append(geog)
            accuracies.append(select_dict[self.selection_metric])
            selections.append(select_dict['estimator'].upper())
        if expand_geog:
            geogs=self.expandGeogsToMultiIndex(geogs)
            
        hash_id=self.hash_id
        #correction_selection_names=pd.Series(selections,name='best estimator',index=geogs)
        #correction_selection_names.to_csv(os.path.join(self.mct_results_folder,f'correction_selection_names_{hash_id}.csv'))
        
        geogs_with_id=pd.MultiIndex.from_tuples(
            [tuple([self.div_id_dict[mtup[0]]]+list(mtup))for mtup in geogs],# invariant to _V
            names=['division ID']+list(geogs.names))
        correction_selection_score=pd.DataFrame(
            {
                'selected estimator':selections,
                self.selection_metric:accuracies,
                #'ID':list(range(len(selections)))
            },index=geogs_with_id)
        correction_selection_score.to_csv(os.path.join(self.mct_results_folder,f'correction_selection_score_{hash_id}{val_str}.csv'))
        correction_selection_score.round(decimals=3).to_html(os.path.join(self.mct_results_folder,f'correction_selection_score_{hash_id}{val_str}.html'))
    
    
     
    
    
    def plotGeogHybridAccuracy(self,plot_negative=True,cmap=None,use_val_data=True):
        try: self.eco_geog
        except: self.setEcoGeog()
        #try: self.correction_selection_accuracy
        #except: self.setCorrectionSelectionAccuracy()
        try:self.best_model_df
        except: self.selectCorrections()
        
        geog=self.model_geog
        metric=self.selection_metric
        #hybrid_accuracy_series=self.correction_selection_accuracy#.reset_index(name=geog)
        if use_val_data:
            best_model_df=self.best_model_df_V.copy()
        else:
            best_model_df=self.best_model_df.copy()
        best_model_df.index=best_model_df.index.get_level_values(self.model_geog)
        hybrid_geog=self.eco_geog.merge(best_model_df,on=geog,how='left')
        
        plt.rcParams['hatch.linewidth'] = 0.1
        plt.rcParams['axes.facecolor'] = 'lightgrey'
        fig=plt.figure(dpi=300,figsize=[9,3.7])
        fig.patch.set_facecolor('w')
        fig.suptitle(f'Best Correction Model Out Of Sample Average Validation Scores')
        for i,metric in enumerate(best_model_df.columns.to_list()):
            ax=fig.add_subplot(1,2,i+1)
            ax.set_title(f'{metric.upper()}')
            #background=self.eco_geog.plot(color='darkgrey',ax=ax,hatch='/////')
            #pos_geog_acc_df=geog_acc_df[geog_acc_df.loc[:,metric]>0]
            #pos_geog_acc_df.plot(column=metric,ax=ax,cmap='plasma',legend=True,)
            if plot_negative:
                norm = TwoSlopeNorm(vmin=-1,vcenter=0, vmax=1)
                if cmap is None:cmap='RdBu'#'brg'##'plasma'
                cbar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
                #self.geog_acc_df=geog_acc_df
                h=hybrid_geog.plot(column=metric,ax=ax, cmap=cmap, norm=norm,legend=True,missing_kwds={
                    "color": "lightgrey","hatch": "xxxxxxxxx","label": "Missing values",},legend_kwds={'orientation': "horizontal"})
                #fig.colorbar(cbar, ax=ax)
                self.geog_gdf=hybrid_geog
            else:
                if cmap is None:cmap='plasma'
                norm=Normalize(vmin=0,vmax=1)
                cbar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
                metric_series=hybrid_geog.loc[:,metric]
                neg_geog_acc_df=hybrid_geog[(metric_series<=0)]
                n_=neg_geog_acc_df.plot(ax=ax,color='lightgrey',hatch='oooooo',zorder=1,label=f'non-positive {metric.upper()}',legend=True,legend_kwds={'orientation': "horizontal"})
                #neg_leg=plt.legend(n_,f'non-positive {metric}')
                pos_hybrid_geog=hybrid_geog[~(hybrid_geog.loc[:,metric]<=0)]
                #mapscheme=mapclassify.Quantiles(pos_hybrid_geog.loc[:,metric].to_numpy(),k=8).bins
                """hybrid_geog.plot(column=metric,ax=ax,label=f'non-negative {metric.upper()}',missing_kwds={
                    "color": "lightgrey","edgecolor": "red","hatch": "///","label": "Missing values",},legend=True,scheme="User_Defined",classification_kwds={'bins':[0,.3,.45,.6,0.7,metric_series.max()]},legend_kwds={'loc': 'lower right'})"""
                pos_hybrid_geog.plot(column=metric,ax=ax,cmap='plasma',zorder=1,norm=norm,label=f'{metric}',legend=True,missing_kwds={
                    "color": "lightgrey",
                    #"edgecolor": "red",
                    "hatch": "xxxxxxxxx",
                    "label": "Missing values",},legend_kwds={'orientation': "horizontal"})
                #fig.colorbar(cbar,ax=ax)
                #self.geog_gdf=pos_hybrid_geog
                #self.geog_gdf_neg=neg_geog_acc_df
            self.add_states(ax)
            ax.set_xticklabels([])
            ax.set_xticks([])
            ax.set_yticklabels([])
            ax.set_yticks([])
            if i==0:
                handles, labels = ax.get_legend_handles_labels()
                handles=[
                    *handles,
                    mpatches.Patch(
                        hatch='xxxxxxxxx',facecolor='lightgrey',
                        label='<100 CN event days'),
                    mpatches.Patch(
                        hatch='oooooo',facecolor='lightgrey', 
                        label='negative score'),
                ]
                ax.legend(handles=handles,fontsize=6,bbox_to_anchor=(0.3,0.15),frameon=False)
        fig_name=f'{self.model_scale}_hybrid-select_combined.tif'
        if not plot_negative:
            fig_name='pos-score_'+fig_name
            #ax.legend()
            #ax.add_artist(negleg)
        fig_name='cv_'+cmap+fig_name
        fig.tight_layout()
        #plt.legend()
        plt.show()
        fig.savefig(os.path.join(self.mct_results_folder,fig_name))   
    
    def add_states(self,ax):
        try: self.eco
        except: self.setEcoGeog()
        try: self.eco_clip_states
        except:
            states=gpd.read_file(self.states_path)
            eco_d=self.eco.copy()
            eco_d['dissolve_field']=1
            eco_d.dissolve(by='dissolve_field')
            self.eco_clip_states=gpd.clip(states,eco_d)
        self.eco_clip_states.boundary.plot(linewidth=0.3,ax=ax,color=None,edgecolor='k')    
    
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
   
        
if __name__=="__main__":
    assert False,'pickle requires running CompareCorrect.runModelCorrection from another python file. Try run_multi_correct.py or multi_correct.ipynb'
