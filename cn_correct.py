import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import re
import os
import logging
import matplotlib.pyplot as plt
from random import shuffle
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LassoCV,LinearRegression
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.model_selection import RepeatedKFold,GridSearchCV,LeaveOneGroupOut
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import BaseEstimator, TransformerMixin,RegressorMixin
from scipy.stats import pearsonr
from multiprocessing import Process,Queue
from mylogger import myLogger
from data_analysis import get_comid_data
from mp_helper import MpHelper,MpWrapper
from collections import Counter
from time import time



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
            return np.asfortranarray(X.to_numpy(),dtype=np.float32)
        else:
            return np.asfortranarray(X,dtype=np.float32)
    

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
            return X.loc[:,self.unique_>1]
        else:
            return X[:,self.unique_>1]       



class PipeWrapper:
    def __init__(self,data_dict,model_spec_tup):
        #self.data_hash=joblib.hash(data_dict)
        self.model_spec_tup=model_spec_tup
        self.modeldict=self.model_spec_tup[2]
        x=data_dict['x']
        y=data_dict['y']
        if self.modeldict['cross_validate']:
            self.train_comids=dict.fromkeys(x.index.unique(level='comid'))#dict for faster search
        
        myLogger.__init__(self,'PipeWrapper.log')
        self.modeled_runoff_col=self.modeldict['sources']['modeled']
        self.data_filter=self.modeldict['filter']
        if self.data_filter == 'none':
            self.model=PipelineModel(data_dict,self.model_spec_tup)
        elif self.data_filter == 'nonzero':
            modeled_runoff=x.loc[:,self.modeled_runoff_col]
            zero_idx=modeled_runoff==0
            self.model={}
            self.model['zero']=PipelineModel(x[zero_idx],y[zero_idx],('lin-reg',{'max_poly_deg':1,'fit_intercept':False},None))
            if x[~zero_idx].shape[0]>0:
                self.model['nonzero']=PipelineModel(x[~zero_idx],y[~zero_idx],self.model_spec_tup)
            else:
                self.model['nonzero']=NullModel()
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
        full_comid_list=x.index.unique(level='comid')
        keep_comid_list=[comid for comid in full_comid_list if not comid in self.train_comids]
        if len(keep_comid_list)==0:
            return 'none'
        return x.loc[(keep_comid_list,slice(None)),:]
    
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

class PipelineModel(myLogger):
    def __init__(self,x,y,model_spec_tup):
        myLogger.__init__(self,)
        model_name,specs,_=model_spec_tup
        cv=RepeatedKFold(random_state=0,n_splits=10,n_repeats=1)
        if model_name.lower() =='lin-reg':
            deg=specs['max_poly_deg']
            if deg>1:
                pipe=make_pipeline(
                StandardScaler(),
                PolynomialFeatures(include_bias=False),
                DropConst(),       
                LinearRegression(fit_intercept=specs['fit_intercept']))
                param_grid={'polynomialfeatures__degree':np.arange(1,deg+1)}
                self.pipe=GridSearchCV(pipe,param_grid=param_grid,cv=cv,n_jobs=4)
            else:
                self.pipe=make_pipeline(
                StandardScaler(),
                #PolynomialFeatures(include_bias=False),
                DropConst(),       
                LinearRegression(fit_intercept=specs['fit_intercept']))
            #self.pipe.fit(x,y)
        elif model_name.lower() in ['l1','lasso']:
            deg=specs['max_poly_deg']
            lasso_kwargs=dict(random_state=0,fit_intercept=specs['fit_intercept'],cv=cv)
            self.pipe=make_pipeline(
                StandardScaler(),
                PolynomialFeatures(include_bias=False,degree=deg),
                DropConst(),
                ToFortranOrder(),
                LassoCV(**lasso_kwargs,n_jobs=4))
            #self.pipe.fit(x.astype(np.float32),y.astype(np.float32))
        elif model_name.lower()=='gbr':
            if 'kwargs' in specs:
                kwargs=specs['kwargs']
            else:kwargs={}
            param_grid={'max_depth':list(range(3,5)),
                       'n_estimators':[100,500]
                       }
            
            self.pipe=GridSearchCV(GradientBoostingRegressor(random_state=0,**kwargs),param_grid=param_grid,cv=cv,n_jobs=4)
            #self.pipe.fit(x,y)
        else:
            assert False,'model_name not recognized'
        self.pipe.fit(x.astype(np.float32),y.astype(np.float32))
        #self.logger.info(f'fit complete, starting yhat_train prediction')
        #self.yhat_train=pd.DataFrame(self.pipe.predict(x),index=x.index,columns=['yhat'])
        #self.logger.info(f'yhat_train prediction complete.')
            
    def predict(self,x):
        return self.pipe.predict(x)
    
    def set_test_stats(self,xtest,ytest):
        self.yhat_test=self.predict(xtest)
        self.poly_test_stats=SeriesCompare(ytest,self.yhat_test)
        #dself.uncorrected_test_stats=SeriesCompare(ytest,xtest)
        

    
    
    """def make_poly_X(self,x,deg):
        #X=pd.DataFrame(np.ones(x.shape[0]),columns=['constant'])
        X=pd.DataFrame()
        for d in range(1,deg+1):
            X.loc[:,f'x^{d}']=x.values**d
        return X"""
            

    """def make_poly_X(self,x):
        #X=pd.DataFrame(np.ones(x.shape[0]),columns=['constant'])
        X=pd.DataFrame()
        for d in range(1,self.deg+1):
            X.loc[:,f'x^{d}']=x.values**d
        return X"""
            
                

class CatchmentCorrection(myLogger):
    def __init__(self,comid,modeldict):
        myLogger.__init__(self,'catchment_correction.log')
        self.comid=comid
        self.runoff_df=self.make_runoff_df(comid)
        self.modeldict=modeldict
        
        
        
    def make_runoff_df(self,comid):
        df=get_comid_data(comid)
        date=df['date']
        df.index=date
        df.drop(columns='date',inplace=True)
        return df
    
    def run(self):
        self.runCorrection()
        
        
    def set_train_test(self,y_df,x_df):
        train_share=self.modeldict['train_share']
        n=y_df.shape[0]
        split_idx=int(train_share*n)
        x_train=x_df.iloc[:split_idx]
        y_train=y_df.iloc[:split_idx]
        x_test=x_df.iloc[split_idx:]
        y_test=y_df.iloc[split_idx:]
        return x_train,y_train,x_test,y_test
    
    def filter_data(self,y,x,data_filter):
        if data_filter is None or data_filter.lower()=="none":
            return y,x
        if data_filter.lower()=='nonzero':
            non_zero_idx=np.arange(y.shape[0])[x>0]
            y=y[non_zero_idx];x=x[non_zero_idx]
        if data_filter.lower()[:10]=='percentile':
            assert False, not developed
        return y,x
                       
                       
    def runCorrection(self):
        sources=self.modeldict['sources']
        obs_df=self.runoff_df.loc[:,sources['observed']]
        mod_df=self.runoff_df.loc[:,sources['modeled']]
        if mod_df[mod_df>0].shape[0]<100:
            self.logger.error(f'data problem for comid:{self.comid} mod_df[mod_df>0].shape[0]:{mod_df[mod_df>0].shape[0]} and mod_df.shape:{mod_df.shape}')
            self.correction_dict={}
            self.uncorrected_test_stats=None
            return
        
        #data_filter=self.modeldict['filter']
        #obs_df,mod_df=self.filter_data(obs_df,mod_df,data_filter)
        
        if obs_df.shape[0]<32:
            self.logger.error(f'data problem for comid:{self.comid} obs_df.shape:{obs_df.shape} and mod_df.shape:{mod_df.shape}')
            self.correction_dict={}
            self.uncorrected_test_stats=None
            return
        
        x_train,y_train,x_test,y_test=self.set_train_test(obs_df,mod_df)
        
        self.correction_dict={}
        for m_name,mdict in self.modeldict['model_specs'].items():
            model=PipeWrapper(x_train,y_train,deg=deg,model_spec_tup=(m_name,mdict,self.modeldict))  
            model.set_test_stats(x_test,y_test)
            self.correction_dict[f'{m_name}']=model
        
        self.uncorrected_test_stats=SeriesCompare(y_test,x_test)
 

class ComidData(myLogger):            
    def __init__(self,comid,sources):
        myLogger.__init__(self,'catchment_data.log')
        self.comid=comid
        self.sources=sources
        self.runoff_df=self.make_runoff_df(comid,multi_index=True) 
        #self.modeldict=modeldict  
        
        #sources=self.modeldict['sources']
        obs_src=sources['observed']
        mod_src=sources['modeled']
        self.runoff_model_data_df=self.runoff_df.loc[:,[obs_src,mod_src]]
        nonzero=self.runoff_model_data_df.loc[:,mod_src]>0
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
    
    def set_train_test(self,train_share):
        df=self.runoff_model_data_df
        #train_share=self.modeldict['train_share']
        n=df.shape[0]
        split_idx=int(train_share*n)
        y_df=df.loc[:,self.sources['observed']]
        x_df=df.drop(self.sources['observed'],axis=1,inplace=False)
    
        self.x_train=x_df.iloc[:split_idx]
        self.y_train=y_df.iloc[:split_idx]
        self.x_test=x_df.iloc[split_idx:]
        self.y_test=y_df.iloc[split_idx:]
 

class GroupDFSplitter(myLogger):
    def __init__(self,n_reps,num_groups=None):
        myLogger.__init__(self)
        self.n_reps=n_reps
        self.num_groups=num_groups
    def get_df_split(self,groups):
        if type(groups) in [np.ndarray, pd.Series]:
            groups=pd.DataFrame(groups,columns=['group'])
        inferred_group_name=groups.columns.to_list()[0]
        if self.num_groups is None:
            self.num_groups=groups.value_counts(subset=inferred_group_name).min()
        print(f'split leaving one member out grouping:{inferred_group_name}')
        n=groups.shape[0]
        for seed in range(self.n_reps):
            shuf_grp=groups.sample(frac=1,replace=False,random_state=seed)
            grp=shuf_grp.groupby(by=inferred_group_name).cumcount()
            for i in range(self.num_groups):
                grp_bool=grp!=pd.Series([i]*n,name=grp.name,index=grp.index)
                yield grp_bool
            
class Runner(myLogger):
    def __init__(self,X,y,m_name,specs,modeldict,bool_idx=None,return_save_string=True):
        self.X=X;self.y=y
        self.m_name=m_name
        self.specs=specs
        self.modeldict=modeldict
        self.bool_idx=bool_idx
        self.return_save_string=return_save_string
        
    def run(self):
        myLogger.__init__(self)
        self.model=self.runmodel()
        
    def runmodel(self):
        try:
            self.logger.info(f'starting {self.m_name}')
            t0=time()
            if not self.bool_idx is None:
                self.X=self.X[self.bool_idx]
                self.y=self.y[self.bool_idx]
            data_dict={'x':self.X,'y':self.y}
            args=[data_dict,(self.m_name,self.specs,self.modeldict)]
            name=os.path.join('results',f'pipe-{joblib.hash(args)}.pkl')
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
        name=os.path.join('results','comiddata.pkl')
        if os.path.exists(name):
            try:
                with open(name,'rb') as f:
                    self.comid_data_object_dict,self.failed_comid_dict=pickle.load(f)
                return
            except:
                print('load failed, building comiddata')
            
        for comid in self.comidlist:
            comid_data_obj=ComidData(comid,self.modeldict['sources'])
            if comid_data_obj.nonzero_count>100:
                self.comid_data_object_dict[comid]=comid_data_obj
            else:
                self.failed_comid_dict[comid]=f'runoff_model_data_df too small with shape:{comid_data_obj.runoff_model_data_df.shape}'
        if len(self.failed_comid_dict)>0:
            self.logger.info(f'failed comids:{self.failed_comid_dict}')
        savetup=(self.comid_data_object_dict,self.failed_comid_dict)
        with open(name,'wb') as f:
            pickle.dump(savetup,f)
        
    
    def addGeogCols(self,):
         
        
        for comid,obj in self.comid_data_object_dict.items():
            geog_dict=self.comid_geog_dict[comid]
            for col_name,val in geog_dict.items():
                try:
                    obj.runoff_model_data_df.loc[:,col_name]=val
                except:
                    print(f'comid:{comid},col_name:{col_name},val:{val}.')
                    assert False,f'comid:{comid},col_name:{col_name},val:{val}.'
    
    def setComidTrainTest(self):
        train_share=self.modeldict['train_share']
        for comid,obj in self.comid_data_object_dict.items():
            obj.set_train_test(train_share)

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
            self.logger.info(f'making bool_idx for cv')
            bool_idx_list=[bool_idx for bool_idx in GroupDFSplitter(reps,num_groups=5).get_df_split(group_indicator)]
            self.logger.info(f'bool_idx_list complete')
        for m_name,specs in self.modeldict['model_specs'].items():
            single_modeldict=self.modeldict
            single_modeldict['model_specs']={m_name:specs}
            if not cv:
                self.model_results[m_name]=[self.Runner(X,y,m_name,specs,single_modeldict).run().model]
            else:
                self.model_results[m_name]=[]
                
                args_list=[];kwargs_list=[]
                for bool_idx in bool_idx_list:
                    #self.logger.info(f'cv run_{i} starting')
                    #self.logger.info(f'building args_list for {specs}')
                    ##args=[X[bool_idx],y[bool_idx],m_name,specs,single_modeldict]
                    args=[X,y,m_name,specs,single_modeldict]
                    args_list.append(args)
                    kwargs={'bool_idx':bool_idx}
                    kwargs_list.append(kwargs)
                    #model=self.run_it(X[bool_idx],y[bool_idx],m_name,specs,single_modeldict).run()
                    #self.logger.info(f'cv run_{i} complete')
                    #self.model_results[m_name].append(model)
                self.logger.info(f'starting multiproc Runners for {m_name}')
                results=MpHelper().runAsMultiProc(Runner,args_list,kwargs_list=kwargs_list,proc_count=2)
                self.model_results[m_name].extend([result.model for result in results])
                self.logger.info(f'Runners complete for {m_name}')
                    

    def runTestData(self):
        for m_name,model_list in self.model_results.items():
            if not self.modeldict['cross_validate']: 
                if not type(model_list) is list:model_list=[model_list]
            for model in model_list:
                if type(model) is str:
                    with open(model,'rb') as f:
                        model=pickle.load(f)
                for obj in self.comid_modeling_objects:
                    if not m_name in obj.test_results:
                        obj.test_results[m_name]=[]
                    yhat_test,test_stats=model.get_prediction_and_stats(obj.x_test_float,obj.y_test)
                    if type(test_stats) is str:
                            self.logger.warning(f'test data from comid:{obj.comid}: {(yhat_test,test_stats)}')
                    else:
                        obj.test_results[m_name].append({'test_stats':test_stats,'yhat_test':yhat_test})
            for obj in self.comid_modeling_objects:
                uncorr_yhat=obj.x_test.loc[:,self.modeldict['sources']['modeled']]
                obj.test_results['uncorrected']=[{'test_stats':SeriesCompare(obj.y_test.values,uncorr_yhat.values),'yhat_test':uncorr_yhat}]
            
    def plotGeoTestData(self):
        geog=self.modeldict['model_geog']
        bigger_geog=self.geog_names[self.geog_names.index(geog)-1]
        try: self.eco
        except:
            self.eco=gpd.read_file(self.physio_path)
            self.eco.columns=[col.lower() for col in self.eco.columns.to_list()]
            null_bool=self.eco.loc[:,geog].isnull()#fix missing sections
            self.eco[null_bool].loc[:,geog]=self.eco[null_bool].loc[:,bigger_geog]
            eco_geog=self.eco.dissolve(by=geog)
            self.eco_geog=eco_geog
        self.acc_df_dict={};self.geog_acc_df_dict={}  
        for m_name in self.model_results.keys():
            data_dict={'nse':[],'pearson':[],geog:[]}
            for obj in self.comid_modeling_objects:
                for result_dict in obj.test_results[m_name]:
                    data_dict['nse'].append(result_dict['test_stats'].nse)
                    data_dict['pearson'].append(result_dict['test_stats'].nse)
                    data_dict[geog].append(self.comid_geog_dict[obj.comid][geog])
            mean_acc_df=pd.DataFrame(data_dict).groupby(geog).mean()
            #self.acc_df_dict[m_name]=mean_acc_df
            geog_acc_df=self.eco_geog.merge(mean_acc_df,on=geog)
            #self.geog_acc_df_dict[m_name]=
            for metric in ['nse','pearson']:
                fig=plt.figure(dpi=300,figsize=[9,6])
                fig.suptitle(f'modeldict:{self.modeldict}')
                ax=fig.add_subplot(1,1,1)
                ax.set_title(f'{m_name}_{metric}')
                pos_geog_acc_df=geog_acc_df[geog_acc_df.loc[:,metric]>0]
                pos_geog_acc_df.plot(column=metric,ax=ax,cmap='plasma',legend=True,)
                self.add_states(ax)
                fig.savefig(os.path.join('print',f'accuracy_{m_name}_{metric}.png'))
        
            
    def add_states(self,ax):
        try: self.eco_clip_states
        except:
            states=gpd.read_file(self.states_path)
            eco_d=self.eco.copy()
            eco_d['dissolve_field']=1
            eco_d.dissolve(by='dissolve_field')
            self.eco_clip_states=gpd.clip(states,eco_d)
        self.eco_clip_states.boundary.plot(linewidth=1,ax=ax,color=None,edgecolor='k')                 
            
        
                
                
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
    def __init__(self,):
        myLogger.__init__(self,'comparecorrect.log')
        self.proc_count=2
        if not os.path.exists('results'):
            os.mkdir('results')
        if not os.path.exists('print'):
            os.mkdir('print')
        self.modeldict={
            'cross_validate':{'n_reps':3,'strategy':'leave_one_member_out'},#False,#
            'model_geog':'section',
            'sources':{'observed':'nldas','modeled':'cn'}, #[observed,modeled]
            'filter':'nonzero',#'none',#'nonzero'
            'train_share':0.80,
            'split_order':'chronological',#'random'
            'model_scale':'conus',#'comid'
            'model_specs':{
                #no intercept b/c no dummy drop
                #'lasso':{'max_poly_deg':3,'fit_intercept':False},
                'gbr':{
                    'kwargs':{},#these pass through to sklearn's gbr
                    #'n_estimators':10000,
                    #'subsample':1,
                    #'max_depth':3
                },
                'lin-reg':{'max_poly_deg':3,'fit_intercept':False},
            }
        } 
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
    
    
    def runBigModel(self,):
        args=[[comid,self.modeldict] for comid in self.comidlist]
        save_hash=joblib.hash(args)
        save_path=os.path.join('results',f'data-collection_{save_hash}')
        loaded=False
        if os.path.exists(save_path):
            with open(save_path,'rb') as f:
                self.dc=pickle.load(f)
            loaded=True
        else:
            print('running big model')
        if not loaded:
            self.dc=DataCollection(self.comidlist,self.modeldict,self.makeComidGeogDict())
            self.dc.build()
            self.dc.runModel()
            self.dc.runTestData()
            with open(save_path,'wb') as f:
                pickle.dump(self.dc,f)
        self.dc.plotGeoTestData()
        
    def makeComidGeogDict(self):
        geog=self.modeldict['model_geog']
        if not type(geog) is list:
            geogs=[geog]
        else:
            geogs=geog
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
    
    def makeAccuracyDF(self,):
        self.runModelCorrection(try_load=True)
        prefix=f"{self.modeldict['sources']['observed']}_{self.modeldict['filter']}_"
        nse_key=prefix+'nse'
        pear_key=prefix+'pearsons'
        data_dict={nse_key:[],pear_key:[]}
        idx_tup_list=[]
        #idx=[]
        for obj in self.comid_obj_list:
            comid=obj.comid
            if len(obj.correction_dict)==0:
                self.logger.warning(f'no results for comid:{comid}')
                continue
            for model_name,model in obj.correction_dict.items():
                idx_tup_list.append((comid,model_name))
                #idx.append(comid)
                stats=model.poly_test_stats
                data_dict[nse_key].append(stats.nse)
                data_dict[pear_key].append(stats.pearsons)
            data_dict[nse_key].append(obj.uncorrected_test_stats.nse)
            data_dict[pear_key].append(obj.uncorrected_test_stats.pearsons)
            idx_tup_list.append((comid,'uncorrected'))
        midx=pd.MultiIndex.from_tuples(idx_tup_list,names=['comid','model_name'])
        self.accuracy_df=pd.DataFrame(data_dict,index=midx)
            
    def plotAccuracyDF(self,model_spec={'lasso':{'max_poly_deg':5}},geography='section'):
        try: self.eco
        except:
            self.eco=gpd.read_file(self.physio_path)
            self.eco.columns=[col.lower() for col in self.eco.columns.to_list()]
        try: self.accuracy_df
        except: self.makeAccuracyDF()
        model_name_list=self.accuracy_df.index.get_level_values('model_name').unique()
        if not model_name in model_name_list:
            found=False
            for model in model_n in model_name_list:
                if re.search(model_name,model_n):
                    self.logger.info(f'model_name:{model_name} not found. plotting {model_n} instead')
                    model_name=model_n
                    found=True
                    break
            if not found:
                model_self.logger.critical(f'model_name:{model_name} not found, nothing similar found')
                return
            
        model_accuracy_df=self.accuracy_df.loc[(slice(None),model_name),:]
        model_accuracy_df.index=model_accuracy_df.index.droplevel(level='model_name')
        self.model_accuracy_df=model_accuracy_df
        
        
        acc=model_accuracy_df.merge(self.comid_physio,on='comid')
        self.acc=acc
        for col in self.geog_names:
            if col!=geography:
                acc.drop(col,axis=1,inplace=True)
        model_accuracy_df_geog_mean=acc.groupby(geography).mean()
        self.model_accuracy_df_geog_mean=model_accuracy_df_geog_mean
        
        eco_geog=self.eco.dissolve(by=geography)
        self.eco_geog=eco_geog
        #eco_geog.index=[idx.lower() for idx in eco_geog.index.to_list()]
        eco_geog_data=eco_geog.merge(model_accuracy_df_geog_mean,on=geography)
        self.eco_geog_data=eco_geog_data
        for col in model_accuracy_df.columns:
            fig=plt.figure(dpi=300,figsize=[12,6])
            fig.suptitle(f'modeldict:{self.modeldict}')
            
            ax=fig.add_subplot(1,1,1)

            ax.set_title(f'{model_name}_{col}')
            pos_eco_geog_data=eco_geog_data[eco_geog_data.loc[:,col]>0]
            pos_eco_geog_data.plot(column=col,ax=ax,cmap='plasma',legend=True,)
            #self.param_gdf.boundary.plot(ax=ax,edgecolor='w',linewidth=1)
            self.add_states(ax)
            fig.savefig(os.path.join('print',f'accuracy_{model_name}_{col}.png'))
        #ax.legend()
        
        
    def add_states(self,ax):
        try: self.eco_clip_states
        except:
            states=gpd.read_file(self.states_path)
            eco_d=self.eco.copy()
            eco_d['dissolve_field']=1
            eco_d.dissolve(by='dissolve_field')
            self.eco_clip_states=gpd.clip(states,eco_d)
        self.eco_clip_states.boundary.plot(linewidth=1,ax=ax,color=None,edgecolor='k')                
    
    def runModelCorrection(self,try_load=False):
        args=[[comid,self.modeldict] for comid in self.comidlist]
        save_hash=joblib.hash(args)
        save_path=os.path.join('results',f'comid_correction_list_hash-{save_hash}')
        if try_load:
            if os.path.exists(save_path):
                with open(save_path,'rb') as f:
                    comid_obj_list=pickle.load(f)
                if len(comid_obj_list)==len(self.comidlist):
                    self.logger.info(f'load complete for {save_path}')
                    self.comid_obj_list=comid_obj_list
                else:
                    assert False,f'save_path:{save_path} has wrong length'
                    self.comid_obj_list_err=comid_obj_list
                return
            else:
                self.logger.debug('no file to load')
        else:
            assert not os.path.exists(save_path),f'save_path:{save_path} already exists, halting'                            
        comid_obj_list=MpHelper().runAsMultiProc(CatchmentCorrection,args,proc_count=self.proc_count)
        self.comid_obj_list=comid_obj_list
        
        with open(save_path,'wb') as f:
            pickle.dump(comid_obj_list,f)
            
    def build_test_metric_df(self):
        assert self.comid_obj_list,'runModelCorrection must be run first'
        correction_dict_list=[]
        uncorrected_test_stats_list=[]
        for o,obj in enumerate(self.comid_obj_list):
            assert self.comidlist[o]==obj.comid,f'comid mismatch!!!'
            correction_dict_list.append(obj.correction_dict)
            uncorrected_test_stats_list.append(obj.uncorrected_test_stats)
   
        
if __name__=="__main__":
    assert False,'pickle requires running CompareCorrect.runModelCorrection from another python file'
