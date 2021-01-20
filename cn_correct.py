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
from sklearn.model_selection import RepeatedKFold,GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import pearsonr
from multiprocessing import Process,Queue
from mylogger import myLogger
from data_analysis import get_comid_data
from mp_helper import MpHelper



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
        
       

class PolyModel(myLogger):
    # a single polynomial model of the specified degree
    def __init__(self,x,y,deg,model_name=None):
        myLogger.__init__(self,'polymodel.log')
        self.deg=deg;self.model_name=model_name
        X=self.make_poly_X(x)   
        poly_kwargs={'include_bias':False}
        param_grid={'polynomialfeatures__degree':np.arange(1,deg+1)}
        cv=RepeatedKFold(random_state=0,n_splits=10,n_repeats=5)
        if model_name.lower() =='lin-reg':
            pipe=make_pipeline(
                StandardScaler(),
                PolynomialFeatures(**poly_kwargs),
                LinearRegression(fit_intercept=True))
            self.est=GridSearchCV(pipe,param_grid=param_grid,cv=cv)
        elif model_name.lower() in ['l1','lasso']:
            lasso_kwargs=dict(
                random_state=0,
                fit_intercept=True,
                cv=cv
                
            )
            self.est=make_pipeline(StandardScaler(),PolynomialFeatures(**poly_kwargs,degree=deg),LassoCV(**lasso_kwargs))
        elif model_name.lower()=='gbr':
            self.est=GradientBoostingRegressor(random_state=0)
        else:
            assert False,'model_name not recognized'
        self.est.fit(X,y)
        self.yhat_train=self.est.predict(X)
            
    def predict(self,x):
        X=self.make_poly_X(x)
        return self.est.predict(X)
    
    def set_test_stats(self,xtest,ytest):
        self.yhat_test=self.predict(xtest)
        self.poly_test_stats=SeriesCompare(ytest,self.yhat_test)
        #dself.uncorrected_test_stats=SeriesCompare(ytest,xtest)
    
    def make_poly_X(self,x):
        #X=pd.DataFrame(np.ones(x.shape[0]),columns=['constant'])
        X=pd.DataFrame()
        for d in range(1,self.deg+1):
            X.loc[:,f'x^{d}']=x.values**d
        return X
            
            
            
            
                

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
        if obs_df.shape[0]<100:
            self.logger.error(f'data problem for comid:{self.comid} obs_df.shape:{obs_df.shape} and mod_df.shape:{mod_df.shape}')
            self.correction_dict={}
            self.uncorrected_test_stats=None
            return
        
        data_filter=self.modeldict['filter']
        obs_df,mod_df=self.filter_data(obs_df,mod_df,data_filter)
        
        if obs_df.shape[0]<32:
            self.logger.error(f'data problem for comid:{self.comid} obs_df.shape:{obs_df.shape} and mod_df.shape:{mod_df.shape}')
            self.correction_dict={}
            self.uncorrected_test_stats=None
            return
        
        x_train,y_train,x_test,y_test=self.set_train_test(obs_df,mod_df)
        
        m_names=self.modeldict['model_name']
        self.correction_dict={}
        for m_name in m_names:
            deg=self.modeldict['max_poly_deg']
            model=PolyModel(x_train,y_train,deg,model_name=m_name)  
            model.set_test_stats(x_test,y_test)
            self.correction_dict[f'{m_name}']=model
        
        self.uncorrected_test_stats=SeriesCompare(y_test,x_test)
        
        

class CompareCorrect(myLogger):
    def __init__(self,):
        myLogger.__init__(self,'comparecorrect.log')
        self.proc_count=12
        if not os.path.exists('results'):
            os.mkdir('results')
        self.modeldict={
            'sources':{'observed':'nldas','modeled':'cn'}, #[observed,modeled]
            'filter':'nonzero',#'none',#'nonzero'
            'train_share':0.80,
            'split_order':'chronological',#'random'
            'max_poly_deg':5,
            'model_name':['lin-reg','lasso','gbr'],#['lin-reg','lasso']# 'l1' or 'lasso', 'l2' or 'ridge'
            
            
        }
        #self.logger=logging.getLogger(__name__)
        self.comidlist=list(dict.fromkeys(
            pd.read_csv(
                'catchments-list-cleaned.csv')['comid'].to_list())) 
        #keep order, but remove duplicates
        
        self.geog_names=['division','province','section'] #descending size order
        self.expand_geog_names=False #True will append larger geogs names in hierarchy to smaller ones. e.g., to see which province a section is, etc.
        self.physio_path='ecoregions/physio.dbf'
        if not os.path.exists(self.physio_path):
            print(f'cannot locate {self.physio_path}, the physiographic boundary shapefile. download it and unzip it in a folder called "ecoregions".')
        self.states_path='geo_data/states/cb_2017_us_state_500k.dbf'
        
    
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
            
    def plotAccuracyDF(self,model_name='lasso',geography='section'):
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
        clist_df=pd.read_csv('catchments-list-cleaned.csv')
        self.model_accuracy_df=model_accuracy_df
        self.clist_df=clist_df
        
        comid_physio=clist_df.drop('comid',axis=1)
        comid_physio.index=clist_df.loc[:,'comid']
        acc=model_accuracy_df.merge(comid_physio,on='comid')
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
            fig.savefig(f'accuracy_{model_name}_{col}.png')
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
