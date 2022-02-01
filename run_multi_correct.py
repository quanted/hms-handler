from cn_correct import CompareCorrect as CC
from cn_correct import MultiCorrectionTool as MCT
from warnings import filterwarnings
from traceback import format_exc
import numpy as np

if __name__=='__main__':
    filterwarnings('ignore')
    
    max_deg=5
    l1_list=np.linspace(1,9,10)**2#for elastic-net
    l1_list=list(1-l1_list/(max(l1_list)+1))
    l2_alphas=list(np.logspace(-5,1.4,50))# for ridge
    i=5
    model_spec_list=[
        {f'lasso-{i}':{
            'kwargs':{'max_iter':5000,'tol':1e-6},
            'inner_cv':{'n_repeats':2,'n_splits':10,'n_jobs':1},
            'max_poly_deg':i,'poly_search':False,
            'fit_intercept':False,'n_alphas':50
            } for i in range(1,max_deg+1)},

        {f'ridge-{i}':{
            'kwargs':{'max_iter':5000,'tol':1e-6},
            'inner_cv':{'n_repeats':2,'n_splits':10,'n_jobs':1},
            'max_poly_deg':i,'poly_search':False,
            'fit_intercept':False,'alphas':l2_alphas
            } for i in range(1,max_deg+1)},

        {f'lin_reg-{i}':{
            'max_poly_deg':i,
            'poly_search':False,
            'fit_intercept':False
            } for i in range(1,max_deg+1)},

        {f'elastic_net-{i}':{
            'kwargs':{'max_iter':5000,'tol':1e-6},
            'n_alpha':10,
            'l1_ratio':l1_list,#list(1-np.logspace(-2,-.03,7)),
            'inner_cv':{'n_repeats':2,'n_splits':10,'n_jobs':1},
            'max_poly_deg':i,'poly_search':False,'fit_intercept':False
            } for i in range(1,max_deg+1)},

        {f'gbr-{n}_{l}_{s}_{d}':{
            'kwargs':{
                'n_estimators':n,'learning_rate':l,
                'subsample':s,'max_depth':d
                }
            } for n in [100,200] for l in [0.05,.1] for s in [0.6,0.8] for d in [2,4,8]
        }
    ]





        
    mct=MCT(model_specs=model_spec_list,plot=False)
    mct.runCorrections(load=True )
    mct.buildCorrectionResultsDF()
    mct.selectCorrections()
    mct.setCorrectionSelectionAccuracy()
    mct.setSortOrder()   
    mct.plotCorrectionRunoffComparison()
    mct.plotCorrectionRunoffComparison(split_zero=False)
    mct.plotCorrectionRunoffComparison(sort=True)
    #mct.plotCorrectionRunoffComparison(time_range=slice(-365,None))
    mct.plotCorrectionRunoffComparison(split_zero=False,time_range=slice(-365,None))
    #mct.plotCorrectionRunoffComparison(sort=True,time_range=slice(-365,None))
    #mct.plotCorrectionRunoffComparison(sort=True,time_range=slice(-365,None),split_zero=False)
    mct.saveCorrectionSelectionTable() 
    mct.plotCorrectionResultLines()
    mct.plotGeogHybridAccuracy(plot_negative=False)
            
    """{'stackingregressor':{
            'stack_specs':[
                {'lasso':{'max_poly_deg':5,'fit_intercept':False}},
                #{'lassolars':{'max_poly_deg':5,'fit_intercept':False,'inner_cv':{'n_repeats':10,'n_splits':10,'n_jobs':1}}},
                {'gbr':{'kwargs':{},
                        #'inner_cv':{'n_repeats':10,'n_splits':10,'n_jobs':1},
                        #'param_grid':{
                        #    'n_estimators':[100,400],'subsample':[1,0.8],'max_depth':[3,4]}#these pass through to sklearn's gbr
                        #'n_estimators':10000,
                        #'subsample':1,
                        #'max_depth':3
                        }},
                #{'lin-reg':{'max_poly_deg':5,'fit_intercept':False}},
                ]
        }}"""
    

    
