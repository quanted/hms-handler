from cn_correct import CompareCorrect as CC
from warnings import filterwarnings
from traceback import format_exc
import numpy as np

if __name__=='__main__':
    filterwarnings('ignore')
    l1_list=np.linspace(1,9,10)**2
    l1_list=list(1-l1_list/(max(l1_list)+1))
    model_spec_list=[
        {'elastic-net':{
            'l1_ratio':l1_list,#list(1-np.logspace(-2,-.03,7)),
            'inner_cv':{'n_repeats':2,'n_splits':10,'n_jobs':1},
            'max_poly_deg':5,'fit_intercept':False}
            },
        {'lasso':{'inner_cv':{'n_repeats':2,'n_splits':10,'n_jobs':1},
                  'max_poly_deg':5,'fit_intercept':False,
                 'n_alphas':200
                 }
            },
        {'lin-reg':{'inner_cv':{'n_repeats':2,'n_splits':10,'n_jobs':1},
                    'max_poly_deg':5,'fit_intercept':False}},
        {'gbr':{
            'kwargs':{},
            'inner_cv':{'n_repeats':2,'n_splits':10,'n_jobs':1},
            'param_grid':{
                'ccp_alpha':[0.001,0.005,0.01],
                'n_estimators':[50,100,400],
                'learning_rate':[0.05,0.025],
                'subsample':[0.7,1],'max_depth':[2,3]}#these pass through to sklearn's gbr
                }
            },
        ]
    for model_spec in model_spec_list:
        try:
            cc=CC(model_specs=model_spec)
            print('modeldict',cc.modeldict)
            cc.runBigModel()
            cc.plotGeoTestData(plot_negative=False)
            cc.plotGeoTestData(plot_negative=True)
            print('complete')
        except:
            print(format_exc())
            
            
            
            
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
    

    
