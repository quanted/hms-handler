from cn_correct import CompareCorrect as CC
from warnings import filterwarnings
from traceback import format_exc

if __name__=='__main__':
    filterwarnings('ignore')
    model_spec_list=[
        'stackingregressor':{
            'stack_specs':[
                {'lasso':{'max_poly_deg':5,'fit_intercept':False,'inner_cv':{'n_repeats':1,'n_splits':10,'n_jobs':1}}},
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
        }
        ]
    for model_spec in model_spec_list:
        try:
            cc=CC(model_specs=model_spec)
            print('modeldict',cc.modeldict)
            cc.runBigModel()
            cc.plotGeoTestData()
            print('complete')
        except:
            print(format_exc())
            
    
    

    
