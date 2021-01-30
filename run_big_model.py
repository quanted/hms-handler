from cn_correct import CompareCorrect as CC
from warnings import filterwarnings
from traceback import format_exc

if __name__=='__main__':
    filterwarnings('ignore')
    model_spec_list=[
        {'gbr':{
            'kwargs':{},#these pass through to sklearn's gbr
                #'n_estimators':10000,
                #'subsample':1,
                #'max_depth':3
                }},
        {'lin-reg':{'max_poly_deg':5,'fit_intercept':False}},
        {'lasso':{'max_poly_deg':5,'fit_intercept':False}},
        
        
    ]
    for model_spec in model_spec_list:
        try:
            cc=CC()
            cc.modeldict['model_specs']=model_spec
            print('modeldict',cc.modeldict)
            cc.runBigModel()
            cc.plotGeoTestData()
            print('complete')
        except:
            print(format_exc())
            
    
    

    
