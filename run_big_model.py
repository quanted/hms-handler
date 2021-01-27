from cn_correct import CompareCorrect as CC
from warnings import filterwarnings

if __name__=='__main__':
    filterwarnings('ignore')
    model_specs_list=[
        {'lin-reg':{'max_poly_deg':4,'fit_intercept':False}},
        #{'lasso':{'max_poly_deg':3,'fit_intercept':False}},
        {'gbr':{'kwargs':{}}}]
    for model_specs in model_specs_list:
        cc=CC()
        cc.modeldict['model_specs']=model_specs    
        print('modeldict',cc.modeldict)
        cc.runBigModel()
    print('complete')
    
    

    
