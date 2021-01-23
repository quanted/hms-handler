from cn_correct import CompareCorrect as CC
from warnings import filterwarnings

if __name__=='__main__':
    filterwarnings('ignore')
    cc=CC()
    print('modeldict',cc.modeldict)
    cc.runBigModel()
    print('complete')
    

    
