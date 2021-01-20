from cn_correct import CompareCorrect as CC
from warnings import filterwarnings

if __name__=='__main__':
    filterwarnings('ignore')
    cc=CC()
    cc.modeldict['filter']='none'
    print('modeldict',cc.modeldict)
    cc.runModelCorrection(try_load=False)
    print('complete')
    cc.modeldict['filter']='nonzero'
    print('modeldict',cc.modeldict)
    cc.runModelCorrection(try_load=False)
    print('complete')

    
