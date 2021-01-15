import os
import re
from multiprocessing import Process,Queue
from mylogger import myLogger
from time import time,sleep
import numpy as np
import pandas as pd


class MpWrapper(Process,myLogger):
    def __init__(self,q,i,run_object,pass_args=[],pass_kwargs={}):
        super().__init__()
        self.q=q;self.i=i
        self.run_object=run_object
        self.pass_args=pass_args
        self.pass_kwargs=pass_kwargs
        
    def run(self):
        myLogger.__init__(self,'mpwrapper.log')
        the_obj=self.run_object(*self.pass_args,**self.pass_kwargs)
        the_obj.run()
        self.q.put((self.i,the_obj))
        
            
class MpHelper(myLogger):   
    def __init__(self):
        myLogger.__init__(self,name='mphelper.log')
        
            
    def runAsMultiProc(self,mp_object,args_list,proc_count=4,kwargs_list={},):
        try:
            starttime=time()
            q=Queue()
            I=len(args_list)
            if type(kwargs_list) is dict:
                kwargs_list=[kwargs_list]*I
            #q_args_list=[[q,i,*args_list[i]] for i in range(I)]
            i_todo_list=list(range(I))
            outlist=[None for _ in range(I)]
            procs={}
            for i in range(I):
                process=MpWrapper(
                    q,i,mp_object,
                    pass_args=args_list[i],pass_kwargs=kwargs_list[i])
                procs[i]=process
            for _ in range(proc_count):
                i=i_todo_list.pop()
                procs[i].start()
            countdown=I
        except:
            self.logger.exception('error in runasmultiproc')
            assert False,'unexpected error'
        while countdown:
            try:
                
                self.logger.info(f'multiproc checking q. countdown:{countdown}')
                i,result=q.get(True,20)
                self.logger.info(f'multiproc has something from the q!')
                outlist[i]=result
                countdown-=1
                procs[i].join()
                if i_todo_list:
                   ii=i_todo_list.pop()
                   procs[ii].start()
                
                
                self.logger.info(f'proc completed. countdown:{countdown}')
            except:
                #self.logger.exception('error')
                if not q.empty(): self.logger.exception(f'error while checking q, but not empty')
                else: sleep(1)
        #[proc.join() for proc in procs]
        q.close()
        self.logger.info(f'all procs joined sucessfully')
        endtime=time()
        self.logger.info(f'pool complete at {endtime}, time elapsed: {(endtime-starttime)/60} minutes')
        return outlist
    