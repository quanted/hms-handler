import os
import re
from multiprocessing import Process,Queue
from mylogger import myLogger
from time import time,sleep
import numpy as np
import pandas as pd


class MpWrapper(Process,myLogger):
    def __init__(self,q,jobq,run_object):#,pass_args=[],pass_kwargs={}):
        super().__init__()
        self.q=q;
        self.jobq=jobq
        #self.i=i
        self.run_object=run_object
        #self.pass_args=pass_args
        #self.pass_kwargs=pass_kwargs
        
    def run(self):
        myLogger.__init__(self,'mpwrapper.log')
        while True:
            job=self.jobq.get()
            if type(job) is str: break
            the_obj=self.run_object(*job['args'],**job['kwargs'])
            the_obj.run()
            self.q.put((job['i'],the_obj))

            
class MpHelper(myLogger):   
    def __init__(self):
        myLogger.__init__(self,name='mphelper.log')
        
            
    def runAsMultiProc(self,mp_object,args_list,proc_count=4,kwargs_list={},):
        try:
            starttime=time()
            q=Queue()
            jobq=Queue()
            I=len(args_list)
            if type(kwargs_list) is dict:
                kwargs_list=[kwargs_list]*I
            #q_args_list=[[q,i,*args_list[i]] for i in range(I)]
            for i in range(I):
                jobq.put({'i':i,'args':args_list[i],'kwargs':kwargs_list[i]})
            #i_todo_list=list(range(I))
            outlist=['empty' for _ in range(I)]
            procs=[]
            for _ in range(proc_count):
                procs.append(MpWrapper(q,jobq,mp_object))
                procs[-1].start()
            countdown=I
        except:
            self.logger.exception('error in runasmultiproc')
            assert False,'unexpected error'
        pct_complete=0
        while countdown:
            try:
                self.logger.info(f'multiproc checking q. countdown:{countdown}')
                try:
                    i,result=q.get(True,20)
                except:
                    if not q.empty():
                        self.logger.exception(f'q not empty, but error encountered ')
                    continue
                self.logger.info(f'multiproc has something from the q!')
                outlist[i]=result
                countdown-=1
                completion=100*(I-countdown)/I
                if completion-pct_complete>10:
                    pct_complete=completion
                    self.logger.info(f'{pct_complete}%')
                '''procs[i].terminate()
                procs[i].join()
                if i_todo_list: #start the next process
                    ii=i_todo_list.pop(-1)
                    procs[ii]=MpWrapper(
                        q,ii,mp_object,
                        pass_args=args_list[ii],pass_kwargs=kwargs_list[ii])
                    procs[ii].start()
                '''
                self.logger.info(f'proc completed. countdown:{countdown}')
            except:
                #self.logger.exception('error')
                self.logger.exception('unexpected error')
                assert False, 'halt'
        [jobq.put('close') for _ in procs]
        [proc.join() for proc in procs]
        q.close()
        jobq.close()                             
        self.logger.info(f'all procs joined sucessfully')
        endtime=time()
        self.logger.info(f'pool complete at {endtime}, time elapsed: {(endtime-starttime)/60} minutes')
        return outlist
    