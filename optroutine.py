#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 23:07:14 2018

@author: Zhan
"""

import numpy as np
from enum import Enum

class OptExitType(Enum):
    running = 'running'
    maxiter = 'MaximumIterationReached'
    earlystop = 'earlystopped'
    epsilonstop = 'epsilonreached'
    
class OptStatus(OptExitType):
    
    def __init__(self):
        self.xhistory = np.array([])
        self.fval = np.array([])
        self.exitflag = OptExitType.running
        self.epochcount = 0
        
class SGD:
    
    def __init__(self,fobj,fgrad, feature, label):
        self.fgrad = fgrad
        self.fobj = fobj
        self.feature = feature
        self.label = label
        self.n = label.shape[0]
        if feature.shape[0] != self.n:
            raise ValueError("number of data in x and label does not match")
        self.optstatus = OptStatus()
        
    def iteratesSGD(self, x0, stepsize, maxepochs, epochsize=None,
                    batchsize=1, epsilon = 10**(-10), earlystop=False, 
                    constraintfunc=None, c=None):
        
        if epochsize is None:
            epochsize = self.n
              
        self.optstatus.xhistory = np.zeros((np.shape(x0)[0],maxepochs+1))
        self.optstatus.xhistory[:] = np.nan
        xnew = np.array(x0)
        for t in range(maxepochs):
            self.optstatus.xhistory[:,t] = xnew
            # reduce stepsize each epoch
            stepsize = stepsize/(1+t);
            for i in range(epochsize):
                # one epoch
                xold = np.array(xnew)
                grad = np.zeros(np.shape(x0))
                for b in range(batchsize):
                    # one batch of gradient
                    sampleID = np.random.randint(self.n)
                    sample = np.concatenate(([self.label[sampleID]],self.feature[sampleID,:]))
                    grad += self.fgrad(xold,sample)
                xnew = xold - stepsize * grad;
                               
                if np.abs(xnew-xold) < epsilon:
                    # regular termination,change smaller than epsilon
                    self.optstatus.exitflag = OptExitType.epsilonstop
                    break
              
            self.optstatus.epochcount += 1
            if self.optstatus.exitflag != OptExitType.running:
                break
            # early stop                
            if constraintfunc(xnew) > c:
                self.optstatus.exitflag = OptExitType.earlystop
                break            

        # if maximum iteration reached 
        if self.optstatus.exitflag == OptExitType.running:
            self.optstatus.exitflag = OptExitType.maxiter
        
        # projection if early stop triggered
        if self.optstatus.exitflag == OptExitType.earlystop:
            #TODO: projection
            projection = 0
            
            
            
        
        
                
            
            
            