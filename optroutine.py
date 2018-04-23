#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 23:07:14 2018

@author: Zhan
"""

import numpy as np
from enum import Enum
from cvxopt.solvers import coneqp
from cvxopt import matrix, solvers


class OptExitType(Enum):
    running = 'running'
    maxiter = 'MaximumIterationReached'
    earlystop = 'earlystopped'
    epsilonstop = 'epsilonreached'
    
class OptStatus:
    
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
                               
                if np.linalg.norm(xnew-xold,2)**2 < epsilon:
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
            
            
def l2projection(theta0,center,c):
    """
    projection onto l2 ball with radius sqrt(c). i.e.:
        argmin ||u-theta0|| st ||u - center||^2 <= c
    Input:
        theta0: original variable before projection
        center: center of l2 ball
        c: constraint upper bound, sqrt(c) is radius of l2 ball
    """
    if np.shape(center) != np.shape(theta0):
        raise ValueError("dimension of theta and center must match")
    theta_proj =  c**(0.5) * (theta0-center)/max(c**(0.5), np.linalg.norm(theta0-center,2)) + center
    return theta_proj


def ellipsoidprojection(theta0,center,weights,c):
    """
    projection onto ellipsoid defined by a diagonal quadratic function
        argmin ||u-theta0|| st (u-center)' * diag(weights) * (u-center) <= c  
    
    """
    if not np.shape(theta0)[0] == np.shape(center)[0] == np.shape(weights)[0]:
        raise ValueError("dimension of theta, center and diagonal weights must match")
        
    if np.sum(weights*(theta0-center)**2) > c:
        # outside ellipsoid, projection
        weights_inv = 1/weights
        weights_inv_sqrt = weights_inv**(0.5)
        b = theta0 - center
        epsilon = 10^(-6)
        theta_proj = np.array(theta0)
        stepsize = 1/max(weights_inv)
        f = -weights_inv_sqrt * b
    
        for t in range(100):            
            thetaold = np.array(theta_proj)
            thetanew = thetaold- stepsize * (weights_inv * thetaold + f) 
            theta_proj = l2projection(thetanew,np.zeros(thetanew.shape),c)
            if np.linalg.norm(theta_proj-thetaold,2)**2 <= epsilon:
                break
            
        theta_proj = weights_inv_sqrt * theta_proj + center
    else:
        # inside ellipsoid
        theta_proj = theta0
    return theta_proj
    
#examples:
n=4
Q = np.random.rand(n)    
center = np.random.rand(n)   
x0 = np.random.rand(n)   

xproj=ellipsoidprojection(x0,center,Q,0.05)
l2xproj = l2projection(x0,center,0.05)
print(l2xproj)
print(xproj)
            
            
            
        
        
                
            
            
            