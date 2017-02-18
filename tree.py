# -*- coding: utf-8 -*-
"""
Created on Wed Sep 07 04:46:30 2016

@author: yuan
"""

import numpy
from scipy.special import ndtr

def CRR(n, s0, K, r, s, T):
    
    s0, K, r, s, T = map(float, (s0, K, r, s, T))    
    
    dt = T/n
    u = numpy.exp(s*numpy.sqrt(dt))
    d = 1.0/u
    p = (numpy.exp(r*dt) - d)/(u-d)
    
    payoff = lambda nn: numpy.maximum(K-s0*numpy.logspace(-nn, nn, num=nn+1, base=d), numpy.zeros(nn+1))
    put = payoff(n)
    
    for ii in numpy.arange(n, 0, -1):
        put = numpy.exp(-r*dt)*(p*numpy.delete(put, -1) + (1-p)*numpy.delete(put, 0))
        put = numpy.maximum(put, payoff(ii-1))
        
    return put[0]

def HW(n, s0, K, r, s, T):
    
    s0, K, r, s, T = map(float, (s0, K, r, s, T))    
    
    dt = T/n
    u = numpy.exp(s*numpy.sqrt(dt))
    d = 1.0/u
    p = (numpy.exp(r*dt) - d)/(u-d)
    
    payoff = lambda nn: numpy.maximum(K-s0*numpy.logspace(-nn, nn, num=nn+1, base=d), numpy.zeros(nn+1))
    put = payoff(n)    
    for ii in numpy.arange(n, 0, -1):
        put = numpy.exp(-r*dt)*(p*numpy.delete(put, -1) + (1-p)*numpy.delete(put, 0))
        put = numpy.maximum(put, payoff(ii-1))
    
    putA = put[0]
    
    put = payoff(n)
    for ii in numpy.arange(n, 0, -1):
        put = numpy.exp(-r*dt)*(p*numpy.delete(put, -1) + (1-p)*numpy.delete(put, 0))
    
    putBS = put[0]
    
    return putA - putBS + BSput(s0, K, r, s, T)

def CVslow(n, s0, K, r, s, T):
    
    s0, K, r, s, T = map(float, (s0, K, r, s, T))
    
    dt = T/n
    u = numpy.exp(s*numpy.sqrt(dt))
    d = 1.0/u
    p = (numpy.exp(r*dt) - d)/(u-d)
    
    payoff = lambda nn: numpy.maximum(K-s0*numpy.logspace(-nn, nn, num=nn+1, base=d), numpy.zeros(nn+1))
    sgrid =  lambda nn: s0*numpy.logspace(-nn, nn, num=nn+1, base=d)
    put = payoff(n)
    
    for ii in numpy.arange(n, 0, -1):
        put = numpy.maximum(put, payoff(ii))
        put -= map(lambda ss0: BSput(ss0, K, r, s, (n-ii)*dt), sgrid(ii))
        put = numpy.exp(-r*dt)*(p*numpy.delete(put, -1) + (1-p)*numpy.delete(put, 0))
        put += map(lambda ss0: BSput(ss0, K, r, s, (n-ii+1)*dt), sgrid(ii-1))
        
    return put[0]

def CV(n, s0, K, r, s, T):
    
    s0, K, r, s, T = map(float, (s0, K, r, s, T))
    
    dt = T/n
    u = numpy.exp(s*numpy.sqrt(dt))
    d = 1.0/u
    p = (numpy.exp(r*dt) - d)/(u-d)
    
    payoff = lambda nn: numpy.maximum(K-s0*numpy.logspace(-nn, nn, num=nn+1, base=d), numpy.zeros(nn+1))
    
    ss = lambda n, i: s0*(d**(-n + 2*i))
    ind = payoff(n).nonzero()[0].min()  # index of the 1st node in the early-exercise region
    
    premium = numpy.zeros(ind+1)

    for ii in numpy.arange(n, 0, -1):    
        premium[ind:] = map(lambda jj: K - ss(ii, jj) - BSput(ss(ii, jj), K, r, s, (n-ii)*dt), range(ind, len(premium)))
                
        if ind <= ii:
            ind -= 1
            while True:
                if len(premium) <= ind+1:
                    premium = numpy.append(premium, K - ss(ii, ind+1) - BSput(ss(ii, ind+1), K, r, s, (n-ii)*dt))
                next_prem = numpy.exp(-r*dt)*(p*premium[ind] + (1-p)*premium[ind+1])
                if next_prem > K - ss(ii-1, ind) - BSput(ss(ii-1, ind), K, r, s, (n-ii+1)*dt): 
                    ind += 1
                else: 
                    break
        
        premium = numpy.exp(-r*dt)*(p*numpy.delete(premium, -1) + (1-p)*numpy.delete(premium, 0))
            
    return premium[0] + BSput(s0, K, r, s, T)

def BSput(s0, K, r, s, T):
    d1 = (numpy.log(float(s0)/float(K))+(r + 0.5*s*s)*T)/(s*numpy.sqrt(T))
    d2 = d1 - s*numpy.sqrt(T)
    return max(K-s0, 0) if numpy.isclose(T, 0) else -s0*ndtr(-d1)+K*numpy.exp(-r*T)*ndtr(-d2)

if __name__ == '__main__':
    # proof that treeCVslow gives the same result as treeCVfast
    numStep = 40
    print (treeCVslow(numStep, 100, 100, 0.1, 0.3, 1) - treeCVfast(numStep, 100, 100, 0.1, 0.3, 1))