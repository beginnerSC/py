# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 19:38:44 2016

@author: yuan
"""

import numpy
from scipy.signal import convolve
from scipy.interpolate import interp1d
from scipy.stats import norm
import matplotlib.pyplot as plt


def DOput(m, c, s0, K, r, s, T, H, n):
    '''
    down and out barrier put option. H < s0
    '''
    c, s0, K, r, s, T, H = map(float, (c, s0, K, r, s, T, H))
    dt = T/n
        
    # convolve implies the use of the midpoint rule. 
    # to obtain second order convergence, pick an h such that log(H/s0) is (n + 0.5)h

    h = 2*c/m
    #h = numpy.abs(numpy.log(H/s0))/(round(numpy.abs(numpy.log(H/s0))/h - 0.5) + 0.5)
    
    x = h*numpy.arange(-m/2, m/2+1)

    payoff = map(lambda x: max(K - s0*numpy.exp(x), 0), x)
    put = numpy.copy(payoff)
    f = norm.pdf(x, loc=-(r-0.5*s*s)*dt, scale=s*numpy.sqrt(dt))

    for ii in range(n):
        #if ii is not 0:
            #plt.plot(x, put)
        put[ x < numpy.log(H/s0) ] = 0
        put = numpy.exp(-r*dt)*h*(convolve(put, f)[m/2 : 3*m/2+1])

    putFnc = interp1d( x, put )
    return putFnc(0) if s0>H else 0


def DOputCV(m, c, s0, K, r, s, T, H, n):
    '''
    down and out barrier put option. H < s0
    '''
    c, s0, K, r, s, T, H = map(float, (c, s0, K, r, s, T, H))
    dt = T/n
        
    # convolve implies the use of the midpoint rule. 
    # to obtain second order convergence, pick an h such that log(H/s0) is (n + 0.5)h
    h = 2*c/m
    #h = numpy.abs(numpy.log(H/s0))/(round(numpy.abs(numpy.log(H/s0))/h - 0.5) + 0.5)
    
    x = h*numpy.arange(-m/2, m/2+1)

    payoff = map(lambda x: max(K - s0*numpy.exp(x), 0), x)
    put = numpy.copy(payoff)
    
    mf = -(r-0.5*s*s)*dt
    sf = s*numpy.sqrt(dt)

    f = norm.pdf(x, loc=mf, scale=sf)

    # before the first iteration put doesn't look like a bell shaped
    put[ x < numpy.log(H/s0) ] = 0
    put = numpy.exp(-r*dt)*h*(convolve(put, f)[m/2 : 3*m/2+1])

    for ii in range(n-1):
        
        sc = h*put.sum()
        mp = h*(put*x).sum()/sc
        sp = numpy.sqrt(h*(put*x*x).sum()/sc - mp*mp)
        
        cv = sc*norm.pdf(x, loc=mp, scale=sp)
        
        diff = put - cv
        
        #plt.plot(x, diff)
        
        diff[ x < numpy.log(H/s0) ] = 0

        tmps = numpy.sqrt(sp**2 + sf**2)
        put = numpy.exp(-r*dt)*(h*(convolve(diff, f)[m/2 : 3*m/2+1]) \
            + sc*norm.pdf(x, loc=(mp+mf), scale=tmps) \
                *(1 - norm.cdf( ((numpy.log(H/s0) - mp)*sf/sp + (numpy.log(H/s0)-x + mf)*sp/sf)/tmps )))
        
    putFnc = interp1d( x, put )
    return putFnc(0) if s0>H else 0


def UOcall(m, c, s0, K, r, s, T, H, n):
    '''
    down and out barrier put option. H < s0
    '''
    c, s0, K, r, s, T, H = map(float, (c, s0, K, r, s, T, H))
    dt = T/n
        
    # convolve implies the use of the midpoint rule. 
    # to obtain second order convergence, pick an h such that log(H/s0) is (n + 0.5)h

    h = 2*c/m
    h = numpy.abs(numpy.log(H/s0))/(round(numpy.abs(numpy.log(H/s0))/h - 0.5) + 0.5)
    h = numpy.abs(numpy.log(H/s0))/(round(numpy.abs(numpy.log(H/s0))/h ))
    
    x = h*numpy.arange(-m/2, m/2+1)

    payoff = map(lambda x: max(s0*numpy.exp(x) - K, 0), x)
    call = numpy.copy(payoff)
    f = norm.pdf(x, loc=-(r-0.5*s*s)*dt, scale=s*numpy.sqrt(dt))

    for ii in range(n):
        #if ii is not 0:
            #plt.plot(x, put)
        call[ x > numpy.log(H/s0) ] = 0
        call = numpy.exp(-r*dt)*h*(convolve(call, f)[m/2 : 3*m/2+1])

    callFnc = interp1d( x, call )
    return callFnc(0) if s0<H else 0


def UOcallCV(m, c, s0, K, r, s, T, H, n):
    '''
    down and out barrier put option. H < s0
    '''
    c, s0, K, r, s, T, H = map(float, (c, s0, K, r, s, T, H))
    dt = T/n
        
    # convolve implies the use of the midpoint rule. 
    # to obtain second order convergence, pick an h such that log(H/s0) is (n + 0.5)h
    h = 2*c/m
    h = numpy.abs(numpy.log(H/s0))/(round(numpy.abs(numpy.log(H/s0))/h - 0.5) + 0.5)
    h = numpy.abs(numpy.log(H/s0))/(round(numpy.abs(numpy.log(H/s0))/h ))
    
    x = h*numpy.arange(-m/2, m/2+1)

    payoff = map(lambda x: max(s0*numpy.exp(x) - K, 0), x)
    call = numpy.copy(payoff)
    
    mf = -(r-0.5*s*s)*dt
    sf = s*numpy.sqrt(dt)

    f = norm.pdf(x, loc=mf, scale=sf)

    # before the first iteration put doesn't look like a bell shaped
    call[ x > numpy.log(H/s0) ] = 0
    call = numpy.exp(-r*dt)*h*(convolve(call, f)[m/2 : 3*m/2+1])

    for ii in range(n-1):
        
        sc = h*call.sum()
        mp = h*(call*x).sum()/sc
        sp = numpy.sqrt(h*(call*x*x).sum()/sc - mp*mp)
        
        cv = sc*norm.pdf(x, loc=mp, scale=sp)
        
        diff = call - cv
        
        #plt.plot(x, diff)
        
        diff[ x > numpy.log(H/s0) ] = 0

        tmps = numpy.sqrt(sp**2 + sf**2)
        call = numpy.exp(-r*dt)*(h*(convolve(diff, f)[m/2 : 3*m/2+1]) \
            + sc*norm.pdf(x, loc=(mp+mf), scale=tmps) \
                *(1 - norm.cdf( -((numpy.log(H/s0) - mp)*sf/sp + (numpy.log(H/s0)-x + mf)*sp/sf)/tmps )))
        
    callFnc = interp1d( x, call )
    return callFnc(0) if s0<H else 0



if __name__ == '__main__':
    
    s0 = 100
    K = 120
    r = 0.1
    s = 0.3
    T = 1
    H = 90

    n = 12
    m = 10000
    
    print DOput(m, 2.0, s0, K, r, s, T, H, n)
    print DOputCV(m, 2.0, s0, K, r, s, T, H, n)
    
    
    H = 130
    
    print UOcall(m, 2.0, s0, K, r, s, T, H, n)
    print UOcallCV(m, 2.0, s0, K, r, s, T, H, n)
    
    print UOcall(102400, 2, s0, K, r, s, T, H, n)

''' 
    s0 = 100
    K = 120
    r = 0.1
    s = 0.3
    T = 1
    H = 90

    n = 4
    m = 100000   
    
    DOput   gives 2.95308508889
    DOputCV gives 2.95308502782
'''