# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 19:38:44 2016

@author: yuan
"""

import numpy
from scipy.signal import convolve
from scipy.interpolate import interp1d
from scipy.stats import norm
from scipy.special import ndtr
import matplotlib.pyplot as plt


def Bermudan(m, c, s0, K, r, s, T, n):
    c, s0, K, r, s, T = map(float, (c, s0, K, r, s, T))
    dt = T/n
    h = 2*c/m
    x = numpy.linspace(-c, c, num=m+1)

    payoff = map(lambda x: max(K - s0*numpy.exp(x), 0), x)
    put = numpy.copy(payoff)
    f = norm.pdf(x, loc=-(r-0.5*s*s)*dt, scale=s*numpy.sqrt(dt))

    for ii in range(n):
        put = numpy.exp(-r*dt)*h*(convolve(put, f)[m/2 : 3*m/2+1])
        put = numpy.maximum(put, payoff)
        
    putFnc = interp1d( x, put )
    return putFnc(0)


def BSCV(m, c, s0, K, r, s, T, n):
    c, s0, K, r, s, T = map(float, (c, s0, K, r, s, T))
    dt = T/n
    h = 2*c/m
    x = numpy.linspace(-c, c, num=m+1)

    payoff = map(lambda x: max(K - s0*numpy.exp(x), 0), x)
    put = numpy.copy(payoff)
    f = norm.pdf(x, loc=-(r-0.5*s*s)*dt, scale=s*numpy.sqrt(dt))

    for ii in range(n):
        
        #cv = map(lambda x: max(K-s0*numpy.exp(x), 0), x) if ii==0 else BSputLogSc(x, s0, K, r, s, ii*dt)
        cv = payoff if ii==0 else BSputLogSc(x, s0, K, r, s, ii*dt)
                
        put = numpy.exp(-r*dt)*h*(convolve(put - cv, f)[m/2 : 3*m/2+1]) \
                + BSputLogSc(x, s0, K, r, s, (ii+1)*dt)
                
        put = numpy.maximum(put, payoff)
        
    putFnc = interp1d( x, put )
    return putFnc(0)


def expCV(m, c, s0, K, r, s, T, n):
    c, s0, K, r, s, T = map(float, (c, s0, K, r, s, T))
    dt = T/n
    h = 2*c/m
    x = numpy.linspace(-c, c, num=m+1)

    payoff = numpy.array(map(lambda x: max(K - s0*numpy.exp(x), 0), x))
    put = numpy.copy(payoff)
    f = norm.pdf(x, loc=-(r-0.5*s*s)*dt, scale=s*numpy.sqrt(dt))

    for ii in range(n):
        
        if ii==0:
            cv = payoff
        else:
            # coefficients that make cv smooth
            #c2 = -s0*numpy.exp(xstar)/(K-s0*numpy.exp(xstar))
            #c1 = (K-s0*numpy.exp(xstar))*numpy.exp(-c2*xstar)            

            # coefficients that make diff smooth
            c2 = (numpy.log(put[ind+1]) - numpy.log(put[ind]))/(x[ind+1] - x[ind])
            c1 = put[ind]*numpy.exp(-c2*x[ind])

            cv =  numpy.concatenate((payoff[x < xstar], c1*numpy.exp(c2*x)[x >= xstar]))
            
        put = numpy.exp(-r*dt)*h*(convolve(put - cv, f)[m/2 : 3*m/2+1])
        
        if ii==0:
            put += BSputLogSc(x, s0, K, r, s, dt)
        else:
            put += cvConv(x, s0, K, xstar, r, s, dt) + numpy.exp(-r*dt)*c1*numpy.exp(c2*(x + (r-(1-c2)*0.5*s*s)*dt))*ndtr((x + c2*s*s*dt + (r-0.5*s*s)*dt - xstar)/(s*numpy.sqrt(dt)))

        ind = numpy.argwhere(put > payoff)[0, 0]
        a = payoff[ind-1] - put[ind-1]
        b = put[ind] - payoff[ind]
        xstar = x[ind-1] + h*a/(a+b)
        
        put = numpy.maximum(put, payoff)
        
    putFnc = interp1d( x, put )
    return putFnc(0)


def BSputLogSc(x, s0, K, r, s, tau):
    '''
    x = log S/K, can be a vector
    '''
    d1 = (x + numpy.log(s0/K) + (r + 0.5*s*s)*tau)/(s*numpy.sqrt(tau))
    d2 = d1 - s*numpy.sqrt(tau)
    return -s0*numpy.exp(x)*ndtr(-d1) + K*numpy.exp(-r*tau)*ndtr(-d2)


def cvConv(x, s0, K, xstar, r, s, dt):
    '''
    x = log S/K, can be a vector
    '''
    d1 = (x - xstar + (r + 0.5*s*s)*dt)/(s*numpy.sqrt(dt))
    d2 = d1 - s*numpy.sqrt(dt)
    return -s0*numpy.exp(x)*ndtr(-d1) + K*numpy.exp(-r*dt)*ndtr(-d2)


if __name__ == '__main__':
    
    s0 = 100
    K = 100
    r = 0.1
    s = 0.3
    T = 1

    n = 12
    m = 1000
    
    print Bermudan(m, 2.0, s0, K, r, s, T, n)    
    print BSCV(m, 2.0, s0, K, r, s, T, n)
    print expCV(m, 2.0, s0, K, r, s, T, n)
    
    
    #print Bermudan(102400, 2.0, s0, K, r, s, T, n)