# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 10:45:41 2016

@author: yuan
"""

import numpy
from scipy.signal import convolve
from scipy.interpolate import interp1d
from scipy.stats import norm
import matplotlib.pyplot as plt

def CC(m, c, s0, K, r, s, T, n):
    c, s0, K, r, s, T = map(float, (c, s0, K, r, s, T))
    dt = T/n
    h = 2*c/m
    x = numpy.linspace(-c, c, num=m+1)

    fR = norm.pdf(x, loc=(r-0.5*s*s)*dt, scale=s*numpy.sqrt(dt))
    fB = numpy.copy(fR)

    for ii in range(n-1):
        fnc = interp1d( x, fB, fill_value='extrapolate' )
        fY = fnc( numpy.log(numpy.exp(x) - 1) )*numpy.exp(x)/(numpy.exp(x) - 1)
        fY[ numpy.invert(numpy.isfinite(fY)) ] = 0
        fB = h*(convolve(fR, fY)[m/2 : 3*m/2+1])

    # change fB; using a new grid with nonzero integrand
    fnc = interp1d( x, fB )
    x = numpy.linspace( numpy.log(K*(n+1)/s0 - 1), c, num=m+1)
    h = x[1] - x[0]
    fB = fnc(x)

    integrand = (s0*(1 + numpy.exp(x))/(n+1) - K)*fB    
    integrand[0] *= 0.5
    integrand[-1] *= 0.5
    
    return numpy.exp(-r*T)*h*integrand.sum()


def Benh(m, c, s0, K, r, s, T, n):
    c, s0, K, r, s, T = map(float, (c, s0, K, r, s, T))
    dt = T/n
    h = 2*c/m
    x = numpy.linspace(-c, c, num=m+1)

    mu = [(r-0.5*s*s)*dt]
    for ii in range(n):
        mu += [mu[0] + numpy.log(1 + numpy.exp(mu[-1]))]
    
    fR = norm.pdf(x, loc=mu[0],   scale=s*numpy.sqrt(dt))
    fD = norm.pdf(x, loc=0,       scale=s*numpy.sqrt(dt))
    
    for ii in range(n-1):
        fnc = interp1d( x, fD, fill_value='extrapolate' )
        fZ = fnc( numpy.log(numpy.exp(x + mu[ii+1]) - 1) - mu[ii] )*numpy.exp(x + mu[ii+1])/(numpy.exp(x + mu[ii+1]) - 1)
        fZ[ numpy.invert(numpy.isfinite(fZ)) ] = 0
        fD = h*(convolve(fR, fZ)[m/2 : 3*m/2+1])

    # change fB; using a new grid with nonzero integrand
    fnc = interp1d( x, fD )
    x = numpy.linspace( numpy.log(K*(n+1)/s0 - 1) - mu[n-1], c, num=m+1)
    h = x[1] - x[0]
    fD = fnc(x)

    integrand = (s0*(1 + numpy.exp(x + mu[n-1]))/(n+1) - K)*fD
    integrand[0] *= 0.5
    integrand[-1] *= 0.5
    
    return numpy.exp(-r*T)*h*integrand.sum()

def BenhCV(m, c, s0, K, r, s, T, n):
    c, s0, K, r, s, T = map(float, (c, s0, K, r, s, T))
    dt = T/n
    h = 2*c/m
    x = numpy.linspace(-c, c, num=m+1)

    mu = [(r-0.5*s*s)*dt]
    for ii in range(n):
        mu += [mu[0] + numpy.log(1 + numpy.exp(mu[-1]))]
    
    mR = mu[0]
    sR = s*numpy.sqrt(dt)
    
    fR = norm.pdf(x, loc=mR, scale=sR )
    fD = norm.pdf(x, loc=0,  scale=sR )
    
    for ii in range(n-1):
        mD = h*(fD*x).sum()
        sD = numpy.sqrt(h*(fD*x*x).sum() - mD*mD)
        cvD = lambda x: norm.pdf(x, loc=mD, scale=sD)
        
        diffD = interp1d( x, fD-cvD(x), fill_value='extrapolate' )
        fZ = (diffD( numpy.log(numpy.exp(x + mu[ii+1]) - 1) - mu[ii] ) + \
                cvD( numpy.log(numpy.exp(x + mu[ii+1]) - 1) - mu[ii] ))*numpy.exp(x + mu[ii+1])/(numpy.exp(x + mu[ii+1]) - 1)
        fZ[ numpy.invert(numpy.isfinite(fZ)) ] = 0
        
        mZ = h*(fZ*x).sum()
        sZ = numpy.sqrt(h*(fZ*x*x).sum() - mZ*mZ)
        cvZ = lambda x: norm.pdf(x, loc=mZ, scale=sZ)
    
        fD = h*(convolve(fR, fZ-cvZ(x))[m/2 : 3*m/2+1]) + norm.pdf(x, loc=mR+mZ,  scale=numpy.sqrt(sR**2 + sZ**2) )
        
    # change fB; using a new grid with nonzero integrand
    fnc = interp1d( x, fD )
    x = numpy.linspace( numpy.log(K*(n+1)/s0 - 1) - mu[n-1], c, num=m+1)
    h = x[1] - x[0]
    fD = fnc(x)

    integrand = (s0*(1 + numpy.exp(x + mu[n-1]))/(n+1) - K)*fD
    integrand[0] *= 0.5
    integrand[-1] *= 0.5
    
    return numpy.exp(-r*T)*h*integrand.sum()


if __name__ == '__main__':
    
    s0 = 100.0
    K = 120.0
    r = 0.1
    s = 0.3
    T = 1.0

    n = 12
    m = 1000

    print CC(m, 5.0, s0, K, r, s, T, n)
    
    m = 102400
    
    print Benh(m, 2.0, s0, K, r, s, T, n)
    print BenhCV(m, 2.0, s0, K, r, s, T, n)
    
    # for m=102400, 
    # Benh gives 2.31345342291
    # BenhCV gives 2.313453335

