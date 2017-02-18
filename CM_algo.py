# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 18:23:26 2016

@author: yuan
"""

import math, cmath
import numpy
import fnc
from scipy.special import ndtr
from scipy.stats import norm


class equityModel:
    
    def __init__(self, *arg):
        pass
    
    def setIR(self, r):
        self.r = float(r)
    
    def chf(self, u, T):
        '''
        chf of log(sT/s0)
        '''
        pass
    

class JD(equityModel):
    
    def __init__(self, alpha, beta, lambdaa, sigma):

        self.alpha = alpha
        self.beta = beta
        self.lambdaa = lambdaa
        self.sigma = sigma
        
    def cumulants(self, T):
        
        T = float(T)
        r = self.r
        a = self.alpha
        b = self.beta
        ll = self.lambdaa
        s = self.sigma
        
        return  (r - ll*(numpy.exp(a + 0.5*b*b) - 1 - a) - 0.5*s*s)*T, \
                s*s*T + ll*T*(a*a + b*b), \
                ll*T*a*(a*a + 3*b*b), \
                ll*T*(a**4 + 6*a**2*b**2 + 3*b**4)
    
    def vol(self):
        '''
        sd of time 1 distribution of log(sT/s0)
        equalent to sigma in BS model
        '''
        return numpy.sqrt(self.cumulants(1)[1])
    
    def chf(self, u, T):

        T = float(T)
        r = self.r
        a = self.alpha
        b = self.beta
        ll = self.lambdaa
        s = self.sigma
        
        return numpy.exp( \
            1j*u*(r - ll*(numpy.exp(a + 0.5*b*b)-1) - 0.5*s*s)*T - 0.5*s*s*u*u*T \
            + ll*T*(numpy.exp(1j*a*u - 0.5*b*b*u*u) - 1) \
        )

    def call(self, s0, K, T, n=10):

        s0, K, T = map(float, (s0, K, T))
        
        r = self.r
        alpha = self.alpha
        beta = self.beta
        lambdaa = self.lambdaa
        sigma = self.sigma
        
        price = 0;
        fac_k = 1;  # k!
        mu = r - lambdaa*(numpy.exp(alpha + 0.5*beta*beta) - 1);

        for k in range(n):
        
            price += numpy.exp(-lambdaa*T)*(lambdaa*T)**k/fac_k *           \
                (s0*numpy.exp(mu*T + k*(alpha + 0.5*beta*beta))*ndtr(       \
                    (numpy.log(s0/K) + (mu + 0.5*sigma*sigma)*T + k*(alpha + beta*beta))  \
                    /numpy.sqrt(sigma*sigma*T + beta*beta*k))               \
                - K*ndtr(                                                   \
                    (numpy.log(s0/K) + (mu - 0.5*sigma*sigma)*T + k*alpha)  \
                    /numpy.sqrt(sigma*sigma*T + beta*beta*k)));

            fac_k *= (k+1);
    
        return numpy.exp(-r*T)*price;
        
        
class BS(equityModel):
    
    def __init__(self, sigma):
        self.sigma = sigma
    
    def chf(self, u, T):
        r, sigma = self.r, self.sigma
        return numpy.exp((r - 0.5*sigma**2)*T*u*1j - 0.5*(sigma**2)*T*(u**2))

    def call(self, s0, K, T):
        s0, K, T = map(float, (s0, K, T))
        r, s = self.r, self.sigma
        d1 = (numpy.log(float(s0)/float(K))+(r + 0.5*s*s)*T)/(s*numpy.sqrt(T))
        d2 = d1 - s*numpy.sqrt(T)
        return max(s0-K, 0) if numpy.isclose(T, 0) else s0*ndtr(d1) - K*numpy.exp(-r*T)*ndtr(d2)



class Edgeworth(equityModel):
    
    def __init__(self, c1, c2, c3, c4):
        '''
        inputs are raw moments; switch to cumulants
        '''
        self.m = c1
        self.s = numpy.sqrt(c2)
        self.c3 = c3
        self.c4 = c4
        
    def chf(self, u, T):
        r, m, s, c3, c4 = self.r, self.m, self.s, self.c3, self.c4
        self.c5 = c5 = 120.0*(numpy.exp(r*T-m-0.5*s*s) - (1.0+c3/6+c4/24))
        self.c5 = c5 = 0        
        
        return numpy.exp(1j*m*u - 0.5*s*s*u*u)*((1 + c4*u**4/24) - 1j*(c3*u**3/6 - c5*u**5/120))

    def call(self, s0, K, T):
        s0, K, T = map(float, (s0, K, T))
        r, m, s, c3, c4 = self.r, self.m, self.s, self.c3, self.c4
        self.c5 = c5 = 120.0*(numpy.exp(r*T-m-0.5*s*s) - (1.0+c3/6+c4/24))
        self.c5 = c5 = 0
        
        pdf = lambda x: norm.pdf(x, m, s)
        p1 = lambda x: (m-x)/(s*s)
        p2 = lambda x: (m*m - s*s - 2*m*x + x*x)/(s**4)
        p3 = lambda x: (m-x)*(m*m - 3*s*s - 2*m*x + x*x)/(s**6)
        
        return s0*(1.0 + c3/6 + c4/24 + c5/120)*numpy.exp(m + 0.5*s*s - r*T)*ndtr((m + s*s - numpy.log(K/s0))/s) \
                - K*numpy.exp(-r*T)*ndtr((m - numpy.log(K/s0))/s)   \
                + K*numpy.exp(-r*T)*pdf(numpy.log(K/s0))*(          \
                    -(c5/120)*p3(numpy.log(K/s0))                   \
                    +(c5/120 + c4/24)*p2(numpy.log(K/s0))           \
                    -(c5/120 + c4/24 + c3/6)*p1(numpy.log(K/s0))    \
                    +(c5/120 + c4/24 + c3/6)                        \
                )


def Carr(n, s0, K, T, model, a=100):
    s0, K, T, a = map(float, (s0, K, T, a))
    k = math.log(K)
    b = 1.5
    r = model.r
    
    integrand = lambda u: (cmath.exp(1j*(-u*k + (u-(b+1)*1j)*math.log(s0)))*math.exp(-r*T)*model.chf(u - (b+1)*1j, T)/(b**2 + b - u**2 + (2*b+1)*u*1j)).real
    
    return math.exp(-b*k)*fnc.trap(n, integrand, 0, a)/math.pi


def BSstyle(n, s0, K, T, model, a=100):
    s0, K, T, a = map(float, (s0, K, T, a))
    r = model.r
    
    integrand = lambda u: (cmath.exp(-1j*u*math.log(K/s0))*model.chf(u-1j, T)/(1j*u*model.chf(-1j, T))).real
    p1 = 0.5 + fnc.midpt(n, integrand, 0, a)/math.pi

    integrand = lambda u: (cmath.exp(-1j*u*math.log(K/s0))*model.chf(u, T)/(1j*u)).real
    p2 = 0.5 + fnc.midpt(n, integrand, 0, a)/math.pi
    
    return s0*p1 -K*math.exp(-r*T)*p2


def Lewis(n, s0, K, T, model, a=100):
    
    s0, K, T = map(float, [s0, K, T])
    r = model.r
    
    integrand = lambda u: (cmath.exp(u*(math.log(s0/K) )*(1j) - 0.5*r*T)*model.chf(u-0.5j, T)/(u**2 + 0.25)).real
    return s0 - math.sqrt(s0*K)*math.exp(-0.5*r*T)*fnc.trap(n, integrand, 0, a)/math.pi
    

if __name__ == '__main__':

    s0 = 100
    K = 120
    T = 1
    r = 0.1

    r = 0.0367
    sigma = 0.126349
    lambdaa = 0.174814
    alpha = -0.390078
    beta = 0.338796 

    model = JD(alpha, beta, lambdaa, sigma)
    model.setIR(r)
        
    modelCV = Edgeworth( *model.cumulants(T) )
    modelCV.setIR(r)
    
    print    
    print 'Edgeworth expansion, phi(-i) != rT'
    print 'closed form: \t', modelCV.call(s0, K, T)
    print 'Carr & Madan: \t', Carr(1000, s0, K, T, modelCV)
    print 'BSstyle: \t', BSstyle(1000, s0, K, T, modelCV)
    print 'Lewis algo: \t', Lewis(1000, s0, K, T, modelCV)
    
    print
    print 'JD model, phi(-i) == rT'
    print 'closed form: \t', model.call(s0, K, T)
    print 'Carr & Madan: \t', Carr(1000, s0, K, T, model)
    print 'BSstyle: \t', BSstyle(1000, s0, K, T, model)
    print 'Lewis algo: \t', Lewis(1000, s0, K, T, model)
    
    '''
    ################################# BS model #######################################
    s = 0.3

    model = BS(s)
    model.setIR(r)
    
    print 'Black-Scholes Model'
    print 'Formula: \t', model.call(s0, K, T)
    print 'Carr & Madan: \t', Carr(1000, s0, K, T, model)
    print
    
    ################################# JD model #######################################
    r = 0.0367
    sigma = 0.126349
    lambdaa = 0.174814
    alpha = -0.390078
    beta = 0.338796 

    model = JD(alpha, beta, lambdaa, sigma)
    model.setIR(r)
        
    print 'Merton\'s Jump Diffusion Model'
    print 'Formula: \t', model.call(s0, K, T)
    print 'Carr & Madan: \t', Carr(1000, s0, K, T, model)
    print
    '''
    
    
    
    