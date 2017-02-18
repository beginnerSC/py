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
        
    def cumulants(self, T):
        '''
        cumulants of log(sT/s0)
        '''
        pass
    
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
        
        
class Heston(equityModel):
    
    def __init__(self, kappa, theta, sigma, rho, v0): 

        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.v0 = v0
        
    def cumulants(self, T):
        
        T = float(T)
        r = self.r
        
        k = self.kappa
        th = self.theta
        s = self.sigma
        rho = self.rho
        v0 = self.v0
        
        return  -(v0 - v0*numpy.exp(-T*k) - 2*r*T*k + th*(-1 + numpy.exp(-T*k) + T*k))/(2.*k), \
                (((-2*v0 + th)*s**2)*numpy.exp(-T*k) + 2*v0*(4*k**2 - 4*k*rho*s + s**2) + th*(8*T*k**3 - 5*s**2 + 2*k*s*(8*rho + T*s) - 8*k**2*(1 + T*rho*s)) + (4*(th*(s**2 + k*s*(-4*rho + T*s) + k**2*(2 - 2*T*rho*s)) + v0*k*(s*(2*rho - T*s) + 2*k*(-1 + T*rho*s))))*numpy.exp(-T*k))/(8.*k**3), \
                (s*(((3*v0 - th)*s**3)*numpy.exp(-3*T*k) + (6*s*(v0*(s**2 + 2*k*s*(-3*rho + T*s) + k**2*(4 - 4*T*rho*s)) + th*(-s**2 + k*s*(4*rho - T*s) + 2*k**2*(-1 + T*rho*s))))*numpy.exp(-2*T*k) + 2*(3*v0*(8*k**3*rho + 6*k*rho*s**2 - s**3 - 4*k**2*(s + 2*rho**2*s)) + th*(24*T*k**4*rho + 11*s**3 - 3*k*s**2*(20*rho + T*s) + 6*k**2*s*(5 + 12*rho**2 + 3*T*rho*s) - 12*k**3*(4*rho + T*s + 2*T*rho**2*s))) + (3*(v0*(-s**3 + 2*T*k*s**3 + 8*T*k**4*rho*(-2 + T*rho*s) + 2*k**2*s*(8*rho**2 - 8*T*rho*s + T**2*s**2) + 8*k**3*(2*T*s + 2*T*rho**2*s - rho*(2 + T**2*s**2))) - th*(5*s**3 + 8*T*k**4*rho*(-2 + T*rho*s) + 2*k**2*s*(8 + 24*rho**2 - 16*T*rho*s + T**2*s**2) + k*(-32*rho*s**2 + 6*T*s**3) + 8*k**3*(2*T*s + 4*T*rho**2*s - rho*(4 + T**2*s**2)))))*numpy.exp(-T*k)))/(16.*k**5), \
                (s**2*((3*(-4*v0 + th)*s**4)*numpy.exp(-4*T*k) + (24*s**2*(th*(s**2 + k*s*(-4*rho + T*s) + k**2*(2 - 2*T*rho*s)) + v0*(-2*s**2 + k*s*(10*rho - 3*T*s) + 6*k**2*(-1 + T*rho*s))))*numpy.exp(-3*T*k) + 3*(4*v0*(16*k**4*(1 + 4*rho**2) - 32*k**3*rho*(3 + 2*rho**2)*s + 24*k**2*(1 + 4*rho**2)*s**2 - 40*k*rho*s**3 + 5*s**4) + th*(64*k**5*(T + 4*T*rho**2) - 93*s**4 + 4*k*s**3*(176*rho + 5*T*s) - 32*k**2*s**2*(11 + 50*rho**2 + 5*T*rho*s) + 32*k**3*s*(40*rho + 32*rho**3 + 3*T*s + 12*T*rho**2*s) - 32*k**4*(5 + 24*rho**2 + 12*T*rho*s + 8*T*rho**3*s))) - (12*(4*v0*(s**4 + k*s**3*(-10*rho + 3*T*s) + 2*k**2*s**2*(3 + 12*rho**2 - 10*T*rho*s + T**2*s**2) + 4*k**4*(1 - 4*T*rho*s + 2*T**2*rho**2*s**2) + 4*k**3*s*(3*T*s + 6*T*rho**2*s - 2*rho*(3 + T**2*s**2))) - th*(7*s**4 + 2*k*s**3*(-24*rho + 5*T*s) + 4*k**2*s**2*(6 + 20*rho**2 - 14*T*rho*s + T**2*s**2) + 8*k**4*(1 - 4*T*rho*s + 2*T**2*rho**2*s**2) + 8*k**3*s*(3*T*s + 8*T*rho**2*s - 2*rho*(4 + T**2*s**2)))))*numpy.exp(-2*T*k) - (8*(v0*(-2*T**3*k**3*(2*k*rho - s)**3*s + 6*(16*k**4*rho**2 - 16*k**3*rho**3*s - 3*k**2*s**2 + 5*k*rho*s**3 - s**4) + 3*T*k*(16*k**4*(1 + 2*rho**2) - 32*k**3*rho*(2 + rho**2)*s + 12*k**2*(1 + 4*rho**2)*s**2 - 14*k*rho*s**3 + s**4) + 6*T**2*k**2*(8*k**4*rho**2 - 8*k**3*rho*(2 + rho**2)*s + 2*k**2*(3 + 8*rho**2)*s**2 - 8*k*rho*s**3 + s**4)) + th*(-21*s**4 + 9*k*s**3*(20*rho - 3*T*s) + 16*T**2*k**6*rho**2*(-3 + T*rho*s) - 6*k**2*s**2*(15 + 80*rho**2 - 35*T*rho*s + 2*T**2*s**2) - 24*T*k**5*(2 - 4*T*rho*s - 4*T*rho**3*s + rho**2*(8 + T**2*s**2)) + 12*k**4*(-4 + 24*T*rho**3*s - 3*T**2*s**2 + T*rho*s*(32 + T**2*s**2) - 2*rho**2*(12 + 7*T**2*s**2)) - 2*k**3*s*(-192*rho**3 + 240*T*rho**2*s + T*s*(54 + T**2*s**2) - 6*rho*(32 + 7*T**2*s**2)))))*numpy.exp(-T*k)))/(64.*k**7)

    def vol(self):
        '''
        sd of time 1 distribution of log(sT/s0)
        equalent to sigma in BS model
        '''
        return numpy.sqrt(self.cumulants(1)[1])
    
    def chf(self, u, T):
        '''
        Lord & Kahl's formulation
        '''
        T = float(T)
        r = self.r
        
        k = self.kappa
        th = self.theta
        s = self.sigma
        rho = self.rho
        v0 = self.v0
        
        d = numpy.sqrt((k-1j*rho*s*u)**2 + s*s*(1j*u+u*u))
        g = (k - 1j*rho*s*u - d)/(k - 1j*rho*s*u + d)
        D = (k - 1j*rho*s*u - d)*((1 - numpy.exp(-d*T)) / (1 - g*numpy.exp(-d*T)))/(s*s)
        C = 1j*r*u*T + (k*th/(s*s))*( (k - 1j*rho*s*u - d)*T - 2*numpy.log( (1 - g*numpy.exp(-d*T))/(1 - g) ))
        
        return numpy.exp( C + D*v0 )


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
        
        return numpy.exp(1j*m*u - 0.5*s*s*u*u)*((1 + c4*u**4/24) - 1j*(c3*u**3/6 - c5*u**5/120))

    def call(self, s0, K, T):
        s0, K, T = map(float, (s0, K, T))
        r, m, s, c3, c4 = self.r, self.m, self.s, self.c3, self.c4
        self.c5 = c5 = 120.0*(numpy.exp(r*T-m-0.5*s*s) - (1.0+c3/6+c4/24))
        
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

class BSCV(equityModel):
    '''
    只合分布的 mean 和 variance
    '''
    
    def __init__(self, c1, c2):
        '''
        inputs are raw moments; switch to cumulants
        '''
        self.m = c1
        self.s = numpy.sqrt(c2)
        
    def chf(self, u, T):
        r, m, s = self.r, self.m, self.s
        
        return numpy.exp(1j*m*u - 0.5*s*s*u*u)

    def callforBSstyle(self, s0, K, T):
        s0, K, T = map(float, (s0, K, T))
        r, m, s = self.r, self.m, self.s
        
        return  s0*ndtr((m + s*s - numpy.log(K/s0))/s ) - K*numpy.exp(-r*T)*ndtr((m - numpy.log(K/s0))/s )

    def callforCarr(self, s0, K, T):
        s0, K, T = map(float, (s0, K, T))
        r, m, s = self.r, self.m, self.s

        return  s0*numpy.exp(m + 0.5*s*s -r*T)*ndtr((m + s*s - numpy.log(K/s0))/s ) - K*numpy.exp(-r*T)*ndtr((m - numpy.log(K/s0))/s )

def Lewis(n, s0, K, T, model, a=100):
    
    s0, K, T = map(float, [s0, K, T])
    r = model.r
    
    integrand = lambda u: (cmath.exp(u*(math.log(s0/K) )*(1j) - 0.5*r*T)*model.chf(u-0.5j, T)/(u**2 + 0.25)).real
    return s0 - math.sqrt(s0*K)*math.exp(-0.5*r*T)*fnc.trap(n, integrand, 0, a)/math.pi


def Carr(n, s0, K, T, model, a=100):
    s0, K, T, a = map(float, (s0, K, T, a))
    k = math.log(K)
    b = 1.5
    r = model.r
    
    integrand = lambda u: (cmath.exp(1j*(-u*k + (u-(b+1)*1j)*math.log(s0)))*math.exp(-r*T)*model.chf(u - (b+1)*1j, T)/(b**2 + b - u**2 + (2*b+1)*u*1j)).real
    
    return math.exp(-b*k)*fnc.trap(n, integrand, 0, a)/math.pi


def CarrEWCV(n, s0, K, T, model, a=100):
    '''
    input model must have cumulants() implemented
    '''    
    s0, K, T, a = map(float, (s0, K, T, a))
    k = math.log(K)
    b = 1.5
    r = model.r
    
    modelCV = Edgeworth( *model.cumulants(T) )
    modelCV.setIR(r)
    
    intgDiff   = lambda u: (cmath.exp(1j*(-u*k + (u-(b+1)*1j)*math.log(s0)))*math.exp(-r*T)*(model.chf(u - (b+1)*1j, T) - modelCV.chf(u - (b+1)*1j, T))/(b**2 + b - u**2 + (2*b+1)*u*1j)).real
    
    return math.exp(-b*k)*fnc.trap(n, intgDiff, 0, a)/math.pi + modelCV.call(s0, K, T)


def CarrBSCV(n, s0, K, T, model, a=100):
    '''
    input model must have vol() implemented
    '''    
    s0, K, T, a = map(float, (s0, K, T, a))
    k = math.log(K)
    b = 1.5
    r = model.r
    
    c1, c2 = model.cumulants(1)[:2]
    modelCV = BSCV( c1, c2 )
    modelCV.setIR(r)

    '''    
    intgDiff   = lambda u: ((cmath.exp(1j*(-u*k + (u-(b+1)*1j)*math.log(s0)))*math.exp(-r*T)     \
                    *model.chf(u - (b+1)*1j, T)     \
                            - cmath.exp(1j*(-u*k + (u-(b+1)*1j)*(math.log(s0) )))*math.exp(-r*T)   \
                    *modelCV.chf(u - (b+1)*1j, T))    \
                    /(b**2 + b - u**2 + (2*b+1)*u*1j)).real
    '''
    
    intgDiff   = lambda u: (cmath.exp(1j*(-u*k + (u-(b+1)*1j)*math.log(s0)))*math.exp(-r*T)*(model.chf(u - (b+1)*1j, T) - modelCV.chf(u - (b+1)*1j, T))/(b**2 + b - u**2 + (2*b+1)*u*1j)).real
    
    return math.exp(-b*k)*fnc.trap(n, intgDiff, 0, a)/math.pi + modelCV.callforCarr(s0, K, T)


def CarrRealBSCV(n, s0, K, T, model, a=100):
    '''
    input model must have vol() implemented
    '''    
    s0, K, T, a = map(float, (s0, K, T, a))
    k = math.log(K)
    b = 1.5
    r = model.r
    
    modelCV = BS( model.vol() )
    modelCV.setIR(r)
    
    intgDiff   = lambda u: (cmath.exp(1j*(-u*k + (u-(b+1)*1j)*math.log(s0)))*math.exp(-r*T)*(model.chf(u - (b+1)*1j, T) - modelCV.chf(u - (b+1)*1j, T))/(b**2 + b - u**2 + (2*b+1)*u*1j)).real
    
    return math.exp(-b*k)*fnc.trap(n, intgDiff, 0, a)/math.pi + modelCV.call(s0, K, T)


def BSstyle(n, s0, K, T, model, a=100):
    s0, K, T, a = map(float, (s0, K, T, a))
    r = model.r
    
    integrand = lambda u: (cmath.exp(-1j*u*math.log(K/s0))*model.chf(u-1j, T)/(1j*u*model.chf(-1j, T))).real
    p1 = 0.5 + fnc.midpt(n, integrand, 0, a)/math.pi

    integrand = lambda u: (cmath.exp(-1j*u*math.log(K/s0))*model.chf(u, T)/(1j*u)).real
    p2 = 0.5 + fnc.midpt(n, integrand, 0, a)/math.pi
    
    return s0*p1 -K*math.exp(-r*T)*p2


def BSstyleEWCV(n, s0, K, T, model, a=100):
    '''
    input model must have cumulants() implemented
    '''
    s0, K, T, a = map(float, (s0, K, T, a))
    r = model.r

    modelCV = Edgeworth( *model.cumulants(T) )
    modelCV.setIR(r)
    
    intgDiff = lambda u: (cmath.exp(-1j*u*math.log(K/s0))*(model.chf(u-1j, T) - modelCV.chf(u-1j, T))/(1j*u*model.chf(-1j, T))).real
    p1Diff = fnc.midpt(n, intgDiff, 0, a)/math.pi

    intgDiff = lambda u: (cmath.exp(-1j*u*math.log(K/s0))*(model.chf(u, T) - modelCV.chf(u, T))/(1j*u)).real
    p2Diff = fnc.midpt(n, intgDiff, 0, a)/math.pi
    
    return s0*p1Diff -K*math.exp(-r*T)*p2Diff + modelCV.call(s0, K, T)
    

def BSstyleBSCV(n, s0, K, T, model, a=100):
    '''
    input model must have vol() implemented
    '''
    s0, K, T, a = map(float, (s0, K, T, a))
    r = model.r

    modelCV = BSCV( *model.cumulants(1)[:2] )
    modelCV.setIR(r)
    
    '''
    如果拿掉 log return 期望值為 rT 的假設，期望值的分母就不一樣了，得分開減
    '''    
    
    intgDiff = lambda u: (cmath.exp(-1j*u*math.log(K/s0))*  \
                            (model.chf(u-1j, T)/(1j*u*model.chf(-1j, T)) - modelCV.chf(u-1j, T)/(1j*u*modelCV.chf(-1j, T)))     \
                        ).real

    p1Diff = fnc.midpt(n, intgDiff, 0, a)/math.pi

    intgDiff = lambda u: (cmath.exp(-1j*u*math.log(K/s0))*(model.chf(u, T) - modelCV.chf(u, T))/(1j*u)).real
    p2Diff = fnc.midpt(n, intgDiff, 0, a)/math.pi
    
    return s0*p1Diff -K*math.exp(-r*T)*p2Diff + modelCV.callforBSstyle(s0, K, T)


def BSstyleRealBSCV(n, s0, K, T, model, a=100):
    '''
    input model must have vol() implemented
    '''
    s0, K, T, a = map(float, (s0, K, T, a))
    r = model.r

    modelCV = BS( model.vol())
    modelCV.setIR(r)
    
    intgDiff = lambda u: (cmath.exp(-1j*u*math.log(K/s0))*(model.chf(u-1j, T) - modelCV.chf(u-1j, T))/(1j*u*model.chf(-1j, T))).real
    p1Diff = fnc.midpt(n, intgDiff, 0, a)/math.pi

    intgDiff = lambda u: (cmath.exp(-1j*u*math.log(K/s0))*(model.chf(u, T) - modelCV.chf(u, T))/(1j*u)).real
    p2Diff = fnc.midpt(n, intgDiff, 0, a)/math.pi
    
    return s0*p1Diff -K*math.exp(-r*T)*p2Diff + modelCV.call(s0, K, T)    
    
    
def CarrVec(n, s0, K, T, model, a=100):
    s0, K, T, a = map(float, (s0, K, T, a))
    k = math.log(K)
    b = 1.5
    r = model.r
    
    u = numpy.linspace(0, a, n+1)
    integrand = (numpy.exp(1j*(-u*k + (u-(b+1)*1j)*numpy.log(s0)))*numpy.exp(-r*T)*model.chf(u - (b+1)*1j, T)/(b**2 + b - u**2 + (2*b+1)*u*1j)).real

    integrand[0] *= 0.5
    integrand[-1] *= 0.5
    
    return numpy.exp(-b*k)*integrand.sum()*a/(n*math.pi)


if __name__ == '__main__':

    s0 = 100
    K = 120
    T = 1

    r = 0.1
    
    '''
    ################################# BS model #######################################
    s = 0.3

    model = BS(s)
    model.setIR(r)
    
    print model.call(s0, K, T)
    print Carr(1000, s0, K, T, model), BSstyle(1000, s0, K, T, model)
    '''    
    '''
    ################################# Edgeworth model #######################################
    model = Edgeworth(0, 1, 0, 3) 
    model.setIR(r)
    
    print model.call(s0, K, T)
    print Carr(1000, s0, K, T, model), BSstyle(1000, s0, K, T, model)
    '''    
    '''
    ################################# JD model #######################################
    r = 0.0367
    sigma = 0.126349
    lambdaa = 0.174814
    alpha = -0.390078
    beta = 0.338796 

    model = JD(alpha, beta, lambdaa, sigma)
    model.setIR(r)
    
    n = 20
        
    print model.call(s0, K, T)
    print Carr(n, s0, K, T, model), CarrEWCV(n, s0, K, T, model), CarrBSCV(n, s0, K, T, model)
    print BSstyle(n, s0, K, T, model), BSstyleEWCV(n, s0, K, T, model), BSstyleBSCV(n, s0, K, T, model)
    '''    
    
    ################################# Heston model #######################################
    
    r = 0.1
    kappa = 2
    theta = 0.04
    sigma = 0.5
    rho = -0.7
    v0 = 0.04
    
    model = Heston(kappa, theta, sigma, rho, v0)
    model.setIR(r)
    
    #print Carr(10000, s0, K, T, model, a=1000), BSstyle(10000, s0, K, T, model, a=1000)
    
    print CarrBSCV(1000, s0, K, T, model)
    print Carr(1000, s0, K, T, model)
    