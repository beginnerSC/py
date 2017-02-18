# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 20:24:10 2016

@author: yuan
"""

import numpy

def trap(n, f, a, b):

    h = (float(b)-float(a))/n
    flist = numpy.vectorize(f)(numpy.linspace(a, b, n+1))
        
    flist[0] /= 2
    flist[-1] /= 2
    
    return h*flist.sum()
    
    
def midpt(n, f, a, b):

    h = (float(b)-float(a))/n
    flist = numpy.vectorize(f)(numpy.linspace(a+h/2, b-h/2, n))
    
    return h*flist.sum()


def simp(n, f, a, b):
    
    if n%2 == 1 :
        print 'number of partitions must be even'
        return None
        
    h = (float(b)-float(a))/n
    flist = numpy.vectorize(f)(numpy.linspace(a, b, n+1))
    coeff = numpy.concatenate(([1, 4], numpy.array([[2, 4] for ii in range(n/2 - 1)]).flatten(), [1]))
    
    return (coeff*flist).sum()*h/3