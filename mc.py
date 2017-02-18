# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 00:22:39 2017

@author: yuan
"""

import numpy as np
from scipy.stats import norm

s0 = 100
K = 120
s = 0.3
r = 0.1
T = 1.0

n = 1
m = 100000

dt = T/n
t = np.arange(0, T+0.5*dt, dt)
payoff = np.empty(m)

for i in range(m):
    bm = np.cumsum(np.insert(norm.rvs(size=n, scale=np.sqrt(dt)), 0, 0))
    
    path = s0*np.exp((r-0.5*s*s)*t + s*bm)
    payoff[i] = np.exp(-r*T)*max(path[-1] - K, 0)
    
print np.average(payoff)