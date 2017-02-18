# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 18:54:00 2016

@author: yuan
"""

from tablemaker import nrmaker
from ft_pricing import *


s0 = 100
K = 120
T = 1

r = 0.1
s = 0.3


kappa = 2
theta = 0.04
sigma = 0.5
rho = -0.7
v0 = 0.04

model = Heston(kappa, theta, sigma, rho, v0)
model.setIR(r)

ben = Carr(10000, s0, K, T, model, a=200)



flist = [Carr, CarrBSCV, CarrEWCV]
arg = (s0, K, T, model, 200)
nlist = [20, 30, 40, 50, 60, 70, 80, 90, 100]
colLabels = ['CM', 'CM-BS-CV', 'CM-EW-CV']

nr = nrmaker(nlist, flist, arg, benchmark=ben, fLabList=colLabels)
nr.path = 'C:\\Users\\yuan\\Desktop\\nctu\\papers\\GCV\\PhD thesis_GCV_db_old\\2016\\eps'

col=['CPU (ms)', 'log error']
cap = 'Efficiency and Accuracy Comparison of Carr and Madan\'s and Its Control Variate Modifications.'
label = 'tab:CM'

nr.genTable(col, cap, label)



nlist = range(20, 300, 20)
nr = nrmaker(nlist, flist, arg, benchmark=ben, fLabList=colLabels)
nr.path = 'C:\\Users\\yuan\\Desktop\\nctu\\papers\\GCV\\PhD thesis_GCV_db_old\\2016\\eps'


cap = 'Efficiency and Accuracy Comparison of Carr and Madan\'s and Its Control Variate Modifications.'
label = 'fig:CM'
nr.genFig('CPU (ms)', 'log error', cap, label, domain=(0, 8))

cap = 'Accuracy Comparison of Carr and Madan\'s and Its Control Variate Modifications.'
figlabel = ('number of partitions', 'log error')
latexlabel = 'fig:CMn'
nr.genFig('num', 'log error', cap, latexlabel, domain=(0, 130), xylabels=figlabel)




'''

flist = [BSstyle, BSstyleBSCV, BSstyleEWCV]
nlist = range(20, 41, 2)


nr = nrmaker(nlist, flist, arg, benchmark=ben)
nr.path = 'C:\\Users\\yuan\\Desktop\\nctu\\papers\\GCV\\PhD thesis_GCV_db_old\\2016\\eps'

col=['CPU (ms)', 'log error']
cap = 'Efficiency and Accuracy Comparison of Black-Scholes Style Formula and Its Control Variate Modifications.'
label = 'tab:BSstyle'
nr.genTable(col, cap, label)

cap = 'Efficiency and Accuracy Comparison of Black-Scholes Style Formula and Its Control Variate Modifications.'
label = 'fig:BSstyle'
nr.genFig('CPU (ms)', 'log error', cap, label)

cap = 'Accuracy Comparison of Black-Scholes Style Formula and Its Control Variate Modifications.'
xylabel = ('number of partition', 'log error')
label = 'fig:BSstylen'
nr.genFig('num', 'log error', cap, label)

'''
