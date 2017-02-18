# -*- coding: utf-8 -*-
"""
Created on Thu Sep 01 11:43:32 2016

@author: yuan
"""

import math
from tablemaker import nrmaker
from fnc import *
from tree import *

#############################################
'''
flist = [trap, midpt, simp]
arg = (math.exp, 0, 1)
nlist = [10, 100, 1000, 10000]

nr = nrmaker(nlist, flist, arg)

col=['CPU (ms)', 'Abs error']
cap = 'Numerical Result Comparison of Various Integration Methods'
label = 'tab:quadrature'

nr.genTable(col, cap, label)
'''
#############################################

flist = [CVfast, CRR, CVWhite]
arg = (40, 40, 0.05, 0.1, 3)


nlist = [100, 200, 400, 800, 1600, 3200, 6400, 12800]
nlist = [100, 200, 400, 800]


nr = nrmaker(nlist, flist, arg)
nr.path = 'C:\\Users\\yuan\\Desktop\\nctu\\papers\\GCV\\PhD thesis\\2016\\eps'

col=['CPU (s)', 'Abs error']
cap = 'Efficiency and Accuracy Comparison of Various Binominal Tree Algorithms'
label = 'tab:Tree'

nr.genTable(col, cap, label)


nlist = range(10, 200, 5)
nr.run(nlist, flist, arg)

cap = 'Pricing Results Against Number of Time Steps'
label = 'fig:Tree'

nr.genFig('num steps', 'result', cap, label)


