# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 18:54:00 2016

@author: yuan
"""

from tablemaker import nrmaker
from fnc import *
from tree import *

flist = [CV, CRR, HW]
arg = (40, 40, 0.05, 0.1, 3)



scott = nrmaker([100, 200, 400, 800], [CV, CVslow], arg)
col=['CPU (s)', 'result']
cap = 'Comparison of'
label = 'tab:CVfastCVslow'

scott.genTable(col, cap, label)



nlist = [100, 200, 400, 800]
nr = nrmaker(nlist, flist, arg, benchmark=1.237687)
nr.path = 'C:\\Users\\yuan\\Desktop\\nctu\\papers\\GCV\\PhD thesis_GCV_db_old\\2016\\eps'



col=['CPU (s)', 'Abs error']
cap = 'Efficiency and Accuracy Comparison of Various Binominal Tree Algorithms'
label = 'tab:Tree'

nr.genTable(col, cap, label)




nlist = range(10, 200, 5)
nr.run(nlist, flist, arg)

cap = 'Pricing Results Against Number of Time Steps'
label = 'fig:Tree'
figlabel = ('number of time steps', 'pricing result')

nr.genFig('num', 'result', cap, label, xylabels=figlabel)


