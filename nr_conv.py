# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 18:54:00 2016

@author: yuan
"""

from tablemaker import nrmaker
from fnc import *
from conv_bermudan import *
from conv_barrier import *
from conv_asian import *


s0 = 100
K = 100
r = 0.1
s = 0.3
T = 1

c = 2
n = 12


H = 130
K = 120



'''

arg = (c, s0, K, r, s, T, n)

#ben = Benh(102400, *arg)
ben = 2.31345342291

flist = [Benh, BenhCV]
nlist = [200, 400, 800, 1600, 3200, 6400, 12800]

asian = nrmaker(nlist, flist, arg, benchmark=ben, repeat=5)
asian.path = 'C:\\Users\\yuan\\Desktop\\nctu\\papers\\GCV\\PhD thesis_GCV_db_old\\2016\\eps'

cap = 'Comparison of Asian Options Pricing Algorithms'
label = 'tab:conv_asian'

col=['CPU (s)', 'log error']
asian.genTable(col, cap, label)


cap = 'Efficiency and Accuracy Comparison of Asian Options Pricing Algorithms'
label = 'fig:conv_asian'
asian.genFig('CPU (s)', 'log error', cap, label)



cap = 'Efficiency and Accuracy Comparison of Asian Options Pricing Algorithms'
label = 'fig:convn_aisan'
figlabel = ('number of partitions', 'log error')
asian.genFig('num', 'log error', cap, label, xylabels=figlabel)






########################### Down and out call ########################### 

arg = (c, s0, K, r, s, T, H, n)

#ben = UOcall(102400, *arg)
ben = 0.176355450733

flist = [UOcall, UOcallCV]
nlist = range(4000, 10500, 1000)
nlist = [200, 400, 800, 1600, 3200, 6400, 12800]


uo = nrmaker(nlist, flist, arg, benchmark=ben, repeat=5)
uo.path = 'C:\\Users\\yuan\\Desktop\\nctu\\papers\\GCV\\PhD thesis_GCV_db_old\\2016\\eps'

cap = 'Comparison of Discrete Barrier Options Pricing Algorithms'
label = 'tab:barrier'

col=['CPU (s)', 'log error']
uo.genTable(col, cap, label)


cap = 'Efficiency and Accuracy Comparison of Various Binominal Tree Algorithms'
label = 'fig:conv_barrier'
uo.genFig('CPU (s)', 'log error', cap, label)



cap = 'Efficiency and Accuracy Comparison of Various Binominal Tree Algorithms'
label = 'fig:convn_barrier'
figlabel = ('number of partitions', 'log error')
uo.genFig('num', 'log error', cap, label, xylabels=figlabel)


'''

########################### Bermudan put ########################### 

K = 100

arg = (c, s0, K, r, s, T, n)

#ben = Bermudan(102400, *arg)
ben = 8.24327435926

flist = [Bermudan, BSCV, expCV]
nlist = [100, 200, 400, 800, 1600, 3200, 6400]


berm = nrmaker(nlist, flist, arg, benchmark=ben, repeat=10)
berm.path = 'C:\\Users\\yuan\\Desktop\\nctu\\papers\\GCV\\PhD thesis_GCV_db_old\\2016\\eps'

cap = 'Comparison of Bermudan Pricing Options'
label = 'tab:berm'

col=['CPU (s)', 'log error']
berm.genTable(col, cap, label)


cap = 'Efficiency and Accuracy Comparison of Various Binominal Tree Algorithms'
label = 'fig:conv'
berm.genFig('CPU (s)', 'log error', cap, label)


cap = 'Efficiency and Accuracy Comparison of Various Binominal Tree Algorithms'
label = 'fig:convn'
figlabel = ('number of partitions', 'log error')
berm.genFig('num', 'log error', cap, label, xylabels=figlabel)




