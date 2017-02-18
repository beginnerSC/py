# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 16:42:24 2016

@author: yuan
"""


from pandas import DataFrame
import matplotlib.pyplot as plt
import time, numpy
    
class nrmaker:
    
    def __init__(self, nlist, flist, arg, benchmark=None, repeat=1, fLabList=None ):
        '''
        run the functions in flist and return the numerical results as a DataFrame
        fLabList is the list of labels of functions on table or in the figure legend; use function name if None
        
        '''
        self.nlist = nlist
        self.fNameList = fLabList if fLabList else [f.__name__ for f in flist]
        self.run(nlist, flist, arg, benchmark, repeat)

    def run(self, nlist, flist, arg, benchmark=None, repeat=1):
        result = {}
        CPUtime = {}
        num = {}
    
        for ii, f in enumerate(flist):
        
            resultList = []
            timeList = []
            nList = []
        
            for n in nlist:
                start = time.clock()            
                for _ in range(repeat):            
                    tmp = f(n, *arg)
                    
                timeList.append((time.clock() - start)/repeat)
                resultList.append(tmp)
                nList.append(n)
        
            result[self.fNameList[ii]] = resultList
            CPUtime[self.fNameList[ii]] = timeList
            num[self.fNameList[ii]] = nList

        self.frameDict = {}
        
        self.frameDict['num'] = DataFrame(
                                        numpy.array([num[name] for name in self.fNameList]).T, 
                                        columns = [['num' for _ in self.fNameList], self.fNameList], 
                                        index = nlist )
    
        self.frameDict['result'] = DataFrame(
                                        numpy.array([result[name] for name in self.fNameList]).T, 
                                        columns = [['result' for _ in self.fNameList], self.fNameList], 
                                        index = nlist )
        #print 'dbug'
        #print self.frameDict['result']
        
        self.frameDict['CPU (s)'] = DataFrame(
                                        numpy.array([CPUtime[name] for name in self.fNameList]).T, 
                                        columns = [['CPU (s)' for _ in self.fNameList], self.fNameList], 
                                        index = nlist )        
        # 下面這些應該要改寫成 factory 

        # get Abs error
        self.set_benchmark( benchmark )
        self.frameDict['Abs error'] = numpy.abs(self.frameDict['result'] - self.benchmark)
        self.frameDict['Abs error'].columns = [['Abs error' for _ in self.fNameList], self.fNameList]
        
        # get log error; must have abs error first 
        self.frameDict['log error'] = numpy.log10(self.frameDict['Abs error'])
        self.frameDict['log error'].columns = [['log error' for _ in self.fNameList], self.fNameList]

        # get CPU (ms)
        self.frameDict['CPU (ms)'] = self.frameDict['CPU (s)']*1000
        self.frameDict['CPU (ms)'].columns = [['CPU (ms)' for _ in self.fNameList], self.fNameList]
                
    def set_benchmark(self, benchmark ):
        
        if benchmark == None: 
            self.benchmark = self.frameDict['result'].ix[self.frameDict['result'].index[-1]]
        else:
            self.benchmark = benchmark
        
    def genTable(self, col=['result', 'CPU (s)'], caption=None, label='tab' ):
        '''
        col is a subset of ['result', 'CPU (s)', 'CPU (ms)', 'Abs error', 'log error']
        '''
        tab = DataFrame(
                index = self.nlist, columns = [['result' for _ in self.fNameList], self.fNameList]
                ).fillna(0)
                
        for key in col:
            tab = tab.add(self.frameDict[key], fill_value=0)
    
        if 'result' not in col:
            tab = tab.drop('result', axis=1, level=0)
    
        frame = tab.swaplevel(0, 1, axis=1).sortlevel(0, axis=1)
    
        print '\\begin{figure}[!t]'
        print '\\begin{center}'
        print frame.to_latex().encode('ascii', 'replace')
        print '\\end{center}'
        print '\\caption{'
        print caption
        print '}\label{' + label + '}'
        print '\\end{figure}'
        print
        
    def genFig(self, x='num', y='result', caption=None, figlabel='fig', domain=None, xylabels=None ):
        '''
        choose x, y from ['num', 'result', 'CPU (s)', 'CPU (ms)', 'Abs error', 'log error']
        domain is a 2D list or tuple (a, b); only plot from a to b
        xylabels is a 2D list (xlabel, ylabel)
        '''
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        
        # using frameDict[x][x][fnc] is a terrible coding style
        #plotstyleList = ['ko-', 'k^--', 'ks-.', 'k+:']
        plotstyleList = ['k-', 'k--', 'k-.', 'k:']
        for ii, fnc in enumerate(self.fNameList):
            xlist = self.frameDict[x][x][fnc]
            ylist = self.frameDict[y][y][fnc]
            
            if domain != None:
                a, b = domain
                ind = (xlist > a)&(xlist < b)   # where x values are in the list
                xlist = xlist[ind]
                ylist = ylist[ind]
                
            ax.plot(xlist, ylist, plotstyleList[ii], label=fnc)
        
        if xylabels==None: 
            ax.set_xlabel(x)
            ax.set_ylabel(y)
        else:
            ax.set_xlabel(xylabels[0])
            ax.set_ylabel(xylabels[1])
            
        ax.legend(loc='best')
        
        filename = figlabel[4:] if figlabel[:4] == 'fig:' else figlabel
        plt.savefig(self.path + '\\\\' + filename + '.eps', dpi=400)
        
        print '\\begin{figure}[!t]'
        print '\\begin{center}'
        print '    \\parbox[t]{0.8\\textwidth}{'
        print '        \\centerline{\\epsfig{figure=eps/' + filename + '.eps' + ', width=0.8\\textwidth}}'
        print '    }\\hfill'
        print '\\end{center}'
        print '\\caption{'
        print caption
        print '}\label{' + figlabel + '}'
        print '\\end{figure}'
        print
    