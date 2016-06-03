# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 10:59:08 2013

@author: gf
"""

import os
import pickle
import numpy as np
import itertools
import getLogDistributions as gLD
reload(gLD)
import matplotlib.pyplot as plt
import tables

NN = {1:(1,1),
      2:(2,1),
      3:(3,1),
      4:(2,2),
      5:(3,2),
      6:(3,2),
      7:(3,3),      
      8:(3,3),
      9:(3,3),      
      10:(5,2),
      11:(4,3),
      12:(4,3),
      13:(4,4),
      14:(4,4),
      15:(5,3),
      16:(4,4)
      }

mainDir = "/run/media/gf/DATA/meas/Barkh/Films/NiO_Fe/"
f5filename = os.path.join(mainDir, "NiO_Fe_results.h5")

f5 = tables.openFile(f5filename, mode="r")

runNode = f5.root.NiO80.M10x.run21

area = 1000.*1000.
log_step = 1./10

Axy_labels = []
# Collect Axy labels first
#for seqNode in f5.listNodes(runNode):
    #AxyNode = seqNode.clus
    #for label in f5.listNodes(AxyNode):
        #Axy_labels.append(label.name)
#Axy_labels = set(Axy_labels)

aval_clus = 'clus'
Axy = {}
Axy[aval_clus] = {}

for seqNode in f5.listNodes(runNode):
    clusNode = f5.getNode(seqNode, aval_clus)
    for lb in f5.listNodes(clusNode):
        Axy[aval_clus][lb.name] = np.concatenate((
            Axy[aval_clus].get(lb,np.array([])),lb.read()))
f5.close()

fig = plt.figure()
fig.set_size_inches((20,10),forward=True)
rows, cols = NN[len(Axy[aval_clus])]
for n, key in enumerate(Axy[aval_clus]):
    selected_Axy = Axy[aval_clus][key]
    print "==============="
    print "Cluster of type %s" % key
    N_S, bins = np.histogram(selected_Axy, np.arange(1, max(selected_Axy)+2), 
                             normed=False)
    S = bins[:-1]
    # Compress the data to non-zero values
    #cond = N_S!=0
    #N_S = np.compress(cond, N_S)
    #S = np.compress(cond, S)i
    yAverage = []
    # Get log bins
    SlogBins, logBins = gLD.getLogBins(bins[0], bins[-1], log_step)
    for i, j in zip(logBins[:-1],logBins[1:]):
        q1, q2 = np.greater_equal(S, i), np.less(S, j)
        q = np.logical_and(q1, q2)
        if sum(q) == 0:
            averageValue = np.NaN
        else:
            #allElements = [val for val in itertools.chain(*A_S[q])]
            averageValue = np.sum(S[q]*N_S[q])/area/(j-i)
        yAverage.append(averageValue)
    yAverage =  np.asanyarray(yAverage)
    # Check if there are NaN values
    iNan = np.isnan(yAverage)
    x = SlogBins[~iNan]
    y = yAverage[~iNan]
    
    plt.subplot(rows, cols, n+1)
    plt.loglog(x, y, 'o', label=key)
    # Plot the fit
    q = np.logical_and(np.greater_equal(x, 10), np.less(x, 1000))
    if sum(q)!=0:
        X, Y = np.log10(x[q]), np.log10(y[q])
        m, k = np.polyfit(X, Y, 1)
        lin_x = np.linspace(10,1000)
        new_y = 10**k*lin_x**m
        plt.plot(lin_x, new_y, '-')
        print "exponent: %f" % m
    x, PS, PSerror = gLD.logDistribution(selected_Axy, log_step, normed=False)
    plt.loglog(x, x*PS/area, '^')
    plt.loglog(x, PS, 'v', label="P(S)")
    plt.errorbar(x, PS, PSerror,fmt=None)
    plt.legend()
plt.show()
