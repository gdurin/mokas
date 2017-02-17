import pickle
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure

def longest_contour(swTimes2D, n):
    cnts = measure.find_contours(swTimes2D == n,.5)
    i = np.argmax(np.array([len(c) for c in cnts]))
    return cnts[i]

def common_pixels(cnts1, cnts2, factor=1000000):
    """
    In time, contour 1 (cnts1) preceeds cnts2
    """
    # Transform into a array of single values
    s1 = cnts1[:,0] + factor * cnts1[:,1]
    s2 = cnts2[:,0] + factor * cnts2[:,1]
    common = np.array([s in s2 for s in s1])
    return cnts1[common]


with open("swTimes2D.pkl", 'rb') as f:
    swTimes2D = pickle.load(f)

n0, nmax = 14, 15
cnts1 = longest_contour(swTimes2D, n0)

fig, ax = plt.subplots()
for i in range(n0+1,nmax+1):
    cnts2 = longest_contour(swTimes2D, i)
    for cnts in [cnts1, cnts2]:
        X,Y = cnts[:,1], cnts[:,0]
        ax.plot(X,Y)
    c_line = common_pixels(cnts1, cnts2)
    if len(c_line):
        X,Y = c_line[:-1,1], c_line[:-1,0]
        res = np.polyfit(X,Y,2) 
        print(res)
        p = np.poly1d(res)
        #ax.plot(X,Y, 'r', lw=1)
        #ax.plot(X,p(X), 'ok', lw=2)
    else:
        print("Common contour %i not found" % i)
    cnts1 = cnts2
plt.show()
