import numpy as np
import pickle
from skimage import measure
import matplotlib.pyplot as plt
import heapq
import getAxyLabels as gal
from matplotlib.path import Path
import matplotlib.patches as patches

class CustomPatches:

    def TriangleLeft(self, size=1, color='orange', lw=1):
        verts = [(0., 0.), # left, bottom
                (0., size), # left, top
                (size, 0.), # right, top
                (0., 0.)]

        codes = [Path.MOVETO,
                Path.LINETO,
                Path.LINETO,
                Path.CLOSEPOLY]

        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor=color, lw=lw)
        return patch

    def TriangleRight(self, size=1, color='orange',lw=1):
        verts = [(0., 0.), # left, bottom
                (size, size), # left, top
                (size, 0.), # right, top
                (0., 0.)]

        codes = [Path.MOVETO,
                Path.LINETO,
                Path.LINETO,
                Path.CLOSEPOLY]

        patch = patches.PathPatch(path, facecolor=color, lw=lw)
        return patch




def select_contour(cluster, position='bottom'):
    """
    select the two longest contours in a list
    then get the one corresponding to the position
    """
    cnts = measure.find_contours(cluster, 0.5)
    lens = [len(cnt) for cnt in cnts]
    l_values = heapq.nlargest(2, lens)
    i0, i1 = [lens.index(l) for l in l_values]
    if l_values[0] == l_values[1]:
        i1 = len(lengths) - 1 - lengths[::-1].index(l_values[0])
    Y0, Y1 = cnts[i0][:,0], cnts[i1][:,0]
    m0, m1 = np.mean(Y0), np.mean(Y1)
    if m0 > m1:
        l_bottom = cnts[i0]
        l_top = cnts[i1]
    else:
        l_bottom = cnts[i1]
        l_top = cnts[i0]
    if position == 'bottom':
        return l_bottom
    else:
        return l_top



filename = "cluster2D.pkl"
n = 30

with open(filename, 'r') as f:
    cluster2D = pickle.load(f)

cluster_switches = np.unique(cluster2D)[1:]

cluster = cluster2D == cluster_switches[0]
angle_left = []
angle_right = []
sw_10 = []
sw_01 = []
angle_10 = []
angle_01 = []
sizesAxy = {}
colors = {'0000': 'r', '1000': 'b', '0100': 'c', '1100': 'g'}
plt.figure()
for switch in cluster_switches[1:]:
    cluster0 = (cluster2D == switch)
    cluster_type = gal.getAxyLabels(cluster0, 'Bottom_to_top', 1)[0]
    print(cluster_type)
    sizesAxy[cluster_type] = sizesAxy.get(cluster_type,0) + np.sum(cluster0)
    cluster += cluster0
    cnts = select_contour(cluster)
    X, Y = cnts[:,1], cnts[:,0]
    # Left edge
    z = np.polyfit(X[:n], Y[:n], 1)
    z0 = np.arctan(z[0])
    print("left: %f" % z0)
    angle = np.abs(z0) * 180 / np.pi
    angle_left.append(angle)
    mksize = 15./700*(np.sum(cluster0)) + 1
    plt.plot(switch, angle, 'o', c=colors[cluster_type], markersize=mksize, alpha=0.5)
    print(np.sum(cluster0))
    if cluster_type == '0100':
        angle_01.append(angle)
        sw_01.append(switch)
    z = np.polyfit(X[-n:], Y[-n:], 1)
    z0 = np.arctan(z[0])
    # Right edge
    print("right: %f" % z0)
    angle = abs(z0) * 180 / np.pi
    angle_right.append(angle)
    if cluster_type == '1000':
        angle_10.append(angle)
        sw_10.append(switch)
    plt.plot(switch, angle, 'o', c=colors[cluster_type], markersize=mksize, alpha=0.5)

angle_left = np.array(angle_left)
angle_right = np.array(angle_right)

times = cluster_switches[1:]

plt.plot(times, angle_left, '-b', lw=1, label='left')
plt.plot(times, angle_right, '-g', lw=1, label='right')
plt.legend()
plt.show()

total_area = np.sum([sizesAxy[k] for k in sizesAxy])
for k in sizesAxy:
    print("%s: %.2f" % (k, sizesAxy[k]*100/total_area))