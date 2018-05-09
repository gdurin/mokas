# Make a movie

import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mokas_colors import getPalette
from skimage import measure

class loadHdf5:
    def __init__(self, fname, baseGroup, limits=None):
        self.fname = fname
        self.baseGroup = baseGroup
        self.edge_trim_percent = 8
        if limits is not None:
             self.limits = limits
             self.is_limits = True


    def load_raw_images(self):
        if self.is_limits:
            r0,r1,c0,c1 = self.limits
        with h5py.File(self.fname, 'r') as f:
            grp0 = f[self.baseGroup]
            if 'images' in grp0:
                if self.is_limits:
                    images = grp0['images'][:,r0:r1+1,c0:c1+1]
                else:
                    images = grp0['images'][...]
                imageNumbers = grp0['imageNumbers'][...]
            else:
                print("No data available, please check")
                images, imageNumbers = None, None
        return images, imageNumbers

    def load_2D(self):
        with h5py.File(self.fname, 'r') as f:
            grp0 = f[self.baseGroup]
            if 'cluster2D_end' in grp0:
                if self.is_limits:
                    r0,r1,c0,c1 = self.limits
                    cluster2D_end = grp0['cluster2D_end'][r0:r1+1,c0:c1+1]
                    cluster2D_start = grp0['cluster2D_start'][r0:r1+1,c0:c1+1]
                    switchTimes2D = grp0['switchTimes2D'][r0:r1+1,c0:c1+1]
                else:
                    cluster2D_end = grp0['cluster2D_end'][...]
                    cluster2D_start = grp0['cluster2D_start'][...]
                    switchTimes2D = grp0['switchTimes2D'][...]
                return cluster2D_start, cluster2D_end, switchTimes2D
            else:
                print("No data available, please check")

    def _get_edges(self, im):
        """
        Calculate the position of the edge
        from the first row image
        edge_trim_percent reduces the width of the wire
        from both sides
        """
        gray_profile = np.mean(im, 0)
        L2 = len(gray_profile) / 2
        p1 = np.argmin(gray_profile[:L2])
        p2 = np.argmin(gray_profile[L2:]) + L2
        distance = p2 - p1
        p1 += distance * self.edge_trim_percent / 100
        p2 -= distance * self.edge_trim_percent / 100
        return p1, p2



mainDir = "/data/Meas/Creep/CoFeB/Film/SuperSlowCreep/NonIrr/Feb2018/"
current, set_n, n_exp = "0.146", "Set1", "03"
fname5 = "{0}A/{0}A.hdf5".format(current)
fname5 = os.path.join(mainDir, fname5)
baseGroup = "/%s/%sA/%s" % (set_n, current, n_exp)
outDir = "Movies/%sA/%s/n_exp%s" % (current, set_n, n_exp)
outDir = "Movies"
outDir = os.path.join(mainDir, outDir)
if not os.path.isdir(outDir):
    os.mkdir(outDir)
#limits = (140,220,180,530)
limits = (180,260,180,530)
r0, r1, c0, c1 = limits
hdf5 = loadHdf5(fname5, baseGroup, limits)
images, imageNumbers = hdf5.load_raw_images()
cluster2D_start, cluster2D_end, switchTimes2D = hdf5.load_2D()
# n=1.45
# white = np.array([1,1,1])
# z0, z1 = 800, 1401
# images = images[:,z0:z1]
# cluster2D_start = cluster2D_start[z0:z1]
# cluster2D_end = cluster2D_end[z0:z1]
# cluster2D_start_switches = np.unique(cluster2D_start)[1:]
# cluster2D_end_switches = np.unique(cluster2D_end)[1:]
# # Initialize color map
# im0 = cluster2D_start > 0
# switchTimes2D = switchTimes2D[z0:z1]
# dimX, dimY = im0.shape
# q = np.zeros((dimX,dimY,3))
# q[im0] = white

# is_in_cluster = False
qq = cluster2D_start != -1
contours = measure.find_contours(qq, .5, 'low')
X, Y = contours[2][:,1], contours[2][:,0]



# fsize = (9.75,4.5)
fsize = (8.14, 4.25)
# fig, axs = plt.subplots(2, 1, sharex=True, sharey=True, figsize=fsize)
switches_start = np.unique(cluster2D_start)[1:]
switches = np.unique(switchTimes2D)[1:]
n = switches[-1]
#p = getPalette(n, 'random', 'lightgray')
p = getPalette(n, 'random', 'black')
cm = colors.ListedColormap(p, 'pColorMap')
bounds = np.append(switches, switches[-1]+1)-0.5
norm = colors.BoundaryNorm(bounds, len(bounds)-1)

for switch in switches[:]:
    print(switch)
    fig, axs = plt.subplots(2, 1, figsize=fsize)
    axs[0].imshow(images[switch] - images[0], 'gray', interpolation='none')
    #axs[0].plot(X, Y, 'k-', lw=2)
    axs[1].plot(X, Y, 'w-', lw=1)
    q = cluster2D_end > switch
    out = np.copy(cluster2D_end)
    out[q] = -1
    axs[1].imshow(out, cmap=cm, norm=norm)
    for ax in axs:
        ax.axis((0, c1 - c0, r1 - r0, 0))
        #ax.grid(True)
    plt.tight_layout()
    
    fname = "frame%04i.jpg" % switch
    outFname = os.path.join(outDir, fname)
    fig.savefig(outFname, facecolor='black', edgecolor='black')
    plt.close(fig)
    print(outFname)
# #plt.show()