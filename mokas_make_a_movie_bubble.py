# Make a movie

import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mokas_colors import getPalette


class loadHdf5:
    def __init__(self, fname, baseGroup):
        self.fname = fname
        self.baseGroup = baseGroup
        self.edge_trim_percent = 8

    def load_raw_images(self):
        with h5py.File(self.fname, 'r') as f:
            grp0 = f[self.baseGroup]
            if 'images' in grp0:
                images = grp0['images'][...]
                imageNumbers = grp0['imageNumbers'][...]
                row_profile = np.mean(images[0], 0)
                #p1, p2 = self._get_edges(images[0])
                #images = images[:, :, p1:p2+1]
            else:
                print("No data available, please check")
                images, imageNumbers = None, None
        return images, imageNumbers

    def load_2D(self):
        with h5py.File(self.fname, 'r') as f:
            grp0 = f[self.baseGroup]
            if 'cluster2D_end' in grp0:
                cluster2D_end = grp0['cluster2D_end'][...]
                #cluster2D_start = grp0['cluster2D_start'][...]
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
outDir = "%s/Movies/%sA/%s/n_exp%s" % (mainDir, current, set_n, n_exp)
hdf5 = loadHdf5(fname5, baseGroup)
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
r0,r1,c0,c1 = (140,220,180,530)
r0,r1,c0,c1 = (180,260,180,530)
#fsize = (9.75,4.5)
fsize = (8.14, 4.25)
#fig, axs = plt.subplots(2, 1, sharex=True, sharey=True, figsize=fsize)
fig, axs = plt.subplots(2, 1, figsize=fsize)
axs[0].imshow(images[-1,r0:r1+1,c0:c1+1]-images[0,r0:r1+1,c0:c1+1],'gray')

n = np.unique(cluster2D_start)[-1]
p = getPalette(n, 'random', 'black')
cm = colors.ListedColormap(p, 'pColorMap')

axs[1].imshow(cluster2D_start[r0:r1+1,c0:c1+1], cmap=cm)
for ax in axs:
    ax.axis((0,c1-c0,r1-r0,0))
plt.tight_layout()
plt.show()



#for i in range(cluster2D_start_switches[0], 1000):
#     axs[0,0].imshow(images[i],'gray')
#     # if i in cluster2D_start_switches:
#     #     sw_start = i
#     #     is_in_cluster = True
#     #     c = np.random.rand(3)
#     #     cluster = cluster2D_start == i
#     #     sw_end = cluster2D_end[cluster].max()
#     # if is_in_cluster:
#     #     q[switchTimes2D==i] = c
#     #     if i == sw_end:
#     #         is_in_cluster = False
#     #         #qq = np.logical_and((cluster2D_start>=sw_start), (cluster2D_end<=sw_end))
#     #         qq = cluster2D_end==sw_end
#     #         q[qq] == c
#     if i in cluster2D_end_switches:
#         q[cluster2D_end==i] = np.random.rand(3)

#     axs[0,1].imshow(q)
#     axs[0,1].axis((0,dimY,dimX,0))
#     for j in range(2):
#         axs[0,j].get_xaxis().set_visible(False)
#         axs[0,j].get_yaxis().set_visible(False)
#     plt.tight_layout()
#     fname = "frame%04i.png" % i
#     outFname = os.path.join(outDir, fname)
#     fig.savefig(outFname, facecolor='gray', edgecolor='gray')
#     print(outFname)
# #plt.show()