# Make a movie

import os
import numpy as np
import h5py
import matplotlib.pyplot as plt


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
                p1, p2 = self._get_edges(images[0])
                images = images[:, :, p1:p2+1]
            else:
                print("No data available, please check")
                images, imageNumbers = None, None
        return images, imageNumbers

    def load_2D(self):
        with h5py.File(self.fname, 'r') as f:
            grp0 = f[self.baseGroup]
            if 'cluster2D_end' in grp0:
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

fname5 = "/home/gf/Meas/Creep/CoFeB/Wires/Arianna/Ta_CoFeB_MgO_wires_IEF_old/Ta_CoFeB_MgO_wires_IEF_old.hdf5"
outDir = "/home/gf/Meas/Creep/CoFeB/Wires/Arianna/Ta_CoFeB_MgO_wires_IEF_old/20um/20um_0.145A/20um_0.145A_10fps_3/Images"
baseGroup = "/20um/0.145A/10fps/wire1/3"
hdf5 = loadHdf5(fname5, baseGroup)
images, imageNumbers = hdf5.load_raw_images()
cluster2D_start, cluster2D_end, switchTimes2D = hdf5.load_2D()
n=1.45
white = np.array([1,1,1])
z0, z1 = 800, 1401
images = images[:,z0:z1]
cluster2D_start = cluster2D_start[z0:z1]
cluster2D_end = cluster2D_end[z0:z1]
cluster2D_start_switches = np.unique(cluster2D_start)[1:]
cluster2D_end_switches = np.unique(cluster2D_end)[1:]
# Initialize color map
im0 = cluster2D_start > 0
switchTimes2D = switchTimes2D[z0:z1]
dimX, dimY = im0.shape
q = np.zeros((dimX,dimY,3))
q[im0] = white

is_in_cluster = False
for i in range(cluster2D_start_switches[0], 1000):
    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, squeeze=False, figsize=(4*n,7*n))
    axs[0,0].imshow(images[i],'gray')
    # if i in cluster2D_start_switches:
    #     sw_start = i
    #     is_in_cluster = True
    #     c = np.random.rand(3)
    #     cluster = cluster2D_start == i
    #     sw_end = cluster2D_end[cluster].max()
    # if is_in_cluster:
    #     q[switchTimes2D==i] = c
    #     if i == sw_end:
    #         is_in_cluster = False
    #         #qq = np.logical_and((cluster2D_start>=sw_start), (cluster2D_end<=sw_end))
    #         qq = cluster2D_end==sw_end
    #         q[qq] == c
    if i in cluster2D_end_switches:
        q[cluster2D_end==i] = np.random.rand(3)

    axs[0,1].imshow(q)
    axs[0,1].axis((0,dimY,dimX,0))
    for j in range(2):
        axs[0,j].get_xaxis().set_visible(False)
        axs[0,j].get_yaxis().set_visible(False)
    plt.tight_layout()
    fname = "frame%04i.png" % i
    outFname = os.path.join(outDir, fname)
    fig.savefig(outFname, facecolor='gray', edgecolor='gray')
    print(outFname)
#plt.show()