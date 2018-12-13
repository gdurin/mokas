import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import h5py
import pandas as pd
from skimage import measure
from collections import OrderedDict
import mahotas
import getLogDistributions as gLD
from mokas_colors import getPalette
import mokas_bestfit as bestfit


currents = ["0.137A", "0.146A", "0.157A", "0.165A"]
fields = {"0.137A": "0.13", "0.146A": "0.14", "0.157A": "0.15", "0.165A": "0.16"}
n_set = "Set1"
d_f = "1.000"
nij_s = {"0.137A": "0.44", "0.146A": "0.33", "0.157A": "0.25", "0.165A": "0.19"}
centers = {"0.137A": (522,558), "0.146A": (530,557), "0.157A": (580,562), "0.165A": (541,575)}
label_size = 18
clrs = ['b', 'orange', 'g', 'r']
multiplier = 6
n_exp = np.array([14, 8, 4, 4])
total_times = n_exp * 800 * 0.2

def adjust_ax(ax):
    ax.get_yaxis().set_tick_params(which='both', direction='in')
    ax.get_xaxis().set_tick_params(which='both', direction='in')
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')

class Formatter(object):
    def __init__(self, im):
        self.im = im
        self.rows, self.cols = im.shape
    def __call__(self, x, y):
        col = int(x + 0.5)
        row = int(y + 0.5)
        if col >= 0 and col < self.cols and row >= 0 and row < self.rows:
            y, x = int(y), int(x)
            z = self.im[y,x]
            return 'x={0:d}, y={1:d}, z={2:d}'.format(x, y, z)

def get_best_fit(x, y, y_err, n_params, p0, min_index=2, max_index=-2):
    sd = bestfit.Size_Distribution(n_params)
    x, y, y_err = x[min_index:max_index], y[min_index:max_index], y_err[min_index:max_index]
    w = y != 0
    x, y, y_err = x[w], y[w], y_err[w]
    model = bestfit.Model(x, y, theory=sd, p0=p0, y_err=None, linlog='log', use_jacobian=False)
    params, errors, ier = model.get_params()
    if ier in range(1,5) and errors is not None:
        for pars in zip(sd.params, params, errors):
            print("%s: %.2f +/- %.2f" % pars)
    else:
        for pars in zip(sd.params, params):
            print("%s: %.2f" % pars)
    x_calc = np.logspace(np.log10(np.min(x)), np.log10(np.max(x)), 2*len(x))
    if ier != 0:
        y_calc = sd.y(params, x_calc)
        return params, errors, ier, x_calc, y_calc
    else:
        return params, errors, ier, None, None

class Klusters:
    """
    upload data and calculate the Kolton's clusters
    """
    def __init__(self, hname, current, deltas_T, ac_time=0.2):
        store = pd.HDFStore(hname)
        self.cluster2D_dt = OrderedDict()
        self.event2D_dt = OrderedDict()
        dt0 = 0.2
        ########################################
        n_ij = nij_s[current]
        group0 = "%s/%s/df_%s/nij_%s" % (current, n_set, d_f, n_ij)
        group = "%s/%s" % (group0, 'cluster2D_nij')
        self.cluster2D = store.get(group).values
        self.cluster2D_dt[dt0] = np.copy(self.cluster2D)
        group = "%s/%s" % (group0, 'event2D')
        self.event2D = store.get(group).values
        self.event2D_dt[dt0] = np.copy(self.event2D)
        group = "%s/%s" % (group0, 'all_events_structure')
        self.all_events_structure = store.get(group)
        group = "%s/%s" % (group0, 'all_events_hierarchy')
        self.all_events_hierarchy = store.get(group)
        group = "%s/%s" % (group0, 'all_clusters_nij_structure')
        self.all_clusters_structure = store.get(group)
        group = "%s/%s" % (group0, 'PS_nij_filtered')
        self.PS_nij = store.get(group)
        store.close()
        #cluster2D_start = get_cluster2D_start(cluster2D, event2D)

        # Get the labels of the clusters_nij from self.cluster2D
        labels = np.unique(self.cluster2D)[1:]
        # Now find the correspondence of the labels with the switching time
        _aes = self.all_events_structure
        times = _aes.switch_time.values
        times = times - times[0]
        #times = np.int16(times*10)/10.
        times = np.round(times/ac_time) * ac_time
        self.times = pd.Series(times, index=_aes.event_label)
        self.deltas_T = deltas_T
        for delta_T in self.deltas_T:
            self.cluster2D_dt[delta_T] = self._remap_image(self.cluster2D, delta_T)
            self.event2D_dt[delta_T] = self._remap_image(self.event2D, delta_T)
        n = np.max(self.event2D.flatten())
        p = getPalette(n, 'random', 'black')
        self.cm = colors.ListedColormap(p, 'pColorMap')

    def _remap_image(self, im, delta_T):
        # Great suggestion from
        # https://stackoverflow.com/questions/13572448/replace-values-of-a-numpy-index-array-with-values-of-a-list
        #
        a = np.copy(im)
        q0 = pd.Series([-1], index=[-1])
        q = np.abs(self.times//delta_T).astype(np.int16)
        qq = pd.concat([q0,q])
        palette = qq.index
        key = qq.values
        index = np.digitize(a.ravel(), palette, right=True)
        return key[index].reshape(a.shape)

    def plot_image_at_deltaT(self, plot_contour=False, limits=None):
        #
        fig, axs = plt.subplots(2,4, sharex=True, sharey=True, figsize=(14,8))
        im = self.cluster2D
        if limits:
            r0, r1, c0, c1 = limits
            im = im[r0:r1, c0:c1]
        if plot_contour:
                self._plot_contour(im, axs[0,0])
        axs[0,0].imshow(im, cmap=self.cm)
        axs[0,0].set_title(r"$\Delta t = %.2f\ s$" % 0.2)
        for i, delta_T in enumerate(self.deltas_T):
            #im = self.cluster2D_dt[delta_T]
            im = self.event2D_dt[delta_T]
            if limits:
                im = im[r0:r1, c0:c1]
            ax = axs[(i+1)//4,(i+1)%4]
            ax.imshow(im, cmap=self.cm)
            if plot_contour:
                self._plot_contour(im, ax)
            ax.set_title(r"$\Delta t = %.2f\ s$" % delta_T)
        plt.show()
        fig.tight_layout()

    def _plot_contour(self, im, ax):
        switches = np.unique(im)[1:]
        for switch in switches:
            q = im == switch
            clusters, n_cluster = mahotas.label(q, np.ones((3,3)))
            for i in range(1, n_cluster+1):
                cluster = clusters == i
                cnts = measure.find_contours(cluster, 0.5)  # need 0.5 because cluste is boolean
                for c in cnts:
                    X, Y = c[:,1], c[:,0]
                    ax.plot(X,Y, 'k-')

    def plot_distributions_at_deltaT(self, log_step=0.05, min_size=5, fig=None):
        if fig is None:
            fig, ax = plt.subplots(1,1)
        else:
            ax = fig.gca()
        fig1, ax1 = plt.subplots(1,1)
        S, PS, PS_err = self.PS_nij.S, self.PS_nij.PS, self.PS_nij.PS_err
        #ax.loglog(S, PS, 'o', label=r"$\Delta t = 0.2\ s$")
        params, errors, ier, S_calc, PS_calc = get_best_fit(S, PS, PS_err, 
                            n_params=3, p0=None, min_index=0, max_index=-1)
        if ier != 0 and False:
            print("N_ij distribution")
            ax.loglog(S_calc, PS_calc, '--', color="C0")
        for i, delta_T in enumerate(deltas_T[::-1]):
            sizes = np.array([])
            print(delta_T)
            #im = self.cluster2D_dt[delta_T]
            im = self.event2D_dt[delta_T]
            switches = np.unique(im)[1:]
            regions = measure.regionprops(im)
            for region in regions:
                q = region.image
            #for switch in switches:
            #    q = im == switch
                clusters, n_cluster = mahotas.label(q, np.ones((3,3)))
                sz = mahotas.labeled.labeled_size(clusters)[1:]
                sizes = np.concatenate((sizes, sz))
            sizes = sizes[sizes > min_size]
            S, PS, PS_err = gLD.logDistribution(sizes, log_step=log_step, normed=True)
            # Plot the distribution of the cluster area
            ax.loglog(S, 10**i*PS, 'o', label=r"$\Delta t = %.2f\ s$" % delta_T)
            ax1.loglog(S, PS, 'o', label=r"$\Delta t = %.2f\ s$" % delta_T)
            params, errors, ier, S_calc, PS_calc = get_best_fit(S, PS, PS_err, 
                            n_params=3, p0=None, min_index=1, max_index=-1)
            if not i:
                params0 = params
            if ier != 0 and True:
                ax.loglog(S_calc, 10**i*PS_calc, '--', color="C%d" % (i))
            PS_linear = 10**i*params0[0] * S[:-5]**(-params0[1])
            ax.loglog(S[:-5], PS_linear, '-', color="C%d" % (i))
        for _ax in [ax, ax1]:
            _ax.legend(loc=0)
            _ax.set_xlabel("$S_{Clust}$", size=20)
            _ax.set_ylabel("$P(S_{Clust})$", size=20)
            _ax.set_title("Size distribution")
            _ax.grid(False)
        plt.show()


def get_cluster2D_start(cluster2D, events2D):
    """
    get the Kolton's clusters
    """
    cluster2D_start = np.copy(cluster2D)
    # 1. Get the min and max time for each cluster of cluster2D
    switches = np.unique(cluster2D)[1:] # These are NOT the times
    _switches = []
    for switch in switches:
        cluster = cluster2D == switch
        _switches.append(np.min(events2D[cluster]))


def plot_images(hname, outDir=False):
    store = pd.HDFStore(hname)
    w, h = 70*multiplier, 40*multiplier
    ff = 1.6
    #xy = [(20,20), (w-68,20), (20,h-12), (w-68,h-12)] # Single image
    xy = [(20,25), (w-100,25), (20,h-20), (w-100,h-20)] # Four images
    limits = [(1,-1), (130,-1), (1,-200), (180, -90)]
    #limits = 4*[(1,-1)]
    fig, axs = plt.subplots(2,2, figsize=(7*ff,4*ff), squeeze=False)
    for i, current in enumerate(currents):
        ax = axs[i//2,i%2]
        #fig, ax = plt.subplots(1,1, figsize=(7*ff,4*ff))
        n_ij = nij_s[current]
        group = "%s/%s/df_%s/nij_%s/%s" % (current, n_set, d_f, n_ij, 'cluster2D_nij')
        print(group)
        lb = "%s mT" % fields[current]
        q = store.get(group)
        im = q.values
        r0, c0 = centers[current]
        #print(-(i<2)*h, (i>1)*h, ((i%2-1)*w), (i%2)*w)
        im = im[r0 - (i<2)*h : r0 + (i>1)*h + 1, c0 + (i%2-1)*w : c0 + (i%2)*w + 1]
        q = np.unique(im)[1:]
        i0, i1 = limits[i]
        n_max = q[i1]
        n_min = q[i0]
        print(n_min, n_max)
        if not i:
            n = n_max - n_min + 1
            p = getPalette(n, 'random', 'black')
            cm = colors.ListedColormap(p, 'pColorMap')
        im[im<n_min] = -1
        im[im>n_max] = -1
        ax.imshow(im, cmap=cm)
        #print(ax.axis())
        ax.axis((0, w, h, 0))
        for label in q:
            out = measure.find_contours(im==label, 0.5)
            for contour in out:
                X, Y = contour[:,1], contour[:,0]
                ax.plot(X,Y, 'k-')
        field = fields[current]
        bbox_props = dict(boxstyle="square,pad=0.2", fc="k", ec="k", lw=1)
        ax.annotate('%s mT' % field, xy=xy[i], size=16, color='white', bbox=bbox_props)
        ax.set_aspect('equal')
        if i == 0:
            px = 50
            x0, y0 = w-15-px, 222
            ax.annotate("",
                xy=(x0, y0), xycoords='data',
                xytext=(x0+px, y0), textcoords='data',
                arrowprops=dict(arrowstyle="|-|, widthA=0.5, widthB=0.5", color='white',
                                connectionstyle="arc3"))
            ax.annotate(r'$%i\ px$' % px, xy=(x0+7, y0-8), 
                xycoords='data', size=11, color='white')
        elif i == 1:
            um, x0, = 20, 15
            xw, y0 = int(um/0.3), 222
            ax.annotate("",
                xy=(x0, y0), xycoords='data',
                xytext=(x0+xw, y0), textcoords='data',
                arrowprops=dict(arrowstyle="|-|, widthA=0.5, widthB=0.5", color='white',
                                connectionstyle="arc3"))
            ax.annotate(r'$%i\ \mu m$' % um, xy=(x0+12, y0-8), 
                xycoords='data', size=11, color='white')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.format_coord = Formatter(im)
        fig.tight_layout()
    plt.show()
    store.close()
    save_data = raw_input("Save images? ")
    if save_data.upper() == 'Y':
        if not outDir:
            outDir = "."
        for _exp in ['pdf', 'png']:
            outname = os.path.join(outDir, "four_images."+_exp)
            fig.savefig(outname)


#####################################################################
if __name__ == "__main__":
    current = sys.argv[1]
    if not current in currents:
        print("Available currents: %s" % currents) 
        sys.exit()
    hname = '/data/Meas/Creep/CoFeB/Film/SuperSlowCreep/NonIrr/Feb2018/Results_NonIrr_Feb2018.hdf5'
    outDir = '/data/Meas/Creep/CoFeB/Film/SuperSlowCreep/NonIrr/Results_Feb2018/General/For_paper'
    #currents = currents[:2]
    #plot_images(store, outDir)
    # sequence == 'power':
    #deltas_T = 2**np.arange(7) * 1.875 # Using original Kolton sequence
    deltas_T = 2**np.arange(7) * 3 # Using original Kolton sequence
    # sequence == 'linear':
    #deltas_T = 2*np.arange(1,8) * 3.75 
    #self.deltas_T = 2*np.arange(1,8) * 0.1
    #self.deltas_T = np.arange(5,12) * 0.05
    deltas_T = [0.5, 0.75, 1., 1.5, 3., 6., 12.]
    kl = Klusters(hname, current=current, deltas_T=deltas_T)
    limits = [150, 550, 150, 550]
    #limits = [600, 1000, 100, 500]
    #limits = None
    kl.plot_image_at_deltaT(False, limits)
    #kl.plot_distributions_at_deltaT()