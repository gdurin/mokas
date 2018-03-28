import sys, os
import numpy as np
import getLogDistributions as gLD
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from collections import OrderedDict, defaultdict
import h5py
import getAxyLabels as gal
import pandas as pd
from skimage import measure
import mokas_cluster_methods as mcm
import mahotas
import mokas_parser as mkp
import mokas_bestfit as bestfit
from mokas_colors import getPalette
import ast


class Clusters:

    def __init__(self, mainDir, hdf5_filename, field, n_experiments, fieldDir=None,
                set_n = None, min_size=5, skip_first_clusters=0, fname_type='date', crops=None):
        """
        it is important to set a min_size of the cluster (in pixels). 
        The value of 10 seems reasonable, but can be larger
        """
        self._fname = os.path.join(mainDir, hdf5_filename)
        if not os.path.isfile(self._fname):
            print(self._fname)
            print("Check the path")
            sys.exit()
        self.min_size = min_size
        self.skip_first_clusters = skip_first_clusters
        self.mainDir = mainDir
        filename, ext = os.path.splitext(hdf5_filename)
        self.n_experiments = n_experiments
        self._len_experiments = len(n_experiments)
        self.title = filename + " - " + " - ".join(field.split("/")) + " - %i exps." % self._len_experiments
        if set_n is None:
            self._baseGroup = field
        else:
            set_n = set_n.capitalize()
            self._baseGroup = "%s/%s" % (set_n, field)
        self.set_n = set_n
        self.crops = crops
        if fieldDir is None:
            self.fieldDir = os.path.join(mainDir, field)
        else:
            self.fieldDir = fieldDir
        self.cluster2D_start = OrderedDict()
        self.cluster2D_end = OrderedDict()
        self.switchTimes2D = OrderedDict()
        self.times = OrderedDict()
        self.last_contour = OrderedDict()
        with h5py.File(self._fname, 'a') as f:
            grp_base = f[self._baseGroup]
            saved_experiments = len(grp_base)
            if len(n_experiments) > saved_experiments:
                print("There are only %i/%i experiments" % (saved_experiments, self._len_experiments))
                self.n_experiments = self.n_experiments[:saved_experiments]
            for n_exp in self.n_experiments:
                grp0 = self._baseGroup + "/%02i" % n_exp
                grp_n_exp = f[grp0]
                if "cluster2D_start" in grp_n_exp:
                    self.cluster2D_start[n_exp] = grp_n_exp['cluster2D_start'][...]
                    self.cluster2D_end[n_exp] = grp_n_exp['cluster2D_end'][...]
                    self.switchTimes2D[n_exp] = grp_n_exp['switchTimes2D'][...]
                else:
                    print("Cluster2D does not exist for exp: %i" % n_exp)
                # Get the last contour
                last_switch = np.unique(self.switchTimes2D[n_exp])[-1]
                grp_n_exp_contour = f[grp0 + "/contours/%i" % last_switch]
                self.last_contour[n_exp] = grp_n_exp_contour[...]
                # Check if times exist
                if 'times' in grp_n_exp:
                    times = grp_n_exp['times'][...]
                else:
                    times = self._get_times(grp_n_exp)
                    grp_n_exp.create_dataset("times", data=times, dtype=np.float16)
                self.times[n_exp] = times
                # read data of measure
                if 'mu_per_pixel' in grp_base.attrs.keys():
                    self.mu_per_pixel = grp_base.attrs['mu_per_pixel']
                else:
                    fname_measure = os.path.join(self.fieldDir, "measure.txt")
                    #print(fname_measure)
                    p = mkp.Parser(fname_measure)
                    data = p.get_data()
                    self.um_per_pixel = data['um_per_pixel']
                    grp_base.attrs.create('um_per_pixel', self.um_per_pixel)
        
    def _get_times(self, grp):
        is_success = False
        root_dir = grp.attrs['root_dir']
        pattern = grp.attrs['pattern']
        pattern = pattern.replace(".ome.tif", "_metadata.txt")
        fname = os.path.join(root_dir, pattern)
        if not os.path.isfile(fname):
            r = root_dir.strip("/")
            q, sub_dir = os.path.split(r)
            fname = os.path.join(self.mainDir, self._baseGroup, sub_dir, pattern)
        with open(fname, 'r') as f:
            q = f.read()
        q = q.replace("null", "False")
        q = q.replace("false", "False")
        d = ast.literal_eval(q)
        times = np.array([float(d[k]["ElapsedTime-ms"]) for k in d if k!='Summary'])
        times.sort()
        times = (times - times[0]) / 1000.
        return times
        
    def get_event_stats(self, min_size=1, former_switches=10):
        """
        get the distribution of the events
        (switches between two frames)
        and the associated Baiesi's statistics
        https://arxiv.org/pdf/cond-mat/0309485.pdf
        """
        print("Get the statistics of the events for each experiment")
        self.event_data = dict()
        self.event_data_shuffle = dict()
        self.event_cdf = dict()
        t0 = 0.
        self.labeledEvents2D = {}
        label = 0
        ######################################
        for j, n_exp in enumerate(self.switchTimes2D):
            print("Experiment: %i" % n_exp)
            event_data = defaultdict(list)
            event_data_shuffle = defaultdict(list)
            switchTimes2D = self.switchTimes2D[n_exp]
            le = -np.ones_like(switchTimes2D)
            event_switches = np.unique(switchTimes2D)[1:] # the -1 are not considered!
            if self.set_n:
                if j == 0:
                    T = self.times[n_exp][-1]
                    delta_T = np.mean(np.diff(self.times[n_exp][-5:]))
                    sw0 = event_switches[0]
                else:
                    t0 += T + delta_T
                    T = self.times[n_exp][-1]
                    delta_T = np.mean(np.diff(self.times[n_exp][-5:]))
                    sw0 += event_switches[-1] + 1
                    print(sw0)
            ############################################
            for switch in event_switches:
                q = switchTimes2D == switch
                clusters, n_cluster = mahotas.label(q, np.ones((3,3)))
                centers_of_mass = mahotas.center_of_mass(q, clusters)
                time = self.times[n_exp][switch] + t0
                sw = switch + sw0
                for i in range(1, n_cluster+1):
                    cluster = clusters == i
                    # Get the area of each cluster
                    area = mahotas.labeled.labeled_size(cluster)[1]
                    if area < min_size:
                        continue
                    position = centers_of_mass[i]
                    # Save the data
                    event_data['n_exp'].append(n_exp)
                    event_data['switch_frame'].append(switch + sw0)
                    event_data['switch_time'].append(time)
                    event_data['event_size'].append(area)
                    event_data['event_positionX'].append(position[0])
                    event_data['event_positionY'].append(position[1])
                    event_data['event_label'].append(label)
                    le[cluster] = label
                    if label == -49:
                        plt.figure()
                        plt.imshow(cluster, 'gray')
                        #plt.imshow(self.labeledEvents2D[n_exp][cluster], 'gray')
                        plt.plot(position[1], position[0], 'ro')
                        plt.show()
                        self.cluster = cluster
                    label += 1
            self.labeledEvents2D[n_exp] = le

            # How to shuffle the rows
            # https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
            #df_events_shuffle = df_events.sample(frac=1).reset_index(drop=True)
            p = np.random.permutation(len(event_data['event_size']))
            es = np.array(event_data['event_size'])
            epx = np.array(event_data['event_positionX'])
            epy = np.array(event_data['event_positionY'])
            epl = np.array(event_data['event_label'])
            event_data_shuffle['event_size'] = list(es[p])
            event_data_shuffle['event_positionX'] = list(epx[p])
            event_data_shuffle['event_positionY'] = list(epy[p])
            event_data_shuffle['event_label'] = list(epl[p])
            event_data_shuffle['n_exp'] = np.copy(event_data['n_exp'])
            event_data_shuffle['switch_time'] = np.copy(event_data['switch_time'])
            event_data_shuffle['switch_frame'] = np.copy(event_data['switch_frame'])
            # ######################################################################
            self.event_data[n_exp] = pd.DataFrame.from_dict(event_data)
            self.event_data_shuffle[n_exp] = pd.DataFrame.from_dict(event_data_shuffle)
        self.all_events = pd.concat([self.event_data[cl] for cl in self.event_data])
        self.all_events_shuffle = pd.concat([self.event_data_shuffle[cl] for cl in self.event_data_shuffle])
        self.cdf = self.get_cdf(self.all_events.event_size)
        return 

    def _get_nij(self, record, df, frac_dim=1):
        """
        find the minumum n_ij for an event j
        identified by record
        and the previous events i
        identified by a sub dataframe 
        (with time < time_j)
        return index and value of the min and the whole n_ij
        """
        t = record.switch_time - df.switch_time
        X0, Y0 = record.event_positionX, record.event_positionY
        X, Y = df.event_positionX, df.event_positionY
        l = ((X-X0)**2 + (Y-Y0)**2)**0.5
        # rounding error
        p = 1 - self.cdf[df.event_size]
        p[p < 0] = np.abs(p)
        n_ij = t * l**frac_dim * p.values
        #pos = n_ij.values.argmin()
        idxmin_n_ij, min_n_ij = int(n_ij.idxmin()), n_ij.min()
        label = df.event_label.loc[idxmin_n_ij]
        if np.isnan(min_n_ij):
            return 3 * [None]
        else:
            return idxmin_n_ij, int(label), min_n_ij

    def get_correlation(self, df, event_size_threshold=1, 
                        frac_dim=1, former_switches=None, label=None):
        """
        df is the dataFrame:
        it can be self.all_events or self.all_events_shuffle
        former_switches: int
            set the number of previous switches to consider n_ij
        """
        #connected_to = []
        s = "Getting the event correlation of "
        if label:
            s += label
        print(s)
        n_ij = {}
        connected_to = defaultdict(list)
        for n_exp in self.n_experiments:
            i_n_exp, record_i, idx_n_ij, nij = [], [], [], []
            q = df[df.n_exp==n_exp]
            q = q[q.event_size >= event_size_threshold]
            times = np.unique(q.switch_time)
            # Find the index of the first event at the second time
            first_index = (q.switch_time==times[1]).idxmax()
            #connected_to += first_index*[np.NaN]
            for i in q.index[q.index >= first_index]:
                record = q.loc[i]
                #time_i = record.switch_time
                #sub_q = q[q.switch_time<time_i]
                switch_i = int(record.switch_frame)
                sub_q = q[q.switch_frame < switch_i]
                if former_switches:
                    sub_q = sub_q[sub_q.switch_frame > (switch_i - former_switches)]
                if sub_q.empty:
                    print("%i frame is empty" % i)
                try:
                    idxmin_n_ij, label_n_ij, min_n_ij = self._get_nij(record, sub_q, frac_dim)
                except AttributeError:
                    print(record)
                    print(sub_q)
                    sys.exit()
                if min_n_ij:
                    connected_to['n_exp'].append(n_exp)
                    connected_to['event_idx'].append(i)
                    connected_to['event_label'].append(int(record.event_label))
                    connected_to['father_idx'].append(idxmin_n_ij)
                    connected_to['father_label'].append(label_n_ij)
                    connected_to['n_ij'].append(min_n_ij)                    
            #n_ij[n_exp] = nij
        cols = ['n_exp', 'event_idx', 'event_label', 'father_idx', 'father_label', 'n_ij']
        connected_to_df = pd.DataFrame(connected_to, columns=cols)
        return connected_to_df

    def show_correlation(self, event_size_threshold=5, frac_dim=1, former_switches=None, dx=0.05):
        label = "real data"
        #self.n_ij = self.get_correlation(self.all_events, event_size_threshold, frac_dim, former_switches, label)
        self.con_to_df = self.get_correlation(self.all_events, event_size_threshold, frac_dim, former_switches, label)
        label = "shuffled data"
        self.con_to_df_shuffled = self.get_correlation(self.all_events_shuffle, event_size_threshold, frac_dim, former_switches, label)
        #self.n_ij_shuffle = self.get_correlation(self.all_events_shuffle, event_size_threshold, frac_dim, former_switches, label)
                            
        # Plot them all
        bins = np.arange(0,5.+dx,dx)
        #if self.set_n:
        if self.set_n:
            # n_ij, n_ij_shuffle = [], []
            # for n_exp in self.n_experiments:
            #     n_ij += self.n_ij[n_exp]
            #     n_ij_shuffle += self.n_ij_shuffle[n_exp]
            n_ij = self.con_to_df['n_ij']
            n_ij_shuffled = self.con_to_df_shuffled['n_ij']
            fig, ax = plt.subplots(1, 1)
            ax.hist(n_ij, bins=bins, alpha=0.5, label='real data')
            ax.hist(n_ij_shuffled, bins=bins, alpha=0.5, label='shuffled')
            ax.legend()
        else:
            n_figs = len(self.n_experiments)
            rows, cols = self._get_rows_cols(n_figs)
            fig, axs = plt.subplots(rows, cols, sharex=True, sharey=True, squeeze=False)
            for i,n_exp in enumerate(self.n_experiments):
                #print(n_ij[n_exp])
                ax = axs[i//cols, i%cols]
                ax.hist(self.n_ij[n_exp], bins=bins, alpha=0.5, label='real data')
                ax.hist(self.n_ij_shuffle[n_exp], bins=bins, alpha=0.5, label='shuffled')
                ax.legend()
                ax.set_title("Experiment n. %i " % n_exp)
        plt.show()

    def _get_gfather_label(self, df, label):
        if label not in df.event_label.values:
            return label
        else:
            idx = df[df.event_label == label].index[0]
            label = df.father_label.loc[idx]
            return self._get_gfather_label(df, label)

    def _get_clusters_nij(self, df, max_nij):
        c = df.copy()
        c = c[c.n_ij <= max_nij]
        q = [self._get_gfather_label(c, label) for label in c.father_label]
        c['gfather_label'] = q
        return c

    def show_clusters_nij(self, df, max_nij):
        c = self._get_clusters_nij(df, max_nij)
        data = zip(c.n_exp, c.event_label, c.gfather_label)
        for n_exp, label, new_label in data:
            q = self.labeledEvents2D[n_exp] == label
            self.labeledEvents2D[n_exp][q] = new_label
        self.plot_cluster_maps(self.cluster2D_start, self.labeledEvents2D)

    def _get_rows_cols(self, n_figs):
        """
        I am supposing there are no more 
        than 10 figs
        """
        if n_figs > 10:
            print("too many figures to do")
            return 6,6
        i = n_figs - 1
        rows = np.array(3*[1] + 5 * [2] + 2 * [3])
        cols = np.array([1,2,3,2,3,3,4,4,3,4])
        return rows[i], cols[i]

    def get_cdf(self, sizes):
        max_size = np.max(sizes)
        counts, bin_edges = np.histogram(sizes, bins=range(max_size+1), normed=True)
        cdf = np.cumsum(counts)
        return pd.Series(cdf, index=bin_edges[1:])

    def get_cluster_stats(self):
        """
        get the distribution of the clusters
        """
        print("Get the statistics of the clusters for each experiment")
        self.cluster_data = dict()
        t0 = 0.
        ######################################
        for j, n_exp in enumerate(self.cluster2D_start):
            print("Experiment: %i" % n_exp)
            cluster_data = defaultdict(list)
            ############################################
            cluster2D_start = self.cluster2D_start[n_exp]
            cluster2D_end = self.cluster2D_end[n_exp]
            cluster_switches = np.unique(cluster2D_start)[self.skip_first_clusters+1:] # the -1 are not considered!
            # To test simply write:
            # cluster_switches = np.unique(cluster2D_start)[1:]
            
            # This is also to test:             
            # areas = np.array([])
            # durations = np.array([])
            # durations_n0 = np.array([])
            delta_T = np.mean(np.diff(self.times[n_exp][-5:]))

            if self.set_n:
                if j == 0:
                    T = self.times[n_exp][-1]
                else:
                    t0 += T + delta_T
                    T = self.times[n_exp][-1]
                    delta_T = np.mean(np.diff(self.times[n_exp][-5:]))

            for switch in cluster_switches:
                time = self.times[n_exp][switch] + t0
                q = cluster2D_start == switch
                clusters, n_cluster = mahotas.label(q, np.ones((3,3)))
                for i in range(1, n_cluster+1):
                    cluster = clusters == i
                    # Get the area of each cluster
                    area = mahotas.labeled.labeled_size(cluster)[1]
                    if area < self.min_size:
                        continue
                    # To test:
                    # areas = np.concatenate((areas, area))
                    # durations = np.concatenate((durations, duration))
                    # durations_n0 = np.concatenate((durations_n0, duration_n0))
                    # assert len(areas) == len(durations)
                    cluster_contour = measure.find_contours(cluster, 0.5)  # need 0.5 because cluste is boolean
                    time_start = switch
                    time_end = np.max(np.extract(cluster, cluster2D_end))
                    duration = (time_end - time_start) * delta_T
                    # Save the data
                    cluster_data['n_exp'].append(n_exp)
                    cluster_data['switch_frame'].append(switch)
                    cluster_data['switch_time'].append(time)
                    cluster_data['cluster_area'].append(area)  
                    cluster_data['cluster_duration'].append(duration)
            
            cluster_cols = ['n_exp', 'switch_frame', 'switch_time', 'cluster_area', 'cluster_duration']
            df = pd.DataFrame.from_dict(cluster_data)
            self.cluster_data[n_exp] = df[cluster_cols]
            #del df
        self.all_clusters = pd.concat([self.cluster_data[cl] for cl in self.cluster_data])
        return df


    def plot_cluster_stats(self, log_step=0.1, n_params=3, p0=[1,1.1,300],
        min_index=2, max_index=-2):
        q = self.all_clusters
        sd = bestfit.Size_Distribution(n_params)
        S, PS, PS_err = gLD.logDistribution(q.cluster_area, log_step=log_step)
        w = PS != 0
        S, PS, PS_err = S[w], PS[w], PS_err[w]
        if min_index:
            S, PS, PS_err = S[min_index:], PS[min_index:], PS_err[min_index:]
        if max_index:
            S, PS, PS_err = S[:max_index], PS[:max_index], PS_err[:max_index]
        model = bestfit.Model(S, PS, sd.theory, p0, 'log')
        params, errors = model.get_params()
        for pars in zip(sd.params, params, errors):
            print("%s: %.2f +/- %.2f" % pars)
        # Plot the distribution of the cluster area
        fig, ax = plt.subplots(1,1)
        ax.loglog(S, PS, 'bo')
        ax.errorbar(S, PS, PS_err, fmt=None)
        S_calc = np.logspace(np.log10(np.min(S)), np.log10(np.max(S)), 2*len(S))
        ax.loglog(S_calc, sd.theory(params, S_calc), '--', label=sd.repr)
        ax.legend()
        ax.set_xlabel("$S_{Clust}$", size=20)
        ax.set_ylabel("$P(S_{Clust})$", size=20)
        ax.set_title("Cluster area distribution")
        ax.grid(True)

        # Plot the distribution of the cluster duration (with the zeros replaced by 0.1)
        fig, ax = plt.subplots(1,1)
        T, PT, PT_err = gLD.logDistribution(q.cluster_duration, log_step=log_step)
        ax.loglog(T, PT, 'bo')
        ax.errorbar(T, PT, PT_err, fmt=None)
        ax.set_xlim([0.5,100]) # Set the limits for the x-axis to avoid the wrong point at 0.1
        ax.set_xlabel("$\Delta t_{Clust}$", size=20)
        ax.set_ylabel("$P(\Delta t_{Clust})$", size=20)
        #ax.set_title("Ta(5 nm)/CoFeB(1 nm)/MgO(2 nm) - IrrID = 16 X 10$^{16}$ He/m$^{2}$ \n Cluster duration distribution")
        ax.grid(True)

        # Plot average cluster duration (the one with the zeros) vs cluster area
        unique_cluster_area = np.unique(q.cluster_area)
        average_cluster_duration = []
        for area in unique_cluster_area:
            # Note the use of.values to extract the values of the array from the df!!
            duration = np.mean(np.extract(q.cluster_area.values == area, q.cluster_duration), dtype=np.float)
            average_cluster_duration.append(duration)

        average_cluster_duration = np.array(average_cluster_duration)

        fig, ax = plt.subplots(1,1)
        plt.plot(unique_cluster_area, average_cluster_duration, 'bo')
        ax.set_xlabel("$S_{Clust}$", size=20)
        ax.set_ylabel("$\Delta t_{Clust} - ave$", size=20)
        #ax.set_title("title goes here")
        plt.show()
        return S, PS

    def _sum_maps(self, cluster2D, axs):
        n_exps = cluster2D.keys()[::-1]
        for i, n_exp in enumerate(n_exps):
            crops = self.crops[n_exp]
            [(r0, c0), (r1, c1)] = crops
            crops = np.array([r0, c0, r1, c1])
            im = cluster2D[n_exp]
            if not i:
                crops_ref = crops
                im[im==-1] = 0
                im_all = im
                i_0 = np.unique(im)[-1]
            else:
                im[im==-1] = 0
                #im[im!=0] += i_0
                r0, c0 = crops[:2] - crops_ref[:2]
                r1, c1 = crops[2:] - crops_ref[:2]
                im_all[c0:c1, r0:r1] += im
                i_0 = np.unique(im)[-1]
                # Plot the contours
                c = self.last_contour[n_exp]
                X,Y = r0 + c[:,1], c0 + c[:,0]
                for k,ax in enumerate(axs):
                    ax.plot(X, Y, 'k', lw=2-k)
        return im_all

    def plot_cluster_maps(self, cluster2D_1, cluster2D_2=None, palette='random'):
        """
        Here we assume that the last image
        is the largest (reasonably)
        """
        cnts = dict()
        i_0 = 0
        fig, axs = plt.subplots(1,2, sharex=True, sharey=True)
        im_all_1 = self._sum_maps(cluster2D_1, axs)
        if cluster2D_2 is not None:
            im_all_2 = self._sum_maps(cluster2D_2, axs)
        n = np.max(im_all_1.flatten())
        p = getPalette(n, 'random', 'black')
        cm = colors.ListedColormap(p, 'pColorMap')
        axs[0].imshow(im_all_1, cmap=cm)
        if cluster2D_2 is None:
            axs[1].imshow(im_all_1!=0, 'gray')
        else:
            axs[1].imshow(im_all_2, cmap=cm)
        plt.show()

def plot_cluster_stats_all_area(clusters, log_step=0.1):
    fig, ax = plt.subplots(1,1)
    for label in clusters:
        q = clusters[label].all_clusters
        
        # Plot the distribution of the cluster area
        x,y,yerr = gLD.logDistribution(q.cluster_area, log_step=log_step)
        plt.loglog(x,y, 'o', label=label)
    ax.set_xlabel("$S_{Clust}$", size=20)
    ax.set_ylabel("$P(S_{Clust})$", size=20)
    #ax.set_title("Ta(5 nm)/CoFeB(1 nm)/MgO(2 nm) - IrrID = 16 X 10$^{16}$ He/m$^{2}$ \n Cluster area distribution")
    ax.legend()
    ax.grid(True)
    plt.show()

        
if __name__ == "__main__":
    #plt.close("all")
    imParameters = {}
    n_experiments = {}
    clusters = {}
    crops = None
    choice = sys.argv[1]
    try:
        irradiation = sys.argv[1]
    except:
        irradiation = 'Irr_800uC'
    
    fieldDir = None
    set_n = None

    if irradiation == 'NonIrr_Dec16':
        # Logic updated Mar 7
        #mainDir = "/data/Meas/Creep/CoFeB/Film/SuperSlowCreep/NonIrr/Dec2016/"
        mainDir = "/data/Meas/Creep/CoFeB/Film/SuperSlowCreep/NonIrr/Dec2016/"
        hdf5_filename = "NonIrr.hdf5"

        #currents = ["0.116", "0.232"]
        currents = ["0.095"]
        ns_experiments = {"0.095": range(1,11)}
        min_size = 5

    elif irradiation == 'NonIrr_Dec18':
        #field = "0.137"
        field = "0.146"
        #field = "0.157"
        #field = "0.165"
        set_n = "Set1"
        mainDir = "/data/Meas/Creep/CoFeB/Film/SuperSlowCreep/NonIrr/Feb2018/%sA/" % field
        fieldDir = os.path.join(mainDir, set_n)
        hdf5_filename = "%sA.hdf5" % field
        currents = [field]
        ns_experiments = {"0.137": [2, 8, 15], "0.146": range(1,9), "0.157": [2,3,4,5], "0.165": range(2,5)}
        if field == "0.146":
            crop_upper_left_pixel = [(300,300),(200,200),(200,200),(200,200),(200,200),(100,100),(0,0),(0,0)]
            crop_lower_right_pixel = [(700,800),(800,900),(800,900),(800,900),(800,900),(900,1000),(1100,1000),(1100,1000)]
            crops = {}
            for i in range(len(crop_lower_right_pixel)):
                crops[i+1] = (crop_upper_left_pixel[i], crop_lower_right_pixel[i])
            nij_max = {"Set1": 0.32}
        else:
            crops = None       
        min_size = 5

    elif irradiation == 'Irr_800uC_Dec16':
        mainDir = "/data/Meas/Creep/CoFeB/Film/SuperSlowCreep/Irr_800uC/Dec2016/"
        hdf5_filename = "Dec2016.hdf5"
        #currents = ["0.116", "0.232"]
        currents = ["0.116"]
        #currents = ["0.232"]
        #ns_experiments = {"0.116": range(2,11), "0.232":range(1,11)}
        #ns_experiments = {"0.232": range(1,11)}
        #ns_experiments = {"0.116": [2,3,4,5,8,9,10]}
        ns_experiments = {"0.116": [2,3,4]}
        min_size = 5
        irradiation = irradiation[:-6]

    elif irradiation == 'Irr_800uC_Dec17':
        mainDir = "/data/Meas/Creep/CoFeB/Film/SuperSlowCreep/Irr_800uC/Dec2017/"
        #currents = ["0.22", "0.31"]
        currents = ["0.22"]
        ns_experiments = {"0.22": [1,2,3,4,6,10,11,12,13,14,15,16,17,18], "0.232":range(1,11)}
        min_size = 5
        irradiation = irradiation[:-6]

    elif irradiation == 'Irr_400uC_Dec16':
        mainDir = "/data/Meas/Creep/CoFeB/Film/SuperSlowCreep/Irr_400uC/Dec2016/"
        currents = ["0.06", "0.1","0.2"]
        #currents = ["0.22"]
        ns_experiments = {"0.06": [1,2,3], "0.1":range(1,11), "0.2":range(1,11)}
        min_size = 5
        irradiation = irradiation[:-6]

    for current in currents:    
        field  = "%sA" % (current)
        print("Analysing %s" % field)
        n_experiments = ns_experiments[current]
        #clusters[current] = Clusters(mainDir, grp0, n_experiments, irradiation, skip_first_clusters=0, min_size=min_size)
        c = Clusters(mainDir, hdf5_filename, field, n_experiments, set_n=set_n,
            fieldDir=fieldDir, skip_first_clusters=0, min_size=min_size, crops=crops)
        clusters[current] = c
        #c.get_cluster_stats()
        #S, PS = c.plot_cluster_stats()
        c.get_event_stats()
        c.show_correlation(event_size_threshold=5, dx=0.01)
        c.show_clusters_nij(c.con_to_df, 0.33)
        
    #plot_cluster_stats_all_area(clusters, log_step=0.1)