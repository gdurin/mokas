import sys, os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import h5py
import pandas as pd
import ast
import mahotas
from collections import OrderedDict, defaultdict, Counter
from skimage import measure
from pathlib import Path
# import local packages
import getAxyLabels as gal
import getLogDistributions as gLD
import mokas_cluster_methods as mcm
import mokas_parser as mkp
import mokas_bestfit as bestfit
from mokas_colors import getPalette


class Clusters:

    def __init__(self, mainDir, hdf5_filename, field, n_experiments, fieldDir=None,
                set_n = None, min_size=5, skip_first_clusters=0, fname_type='date'):
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
        if fieldDir is None:
            self.fieldDir = os.path.join(mainDir, field)
        else:
            self.fieldDir = fieldDir
        self.cluster2D_start = OrderedDict()
        self.cluster2D_end = OrderedDict()
        self.switchTimes2D = OrderedDict()
        self.cluster2D_nij = {}
        self.times = OrderedDict()
        self.last_contour = OrderedDict()
        self.regions = None
        self.centers_of_mass = {}
        self.crops = {}
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
                    sw = self.switchTimes2D[n_exp] > 0
                    self.centers_of_mass[n_exp] = mahotas.center_of_mass(sw)
                    self.crops[n_exp] = eval(grp_n_exp.attrs['crop'])
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
        print(60*"*")
        print("Get the statistics of the events for each experiment")
        self.event_data = dict()
        self.event_data_shuffle = dict()
        self.event_cdf = dict()
        t0 = 0.
        self.labeledEvents2D = {}
        event_label = 0L
        ######################################
        for j, n_exp in enumerate(self.n_experiments):
            print("Experiment: %i" % n_exp)
            event_data = defaultdict(list)
            event_data_shuffle = defaultdict(list)
            switchTimes2D = self.switchTimes2D[n_exp]
            le = -np.ones_like(switchTimes2D).astype(np.int32)
            #print(le.shape)
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
                    #print(sw0)
            ############################################
            print("Event label: %i" % event_label)
            for switch in event_switches:
                q = switchTimes2D == switch
                events, n_events = mahotas.label(q, np.ones((3,3)))
                centers_of_mass = mahotas.center_of_mass(q, events)
                sizes = mahotas.labeled.labeled_size(events)
                time = self.times[n_exp][switch] + t0
                sw = switch + sw0
                for i in range(1, n_events+1):
                    if sizes[i] < min_size:
                        continue
                    event = events == i
                    posX, posY = centers_of_mass[i]
                    size = sizes[i]
                    # Save the data
                    event_data['n_exp'].append(n_exp)
                    event_data['switch_frame'].append(switch + sw0)
                    event_data['switch_time'].append(time)
                    event_data['event_size'].append(size)
                    event_data['event_positionX'].append(posX)
                    event_data['event_positionY'].append(posY)
                    event_data['event_label'].append(event_label)
                    le[event] = event_label
                    event_label += 1L
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
        self.p_large_size = self._get_prob_p()
        self._cluster2D_nij = self._sum_maps(self.labeledEvents2D)
        return 

    def _get_prob_p(self):
        """
        get the probability to be larger than a given s
        it solves the problems when p ~ 1
        """
        p = 1 - self.cdf
        w = p < 1./self.max_size
        p1 = 1./p.index.values
        p[p.index[w]] = p1[w]
        return p


    def _get_nij(self, record, df, frac_dim=1, distance='euclidean'):
        """
        find the minumum n_ij for an event j
        identified by record
        and the previous events i
        identified by a sub dataframe 
        (with time < time_j)
        return index and value of the min and the whole n_ij
        distance : str
            Can be 'euclidean' or 'arc'
        """
        t = record.switch_time - df.switch_time
        X0, Y0 = record.event_positionX, record.event_positionY
        X, Y = df.event_positionX, df.event_positionY
        if distance == 'euclidean':
            l = ((X-X0)**2 + (Y-Y0)**2)**0.5
        elif distance == 'arc':
            n_exp = record.n_exp.astype(int)
            Xc, Yc = self.centers_of_mass[n_exp]
            x0, y0 = (X0 - Xc), (Y0 - Yc)
            theta0 = np.arctan2(y0, x0)
            x, y = (X - Xc), (Y - Yc)
            thetas = np.arctan2(y, x)
            dthetas = np.abs(thetas - theta0)
            R0 = (x0*x0 + y0*y0)**0.5
            Rs = (x*x + y*y)**0.5
            r_s = dthetas * R0
            dR = np.abs(Rs - R0)
            l = (dR * dR + r_s * r_s)**0.5
        elif distance == 'solidity':
            pass
        p = self.p_large_size[df.event_size]
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
        print(60*"*")
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
                    idxmin_n_ij, label_n_ij, min_n_ij = self._get_nij(record, sub_q, frac_dim, distance='arc')
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
            h_ij, h_bins, patches = ax.hist(n_ij, bins=bins, alpha=0.5, label='real data')
            self.h_ij_real = h_ij_real = pd.Series(h_ij, index=h_bins[:-1])
            h_ij, h_bins, patches = ax.hist(n_ij_shuffled, bins=bins, alpha=0.5, label='shuffled')
            self.h_ij_shuffled = pd.Series(h_ij, index=h_bins[:-1])
            ax.set_xlabel(r"$n_{ij}$", size=26)
            ax.set_ylabel(r"$hist(n_{ij})$", size=26)
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
                ax.set_xlabel(r"$n_{ij}$", size=26)
                ax.set_ylabel(r"$hist(n_{ij})$", size=26)
                ax.legend()
                ax.set_title("Experiment n. %i " % n_exp)
        plt.show()



    def _get_gfather_label2(self, _series, label):
        if label not in _series.index:
            return label
        else:
            label = _series[label]
            return self._get_gfather_label2(_series, label)

    def _get_clusters_nij(self, df, max_nij):
        c = df.copy()
        c = c[c.n_ij <= max_nij]
        #q = [self._get_gfather_label(c, label) for label in c.father_label]
        q = []
        for n_exp in self.n_experiments:
            _c = c[c.n_exp == n_exp]
            p = _c.father_label
            p.index = _c.event_label
            q += [self._get_gfather_label2(p, label) for label in p]
        c['gfather_label'] = q
        return c

    def get_clusters_nij(self, df, max_nij, title=None):
        print("Getting cluster with n_ij")
        c = self._get_clusters_nij(df, max_nij)
        print("Preparing the map")
        _cluster2D_nij = self._cluster2D_nij.copy()
        for label, new_label in zip(c.event_label, c.gfather_label):
            q = _cluster2D_nij == label
            _cluster2D_nij[q] = new_label
        print("Plotting the maps")
        return _cluster2D_nij

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
        self.max_size = np.max(sizes)
        counts, bin_edges = np.histogram(sizes, bins=range(self.max_size+1), normed=True)
        cdf = np.cumsum(counts)
        return pd.Series(cdf, index=bin_edges[1:])

    def get_cluster_stats_from_nj(self, image2D, _string=None):
        print(60*"*")
        print("Get the statistics of the clusters for each experiment %s" % _string)
        cluster_data = dict()
        t0 = 0.
        cluster_cols = ['n_exp', 'switch_frame', 'switch_time', 'cluster_size', 'cluster_duration']
        _cluster_data = defaultdict(list)
        ############################################
        cluster2D_start = image2D.copy()
        cluster2D_end = None
        cluster_switches = np.unique(cluster2D_start)[1:] # the -1 are not considered!
        time = np.NAN
        n_exp = np.NAN
        for switch in cluster_switches:
            size = np.sum(cluster2D_start==switch)
            if size < self.min_size:
                continue
            duration = np.NAN
            # Save the data
            _cluster_data['n_exp'].append(n_exp)
            _cluster_data['switch_frame'].append(switch)
            _cluster_data['switch_time'].append(time)
            _cluster_data['cluster_size'].append(size)
            _cluster_data['cluster_duration'].append(duration)        
        all_clusters = pd.DataFrame(_cluster_data, columns=cluster_cols)
        return all_clusters

    def get_cluster_stats_from_nj2(self, image2D, _string=None):
        print(60*"*")
        print("Get the statistics of the clusters for each experiment %s" % _string)
        cluster_cols = ['cluster_label', 'cluster_size', 'cluster_maj_ax_len', 'cluster_min_ax_len']
        _cluster_data = defaultdict(list)
        ############################################
        self.regions = measure.regionprops(image2D)
        regions = [reg for reg in self.regions if reg.area >= self.min_size]
        _cluster_data['cluster_label'] = [props.label for props in regions]
        _cluster_data['cluster_size'] = [props.area for props in regions]
        _cluster_data['cluster_maj_ax_len'] = [props.major_axis_length for props in regions]
        _cluster_data['cluster_min_ax_len'] = [props.minor_axis_length for props in regions]
        all_clusters = pd.DataFrame(_cluster_data, columns=cluster_cols)
        return all_clusters

    def get_cluster_stats_from_dict(self, images2D, _string=None):
        """
        get the distribution of the clusters
        images2D is a dictionary of the cluster maps
        it can be 
        self.cluster2D_start or self.labeledEvents2D
        """
        print(60*"*")
        print("Get the statistics of the clusters for each experiment %s" % _string)
        cluster_data = dict()
        t0 = 0.
        cluster_cols = ['n_exp', 'switch_frame', 'switch_time', 'cluster_size', 'cluster_duration']
        ######################################
        for j, n_exp in enumerate(self.n_experiments):
            print("Experiment: %i" % n_exp)
            _cluster_data = defaultdict(list)
            ############################################
            cluster2D_start = images2D[n_exp]
            if images2D is self.cluster2D_start:
                cluster2D_end = self.cluster2D_end[n_exp]
            else:
                cluster2D_end = None
            cluster_switches = np.unique(cluster2D_start)[self.skip_first_clusters+1:] # the -1 are not considered!
            delta_T = np.mean(np.diff(self.times[n_exp][-5:]))
            if self.set_n:
                if j == 0:
                    T = self.times[n_exp][-1]
                else:
                    t0 += T + delta_T
                    T = self.times[n_exp][-1]
                    delta_T = np.mean(np.diff(self.times[n_exp][-5:]))

            for i, switch in enumerate(cluster_switches):
                if images2D is self.cluster2D_start:
                    time = self.times[n_exp][switch] + t0
                else:
                    time = np.NAN
                # ###################################
                q = cluster2D_start == switch
                clusters, n_cluster = mahotas.label(q, np.ones((3,3)))
                for i in range(1, n_cluster+1):
                    cluster = clusters == i
                    # Get the area of each cluster
                    size = mahotas.labeled.labeled_size(cluster)[1]
                    if size < self.min_size:
                        continue
                    # To test:
                    # areas = np.concatenate((areas, area))
                    # durations = np.concatenate((durations, duration))
                    # durations_n0 = np.concatenate((durations_n0, duration_n0))
                    # assert len(areas) == len(durations)
                    cluster_contour = measure.find_contours(cluster, 0.5)  # need 0.5 because cluste is boolean
                    # time_start = switch
                    if images2D is cluster2D_start:
                         time_end = np.max(np.extract(cluster, cluster2D_end))
                         duration = (time_end - time_start) * delta_T
                    else:
                         duration = np.NAN
                    # Save the data
                    _cluster_data['n_exp'].append(n_exp)
                    _cluster_data['switch_frame'].append(switch)
                    _cluster_data['switch_time'].append(time)
                    _cluster_data['cluster_size'].append(size)  
                    _cluster_data['cluster_duration'].append(duration)            
            cluster_data[n_exp] = pd.DataFrame(_cluster_data, columns=cluster_cols)
        all_clusters = pd.concat([cluster_data[n_exp] for n_exp in self.n_experiments])
        return all_clusters


    def plot_cluster_stats(self, cluster_sizes, log_step=0.1, n_params=3, p0=None,
        min_index=2, max_index=-2, fig=None, lb=None, color='b'):
        sd = bestfit.Size_Distribution(n_params)
        S, PS, PS_err = gLD.logDistribution(cluster_sizes, log_step=log_step)
        S, PS, PS_err = S[min_index:max_index], PS[min_index:max_index], PS_err[min_index:max_index]
        w = PS != 0
        S, PS, PS_err = S[w], PS[w], PS_err[w]
        model = bestfit.Model(S, PS, theory=sd, p0=p0, y_err=None, linlog='log', use_jacobian=False)
        params, errors, ier = model.get_params()
        if ier in range(1,5) and errors is not None:
            for pars in zip(sd.params, params, errors):
                print("%s: %.2f +/- %.2f" % pars)
        else:
            for pars in zip(sd.params, params):
                print("%s: %.2f" % pars)
        # Plot the distribution of the cluster area
        if fig is None:
            fig, ax = plt.subplots(1,1)
        else:
            ax = fig.gca()
        ax.loglog(S, PS, 'o', color=color, label=lb)
        ax.errorbar(S, PS, PS_err, fmt="none")
        S_calc = np.logspace(np.log10(np.min(S)), np.log10(np.max(S)), 2*len(S))
        if ier != 0:
            ax.loglog(S_calc, sd.y(params, S_calc), '--', color=color)
        ax.legend(loc=3)
        ax.set_xlabel("$S_{Clust}$", size=20)
        ax.set_ylabel("$P(S_{Clust})$", size=20)
        ax.set_title("Size distribution")
        ax.grid(True)

        # Plot the distribution of the cluster duration (with the zeros replaced by 0.1)
        # fig, ax = plt.subplots(1,1)
        # T, PT, PT_err = gLD.logDistribution(q.cluster_duration, log_step=log_step)
        # ax.loglog(T, PT, 'bo')
        # ax.errorbar(T, PT, PT_err, fmt=None)
        # ax.set_xlim([0.5,100]) # Set the limits for the x-axis to avoid the wrong point at 0.1
        # ax.set_xlabel("$\Delta t_{Clust}$", size=20)
        # ax.set_ylabel("$P(\Delta t_{Clust})$", size=20)
        # #ax.set_title("Ta(5 nm)/CoFeB(1 nm)/MgO(2 nm) - IrrID = 16 X 10$^{16}$ He/m$^{2}$ \n Cluster duration distribution")
        # ax.grid(True)

        # Plot average cluster duration (the one with the zeros) vs cluster area
        # unique_cluster_area = np.unique(cluster_sizes)
        # average_cluster_duration = []
        # for area in unique_cluster_area:
        #     # Note the use of.values to extract the values of the array from the df!!
        #     duration = np.mean(np.extract(cluster_sizes == area, q.cluster_duration), dtype=np.float)
        #     average_cluster_duration.append(duration)

        # average_cluster_duration = np.array(average_cluster_duration)

        # fig, ax = plt.subplots(1,1)
        # plt.plot(unique_cluster_area, average_cluster_duration, 'bo')
        # ax.set_xlabel("$S_{Clust}$", size=20)
        # ax.set_ylabel("$\Delta t_{Clust} - ave$", size=20)
        # #ax.set_title("title goes here")
        plt.show()
        S, PS, PS_err
        d = {'S': S, 'PS': PS, 'PS_err': PS_err}    
        df_PS = pd.DataFrame(d, columns=['S', 'PS', 'PS_err'])
        return df_PS, fig

    def _sum_maps(self, cluster2D, axs=None):
        n_exps = cluster2D.keys()[::-1]
        for i, n_exp in enumerate(n_exps):
            [(r0, c0), (r1, c1)] = self.crops[n_exp]
            crops = np.array([r0, c0, r1, c1])
            im = cluster2D[n_exp].copy()
            if not i:
                crops_ref = crops
                #im[im==-1] = 0 # if use plus
                im_all = im.copy()
                i_0 = np.unique(im)[-1]
            else:
                #im[im==-1] = 0 # if use plus
                #im[im!=0] += i_0
                r0, c0 = crops[:2] - crops_ref[:2]
                r1, c1 = crops[2:] - crops_ref[:2]
                w = im!=-1
                im_all[c0:c1, r0:r1][w] = im[w] # Uauu!
                i_0 = np.unique(im)[-1]
                # Plot the contours
                if axs is not None:
                    c = self.last_contour[n_exp]
                    X,Y = r0 + c[:,1], c0 + c[:,0]
                    for ax in axs:
                        ax.plot(X, Y, 'k', lw=2)
        return im_all

    def clean_small_clusters(self, _image2D, pxl_step=1):
        """
        clean small clusters, defined as < self.min_size
        pxl_step : int
            n. of pixels around the small cluster
        """
        image2D = _image2D.copy()
        if not self.regions:
            self.regions = measure.regionprops(image2D)
        indx = [i for i,reg in enumerate(self.regions) if reg.area<self.min_size]
        for i in indx:
            reg = self.regions[i]
            r0,c0,r1,c1 = reg.bbox
            qq = image2D[r0-pxl_step:r1+pxl_step, c0-pxl_step:c1+pxl_step]
            b = Counter(qq.flatten())
            for key in [-1,reg.label]:
                b.pop(key, None)
            try:
                new_label, recurrence = b.most_common(1)[0]
                image2D[image2D==reg.label] = new_label
            except IndexError:
                print(b)
        return image2D

    def plot_cluster_maps(self, cluster2D_1, cluster2D_2=None, cluster2D_3=None,
                    palette='random', title=None):
        """
        plot up to 3 images together
        """
        n = 2
        if cluster2D_3 is not None:
                n += 1
        fig, axs = plt.subplots(1, n, sharex=True, sharey=True)
        # Plot left image
        ax = axs[0]
        if isinstance(cluster2D_1, OrderedDict):
            im_all_1 = self._sum_maps(cluster2D_1, axs)
        else:
            im_all_1 = cluster2D_1
        n = np.max(im_all_1.flatten())
        p = getPalette(n, 'random', 'black')
        cm = colors.ListedColormap(p, 'pColorMap')
        ax.imshow(im_all_1, cmap=cm)
        # Plot second image
        ax = axs[1]
        if cluster2D_2 is None:
            ax.imshow(im_all_1!=0, 'gray')
        else:
            if isinstance(cluster2D_2, OrderedDict):
                im_all_2 = self._sum_maps(cluster2D_2, ax)
            else:
                im_all_2 = cluster2D_2
            n = np.max(im_all_2.flatten())
            p = getPalette(n, 'random', 'black')
            cm = colors.ListedColormap(p, 'pColorMap')
            ax.imshow(im_all_2, cmap=cm)
        # Plot third image
        if cluster2D_3 is not None:
            ax = axs[2]
            if isinstance(cluster2D_3, OrderedDict):
                im_all_3 = self._sum_maps(cluster2D_2, ax)
            else:
                im_all_3 = cluster2D_3
            n = np.max(im_all_3.flatten())
            p = getPalette(n, 'random', 'black')
            cm = colors.ListedColormap(p, 'pColorMap')
            ax.imshow(im_all_3, cmap=cm)
        if title:
            fig.suptitle(title, fontsize=30)
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

def plot_cluster_lengths(clusters, log_step=0.1, zeta=0.633, ms=6):
    sizes = clusters.cluster_size
    major_axis_lengths = clusters.cluster_maj_ax_len
    # Plot size vs. lengths
    fig, ax = plt.subplots(1,1, figsize=(8.5,6.5))
    ax.loglog(major_axis_lengths, sizes, 'o', ms=ms, label=r'$S$')
    ax.set_xlabel(r"Major axis length $L$", size=20)    
    ax.set_ylabel(r"Cluster size $S$", size=26)
    ######################
    v = {}  
    lns = np.unique(major_axis_lengths)
    for l in lns:
        w = major_axis_lengths == l
        v[l] = sizes[w]
    l, s = gLD.averageLogDistribution(v, log_step=log_step)
    ax.loglog(l,s, 'ro', label=r'$\langle S \rangle$')
    lb = r'$\zeta$ = %.3f' % zeta
    ax.loglog(l[1:-8], s[5]*(l[1:-8]/l[5])**(1.+zeta), 'k--', label=lb)
    ax.legend()
    ax.grid(True)
    ###############################################
    # Plot distribution lengths
    ll, pll, pll_err = gLD.logDistribution(major_axis_lengths, log_step=log_step)
    fig, ax = plt.subplots(1,1, figsize=(8.5,6.5))    
    ax.loglog(ll, pll, 'o', ms=ms, label=r'$S$')
    ax.set_xlabel(r"Major axis length $L$", size=20)    
    ax.set_ylabel(r"Length distribution $P(L)$", size=20)
    ax.loglog(ll, pll[5]*(ll/ll[5])**(-1.5), 'k--')
    #ax.legend()
    ax.grid(True)
    plt.show()

def print_time(diff):
    sec = diff.seconds
    minutes, seconds = sec//60, sec%60
    if minutes:
        print("*** Time elapsed: %d min, %d s" % (minutes, seconds))
    else:
        print("*** Time elapsed: %d s" % seconds)

def format_coord(x, y, X):
    col = int(x + 0.5)
    row = int(y + 0.5)
    if col >= 0 and col < numcols and row >= 0 and row < numrows:
        z = X[row, col]
        return 'x=%1.4f, y=%1.4f, z=%1.4f' % (x, y, z)
    else:
        return 'x=%1.4f, y=%1.4f' % (x, y)


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
        #currents = ["0.22"]0
        ns_experiments = {"0.06": [1,2,3], "0.1":range(1,11), "0.2":range(1,11)}
        min_size = 5
        irradiation = irradiation[:-6]

    elif irradiation == 'NonIrr_Dec18':
        #field = "0.137"
        field = "0.146"
        #field = "0.157"
        #field = "0.165"
        set_n = "Set1"
        mainDir = "/data/Meas/Creep/CoFeB/Film/SuperSlowCreep/NonIrr/Feb2018/%sA/" % field
        fieldDir = os.path.join(mainDir, set_n)
        hdf5_filename = "%sA.hdf5" % field
        hdf5_filename_results = "Results_NonIrr_Feb2018.hdf5"
        currents = [field]
        ns_experiments = {"0.137": range(2, 16), "0.146": range(1,9), "0.157": [2,3,4,5], "0.165": range(1,5)}

        if field == "0.137":
            #crop_upper_left_pixel = [(None,None), (300,300), (300,300), (300,300), (300,300), (200,200), (0,0)]
            #crop_lower_right_pixel = [(None,None),(800,800), (800,800), (800,800), (900,900), (1040,1392)]
            nij_list, clrs = [0.44], ['r']
        elif field == "0.146":
            #crop_upper_left_pixel = [(300,300),(200,200),(200,200),(200,200),(200,200),(100,100),(0,0),(0,0)]
            #crop_lower_right_pixel = [(700,800),(800,900),(800,900),(800,900),(800,900),(900,1000),(1100,1000),(1100,1000)]
            #nij_max = {"Set1": 0.32}
            nij_list, clrs = [0.33], ['r']           
        elif field == "0.157":
            #crop_upper_left_pixel = [(300,300), (300,300), (200,200), (100,100), (0,0)]
            #crop_lower_right_pixel = [(900,900), (900,900), (1000,1000), (1000,1000), (1040,1040)]
            nij_list, clrs = [0.25], ['r']
        elif field == "0.165":
            #crop_upper_left_pixel = [(200,200),(200,200),(100,100),(0,0)]
            #crop_lower_right_pixel = [(900,900),(900,900),(900,900),(1250,1040)]
            nij_list, clrs = [0.15], ['r']
        
        # crops = {}
        # for i in range(len(crop_lower_right_pixel)):
        #     crops[i+1] = (crop_upper_left_pixel[i], crop_lower_right_pixel[i])
                  
        min_size, hist_dx = 5, 0.01


    for current in currents:    
        field  = "%sA" % (current)
        print("Analysing %s" % field)
        n_experiments = ns_experiments[current]
        start = datetime.datetime.now()
        #clusters[current] = Clusters(mainDir, grp0, n_experiments, irradiation, skip_first_clusters=0, min_size=min_size)
        cl = Clusters(mainDir, hdf5_filename, field, n_experiments, set_n=set_n,
            fieldDir=fieldDir, skip_first_clusters=0, min_size=min_size)
        clusters[current] = cl
        # Get the statistic of the events
        cl.get_event_stats()
        PS_events, fig0 = cl.plot_cluster_stats(cl.all_events.event_size.values, p0=None, lb='raw events', color='g')
        #sys.exit()
        start1 = datetime.datetime.now()
        diff = start1 - start
        print_time(diff)
        #sys.exit()
        # Clusters as usual 
        all_clusters = cl.get_cluster_stats_from_dict(cl.cluster2D_start, 'adiacent events')
        PS_touch, fig0 = cl.plot_cluster_stats(all_clusters.cluster_size.values, 
            fig=fig0, lb='from touching events', color='b')
        start2 = datetime.datetime.now()
        diff = start2 - start1
        print_time(diff)
        # Cluster with n_ij
        cl.show_correlation(event_size_threshold=min_size, dx=hist_dx)
        all_clusters_nij = {}
        cluster2D_nij = {}
        ####################################
        #for clr, nij_ax in zip(['r', 'm'],[0.33, 0.45]):
        for clr, nij_max in zip(clrs,nij_list):
            lb = 'from n_ij = %.2f' % nij_max
            title = r"clusters with $n_{ij} = %.2f$" % nij_max
            cln = cl.get_clusters_nij(cl.con_to_df, nij_max, title=title)
            cluster2D_nij[nij_max] = cln
            ac = cl.get_cluster_stats_from_nj2(cln, 'events with nij: %.2f' % nij_max)
            all_clusters_nij[nij_max] = ac
            PS_nij, fig = cl.plot_cluster_stats(ac.cluster_size.values, fig=fig0, lb=lb, color=clr)
            #cl.plot_cluster_maps(cl.cluster2D_start, cln)
            cln_filtered = cl.clean_small_clusters(cln)
            ac_filtered = cl.get_cluster_stats_from_nj2(cln_filtered, 'events with nij: %.2f, filtered' % nij_max)
            PS_nij_filtered, fig = cl.plot_cluster_stats(ac_filtered.cluster_size.values, 
                fig=fig0, lb=lb+' filtered', color='m', max_index=None)
            
            cl.plot_cluster_maps(cl.cluster2D_start, cln, cln_filtered)
            plot_cluster_lengths(ac_filtered)
        start3 = datetime.datetime.now()
        diff = start3 - start2
        print_time(diff)
        save_data = raw_input("Save data?")
        if save_data.upper() == 'Y':
            # Save to the upper directory into a hdf5
            up_dir = str(Path(mainDir).parent)
            hname = os.path.join(up_dir, hdf5_filename_results)
            store = pd.HDFStore(hname)
            subDir = "%s/%s" % (field, set_n)
            distrs = [PS_events, PS_touch, PS_nij, PS_nij_filtered, cl.h_ij_real, cl.h_ij_shuffled]
            _distrs = ['PS_events', 'PS_touch', 'PS_nij', 'PS_nij_filtered', 'h_ij_real', 'h_ij_shuffled']
            for d, s in zip(distrs, _distrs):
                group = "%s/%s" % (subDir,s)
                store[group] = d
            store.close()