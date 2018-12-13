import sys, os
import datetime
import numpy as np
from scipy import optimize
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
        with h5py.File(self._fname, 'a') as fh:
            try:
                grp_base = fh[self._baseGroup]
            except KeyError:
                print("Key error for %s" % self._baseGroup)
                print(self._fname)
                sys.exit()
            saved_experiments = len(grp_base)
            if len(n_experiments) > saved_experiments:
                print("There are only %i/%i experiments" % (saved_experiments, self._len_experiments))
                self.n_experiments = self.n_experiments[:saved_experiments]
            for n_exp in self.n_experiments:
                grp0 = self._baseGroup + "/%02i" % n_exp
                grp_n_exp = fh[grp0]
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
                grp_n_exp_contour = fh[grp0 + "/contours/%i" % last_switch]
                self.last_contour[n_exp] = grp_n_exp_contour[...]
                grp_n_exp_contour = fh[grp0 + "/contours/0"]
                self.first_contour = grp_n_exp_contour[...]
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
    
    def _f_2(self, c, x, y):
        xc, yc = c
        r_i = np.sqrt((x-xc)**2 + (y-yc)**2)
        return r_i - r_i.mean() 

    def _get_center(self, contour):
        x, y = contour[:,1], contour[:,0]
        center_estimate = np.mean(x), np.mean(y)
        (xc_m,yc_m), ier = optimize.leastsq(self._f_2, center_estimate, args=(x,y))  # done by scipy
        if not ier:
            print("There is a problem to fit the center of the bubble")
        return xc_m, yc_m

    def _get_center_mahotas(self, image):
        q = image > 0
        _cls, n_cls = mahotas.label(~q)
        sizes = mahotas.labeled.labeled_size(_cls)
        idx = np.argmax(sizes[2:]) + 2
        initial_domain = _cls == idx
        center = mahotas.center_of_mass(initial_domain)
        print("Center found at: {1:.2f}, {0:.2f}".format(*center))
        return center


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
        
    def get_event_stats(self, min_size=1):
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
        self.max_label = []
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
            self.max_label.append(event_label)
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
        self.event2D = self._sum_maps(self.labeledEvents2D)
        self.center = self._get_center_mahotas(self.event2D)
        self.max_label = np.array(self.max_label)
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


    def _get_nij(self, _record, df, frac_dim=1, limits=None, distance='arc'):
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
        t = _record.switch_time - df.switch_time
        p = self.p_large_size[df.event_size]
        p = p.values
        X0, Y0 = _record.event_positionX, _record.event_positionY
        X, Y = df.event_positionX, df.event_positionY
        if distance == 'euclidean':
            l = ((X-X0)**2 + (Y-Y0)**2)**0.5
        elif distance == 'arc':
            n_exp = _record.n_exp.astype(int)
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
            if limits is not None:
                angle, pR = limits
                dR_p = dR / R0 * 100
                dthetas = dthetas * 180. / np.pi
                _where = (dR_p <= pR) & (dthetas <= angle)
                if not np.sum(_where):
                    return 4 * [None]
                else:
                    l = l[_where]
                    t = t[_where]
                    p = p[_where]
        elif distance == 'solidity':
            pass
        n_ij = t * l**frac_dim * p
        t_ij = t * p**0.5
        #r_ij = l**frac_dim * p.values**0.5
        #pos = n_ij.values.argmin()
        idxmin_n_ij, min_n_ij = int(n_ij.idxmin()), n_ij.min()
        t_ij = t_ij[idxmin_n_ij]
        label = df.event_label.loc[idxmin_n_ij]
        if np.isnan(min_n_ij):
            return 4 * [None]
        else:
            return idxmin_n_ij, int(label), min_n_ij, t_ij

    def get_correlation(self, df, event_size_threshold=1, 
                        frac_dim=1, previous_frames=None, label=None,
                        internal_reshuffle=False, limits=None):
        """
        df is the dataFrame:
        it can be self.all_events or self.all_events_shuffle
        previous_frames: int
            set the number of previous switches to consider n_ij
        limits : tuple
            set the limits to consider angles (in degree) and % of radius
        """
        #connected_to = []
        print(60*"*")
        s = "Getting the event correlation of "
        if label:
            s += label
        print(s)
        connected_to = defaultdict(list)
        if internal_reshuffle:
            connected_to_shuffled = defaultdict(list)
        for n_exp in self.n_experiments:
            print("Experiment: %i" % n_exp)
            q = df[df.n_exp==n_exp]
            q = q[q.event_size >= event_size_threshold]
            times = np.unique(q.switch_time)
            # Find the index of the first event at the second time
            first_index = (q.switch_time==times[1]).idxmax()
            #connected_to += first_index*[np.NaN]
            for i in q.index[q.index >= first_index]:
                _record = q.loc[i]
                switch_i = int(_record.switch_frame)
                _where = q.switch_frame < switch_i
                if previous_frames:
                    if previous_frames < (switch_i - q.switch_frame.iloc[0]):
                        _where = (q.switch_frame > (switch_i - previous_frames)) & _where
                sub_q = q[_where]
                if sub_q.empty:
                    print(_record)
                    print(switch_i)
                    print(q.switch_frame.iloc[0])
                    print(sub_q)
                    print(i)
                    print("empty")
                    continue
                idxmin_n_ij, label_n_ij, min_n_ij, t_ij = self._get_nij(_record, sub_q, frac_dim, limits)
                # try:
                #     idxmin_n_ij, label_n_ij, min_n_ij, t_ij = self._get_nij(_record, sub_q, frac_dim, limits)
                # except (AttributeError, ValueError):
                #     print(switch_i)
                #     print(q.switch_frame.iloc[0])
                #     print(_record)
                #     print(sub_q)
                #     print(_where)
                #     sys.exit()
                if min_n_ij:
                    connected_to['n_exp'].append(n_exp)
                    connected_to['event_idx'].append(i)
                    connected_to['event_label'].append(int(_record.event_label))
                    connected_to['father_idx'].append(idxmin_n_ij)
                    connected_to['father_label'].append(label_n_ij)
                    connected_to['n_ij'].append(min_n_ij)
                    connected_to['t_ij'].append(t_ij)
                    connected_to['r_ij'].append(min_n_ij/t_ij)
                if internal_reshuffle:
                    is_internal_reshuffle_on_frame = False
                    p = np.random.permutation(len(sub_q))
                    q_shuffled = sub_q.copy()
                    if is_internal_reshuffle_on_frame:
                        q_shuffled.switch_frame.iloc[:] = sub_q.switch_frame.values[p]
                        q_shuffled.switch_time.iloc[:] = sub_q.switch_time.values[p]
                    else:
                        q_shuffled.event_label.iloc[:] = sub_q.event_label.values[p]
                        q_shuffled.event_positionX.iloc[:] = sub_q.event_positionX.values[p]
                        q_shuffled.event_positionY.iloc[:] = sub_q.event_positionY.values[p]
                        q_shuffled.event_size.iloc[:] = sub_q.event_size.values[p]
                    idxmin_n_ij, label_n_ij, min_n_ij, t_ij = self._get_nij(_record, q_shuffled, frac_dim, limits)
                    if min_n_ij:
                        r_ij = min_n_ij/t_ij
                        connected_to_shuffled['n_exp'].append(n_exp)
                        connected_to_shuffled['event_idx'].append(i)
                        connected_to_shuffled['event_label'].append(int(_record.event_label))
                        connected_to_shuffled['father_idx'].append(idxmin_n_ij)
                        connected_to_shuffled['father_label'].append(label_n_ij)
                        connected_to_shuffled['n_ij'].append(min_n_ij)
                        connected_to_shuffled['t_ij'].append(t_ij)
                        connected_to_shuffled['r_ij'].append(r_ij)
        cols = ['n_exp', 'event_idx', 'event_label', 'father_idx', 'father_label', 'n_ij', 't_ij', 'r_ij']
        connected_to_df = pd.DataFrame(connected_to, columns=cols)
        if internal_reshuffle:
            connected_to_shuffled_df = pd.DataFrame(connected_to_shuffled, columns=cols)
            return connected_to_df, connected_to_shuffled_df
        else:
            return connected_to_df

    def show_correlation(self, event_size_threshold=5, frac_dim=1, previous_frames=None, n_ij_max=None, dx=0.05,
                        internal_reshuffle=False, limits=None):

        if internal_reshuffle:
            label = 'internal_reshuffle'
            self.con_to_df, self.con_to_df_shuffled = self.get_correlation(self.all_events, 
                                event_size_threshold=event_size_threshold, 
                                frac_dim=frac_dim, previous_frames=previous_frames, label=label, 
                                internal_reshuffle=internal_reshuffle, limits=limits)   
        else:
            label = "real data"
            self.con_to_df = self.get_correlation(self.all_events, 
                                        event_size_threshold=event_size_threshold, 
                                        frac_dim=frac_dim, previous_frames=previous_frames, 
                                        label=label, internal_reshuffle=False, limits=limits)
            label = "shuffled data"
            self.con_to_df_shuffled = self.get_correlation(self.all_events_shuffle,
                                        event_size_threshold=event_size_threshold, 
                                        frac_dim=frac_dim, previous_frames=previous_frames, 
                                        label=label, internal_reshuffle=False, limits=limits)
                            
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
            self.h_ij_real = pd.Series(h_ij, index=h_bins[:-1])
            h_ij, h_bins, patches = ax.hist(n_ij_shuffled, bins=bins, alpha=0.5, label='shuffled')
            self.h_ij_shuffled = pd.Series(h_ij, index=h_bins[:-1])
            ax.set_xlabel(r"$n_{ij}$", size=26)
            ax.set_ylabel(r"$hist(n_{ij})$", size=26)
            ax.get_yaxis().set_tick_params(which='both', direction='in')
            ax.get_xaxis().set_tick_params(which='both', direction='in')
            ax.xaxis.set_ticks_position('both')
            ax.yaxis.set_ticks_position('both')
            ax.legend()
            # 2D plot
            cols = 2
            fig1, axs = plt.subplots(cols, cols, squeeze=False)
            #ax1.hist2d(r_ij, t_ij, bins=bins)
            labels = ['real', 'shuffled']
            for i, df in enumerate([self.con_to_df, self.con_to_df_shuffled]):
                r_ij, t_ij = df['r_ij'], df['t_ij']    
                axs[i,0].loglog(r_ij, t_ij, 'o', c='C%i' % i, label=labels[i])
                axs[i,0].legend()
                x,y = np.log10(r_ij), np.log10(t_ij)
                axs[i,1].hist2d(x, y, bins=50, norm=colors.LogNorm())
                if n_ij_max:
                    X = np.linspace(np.min(x), np.max(x))
                    Y = -frac_dim * X + np.log10(n_ij_max)
                    axs[i,1].plot(X,Y,'r--') 

                for j in range(cols):
                    axs[i,j].set_xlabel(r"$r^{*}$", size=26)
                    axs[i,j].set_ylabel(r"$\tau^{*}$", size=26)
            #ax1.legend()
	        fig.tight_layout()
	        fig1.tight_layout()
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
        _cluster2D_nij = self.event2D.copy()
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


    def get_cluster_stats_from_nj2(self, image2D, _string=None):
        print(60*"*")
        print("Get the statistics of the clusters for each experiment %s" % _string)
        cluster_cols = ['n_exp','label', 'area', 'maj_ax_len', 'min_ax_len']
        cluster_cols += ['solidity', 'centroid', 'dist_to_center']
        _cluster_data = defaultdict(list)
        ############################################
        self.regions = measure.regionprops(image2D)
        regions = [reg for reg in self.regions if reg.area >= self.min_size]
        lb = [props.label for props in regions]
        _cluster_data['label'] = lb
        _cluster_data['n_exp'] = [np.argmax((self.max_label//l).astype(bool))+1 for l in lb]
        _cluster_data['area'] = [props.area for props in regions]
        _cluster_data['maj_ax_len'] = [props.major_axis_length for props in regions]
        _cluster_data['min_ax_len'] = [props.minor_axis_length for props in regions]
        _cluster_data['solidity'] = [props.solidity for props in regions]
        centroids = [props.centroid for props in regions]
        _cluster_data['centroid'] = centroids
        xc, yc = self.center
        _cluster_data['dist_to_center'] = [((c[0]-xc)**2+(c[1]-yc)**2)**0.5 for c in centroids]
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

    def get_best_fit(self, x, y, y_err, n_params, p0, min_index=2, max_index=-2,):
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

    def plot_cluster_stats(self, cluster_sizes, log_step=0.1, n_params=3, p0=None,
        min_index=2, max_index=-2, fig=None, lb=None, color='b'):
        S, PS, PS_err = gLD.logDistribution(cluster_sizes, log_step=log_step)
        params, errors, ier, S_calc, PS_calc = self.get_best_fit(S, PS, PS_err, 
                            n_params, p0, min_index, max_index)
        # Plot the distribution of the cluster area
        if fig is None:
            fig, ax = plt.subplots(1,1)
        else:
            ax = fig.gca()
        ax.loglog(S, PS, 'o', color=color, label=lb)
        ax.errorbar(S, PS, PS_err, fmt="none")
        if ier != 0:
            ax.loglog(S_calc, PS_calc, '--', color=color)
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
            qq = image2D[r0 - pxl_step: r1 + pxl_step, c0 - pxl_step: c1 + pxl_step]
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


    def plot_cluster_lengths(self, clusters, min_solidity=0.7, log_step=0.1, zeta=0.633, ms=8):
        sizes = clusters.area
        major_axis_lengths = clusters.maj_ax_len
        d = {'Length': major_axis_lengths, 'Size': sizes}
        df_S_vs_l = pd.DataFrame(d, columns=['Length', 'Size'])
        # Plot size vs. lengths
        fig, ax = plt.subplots(1,1, figsize=(8.5,6.5))
        ax.loglog(major_axis_lengths, sizes, 'o', ms=6, label=r'$S$')
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
        ax.loglog(l[1:-4], s[5]*(l[1:-4]/l[5])**(1.+zeta), 'k--', ms=8, label=lb)
        ax.legend()
        ax.grid(True)
        d = {'length': l, 'S_mean': s}    
        df_S_mean = pd.DataFrame(d, columns=['length', 'S_mean'])
        ###############################################
        # Plot distribution lengths
        fig, axs = plt.subplots(1,2, figsize=(17,6.5))    
        #for n_exp in self.n_experiments:
        #    c = clusters[clusters.n_exp == n_exp]
        #    major_axis_lengths = c.maj_ax_len
        if min_solidity:
            clusters = clusters[clusters.solidity >= min_solidity]
        ll, pll, pll_err = gLD.logDistribution(clusters.maj_ax_len, log_step=log_step)
        print("################################## Fitting length distribution")
        params, errors, ier, ll_calc, pll_calc = self.get_best_fit(ll, pll, pll_err, n_params=3, p0=None, min_index=2, max_index=None)  
        d = {'length': ll, 'P_length': pll}    
        df_P_lenghts = pd.DataFrame(d, columns=['length', 'P_length'])
        axs[0].loglog(ll, pll, 'o', ms=ms)
        axs[0].set_xlabel(r"Major axis length $L$", size=20)    
        axs[0].set_ylabel(r"Length distribution $P(L)$", size=20)
        if ier != 0:
            axs[0].loglog(ll_calc, pll_calc, 'k--', label = "best fit")
        axs[0].legend()
        axs[0].grid(True)
        # Normalized
        l_d =  clusters.maj_ax_len / clusters.dist_to_center
        ll_norm, pll_norm, pll_err_norm = gLD.logDistribution(l_d, log_step=log_step)
        print("################################## Fitting normalized length distribution")
        params, errors, ier, ll_norm_calc, pll_norm_calc = self.get_best_fit(ll_norm, pll_norm, pll_err_norm, n_params=3, p0=None, min_index=3, max_index=None)
        d = {'length': ll_norm, 'P_length': pll_norm}    
        df_P_lenghts_norm = pd.DataFrame(d, columns=['length', 'P_length'])
        axs[1].set_xlabel(r"Normalized major axis length $L$", size=20)    
        axs[1].set_ylabel(r"Length distribution $P(L)$", size=20)
        axs[1].loglog(ll_norm, pll_norm, 'o', ms=ms)
        if ier != 0:
            axs[1].loglog(ll_norm_calc, pll_norm_calc, 'k--', label = "best fit")
        axs[1].legend()
        axs[1].grid(True)
        plt.show()
        return df_S_vs_l, df_S_mean, df_P_lenghts, df_P_lenghts_norm


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
        current_field = sys.argv[2]
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
        #currents_fields = ["0.137", "0.146", "0.157", "0.165"]
        #current_field = currents_fields[2]
        set_n = "Set1"
        zeta = 0.633
        d_f = 1
        previous_frames = None
        ###########################
        internal_reshuffle = True
        #d_f = 1 + zeta
        limits = (180, 20)
        #limits = None
        ###########################
        mainDir = "/data/Meas/Creep/CoFeB/Film/SuperSlowCreep/NonIrr/Feb2018/%sA/" % current_field
        fieldDir = os.path.join(mainDir, set_n)
        hdf5_filename = "%sA.hdf5" % current_field
        hdf5_filename_results = "Results_NonIrr_Feb2018.hdf5"
        ns_experiments = {"0.137": range(2, 16), "0.146": range(1,9), 
                          "0.157": [2,3,4,5], "0.165": range(1,5)}
        # ns_experiments = {"0.137": range(2, 16), "0.146": range(1,9), 
        #                   "0.157": [2,3], "0.165": range(1,5)}
        if current_field == "0.137":
            n_ij_max, _clr = 0.44, 'r'
            #nij_list, clrs = [1.44], ['r']
        elif current_field == "0.146":
            n_ij_max, _clr = 0.33, 'r'
            #nij_list, clrs = [1.23], ['r']
        elif current_field == "0.157":
            n_ij_max, _clr = 0.25, 'r'
            #nij_list, clrs = [1.25], ['r']
        elif current_field == "0.165":
            n_ij_max, _clr = 0.19, 'r'
            #nij_list, clrs = [1.5], ['r']
        min_size, hist_dx = 5, 0.01

    field  = "%sA" % (current_field)
    print("Analysing %s" % field)
    n_experiments = ns_experiments[current_field]
    start = datetime.datetime.now()
    #clusters[current] = Clusters(mainDir, grp0, n_experiments, irradiation, skip_first_clusters=0, min_size=min_size)
    cl = Clusters(mainDir, hdf5_filename, field, n_experiments, set_n=set_n,
                    fieldDir=fieldDir, skip_first_clusters=0, min_size=min_size)
    clusters[current_field] = cl
    #continue
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
    cl.show_correlation(event_size_threshold=min_size, dx=hist_dx, frac_dim=d_f, 
                    previous_frames=previous_frames, n_ij_max=n_ij_max, 
                    internal_reshuffle=internal_reshuffle, limits=limits)
    start3 = datetime.datetime.now()
    diff = start3 - start2
    print_time(diff)
    up_dir = str(Path(mainDir).parent)
    hname = os.path.join(up_dir, hdf5_filename_results)
    store = pd.HDFStore(hname)
    subDir = "%s/%s/df_%.3f/nij_%.2f" % (field, set_n, d_f, n_ij_max)
    # ################################
    #save_data = raw_input("Save dataFrames? ")
    save_data = "Y"
    if save_data.upper() == 'Y':
        # Save to the upper directory into a hdf5
        distrs = [cl.con_to_df, cl.con_to_df_shuffled]
        _distrs = ['all_events_hierarchy', 'all_events_hierarchy_shuffled']
        for df, _distr in zip(distrs, _distrs):
            group = "%s/%s" % (subDir, _distr)
            store[group] = df

    all_clusters_nij = {}
    cluster2D_nij = {}
    if False:
        store.close()
        sys.exit()
    ####################################
    
    lb = 'from n_ij = %.2f' % n_ij_max
    title = r"clusters with $n_{ij} = %.2f$" % n_ij_max
    cln = cl.get_clusters_nij(cl.con_to_df, n_ij_max, title=title)
    cluster2D_nij[n_ij_max] = cln
    #ac = cl.get_cluster_stats_from_nj2(cln, 'events with nij: %.2f' % n_ij_max)
    #all_clusters_nij[n_ij_max] = ac
    #PS_nij, fig = cl.plot_cluster_stats(ac.area.values, fig=fig0, lb=lb, color=clr)
    #cl.plot_cluster_maps(cl.cluster2D_start, cln)
    cln_filtered = cl.clean_small_clusters(cln)
    ac_filtered = cl.get_cluster_stats_from_nj2(cln_filtered, 
                                        'events with nij: %.2f, filtered' % n_ij_max)
    PS_nij_filtered, fig = cl.plot_cluster_stats(ac_filtered.area.values, 
                            fig=fig0, lb=lb+' filtered', color='m', max_index=None)     
    cl.plot_cluster_maps(cl.cluster2D_start, cln_filtered)
    df_S_vs_l, df_S_mean, df_P_lenghts, df_P_lenghts_norm = cl.plot_cluster_lengths(ac_filtered)
    ##############################
    #save_data = raw_input("Save distributions? ")
    if save_data.upper() == 'Y':
        # Save to the upper directory into a hdf5
        distrs = [PS_events, PS_nij_filtered, cl.h_ij_real, cl.h_ij_shuffled]
        distrs += [df_S_vs_l, df_S_mean, df_P_lenghts, df_P_lenghts_norm]
        _distrs = ['PS_events', 'PS_nij_filtered', 'h_ij_real', 'h_ij_shuffled']
        _distrs += ['S_vs_l', 'S_mean', 'P_lenghts', 'P_lenghts_norm']
        for df, _distr in zip(distrs, _distrs):
            group = "%s/%s" % (subDir, _distr)
            store[group] = df
    #save_data = raw_input("Save cluster2D? ")
    if save_data.upper() == 'Y':
        # Save to the upper directory into a hdf5
        cluster2D_start = cl._sum_maps(cl.cluster2D_start)
        cluster2D_end = cl._sum_maps(cl.cluster2D_end)
        distrs = [cln_filtered, cl.event2D, cluster2D_start, cluster2D_end]
        _distrs = ["cluster2D_nij", 'event2D', 'cluster2D_start', 'cluster2D_end']
        for d, _distr in zip(distrs, _distrs):
            group = "%s/%s" % (subDir, _distr)
            df = pd.DataFrame(d)
            store[group] = df
    #save_data = raw_input("Save dataFrames? ")
    if save_data.upper() == 'Y':
        # Save to the upper directory into a hdf5
        distrs = [ac_filtered, cl.all_events]
        _distrs = ["all_clusters_nij_structure", 'all_events_structure']
        for df, _distr in zip(distrs, _distrs):
            group = "%s/%s" % (subDir, _distr)
            store[group] = df
    print("Center:", cl.center)
    store.close()
