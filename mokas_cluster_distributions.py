import sys, os
import numpy as np
from scipy.integrate import trapz
#from scipy.optimize import fsolve
from scipy import optimize
import getLogDistributions as gLD
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from collections import OrderedDict, defaultdict
import h5py
import getAxyLabels as gal
import pandas as pd
from skimage import measure
import heapq
import mokas_cluster_methods as cmet
import mahotas
from mokas_colors import get_liza_colors
import ast
import mokas_parser as mkp




class Clusters:

    def __init__(self, hdf5_filename, group, n_experiments, min_size=5,
                skip_first_clusters=0, motion='downward'):
        """
        it is important to set a min_size of the cluster. 
        The value of 10 seems reasonable, but can larger
        """
        self.min_size = min_size
        self.skip_first_clusters = skip_first_clusters
        self._fname = hdf5_filename
        p, filename = os.path.split(hdf5_filename)
        filename, ext = os.path.splitext(filename)
        self.n_experiments = n_experiments
        self._len_experiments = len(n_experiments)
        self.title = filename + " - " + " - ".join(group.split("/")) + " - %i exps." % self._len_experiments
        self._baseGroup = group
        self.cluster2D = OrderedDict()
        self.times = OrderedDict()
        with h5py.File(self._fname, 'a') as f:
            grp_base = f[self._baseGroup]
            saved_experiments = len(grp_base)
            if len(n_experiments) > saved_experiments:
                print("There are only %i/%i experiments" % (saved_experiments, self._len_experiments))
                self.n_experiments = self.n_experiments[:saved_experiments]
            for n_exp in self.n_experiments:
                grp0 = self._baseGroup + "/%i" % n_exp
                grp_n_exp = f[grp0]
                if "cluster2D_start" in grp_n_exp:
                    self.cluster2D[n_exp] = grp_n_exp['cluster2D_start'][...]
                else:
                    print("Cluster2D_start does not exist for exp: %i" % n_exp)
                # Check if times exist
                if 'times' in grp_n_exp:
                    times = grp_n_exp['times'][...]
                else:
                    times = self._get_times(grp_n_exp)
                    grp_n_exp.create_dataset("times", data=times, dtype=np.float16)
                self.times[n_exp] = times
                # reaad data of measure
                if 'mu_per_pixel' in grp_base.attrs.keys():
                    self.mu_per_pixel = grp_base.attrs['mu_per_pixel']
                else:
                    sub_dir = grp_n_exp.attrs['root_dir']
                    if sub_dir[-1] == '/':
                        sub_dir, _ = os.path.split(sub_dir[:-1])
                    else:
                        sub_dir, _ = os.path.split(sub_dir)
                    #sub_dir = str(sub_dir)
                    fname_measure = os.path.join(sub_dir, "measure.txt")
                    p = mkp.Parser(fname_measure)
                    data = p.get_data()
                    self.um_per_pixel = data['um_per_pixel']
                    grp_base.attrs.create('um_per_pixel', self.um_per_pixel)

        rows, cols = self.cluster2D[n_exp].shape
        # All the figure have to be roated to get the motion downward
        self.ref_point = (0, cols//2)
        self.motion = motion
        self._Axy_types = ['0000', '0100', '1000','1100']
        self.cluster_data = OrderedDict()
        if motion == 'downward':
            self.direction = 'Bottom_to_top'

    def _get_times(self, grp):
        root_dir = grp.attrs['root_dir']
        pattern = grp.attrs['pattern']
        pattern = pattern.replace(".ome.tif", "_metadata.txt")
        fname = os.path.join(root_dir, pattern)
        with open(fname, 'r') as f: 
            q = f.read()
        q = q.replace("null", "False")
        q = q.replace("false", "False")
        d = ast.literal_eval(q)
        times = np.array([float(d[k]["ElapsedTime-ms"]) for k in d if k!='Summary'])
        times.sort()
        times = (times - times[0]) / 1000.
        return times


    def calc_R(self, xc, yc):
        """ calculate the distance of each 2D points from the center (xc, yc) """
        return np.sqrt((self.X-xc)**2 + (self.Y-yc)**2)

    def f_2(self, c):
        """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
        Ri = self.calc_R(*c)
        return Ri - Ri.mean()

    def angle_persistence(self, delta_angle=0):
        """
        Analyse a pandas.dataframe to seek
        persistence of angles at the edges
        """
        ap = [pd.DataFrame(), pd.DataFrame()]
        lifetimes = ['lifetime_a10', 'lifetime_a01']
        for n_exp in self.n_experiments:
            df_n_exp = self.cluster_data[n_exp]
            switch_times = df_n_exp.switch_time
            dfs_n_exp = [df_n_exp.a_10, df_n_exp.a_01]
            for i in range(2):
                df = pd.DataFrame()
                angle = dfs_n_exp[i]
                # Calculate start and end
                diff = angle.diff()
                # A point is both the start and the end of the previous value
                q = np.logical_and((abs(diff)>delta_angle), (pd.notnull(diff)))
                times = switch_times[q]
                deltas = times.diff()
                deltas = deltas.shift(-1)
                df[angle.name] = angle[q]
                df[lifetimes[i]] = deltas
                ap[i] = pd.concat([ap[i], df])
        return ap



    def get_front_params(self, cluster, method='polyfit_2', n_fit=30):
        """
        get the parameters of the DW front:
        angles, positions at edges, curvature
        """
        rows, cols = cluster.shape
        cnts = self._select_contour(cluster)
        X, Y = cnts[:,1], cnts[:,0]
        if X[0] > X[-1]:
            X, Y = X[::-1], Y[::-1]
        y_left, y_right = Y[0], Y[-1]
        self.X, self.Y = X, Y
        ####################################
        # Left edge
        ####################################
        z = np.polyfit(X[:n_fit], Y[:n_fit], 1)
        z0 = np.arctan(z[0])
        angle_left = np.abs(z0) * 180 / np.pi
        ####################################
        # Right edge
        ####################################
        z = np.polyfit(X[-n_fit:], Y[-n_fit:], 1)
        z0 = np.arctan(z[0])
        angle_right = abs(z0) * 180 / np.pi
        ####################################
        # Curvature, vertex, and deltas
        ####################################
        straightness = self.get_straightness(cnts)
        row_of_min = np.max(Y) # Yeah, it is correct
        r_curvature, x_v, y_v, y_left, y_right = self.get_curvature((X,Y), method=method)
        delta_left = np.abs((y_left) - row_of_min)
        delta_right = np.abs((y_right) - row_of_min)
        l_front = cmet.get_length(cnts)
        # calculate the linear distance between the pinning points at the edges
        L_linear = ((y_left - y_right)**2 + (X[0]-X[-1])**2)**0.5
        # Calculate the Delta_Size for a front which is not linear
        if y_right > y_left:
            y_min = y_left
        else:
            y_min = y_right
        Delta_S = np.abs(trapz((Y-y_min), X)) - 0.5 * np.abs(y_left - y_right) * cols
        Delta_L = l_front - L_linear
        params = [angle_left, angle_right, r_curvature, straightness, x_v, y_v, y_left, y_right]
        params += [delta_left, delta_right, l_front, Delta_L, Delta_S]
        return params

    def get_straightness(self, cnts, method='linear'):
        """
        Methods: 
        cov
        https://gis.stackexchange.com/questions/16322/measuring-straightness-of-a-curve-segment-represented-as-a-polyline
        linear
        Take a straight linear between the first and the last point and 
        calculate the sqrt(mean(error**2))
        """
        if method == 'cov':
            q = np.cov(cnts.transpose())
            eigenvalues, eigenvectors = np.linalg.eigh(q)
            return eigenvalues[0]
        elif method == 'linear':
            X, Y = cnts[:,1], cnts[:,0]
            m = (Y[-1]-Y[0]) / (X[-1]-X[0])
            c = Y[0]- m * X[0]
            y = m*X + c
            return np.mean((Y-y)**2)**0.5

    def get_curvature(self, xy, method='polyfit_2'):
        """
        Calculus of the curvature of the DW front
        """
        X,Y = xy
        if 'polyfit' in method:
            n = np.int(method[-1])
            params = np.polyfit(X, Y, n)
            p = np.poly1d(params)
            # Find the curvature for each point
            p1 = np.polyder(p,1)
            p2 = np.polyder(p,2)
            # Find the min (max)
            crit = p.deriv().r
            r_crit = crit[crit.imag==0].real
            x_v = r_crit[0]
            y_v = p(x_v)
            y_left, y_right = p(0), p(X[-1])
            # Choose the curvature at the mimunum
            r_curvature = (1.+p1(x_v)**2)**1.5/np.abs(p2(x_v))
            # Gaussian curvature
            # https://en.wikipedia.org/wiki/Gaussian_curvature
            K1, K2 = 0,0
        elif method == 'circlefit':
            center_estimate = np.mean(X), np.mean(Y)
            center, ier = optimize.leastsq(self.f_2, center_estimate)
            xc, yc = center
            R_i = self.calc_R(*center)
            r_curvature = R_i.mean()
            # TODO
            x_v, y_v = xc, yc + r_curvature 
            y_left = yc + r_curvature * (1 - (1 - (xc/r_curvature)**2)**0.5)
            y_right = yc + r_curvature * (1 - (1 - ((X[-1] - xc)/r_curvature)**2)**0.5)
        else:
            raise ValueError("Method not implemented")
        return r_curvature, x_v, y_v, y_left, y_right


    def get_experiment_stats(self, curvature_method='polyfit_2'):
        print("Get the statistics for each experiment")
        self.cluster_data = dict()
        self.sub_cluster_data = dict()
        self.cluster2D_color = dict()
        self.last_switch = dict()
        ######################################
        for n_exp in self.cluster2D:
            print("Experiment: %i" % n_exp)
            cluster_data = defaultdict(list)
            sub_cluster_data = defaultdict(list)
            ############################################
            cluster2D = self.cluster2D[n_exp]
            self.cluster2D_color[n_exp] = np.copy(cluster2D)
            cluster_switches = np.unique(cluster2D)[self.skip_first_clusters+1:]
            cluster = np.logical_and((cluster2D > 0), cluster2D < cluster_switches[0])
            rows, cols = cluster2D.shape
            is_not_transient = False
            for switch in cluster_switches:
                #print(switch)
                cluster0 = cluster2D == switch
                cluster += cluster0
                cluster0_type = gal.getAxyLabels(cluster0, self.direction, 1)[0]
                # We need to check if the cluster is touching the top/bottom edge
                # i.e. we check the last two elements of the cluster_type
                if '1' in cluster0_type[2:]:
                    self.cluster2D_color[n_exp][cluster0] = -1
                    continue
                area = np.sum(cluster0)
                if area < self.min_size:
                    self.cluster2D_color[n_exp][cluster0] = -1
                    continue
                last_switch = switch
                # We need to wait untile the full cluster touches both edges
                cluster_type = gal.getAxyLabels(cluster, self.direction, 1)[0]
                if is_not_transient:
                    front_params = self.get_front_params(cluster, method=curvature_method)
                else:
                    if cluster_type == '1100':
                        is_not_transient = True
                        front_params = self.get_front_params(cluster, method=curvature_method)
                    else:
                        # Use a trick to get the initial front
                        im, n_clusters = mahotas.label(cluster2D==-1, np.ones((3,3)))
                        sizes = mahotas.labeled.labeled_size(im)[1:]
                        try:
                            ind = np.argpartition(sizes, -2)[-2:]
                        except:
                            ind = [0]
                        for i in ind:
                            cl = im==i+1
                            cl_type = gal.getAxyLabels(cl, self.direction, 1)[0]
                            if cl_type == '1101': #Yeah it is correct
                                self.upper_cluster = np.logical_or(cl, cluster)
                                front_params = self.get_front_params(self.upper_cluster, method=curvature_method)
                angle_left, angle_right, r_curvature, straightness, x_v, y_v, y_left, y_right = front_params[:8]
                delta_left, delta_right, l_front, Delta_L, Delta_S = front_params[8:]
                # if is_not_transient:
                #     angle_left, angle_right, r_curvature, x_v, y_v, c, c_right = 7 * [np.NaN]
                #     delta_left, delta_right, l_front = 3 * [np.NaN]

                # Here there is a problem
                # Cluster0 can be made of 2 or more subclasters (this is not uncommon)
                # So we need to loop over the sub_clusters
                sub_clusters, n_sub_clusters = mahotas.label(cluster0, np.ones((3,3)))
                # Save the data
                cluster_data['n_exp'].append(n_exp)
                cluster_data['switch_frame'].append(switch)
                cluster_data['switch_time'].append(self.times[n_exp][switch])
                cluster_data['type'].append(cluster0_type)
                cluster_data['area'].append(area)
                cluster_data['n_sub_cl'].append(n_sub_clusters)
                cluster_data['a_01'].append(angle_left)
                cluster_data['a_10'].append(angle_right)
                cluster_data['r_curv'].append(r_curvature)
                cluster_data['straightness'].append(straightness)
                cluster_data['x_v'].append(x_v)
                cluster_data['y_v'].append(y_v)
                cluster_data['y_left'].append(y_left)
                cluster_data['y_right'].append(y_right)
                cluster_data['delta_left'].append(delta_left)
                cluster_data['delta_right'].append(delta_right)
                cluster_data['l_front'].append(l_front)
                cluster_data['Delta_L'].append(Delta_L)
                cluster_data['Delta_S'].append(Delta_S)
                # Loop over the sub_clusters
                for label in range(1, n_sub_clusters+1):
                    sub_cluster = sub_clusters == label
                    #print("Sub_cluster n. %i" % label)
                    # Check again the cluster_type
                    sub_cluster_type = gal.getAxyLabels(sub_cluster, self.direction, 1)[0]
                    # Update the color map
                    q = self.cluster2D_color[n_exp]
                    q[sub_cluster] = np.int(sub_cluster_type[:2], 2)
                    self.cluster2D_color[n_exp] = q
                    area = np.sum(sub_cluster)
                    if area < self.min_size:
                        continue
                    #print(sub_cluster_type)
                    #print(n_exp, switch, cluster_type)
                    # Calculate the lengths of the subcluster
                    # L is the linear distance between the 
                    l0, l1, L_linear, success = cmet.get_upper_and_lower_contour(sub_cluster, sub_cluster_type,
                                                self.ref_point, motion=self.motion)
                    if success is True:
                        l0, l1 = cmet.get_length(l0), cmet.get_length(l1)
                    sub_cluster_data['switch_frame'].append(switch)
                    sub_cluster_data['switch_time'].append(self.times[n_exp][switch])
                    sub_cluster_data['type'].append(sub_cluster_type)
                    sub_cluster_data['area'].append(area)
                    sub_cluster_data['l0'].append(l0)
                    sub_cluster_data['l1'].append(l1)
                    sub_cluster_data['L_linear'].append(L_linear)
                    sub_cluster_data['success'].append(success)

            self.last_switch[n_exp] = last_switch
            cluster_cols = ['n_exp', 'switch_frame', 'switch_time', 'type', 'area', 'n_sub_cl',
                            'a_10', 'a_01', 'r_curv', 'straightness', 'x_v', 'y_v',
                            'y_left', 'y_right', 'delta_left', 'delta_right',
                            'l_front', 'Delta_L', 'Delta_S']
            df = pd.DataFrame.from_dict(cluster_data)
            self.cluster_data[n_exp] = df[cluster_cols]
            sub_cluster_cols = ['switch_frame', 'switch_time','type', 'area', 'l0', 'l1', 'L_linear', 'success']
            df = pd.DataFrame.from_dict(sub_cluster_data)
            self.sub_cluster_data[n_exp] = df[sub_cluster_cols]
            del df
            
    def plot_global_stats(self, curvature='straightness',color='red', whiter=.5):
        clrs, cmap = get_liza_colors(color, whiter)
        # Do it only after exp and global stats
        fig, axs = plt.subplots(2,2)
        q = self.all_clusters
        # Plot the hist of the curvature
        ax = axs[0,0]
        if curvature == 'straightness':
            c = q.straightness.dropna()
            n, bins, patches = ax.hist(c, bins=100, facecolor=clrs[4], alpha=0.75)
            ax.set_xlabel("Straighness")
            ax.set_title(r'${Straightness:}\ \mu=%.2f,\ \sigma=%.2f$' % (c.mean(), c.std()))
        elif curvature == 'radius':
            c = q.r_curv.dropna()
            n, bins, patches = ax.hist(c, bins=300, range=(0,3000), facecolor=clrs[4], alpha=0.75)
            ax.set_xlabel("Radius of Curvature")
            ax.set_title(r'${R.\ of\ Curvature:}\ \mu=%.2f,\ \sigma=%.2f$' % (c.mean(), c.std()))
        # Plot the pie
        ax = axs[0,1]
        labels = [Axy[:2] for Axy in self._Axy_types]
        fracs = [q[q.type==Axy].area.sum() for Axy in self._Axy_types]
        explode=(0.05, 0, 0, 0)
        textprops = {'fontsize':16}
        _, _, autotexts= ax.pie(fracs, explode=explode, labels=labels, colors=clrs[1:],
                autopct='%1.1f%%', shadow=True, startangle=90, textprops=textprops)
        for autotext in autotexts:
            autotext.set_color('white')
        ax.axis('equal')
        # Plot the hist of the left angle
        ax = axs[1,0]
        c = q.a_10
        n, bins, patches = ax.hist(c, bins=90, range=(0,90), facecolor=clrs[3], alpha=0.75)
        ax.set_xlabel("Left angle (degree)")
        ax.set_title(r'$\mathrm{Left\ angle:}\ \mu=%.2f,\ \sigma=%.2f$' % (c.mean(), c.std()))
        ax = axs[1,1]
        c = q.a_01
        n, bins, patches = ax.hist(c, bins=90, range=(0,90), facecolor=clrs[2], alpha=0.75)
        ax.set_xlabel("Right angle (degree)")
        ax.set_title(r'$\mathrm{Right\ angle:}\ \mu=%.2f,\ \sigma=%.2f$' % (c.mean(), c.std()))
        fig.suptitle(self.title, fontsize=20)
        ##########
        plt.show()

    def plot_cluster_maps(self, color='red', whiter=0.5, zoom_in_data=True, Ncols=5):
        clrs0, cmap = get_liza_colors(color, whiter)
        figs, axs_i = [], []
        n_figs = np.int(np.ceil(float(self._len_experiments)/Ncols))
        r = self._len_experiments % Ncols
        ns = self._len_experiments//Ncols*[Ncols] + [r]*(r>0)
        for n in ns:
            fig, axs = plt.subplots(1, n, sharey=False, squeeze=False)
            figs.append(fig)
            axs_i.append(axs)

        dh_max = 0
        h_min = []
        v_s = []
        for i, n_exp in enumerate(self.cluster2D_color):
            cluster2D_color = self.cluster2D_color[n_exp]
            cluster2D = self.cluster2D[n_exp]
            times = self.times[n_exp]
            rows, cols = cluster2D.shape
            n_fig = np.int(i/Ncols)
            n_ax = i % Ncols
            ax = axs_i[n_fig][0,n_ax]
            if zoom_in_data:
                rows_mean_sw = np.mean(cluster2D, axis=1)
                jj = np.where(rows_mean_sw != -1)
                i0, i1 = np.min(jj) - 20, np.max(jj) + 20
                if i0 < 0:
                    i0 = 0
                if i1 > rows:
                    i1 = rows
                dh = i1 - i0
                if dh > dh_max:
                    dh_max = dh
                h_min.append(i0)
            # Check if all the cluster_types are present!
            clrs = clrs0[:1] + [clrs0[i+1] for i in range(4) if i in cluster2D_color]
            cmap = colors.ListedColormap(clrs,'liza')
            ax.imshow(cluster2D_color, cmap=cmap, interpolation='nearest')
            switches = np.unique(cluster2D)[1:]
            for switch in switches:
                cluster = cluster2D==switch
                cnts = measure.find_contours(cluster, 0.5)
                for cnt in cnts:
                    X,Y = cnt[:,1], cnt[:,0]
                    ax.plot(X, Y, c='k', antialiased=True, lw=1)
            # Calculate the velocities
            #cluster = cluster2D < switches[0]
            cluster = np.logical_or((cluster2D == -1), (cluster2D>self.last_switch[n_exp]))
            cnts = measure.find_contours(cluster, 0.5)
            lens = [len(cnt) for cnt in cnts]
            l_values = heapq.nlargest(2, lens)
            i_s = [lens.index(l) for l in l_values]
            d = []
            for i in i_s:
                X,Y = cnts[i][:,1], cnts[i][:,0]
                meanY = np.mean(Y)
                d.append(meanY)
                ax.plot(X, meanY*X/X, '--w', antialiased=True, lw=2)
            print(d)
            distance_pixel = np.abs(d[1]-d[0])
            delta_time = times[self.last_switch[n_exp]] - times[switches[0]]
            print(distance_pixel, delta_time)
            v = distance_pixel / delta_time * self.um_per_pixel
            print(n_exp, v)
            v_s.append(v)
            ax.set_title("[%i] v:%.2f um/s" % (n_exp,v))
        v_mean = np.mean(v_s)
        v_error = np.std(v_s) / len(v_s)**0.5
        print("Average velocity: %.3f +/- %.3f (um/s)" % (v_mean, v_error))

        for i in range(self._len_experiments):
            n_fig = np.floor(i/np.float(Ncols)).astype(np.int)
            ax = axs_i[n_fig][0,i%Ncols]
            if zoom_in_data:
                ax_coords = 0, cols, h_min[i] + dh_max, h_min[i]
            else:
                ax_coords = 0, cols, rows, 0
            ax.axis(ax_coords)
            if Ncols >= 5:
                ax.set_xticks(np.linspace(0,cols,3))
        for fig in figs:
            fig.suptitle(self.title, fontsize=20)
        plt.show()

    def plot_area_vs_length(self, min_size=0):
        fig, axs = plt.subplots(3, 2, sharey=True, sharex=True, squeeze=False)
        q = self.all_sub_clusters
        q = q[q.area > min_size]
        colors_line = ['r', 'b', 'g']
        colors_bullet = [[1,0.5,0.5], [0.5,0.5,1], [0.5,1,0.5]]
        ls = ["l_0", "L_{lin}"]
        for i, Axy in enumerate(self._Axy_types[:3]):
            qq = q[q.type==Axy]
            qq = qq[qq.success==True]
            for j,qx in enumerate([qq.l0, qq.L_linear]):
                ax = axs[i,j]
                ax.loglog(qx, qq.area, 'o', label="type = %s" % Axy, color=colors_bullet[i])
                #x,y = np.log10(qx), np.log10(qq.area)
                q2 = np.vstack((qx, qq.area)).transpose()
                x,y = gLD.averageLogDistribution(q2, log_step=0.125)
                ax.plot(x, y, 'o', ms=10, color=colors_line[i])
                x,y = np.log10(x), np.log10(y)
                e, c = np.polyfit(x[2:],y[2:],1)
                x = 10**np.sort(x)
                ax.plot(x,10**c*x**e, '-', lw=2, color=colors_line[i], label="slope = %.2f" % e)
                ax.grid(True)
                ax.legend(numpoints=1, loc=4)
            axs[i,0].set_ylabel("Cluster area", fontsize=22)
        for j in range(2):
            ax = axs[2,j]
            ax.set_xlabel(r"Cluster length $%s$" % ls[j], fontsize=22)        
        plt.suptitle(self.title, fontsize=20)
        plt.show()

    def plot_0000_distributions(self, log_step=0.125):
        fig, axs = plt.subplots(1, 3, sharey=True, sharex=True, squeeze=False)
        subc = self.all_sub_clusters
        s000 = subc[subc.type == '0000']
        A000 = s000.area
        L000 = s000.l0.dropna()
        # loglog Plot
        x,y,yerr = gLD.logDistribution(A000, log_step=log_step)
        ax.xlabel(r"Cluster Area $A_{00}$", size=20)
        ax.ylabel(r"Cluster Area Distribution $P(A_{00})$", size=20)

        ax.xlabel(r"Cluster length $l_0$", size=20)
        ax.ylabel(r"Length distribution $P(l_{0})$", size=20)


    def get_global_stats(self):
        print("Calculating the global stats")
        self.all_clusters = pd.concat([self.cluster_data[cl] for cl in self.cluster_data])
        self.all_sub_clusters = pd.concat([self.sub_cluster_data[cl] for cl in self.sub_cluster_data])
        self.total_area = np.sum(self.all_clusters.area)
        for Axy in self._Axy_types:
            q = self.all_sub_clusters[self.all_sub_clusters.type==Axy]['area']
            s = q.sum()
            print("Type %s: fraction of area covered is %.2f" % 
                (Axy, s/float(self.total_area)*100.))
        self.is_global_stats = True
        # TODO: plot hist of angles
        # distributions of Pxy and Axy


    def get_angles(self, cluster, n_fit=30):
        cnts = self._select_contour(cluster)
        X, Y = cnts[:,1], cnts[:,0]
        ####################################
        # Left edge
        ####################################
        z = np.polyfit(X[:n_fit], Y[:n_fit], 1)
        z0 = np.arctan(z[0])
        #print("left: %f" % z0)
        angle_left = np.abs(z0) * 180 / np.pi
        #mksize = 15./700*(np.sum(cluster0)) + 1
        #plt.plot(switch, angle, 'o', c=colors[cluster_type], markersize=mksize, alpha=0.5)
        ####################################
        # Right edge
        ####################################
        z = np.polyfit(X[-n_fit:], Y[-n_fit:], 1)
        z0 = np.arctan(z[0])
        #print("right: %f" % z0)
        angle_right = abs(z0) * 180 / np.pi
        return angle_left, angle_right
        

    def _select_contour(self, cluster, position='bottom'):
        """
        select the two longest contours in a list
        then get the one corresponding to the position
        Care has to be used with initial cluster,
        which are typical of 1101 type
        """
        cnts = measure.find_contours(cluster, 0.5)
        lens = [len(cnt) for cnt in cnts]
        cl_type = gal.getAxyLabels(cluster, self.direction, 1)[0]
        if len(lens) >= 2:
            l_values = heapq.nlargest(2, lens)
            if cl_type == '1101':
                i0 = lens.index(l_values[0])
                return cnts[i0]
            else:
                i0, i1 = [lens.index(l) for l in l_values]
                if l_values[0] == l_values[1]:
                    i1 = len(lens) - 1 - lens[::-1].index(l_values[0])
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
        else:
            # Check if this is the upper cluster
            if cl_type == '1101':
                return cnts[0]
            else:
                print("There is a problem with ", cl_type)

if __name__ == "__main__":
    plt.close("all")
    imParameters = {}
    choice = sys.argv[1]
    # As on June 26, this is the example to followcl
    # Added hdf5=True
    # run mokas_cluster_distributions IEF_old 20um 0.145A 1
    width, set_current, n_wire = sys.argv[2:]
    if choice == 'IEF_old':
        # Example: 
        # run mokas_cluster_distributions.py IEF_old 20um 0.145A 1
        fname = "/home/gf/Meas/Creep/CoFeB/Wires/Arianna/Ta_CoFeB_MgO_wires_{0}/Ta_CoFeB_MgO_wires_{0}.hdf5".format(choice)
        grp0  = "%s/%s/10fps/wire%s" % (width, set_current, n_wire)
        if not os.path.isfile(fname):
            print("Check the path")
            sys.exit()
        n_experiments = range(1,11)
        #n_experiments = range(1,3)
        color = 'red'
        min_size = 5
    elif choice == 'IEF_new':
        # Example: 
        # run mokas_cluster_distributions.py IEF_old 20um 0.145A 1
        fname = "/home/gf/Meas/Creep/CoFeB/Wires/Arianna/Ta_CoFeB_MgO_wires_{0}/Ta_CoFeB_MgO_wires_{0}.hdf5".format(choice)
        grp0  = "%s/%s/10fps/wire%s" % (width, set_current, n_wire)
        if not os.path.isfile(fname):
            print("Check the path")
            sys.exit()
        n_experiments = range(1,11)
        #n_experiments = [1]
        color = 'green'
        min_size = 5
    elif choice == 'LPN':
        # Example: 
        # run mokas_cluster_distributions.py LPN 20um 0.145A 1
        fname = "/home/gf/Meas/Creep/CoFeB/Wires/Arianna/Ta_CoFeB_MgO_wires_{0}/Ta_CoFeB_MgO_wires_{0}.hdf5".format(choice)
        grp0  = "%s/%s/10fps/wire%s" % (width, set_current, n_wire)
        if not os.path.isfile(fname):
            print("Check the path")
            sys.exit()
        n_experiments = [1,2,3,4,5,7,8,9,10] # 6 for .145A is bad
        color = 'blue'
        min_size = 5


    clusters = Clusters(fname, grp0, n_experiments, skip_first_clusters=0, min_size=min_size)
    #clusters.get_experiment_stats(curvature_method='polyfit_4')
    clusters.get_experiment_stats(curvature_method='circlefit')
    clusters.get_global_stats()
    clusters.plot_global_stats(color=color, curvature='radius')
    clusters.plot_cluster_maps(color=color, whiter=0.7, Ncols=10, zoom_in_data=False)
    clusters.plot_area_vs_length()
    ap = clusters.angle_persistence(0)