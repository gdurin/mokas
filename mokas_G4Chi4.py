#!/usr/bin/env python
from __future__ import print_function
import sys, os
import pickle, shutil
import re, string, time
import random
import numpy as np
from scipy import optimize
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import angles
import h5py
import polar
import pandas as pd

def calc_R(c, x, y):
    xc, yc = c
    return np.sqrt((x-xc)**2 + (y-yc)**2) # an array of individual distances

def f_2(c, x, y):
    r_i = calc_R(c, x, y)
    return r_i - r_i.mean()   # if they were all equal, this would be zero -> perfect circle


class Bubble_properties():
    """
    class to handle bubble properties
    it works on a sequence of contours
    as a function of time
    """
    def __init__(self, contours, times=None, start_time=0, normalize_angles=False):
        if times is not None:
            self.times = times
        elif isinstance(contours, dict):
            self.times = contours.keys()
        else:
            print("Missing times. Exit")
            sys.exit()
        if start_time not in self.times:
            print("Start time is not correct")
            sys.exit()
        # Here we are assuming the contours is a dictionary
        self.dws = {}
        # Let's also create a pandas df with a set of angles as rows,
        # the (real) times as columns, and the distances from the center
        self.df = pd.DataFrame()
        switches = contours.keys()
        diff_switches = np.diff(switches)
        #fig = plt.figure()
        #ax = fig.add_subplot(111)
        for i, switch in enumerate(switches[:-1]):
            self.dws[switch] = {}
            contour = contours[switch]
            x, y = np.transpose(contour)
            center = self._fit_center(x, y)
            self.dws[switch]['center'] = center
            if not i:
                center0 = center
                thetas = self._get_angles(x, y, center0, normalize=normalize_angles)
                #n_new_thetas = len(thetas)
                k = len(thetas)
                n_new_thetas = 2**(int(np.round((np.log(k)/np.log(2)))))
                new_thetas = np.linspace(-np.pi, np.pi, n_new_thetas)
            thetas = self._get_angles(x, y, center0, normalize=normalize_angles)
            order = np.argsort(thetas)
            thetas = thetas[order]
            self.dws[switch]['radius'] = self._fit_radius(x, y, center0)
            self.dws[switch]['angle'] = thetas
            self.dws[switch]['dist'] = self._get_distances(contour)
            r = self._get_distances_from_center(contour, center0)
            r = r[order]
            self.dws[switch]['dist_from_center'] = r
            self.dws[switch]['dw'] = contour
            if i:
                new_r = np.interp(new_thetas, thetas, r)
                diff_sw = diff_switches[i]
                # check if there are missing switches
                # copy the same contour
                for k in range(diff_sw):
                    tm = times[switch+k]
                    self.df[tm] = new_r
            # if i in [10, 50, 100]:
            #     ax.plot(x, y, '-v')
            #     X = center0[0] + new_r * np.cos(new_thetas) 
            #     Y = center0[1] + new_r * np.sin(new_thetas) 
            #     ax.plot(X,Y,'o')
            #     ax.set_aspect('equal')
            #     ax.grid(True)
            #     plt.show()
        self.df = self.df.set_index(new_thetas, 'thetas')
        print("Setup of bubbles dict done")
        #self._get_max_displacement()

    @property
    def events(self):
        pass

    def _fit_center(self, x, y):
        center_estimate = np.mean(x),np.mean(y)
        (xc_m,yc_m), ier = optimize.leastsq(f_2, center_estimate, args=(x,y))  # done by scipy
        if not ier:
            print("There is a problem to fit the center of the bubble")
        return xc_m, yc_m

    def _fit_radius(self, x, y, center):
        R = calc_R(center, x, y).mean()
        return R

    def _get_angles(self, x, y, center, normalize=False):
        xc, yc = center
        X, Y = x - xc, y -yc
        _angles = np.arctan2(Y, X)
        #_angles = np.arctan(Y, X)
        # angle=[2*np.pi+a if a<0 else a for a in angle]  # to stay in [0:2pi]
        if normalize:
            _angles = np.array([angles.normalize(a, 0, 2*np.pi) for a in _angles])
        return _angles

    def _get_distances(self, contour):
        # distance from one point to the next - dist array has same length as dw array
        # along the contours, each point's distance from the next is 1 or Sqrt[2]/2
        dist = np.sqrt((np.diff(contour,axis=0) ** 2).sum(axis=1))
        dist = np.append(dist, 0.)  # 1st & last point are the same
        return dist

    def _get_distances_from_center(self, contour, center):
        """
        get an array of the distances from the center
        """
        xc, yc = center
        #x, y = np.hsplit(contour, 2)
        x, y = contour[:,0], contour[:,1]
        distCenter = ((x-xc)**2 + (y-yc)**2)**0.5
        return distCenter

    def _get_max_displacement(self):
        tmax, dmax = -1, -1
        for switch in self.dws:
            xc, yc = self.dws[switch]['center']
            x, y = np.hsplit(self.dws[switch]['dw'], 2)
            dnew = np.sqrt((x-x0)**2 + (y-y0)**2)
            dnew = dnew.mean()
            if dnew > dmax: 
                tmax = switch
                dmax = dnew
        str0 = "Max center displacement is %5.3f at switch time %s" % (dmax, tmax)
        str1 = "%5.3f percent of bubble (at t=%d) radius\n\n" % (100*dmax/self.dws[tmax]['radius'], tmax)
        sys.stderr.write("%s, or %s" % (str0, str1))

class CalcG4chi4():
    """
    Calculus of the G4 and chi4 values using a dataframe
    df has the rows given by the angles of the points
    and columns given by the times
    """
    def __init__(self, df):
        self.mean_radius = df.mean().values
        self.times = df.columns
        self.thetas = df.index
        # Calculus of the delta_h for all the thetas and times
        self.h = df
        self.dh = self.h - self.mean_radius # 


    def _calc_interface_fluctuations(self):
        # Calculus of the distribution of the rescaled height
        # of an interface
        # I. One time distribution (unclear yet, not done)
        # h = (h(0,t_2) - R_t)/R_t^(1/3)
        h = self.dh/self.mean_radius**(1/3.)

    def _calc_S_q(self, ref_i=(3,40), zeta=2./3):
        """
        Calculation of the structure factor
        As a matter of fact it is a power spectrum in the q space
        As the data are done for the angles theta, the first S_q
        is calculate for a q which is in terms of angular distance
        The calculus for space (along the circle) has to be performed
        using first the data at theta which give different r
        at different times, so we need to interpolate the data
        """
        slope = 1 + 2 * zeta
        N_thetas, N_times = self.h.shape
        d_theta = self.h.index[1] - self.h.index[0]
         # Calculate the q for the theta angles
        q_theta = np.linspace(0.0, 1.0/(2.0*d_theta), N_thetas//2)
        hq = df.apply(np.fft.fft, axis=0)
        hq_conj = hq.apply(np.conjugate)
        sq = np.real(hq * hq_conj)
        sq = sq[:N_thetas//2, :]
        S_q_theta = np.mean(sq, axis=1) # mean over time
        # Calculation at q for the r
        for i,radius in enumerate(self.mean_radius):
            d_r = d_theta * radius
            q_i = np.linspace(0.0, 1.0/(2.0*d_r), N_thetas//2)
            if i==0:
                q = q_i
            else:
                sq[:,i] = np.interp(q, q_i, sq[:,i])
        S_q = np.mean(sq, axis=1)
        # Plot S(q_theta)
        fig, axs = plt.subplots(1,2)
        axs[0].loglog(q_theta[1:], S_q_theta[1:], 'ko')
        # Plot the low q depinning exponent (-1)
        i0, i1 = ref_i
        fct = S_q_theta[i0]/q_theta[i0]**(-slope)
        axs[0].loglog(q_theta, fct*q_theta**(-slope), 'r-', label='slope: %.2f' % slope)
        # fct = S_q_theta[i1]/q_theta[i1]**(-1.33)
        # t = q_theta[10:-10]
        # axs[0].loglog(t, fct*t**(-1.33), 'r--', label='slope: 1.33')
        axs[0].legend(fontsize=12)
        axs[0].grid(True)
        axs[0].set_xlabel(r"$q_{\theta}$", size=20)
        axs[0].set_ylabel(r"$S(q_{\theta})$", size=20)
        #
        # Plot S(q)
        #
        axs[1].loglog(q[1:], S_q[1:], 'ko')
        # Plot the low q depinning exponent (-1)
        fct = S_q[i0]/q[i0]**(-slope)
        axs[1].loglog(q, fct*q**(-slope), 'r-', label='slope: %.2f' % slope)
        # fct = S_q[i1]/q[i1]**(-1.33)
        # t = q[10:-10]
        # axs[1].loglog(t, fct*t**(-1.33), 'r--', label='slope: 1.33')
        axs[1].legend(fontsize=12)
        axs[1].grid(True)
        axs[1].set_xlabel(r"$q$", size=20)
        axs[1].set_ylabel(r"$S(q)$", size=20)
        sq = pd.Series(S_q, index=q)
        return sq

    def _calc_G4(self, theta_max=45, time_max=None, theta_step=10, time_step=10):
        # Preliminar dataframes
        # calc df at a delta_theta
        # Ns = theta_N, time_N
        # Steps are in units of dtheta and dtime
        if time_max is None:
            time_max = self.times[-1]
        #theta_step, time_step = steps
        dtheta = np.abs(self.thetas[1] - self.thetas[0])
        dtime = np.abs(self.times[1] - self.times[0])
        theta_max = int(theta_max/180. * np.pi / dtheta)
        time_max = int(time_max/dtime)
        i_theta = np.arange(theta_step, theta_max, theta_step)
        j_times = np.arange(time_step, time_max, time_step)
        G4_theta = np.zeros((len(i_theta), len(j_times)))
        G4_r = np.zeros_like(G4_theta)
        C_theta = np.zeros_like(G4_theta)
        C_r = np.zeros_like(G4_theta)
        #W_r = np.zeros_like(G4_theta)
        # Define the minimum step of r (radial distance)
        # by the min angle at the last contour
        r_min = dtheta * self.mean_radius[-1]
        r_radius = self.mean_radius[-1] / self.mean_radius
        rows, cols = self.dh.shape
        for j, j_delta_time in enumerate(j_times):
            print("Times: %i/%i   " % (j, len(j_times)), end="\r")
            sys.stdout.flush()
            dh_0_0 = self.dh.iloc[:, :-j_delta_time].values
            dh_0_t = self.dh.iloc[:, j_delta_time:].values
            # f1 is an np.array
            f1 = dh_0_0 * dh_0_t
            f1_mean = f1.mean()
            for i, i_delta_theta in enumerate(i_theta):
                # Roll over the rows (i.e. thetas)
                dh_theta_0 = np.roll(self.dh, -i_delta_theta, axis=0)
                dh_theta_t = dh_theta_0[:, j_delta_time:]
                dh_theta_0 = dh_theta_0[:, :-j_delta_time]
                # f2 is a np.array
                f2 = dh_theta_0 * dh_theta_t
                f12 = f1 * f2
                G4_theta[i,j] = f12.mean() - (f1_mean * f2.mean())
                C_theta_t = (dh_theta_t - dh_0_t)**2
                C_theta[i,j] = C_theta_t.mean()**0.5
                #
                # Calculus for r distance
                #
                theta_steps = i_delta_theta * r_radius
                rl = -theta_steps.astype(int)
                dh_r_0 = np.copy(self.dh.values)
                # roll the rows for diffent values                 
                # https://stackoverflow.com/questions/40359940/numpy-roll-vertical-in-2d-array
                dh_r_0[:, range(cols)] = dh_r_0[np.mod(np.arange(rows)[:,None]+rl, rows), range(cols)]
                dh_r_t = dh_r_0[:, j_delta_time:]
                dh_r_0 = dh_r_0[:, :-j_delta_time]
                f3 = dh_r_0 * dh_r_t
                f13 = f1 * f3
                G4_r[i,j] = f13.mean() - (f1_mean * f3.mean())
                C_r_t = (dh_r_t - dh_0_t)**2
                C_r[i,j] = C_r_t.mean()**0.5
        print("")
        # Out of the loop
        tm = np.diff(self.times).mean() * j_times
        tm = tm - tm[0]
        theta = dtheta * i_theta * 180 /np.pi
        G4_theta = pd.DataFrame(G4_theta, index=theta, columns=tm)
        C_theta = pd.DataFrame(C_theta, index=theta, columns=tm)
        r = dtheta * i_theta * self.mean_radius[-1]
        G4_r = pd.DataFrame(G4_r, index=r, columns=tm)
        C_r = pd.DataFrame(C_r, index=r, columns=tm)
        # Plot
        self._plot_G4(G4_theta, 'theta')
        self._plot_G4(G4_r, var='r')
        self._plot_C(C_theta, var='theta')
        self._plot_C(C_r, var='r')
        return G4_theta, G4_r, C_theta, C_r
        

    def _plot_C(self, df, var, step=5):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        x = df.index
        for col in df.columns[::step]:
            y = df[col].values
            lb = "%i s" %(int(np.round(col)))
            ax.loglog(x, y, 'o', label=lb)
        if var == 'theta':
            ax.set_xlabel(r'$\theta (degree)$')
            ax.set_ylabel(r'$C(\theta, t)^{0.5}$')
        elif var == 'r':
            ax.set_xlabel('r (pixels)')
            ax.set_ylabel(r'$C(r, t)^{0.5}$')
        if var == 'r':
            ax.plot(x, 1.05*y[5]*(x/x[5])**0.66, 'k--', label=r'$r^{2/3}$')
        ax.grid(True)
        ax.legend()
        plt.show()

    def _plot_G4(self, df, var='theta'):
        x = df.columns
        y = df.index
        X,Y = np.meshgrid(x,y)
        Z = df.values
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        #ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
        ax.plot_surface(X, Y, Z, vmin=1.1*np.min(Z), vmax=1.1*np.max(Z), cmap='viridis', rstride=1, cstride=1)
        ax.set_xlabel('time (s)')
        if var == 'theta':
            ax.set_ylabel(r'$\theta$ (degree)')
            ax.set_zlabel(r'$G4 (\theta, t)$', rotation=180)
            G4_1 = r"<\delta\rho(0,0) \delta\rho(0,t) \delta\rho(\theta,0) \delta\rho(\theta,t)>"
            G4_2 = r"<\delta\rho(0,0) \delta\rho(0,t)> <\delta\rho(\theta,0) \delta\rho(\theta,t)>"
            G4_0 = r"G4(\theta, t)"
        elif var == 'r':
            ax.set_ylabel(r'$r (pixels)$')
            ax.set_zlabel(r'$G4 (r, t)$', rotation=180)
            G4_0 = r"G4(r, t)"
            G4_1 = r"<\delta\rho(0,0) \delta\rho(0,t) \delta\rho(r,0) \delta\rho(r,t)>"
            G4_2 = r"<\delta\rho(0,0) \delta\rho(0,t)> <\delta\rho(r,0) \delta\rho(r,t)>"
        #ax.set_title(r'$%s = %s - %s$' % (G4_0, G4_1, G4_2))
        text2D = r'$%s = %s - %s$' % (G4_0, G4_1, G4_2)
        ax.set_title(text2D)
        #ax.text2D(0.05, 1.2, text2D, transform=ax.transAxes)
        plt.show()

    
#plt.close("all")
if __name__ == "__main__":
    test = 'meas'
    #test = 'circle'
    if test == 'circle':
        N = 361
        k = 5
        x0, y0 = 500, 500
        contour = {}
        phi = np.linspace(0, 2*np.pi, N)
        nbubbles = 150
        step_radius = 8
        R_in = 100
        Rs = np.arange(R_in, R_in + (nbubbles+1) * step_radius, step_radius)
        times = (Rs-Rs[0])/10.
        for t, R in enumerate(Rs):
            #x, y = R * np.cos(phi)+k*np.random.rand(N) + x0, R * np.sin(phi) + k*np.random.rand(N) + y0
            r = R + np.abs(k*np.sin(10*phi))
            x, y = r*np.cos(phi), r*np.sin(phi)
            q = np.array([x,y])
            q = q.transpose()
            contour[t] = q
        bubble = Bubble_properties(contour, times)
        df = bubble.df
        center = bubble._fit_center(x, y)
        print(center)
        print(x0, y0)
        r_est = f_2(center, x, y)
        xc, yc = center
        r_est = np.sqrt((x-xc)**2 + (y-yc)**2).mean()
        print(R, r_est)
        if True:
            fig = plt.figure()
            ax = fig.gca()
            for c in contour:
                q = contour[c]
                x, y = q[:,0], q[:,1]
                ax.plot(x,y,'o')
            ax.plot(xc+r_est * np.cos(phi), yc+r_est * np.sin(phi),'-')
            ax.set_aspect('equal')
        _angles = bubble._get_angles(x,y,center)
        #calc = CalcG4chi4(bubble.dws, t_elements=5)
        c = CalcG4chi4_df(df)
        G4_theta, G4_r = c._calc_G4(theta_max=135, time_max=20., steps=(1,2))
    elif test == 'meas':
        # Irr_880uC
        # str_irr = "Irr_880uC"
        # mainDir = "/data/Meas/Creep/CoFeB/Film/SuperSlowCreep/Irr_800uC/Dec2016/"
        # hdf5_fname = "Dec2016.hdf5"
        # field, n_run = "0.116A", "05"
        # n_set = None
        #field, n_run = "0.232A", "05"
        #baseGroup = "0.116A/05" # This works
        #baseGroup = "0.116A/03"

        # not Irradiated
        str_irr = "NonIrr"
        #mainDir = "/data/Meas/Creep/CoFeB/Film/SuperSlowCreep/NonIrr/Dec2016/"
        mainDir = "/data/Meas/Creep/CoFeB/Film/SuperSlowCreep/NonIrr/Feb2018/0.146A"
        #hdf5_fname = "NonIrr.hdf5"
        hdf5_fname = "0.146A.hdf5"
        #field, n_run = "0.095A", "02"
        n_set, field, n_run = "Set1", "0.146A", "02"
        ###########################################
        if n_set:
            baseGroup = "%s/%s/%s" % (n_set, field, n_run)
        else:
            baseGroup = "%s/%s" % (field, n_run)

        fname = os.path.join(mainDir, hdf5_fname)
        contours = {}
        key_type = np.int
        with h5py.File(fname, 'a') as f:
            grp0 = f[baseGroup]['contours']
            for group in grp0: #yes, it looks like a dictionary
                if key_type:
                    key = key_type(group)
                else:
                    key = group
                contours[key] = grp0[group][...]
            try:
                grp0 = f[baseGroup]['times']
                times = grp0[...]
            except:
                fname = "{0}{1}/{2}_{3}_{1}/times.dat".format(mainDir,field, n_run, str_irr)
                times = np.loadtxt(fname)
                times = (times[:,1] - times[0,1])/1000.

        fig, axs = plt.subplots(1,2)
        bubble_props = Bubble_properties(contours, times)
        df = bubble_props.df
        switches = contours.keys()
        switch0 = switches[0]
        sw_step = 40
        angles = np.linspace(0, 2*np.pi, 90)
        dws = bubble_props.dws
        yc, xc = dws[switch0]['center']
        # Plot
        cmap = mpl.cm.gnuplot
        lsw = float(len(switches[::sw_step]))
        for i,switch in enumerate(switches[::sw_step]):
            time = times[switch]
            contour = contours[switch]
            X, Y = contour[:,1], contour[:,0]
            radius = bubble_props.dws[switch]['radius']
            x,y = xc + radius * np.cos(angles), yc + radius * np.sin(angles)
            axs[0].plot(x,y,'-', color=cmap(i/lsw))
            axs[0].plot(X, Y, lw=2, color=cmap(i/lsw))
            axs[1].plot((X-xc)/radius, (Y-yc)/radius,'-', color=cmap(i/lsw))
        axs[1].plot((x-xc)/radius, (y-yc)/radius, lw=2, color='black')
        for i in range(2): 
            axs[i].set_aspect('equal')
            axs[i].grid(True)
        theta, r, frames = polar.plot_displacement(contours, (yc,xc),reference='center')
        plt.show()
        print("Max time: %d (s), N. time steps: %i" % (times[-1], len(times)))
        c = CalcG4chi4(df)
        #G4_theta, G4_r, C_theta, C_r = c._calc_G4(theta_max=60, steps=(2,5))
        G4_theta, G4_r, C_theta, C_r = c._calc_G4(theta_max=60, theta_step=5, time_step=10)
        S_q = c._calc_S_q(ref_i=(6,40))
        store = pd.HDFStore(fname)
        store.put(baseGroup+"/S_q", S_q)
        store.put(baseGroup+"/G4_theta", G4_theta)
        store.put(baseGroup+"/G4_r", G4_r)
        store.put(baseGroup+"/C_theta", C_theta)
        store.put(baseGroup+"/C_r", C_r)
        store.close()
    plt.show()

