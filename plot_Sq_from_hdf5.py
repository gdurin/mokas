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
        #print("Setup of bubbles dict done")
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

class CalcSq():
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
        hq = self.h.apply(np.fft.fft, axis=0)
        hq_conj = hq.apply(np.conjugate)
        sq = np.real(hq * hq_conj)
        sq = sq[:N_thetas//2, :]
        S_q_theta = np.mean(sq, axis=1) # mean over time
        # Calculation at q for the r
        for i, radius in enumerate(self.mean_radius):
            d_r = d_theta * radius
            q_i = np.linspace(0.0, 1.0/(2.0*d_r), N_thetas//2)
            if i==0:
                q = q_i
            else:
                sq[:,i] = np.interp(q, q_i, sq[:,i])
        S_q = np.mean(sq, axis=1)
        return pd.Series(S_q, index=q)


    
#plt.close("all")
if __name__ == "__main__":
    test = sys.argv[1]
    if test == 'measSq':
        # not Irradiated
        str_irr = "NonIrr"
        n_set = "Set1"
        #field, n_run = "0.146A", "08"
        fields = ['0.137', '0.146', '0.157', '0.165']
        #n_runs = ["10", "08", "05", "04"]
        #n_runs = [range(2,16), range(1,9), range(2,6), range(1,5)] # Full sets
        n_runs = [range(8,16), range(4,9), range(3,6), range(2,5)]
        fields_mT = {"0.137": "0.13", "0.146": "0.14", "0.157": "0.15", "0.165": "0.16"}
        A = {"0.137": 1.1, "0.146": .6, "0.157": .8, "0.165": 1.}
        imaxs = {"0.137": 200, "0.146": 1200, "0.157": 1200, "0.165": 500}
        i0, i1 = (6, 40)
        #slope = 1 + 2 * 2/3.
        slope = 1 + 2 * 0.633
        fig = plt.figure(figsize=(6,7))
        ax = fig.gca()
        key_type = np.int
        mainDir = "/data/Meas/Creep/CoFeB/Film/SuperSlowCreep/NonIrr/Feb2018/"
        hdf5_filename_results = "Results_NonIrr_Feb2018.hdf5"
        hname = os.path.join(mainDir, hdf5_filename_results)
        #store = pd.HDFStore(hname)
        is_k = False
        for field, n_run in zip(fields, n_runs):
            mainDir = "/data/Meas/Creep/CoFeB/Film/SuperSlowCreep/NonIrr/Feb2018/%sA" % field
            hdf5_fname = "%sA.hdf5" % field
            fname = os.path.join(mainDir, hdf5_fname)
            fh = h5py.File(fname, 'a')
            contours = {}
            for nr in n_run:
                baseGroup = "%s/%sA/%02d" % (n_set, field, nr)
                print(baseGroup)
                grp0 = fh[baseGroup]['contours']
                for group in grp0: #yes, it looks like a dictionary
                    if key_type:
                        key = key_type(group)
                    else:
                        key = group 
                    contours[key] = grp0[group][...]
                try:
                    grp0 = fh[baseGroup]['times']
                    times = grp0[...]
                except:
                    fname = "{0}{1}/{2}_{3}_{1}/times.dat".format(mainDir,field, n_run, str_irr)
                    times = np.loadtxt(fname)
                    times = (times[:,1] - times[0,1])/1000.
                bubble_props = Bubble_properties(contours, times)
                df = bubble_props.df
                c = CalcSq(df)
                S_q = c._calc_S_q()
                if nr == n_run[0]:
                    q = S_q.index
                    sq = S_q.values
                else:
                    q_i = S_q.index
                    sq += np.interp(q, q_i, S_q.values)
            #store.put(baseGroup+"/S_q", S_q)
            fh.close()
            sq = sq / len(n_run)
            lb = "%s mT" % fields_mT[field]
            imax = imaxs[field]
            print(field)
            #ax.loglog(q[1:-imax], A[field]*sq[1:-imax], 'o', ms=4, label=lb)
            # Getting the slope
            len2 = len(q)//2
            slp, cnts = np.polyfit(np.log10(q[1:len2]), np.log10(sq[1:len2]),1)
            #ax.loglog(q, 10**cnts * q**(slp), 'k--')
            print(-slp)
            if not is_k:
                k0 = 10**cnts
                k = 1
                is_k = True
            else:
                k = k0/10**cnts
            ax.loglog(q[:512], A[field]*k*sq[:512], 'o', ms=4, label=lb)
            #ax.loglog(q[:512], A[field]*k*sq[:512], 'o', ms=4, label=lb, mfc='none')
        # Plot the low q depinning exponent (-1)
        ax.loglog(q[:512], 14*q[:512]**(-slope), 'k--')
        l = ax.legend(loc=3, fontsize='large', title="Applied fields")
        plt.setp(l.get_title(),fontsize='large')
        ax.grid(False)
        text_size = 18
        a = np.array([1e-3,1,1e2,1e8])
        ax.axis(tuple(a))
        ax.set_xlabel(r"$q\ (px^{-1})$", size=text_size)
        ax.set_ylabel(r"$structure\ factor\ S(q)$", size=text_size)
        ax.get_yaxis().set_tick_params(which='both', direction='in')
        ax.get_xaxis().set_tick_params(which='both', direction='in')
        ax.yaxis.set_ticks_position('both')
        #ax1.get_yaxis().set_tick_params(which='both', direction='in')
        ax2 = ax.twiny()
        #ax2.set_xticklabels()
        a2 = a / 0.3
        ax2.set_xscale("log")
        ax2.set_xlim(a2[:2])
        ax2.set_xlabel(r"$q\ (\mu m^{-1})$", size=text_size)
        ax2.get_yaxis().set_tick_params(which='both', direction='in')
        ax2.get_xaxis().set_tick_params(which='both', direction='in')
        ax.annotate(r'$\sim q^{-(1 + 2 \zeta_{dep})}$', xy=(0.008, 1.5e6), size=24)
        #store.close()
    fig.tight_layout()
    plt.show()

