import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import h5py
import pandas as pd
import mokas_bestfit as bestfit


def adjust_ax(ax):
    ax.get_yaxis().set_tick_params(which='both', direction='in')
    ax.get_xaxis().set_tick_params(which='both', direction='in')
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')

def get_best_fit(x, y, y_err, n_params, p0, min_index=2, max_index=-2,):
    sd = bestfit.Size_Distribution(n_params)
    x, y = x[min_index:max_index], y[min_index:max_index]
    w = y != 0
    if y_err is not None:
        y_err = y_err[min_index:max_index]
        x, y, y_err = x[w], y[w], y_err[w]
    else:
        x, y = x[w], y[w]
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


data = {}
fields = {"0.137A": "0.13", "0.146A": "0.14", "0.157A": "0.15", "0.165A": "0.16"}
currents = ["0.137A", "0.146A", "0.157A", "0.165A"]
n_set = "Set1"
d_f = "1.000"
#PS_types = ["events", "nij", "nij_filtered", "touch"]
#PS_type = PS_types[2]
nij_s = {"0.137A": "0.44", "0.146A": "0.33", "0.157A": "0.25", "0.165A": "0.22"}
#PS_type = "PS_nij_filtered"
#PS_type = "P_lenghts"
PS_type = "PS_nij_filtered"
ac = {"0.137A": .85, "0.146A": .9, "0.157A": 1, "0.165A": 1}
#nij_s = {"0.137A": "0.44", "0.146A": "0.33", "0.157A": "0.25", "0.165A": "0.19"}
#ac = {"0.137A": .85, "0.146A": .9, "0.157A": 1, "0.165A": 1.2}
#####################################################################
#d_f = "1.633"
#nij_s = {"0.137A": "1.44", "0.146A": "1.23", "0.157A": "1.25", "0.165A": "1.50"}

hname = '/data/Meas/Creep/CoFeB/Film/SuperSlowCreep/NonIrr/Feb2018/Results_NonIrr_Feb2018.hdf5'

store = pd.HDFStore(hname)
clrs = ['b', 'orange', 'g', 'r']

fig, ax = plt.subplots(1,1, figsize=(7,5.8))
#fig1, ax1 = plt.subplots(1,1, figsize=(7,5.8))
for i,current in enumerate(currents):
    n_ij = nij_s[current]
    group = "%s/%s/df_%s/nij_%s/%s" % (current, n_set, d_f, n_ij, PS_type)
    print(group)
    #lb = "Field: %s mT, n_ij: %s" % (fields[current], nij_s[current])
    lb = "%s mT" % fields[current]
    q = store.get(group)
    data[current] = q
    if "P_lenghts" in PS_type:
        x, y = q.length, q.P_length
        y_err = None
    elif PS_type == 'S_mean':
        x, y = q.length, q.S_mean
    else:   
        x, y, y_err = q.S, q.PS, q.PS_err
    if PS_type == 'PS_nij_filtered':
        A = ac[current]
        y = x * y
    else:
        A = 1.

    ax.loglog(x, A*y, 'o', label=lb, c=clrs[i], ms=6)
    #ax.legend(loc=3)
    l = ax.legend(loc=3, fontsize='large', title="Applied fields")
    plt.setp(l.get_title(),fontsize='large')
    # Data fitting
    params, errors, ier, x_calc, y_calc = get_best_fit(x, y, y_err, 
                            n_params=3, p0=None, min_index=None, max_index=None)
    #ax1.loglog(x, (i+1)*A*y, 'o', label=lb, c=clrs[i], ms=6)
    #ax1.loglog(x_calc, (i+1)*A*y_calc, '-', c=clrs[i])



if PS_type == 'events':
    xlabel, ylabel = 'event size (S)', "$P(S)$"
    ax.plot(S, .015*(S/10.)**(-1.17) * np.exp(-S/37.), 'k--')
elif PS_type == 'PS_nij_filtered' or PS_type == 'PS_nij':
    xlabel, ylabel = "$S_{Clust}$", "$P(S_{Clust})$"
    A = 0.18 * 10
    tau = 1.28 - 1
    imax = 3
    lw = 1.5
    msize = 18
    ax.plot(x[:-imax], A*(x[:-imax]/10.)**(-tau) * np.exp(-x[:-imax]/210.), 'b--', lw=lw)
    imax = 3
    ax.plot(x[:-imax], A*(x[:-imax]/10.)**(-tau) * np.exp(-x[:-imax]/300.), '--', lw=lw, c='orange')
    ax.plot(x, (A+.01)*(x/10.)**(-tau) * np.exp(-(x/270.)**0.6), 'g--', lw=lw)
    ax.plot(x, (A+.01)*(x/10.)**(-tau) * np.exp(-(x/320.)**0.6), 'r--', lw=lw)
    a = np.array([7,4000,1e-6,0.3])
    ax.axis(a)
    xlabel, ylabel = r"$S_{clust}\ (px^2)$", r"$P(S_{clust})$"
    ax.set_xlabel(xlabel, size=msize)
    ax.set_ylabel(ylabel, size=msize)
    ax.get_yaxis().set_tick_params(which='both', direction='in')
    ax.get_xaxis().set_tick_params(which='both', direction='in')
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax2 = ax.twiny()
    a2 = a * (0.09)
    ax2.set_xscale("log")
    ax2.set_xlim(a2[:2])
    ax2.get_xaxis().set_tick_params(which='both', direction='in')
    xlabel = r"$S_{clust}\ (\mu m^2)$"
    ax2.set_xlabel(xlabel, size=msize)
    
elif PS_type == "P_lenghts":
    xlabel, ylabel = "$L_{Clust}$", "$P(L_{Clust})$"
    tau = 1.5
    A = 0.13
    imin = 3
    lw = 2
    label_size = 18
    ax.plot(x[imin:-1], 0.1*(x[imin:-1]/10.)**(-tau) * np.exp(-(x[imin:-1]/40.)**1.75), 'k--', lw=lw)
    ax.plot(x[imin:-1], 0.1*(x[imin:-1]/10.)**(-tau), 'k--', lw=lw)
    ax.set_xlabel(r"Major axis length $L\ (px)$", size=label_size)
    ax.set_ylabel(r"Length distribution $P(L)$", size=label_size)
    ax.annotate(r'$\sim L^{-1.5}$', xy=(40, 0.02), size=22)
    ax.annotate(r'$\sim L^{-1.5} e^{-(L/L_o)^{1.5}}$', xy=(11, 1e-4), size=22)
    adjust_ax(ax)
    ax2 = ax.twiny()
    a = np.array([7,4000,1e-6,0.3])
    a2 = a * (0.3)
    ax2.set_xscale("log")
    ax2.set_xlim(a2[:2])
    ax2.get_xaxis().set_tick_params(which='both', direction='in')
    ax2.set_xlabel(r"Major axis length $L\ (\mu m)$", size=label_size)

# if "nij" in PS_type:
#     ax.set_title("Size distribution of clusters")
# else:   
#     ax.set_title("Size distribution of %s" % PS_type)
fig.tight_layout()
ax.grid(False)
plt.show()

store.close()