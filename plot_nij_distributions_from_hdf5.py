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
current = currents[3]
n_set = "Set1"
d_f = "1.000"
#PS_types = ["events", "nij", "nij_filtered", "touch"]
#PS_type = PS_types[2]
nij_s = {"0.137A": "0.44", "0.146A": "0.33", "0.157A": "0.25", "0.165A": "0.19"}
#PS_type = "PS_nij_filtered"
#PS_type = "P_lenghts"
#hijs = ["h_ij_real", "h_ij_shuffled"]
hijs = ['all_events_hierarchy', 'all_events_hierarchy_shuffled']
ac = {"0.137A": .85, "0.146A": .9, "0.157A": 1, "0.165A": 1}
#nij_s = {"0.137A": "0.44", "0.146A": "0.33", "0.157A": "0.25", "0.165A": "0.19"}
#ac = {"0.137A": .85, "0.146A": .9, "0.157A": 1, "0.165A": 1.2}
#####################################################################
#d_f = "1.633"
#nij_s = {"0.137A": "1.44", "0.146A": "1.23", "0.157A": "1.25", "0.165A": "1.50"}

hname = '/data/Meas/Creep/CoFeB/Film/SuperSlowCreep/NonIrr/Feb2018/Results_NonIrr_Feb2018.hdf5'

store = pd.HDFStore(hname)
clrs = ['b', 'orange', 'g', 'r']
frac_dim = 1
d_f = "1.000"
cols = 2
lgs = ['real', 'shuffled']
fig, axs = plt.subplots(1,2, figsize=(12,6))
fig1, axs1 = plt.subplots(1,2, figsize=(12,6), sharex=True, sharey=True)
fig2, axs2 = plt.subplots(1,1, figsize=(6,6))
for i,h_ij in enumerate(hijs):
    n_ij = nij_s[current]
    group = "%s/%s/df_%s/nij_%s/%s" % (current, n_set, d_f, n_ij, h_ij)
    print(group)
    lb = "%s mT" % fields[current]
    df = store.get(group)
    r_ij, t_ij = df['r_ij'], df['t_ij'] 
    x, y = np.log10(r_ij), np.log10(t_ij)
    axs[i].hist2d(x, y, bins=50, norm=colors.LogNorm())
    X = np.linspace(np.min(x), np.max(x))
    Y = -frac_dim * X + np.log10(np.float(n_ij))
    axs[i].plot(X,Y,'r--', lw=2) 
    axs[i].set_xlabel(r"$r^{*}$", size=26)
    axs[i].set_ylabel(r"$\tau^{*}$", size=26)
    axs[i].grid(False)
    X = np.linspace(np.min(r_ij), np.max(r_ij))
    Y = np.float(n_ij) / X 
    axs1[i].loglog(r_ij, t_ij, 'o', c='C%i' % i, markersize=0.5, label=lgs[i], alpha=0.5)
    axs1[i].plot(X,Y,'k--', lw=0.75) 
    axs1[i].legend(markerscale=10)
    axs1[i].axis((.1,100,0.01,10))
    axs1[i].set_xlabel(r"$r^{*}$", size=22)
    axs1[i].set_ylabel(r"$\tau^{*}$", size=22)

for i,h_ij in enumerate(hijs[::-1]):
    n_ij = nij_s[current]
    group = "%s/%s/df_%s/nij_%s/%s" % (current, n_set, d_f, n_ij, h_ij)
    print(group)
    df = store.get(group)
    r_ij, t_ij = df['r_ij'], df['t_ij'] 
    lb = "%s mT" % fields[current]
    df = store.get(group)
    axs2.loglog(r_ij, t_ij, 'o', c='C%i' % (1-i), markersize=0.5, label=lgs[i], alpha=0.8)
    axs2.plot(X,Y,'k--', lw=0.75) 
    axs2.legend()
    axs2.axis((.1,100,0.01,10))
    axs2.set_xlabel(r"$r^{*}$", size=22)
    axs2.set_ylabel(r"$\tau^{*}$", size=22)
for ax in axs1:
    ax.get_yaxis().set_tick_params(which='both', direction='in')
    ax.get_xaxis().set_tick_params(which='both', direction='in')
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')

fig.tight_layout()
fig1.tight_layout()    

plt.show()

store.close()