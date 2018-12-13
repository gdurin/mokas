import sys, os
from pathlib import Path
import pandas as pd
from run_bubble_stats import Clusters


if __name__ == "__main__":
    #plt.close("all")
    imParameters = {}
    n_experiments = {}
    crops = None
    choice = sys.argv[1]
    try:
        irradiation = sys.argv[1]
        current_field = sys.argv[2]
    except:
        irradiation = 'Irr_800uC'
    
    fieldDir = None
    set_n = None

    
    if irradiation == 'NonIrr_Dec18':
        set_n = "Set1"
        zeta = 0.633
        d_f = 1
        mainDir = "/data/Meas/Creep/CoFeB/Film/SuperSlowCreep/NonIrr/Feb2018/%sA/" % current_field
        fieldDir = os.path.join(mainDir, set_n)
        hdf5_filename = "%sA.hdf5" % current_field
        hdf5_filename_results = "Results_NonIrr_Feb2018.hdf5"
        ns_experiments = {"0.137": range(2, 16), "0.146": range(1,9), 
                          "0.157": [2,3,4,5], "0.165": range(1,5)}

        if current_field == "0.137":
            nij_list, clrs = [0.44], ['r']
            #nij_list, clrs = [1.44], ['r']
        elif current_field == "0.146":
            nij_list, clrs = [0.33], ['r']
            #nij_list, clrs = [1.23], ['r']
        elif current_field == "0.157":
            nij_list, clrs = [0.25], ['r']
            #nij_list, clrs = [1.25], ['r']
        elif current_field == "0.165":
            nij_list, clrs = [0.19], ['r']
            #nij_list, clrs = [1.5], ['r']
        min_size, hist_dx = 5, 0.01

    field  = "%sA" % (current_field)
    print("Analysing %s" % field)
    n_experiments = ns_experiments[current_field]
    cl = Clusters(mainDir, hdf5_filename, field, n_experiments, set_n=set_n,
                fieldDir=fieldDir, skip_first_clusters=0, min_size=min_size)
    # Get the statistic of the events
    cl.get_event_stats()
    PS_events, fig0 = cl.plot_cluster_stats(cl.all_events.event_size.values, p0=None, lb='raw events', color='g')
    # Cluster with n_ij
    cl.show_correlation(event_size_threshold=min_size, dx=hist_dx, frac_dim=d_f)
    all_clusters_nij = {}
    cluster2D_nij = {}
    ####################################
    for clr, nij_max in zip(clrs,nij_list):
        lb = 'from n_ij = %.2f' % nij_max
        title = r"clusters with $n_{ij} = %.2f$" % nij_max
        cln = cl.get_clusters_nij(cl.con_to_df, nij_max, title=title)
        cluster2D_nij[nij_max] = cln
        #ac = cl.get_cluster_stats_from_nj2(cln, 'events with nij: %.2f' % nij_max)
        #all_clusters_nij[nij_max] = ac
        #PS_nij, fig = cl.plot_cluster_stats(ac.area.values, fig=fig0, lb=lb, color=clr)
        #cl.plot_cluster_maps(cl.cluster2D_start, cln)
        cln_filtered = cl.clean_small_clusters(cln)
        # ac_filtered = cl.get_cluster_stats_from_nj2(cln_filtered, 
        #                                     'events with nij: %.2f, filtered' % nij_max)
        # PS_nij_filtered, fig = cl.plot_cluster_stats(ac_filtered.area.values, 
        #                         fig=fig0, lb=lb+' filtered', color='m', max_index=None)
        
        cl.plot_cluster_maps(cl.cluster2D_start, cln_filtered)
        # df_S_vs_l, df_S_mean, df_P_lenghts, df_P_lenghts_norm = cl.plot_cluster_lengths(ac_filtered)
        # start3 = datetime.datetime.now()
        # diff = start3 - start2
        # print_time(diff)
        # Save to the upper directory into a hdf5
        up_dir = str(Path(mainDir).parent)
        hname = os.path.join(up_dir, hdf5_filename_results)    
        save_data = raw_input("Save data?")
        if save_data.upper() == 'Y':
            store = pd.HDFStore(hname)
            subDir = "%s/%s/df_%.3f/nij_%.2f" % (field, set_n, d_f, nij_max)
            distrs = [PS_events, PS_nij_filtered, cl.h_ij_real, cl.h_ij_shuffled]
            distrs += [df_S_vs_l, df_S_mean, df_P_lenghts, df_P_lenghts_norm]
            _distrs = ['PS_events', 'PS_nij_filtered', 'h_ij_real', 'h_ij_shuffled']
            _distrs += ['S_vs_l', 'S_mean', 'P_lenghts', 'P_lenghts_norm']
            for d, _distr in zip(distrs, _distrs):
                group = "%s/%s" % (subDir, _distr)
                store[group] = d
            store.close()
