from __future__ import print_function
import sys
import os
import glob, re
import numpy as np
import matplotlib.pyplot as plt
import mokas_wires as mkwires
from mokas_colors import get_colors
from natsort import natsorted


class RunWires:
    def __init__(self, rootDir, subdir_pattern, n_wire=1, 
                erase_small_events_percent=None):
        self.rootDir = rootDir
        # Get the initial parameters
        wire_ini = mkwires.Wires_ini(rootDir, n_wire)
        self.imParameters = wire_ini.imParameters
        self.experiments = wire_ini.experiments
        self.threshold = wire_ini.threshold
        self.rotation = self.imParameters['rotation']
        self.erase_small_events_percent = erase_small_events_percent
        self.wireParameters = {'motion' : wire_ini.motion, 
                            'edge_trim_percent' : wire_ini.edge_trim_percent, 
                            'zoom_in_data' : True}
        # Get the directories based on the pattern
        sub_dirs = natsorted(glob.glob1(rootDir, subdir_pattern))
        sub_dirs = np.array(sub_dirs)
        # Prepare to find a match
        q = subdir_pattern.replace("*", "(.*)")
        all_experiments = [np.int(re.search(q, sd).group(1)) for sd in sub_dirs]
        jj = np.array([x in self.experiments for x in all_experiments])
        self.sub_dirs = sub_dirs[jj]
        #sdirs = [subdir_pattern.replace("*", str(i)) for i in self.experiments]
        #self.sub_dirs = [sd for sd in sdirs if sd in sub_dirs]
        self.filenames = [d + wire_ini.filename_suffix for d in self.sub_dirs]
        print("There are %i files to analyse on the wire %i " % (len(self.filenames), n_wire))
        if self.experiments is None:
            self.n_experiments = len(self.sub_dirs)
            self.experiments = range(self.sub_dirs)
        else:
            self.n_experiments = len(self.experiments)
        print(self.filenames[0])
        self.full_title = ", ".join(self.filenames[0].split("_")[1:4])
        

    def plot_results(self, plot_contours=True):
        """
        Plot the different images for creep in wires
        """
        # Prepare to plot
        plt.close("all")
        self.figs = []
        self.imArray_collector = {}
        # We choose to plot ALL measurements in the downward direction ONLY!
        rows1, cols1 = 1, self.n_experiments
        self.fig1, self.axs1 = plt.subplots(rows1, cols1, sharex=True, sharey=True, squeeze=False) # ColorImages of the wires
        self.figs.append(self.fig1)
        self.fig2, self.axs2 = plt.subplots(self.n_experiments, 1, sharey=True, squeeze=False) # Histograms
        self.figs.append(self.fig2)
        self.fig3, self.axs3 = plt.subplots(cols1, 3*rows1, sharex=True, sharey=True, squeeze=False) # events and clusters
        self.figs.append(self.fig3)
        #for n in range(self.n_experiments):
        allParameters = self.wireParameters.copy()
        for n, experiment in enumerate(self.experiments):
            sub_dir = self.sub_dirs[n]
            #trial = sub_dir[:2]
            title = str(experiment).rjust(2, "0")
            self.imParameters['subDirs'] = [self.rootDir, sub_dir, "", "", ""]
            filename = self.filenames[n]
            self.imParameters['pattern'] = filename
            print("#" * 50)
            print("Experiment # %i: %s" % (experiment, filename))
            #imArray = StackImages(**self.imParameters)
            allParameters.update(self.imParameters)
            imArray = mkwires.Wires(**allParameters)
            if n == 0:
                nImages, rows, cols = imArray.shape
                pColor = get_colors(nImages, 'pastel', norm=True)
            self.imArray_collector[experiment] = imArray
            imArray.showColorImage(self.threshold, palette=pColor, 
                                    plot_contours=True, plotHist=None, 
                                    erase_small_events_percent=self.erase_small_events_percent, 
                                    fig=self.fig1, ax=self.axs1[0, n], 
                                    title=title, noSwitchColor='black')
            imArray.plotHistogram(imArray._switchTimesOverThreshold,
                                    fig=self.fig2, ax=self.axs2[n, 0],
                                    title=title, ylabel=None)
            # imArray.find_contours(lines_color='k', remove_bordering=True, plot_centers_of_mass=False,
            #                          invert_y_axis=False, plot_rays=False,
            #                          fig=self.fig3, ax=self.axs3[n], title=title)
            #imArray.get_stats_prop()
            imArray.getEventsAndClusters(get_clusters_method='limits', cluster_threshold=30)
            # TO BE FIXED
            axs = self.axs3[n,0], self.axs3[n,1], self.axs3[n,2]
            imArray.plot_cluster_maps(imArray._colorMap, zoom_in_data=True, 
                                        fig=self.fig3, axs=axs, 
                                        title=title, with_cluster_number=False)
                    
        suptitle = " - ".join(self.rootDir.split("/")[-2:])
        for fig in self.figs:
            fig.suptitle(suptitle, fontsize=30)
        plt.show()

    def save_hdf5(self):
        for experiment in self.imArray_collector:
            wire = self.imArray_collector[experiment]
            data = [wire.cluster2D_start, wire.cluster2D_end, wire._switchTimes2D, wire._switchSteps2D]
            labels = ['cluster2D_start', 'cluster2D_end', 'switchTimes2D', 'switchSteps2D']
            wire.hdf5.save_data(data, labels, dtype=np.int16)
            # Save histogram
            hist = [wire.N_hist, wire.bins_hist]
            hist_labels = ['N_hist', 'bins_hist']
            wire.hdf5.save_data(hist, hist_labels, dtype=np.float32)

    def save_figs(self):
        res_dir = os.path.join(self.rootDir, 'Results')
        if not os.path.isdir(res_dir):
            os.mkdir(res_dir)
        out_string = "_".join([str(e) for e in self.experiments])
        filename = os.path.join(res_dir, "events_and_clusters_exp%s.png" % out_string)
        self.fig3.savefig(filename)


    def _get_averages(self, lengths, sizes):
        # Calculus of the avalanche sizes and durations
        average_lengths = np.unique(lengths)
        average_sizes = []
        for l in average_lengths:
            size = np.mean(np.extract(lengths == l, sizes))
            average_sizes.append(size)
        average_sizes = np.array(average_sizes)
        return average_lengths, average_sizes

if __name__ == "__main__":
    plt.close("all")
    imParameters = {}
    try:
        choice = sys.argv[1]
    except:
        choice = 'irr'
    if choice == 'half_moon':
        rootDir = "/home/gf/Meas/Creep/CoFeB/Film/Non-irradiated/Half-moon/run3/02_nonirradiatedFilm_0.14A"
        #imParameters['imCrop'] = (200,1040,500,1390)
        #imParameters['imCrop'] = (0,1392,0,1040)
        crop_upper_left_pixel = (0,0)
        crop_lower_right_pixel = (1392,1040)
        imParameters['imCrop'] = [crop_upper_left_pixel, crop_lower_right_pixel]
        #imParameters['imCrop'] = None
        imParameters['pattern'] = "02_nonirradiatedFilm_0.14A_MMStack_Pos0.ome.tif"
        imParameters['firstIm'] = 1
        imParameters['lastIm'] = 250
        imParameters['filtering'] = 'gauss'
        #imParameters['filtering'] = None
        imParameters['sigma'] = 0.5
        imParameters['resize_factor'] = None
        threshold = 30

    elif choice == 'non_irr':
        set_current = ["0.20","0.22","0.24"][0]
        rootDir = "/home/gf/Meas/Creep/CoFeB/Wires/Yuting/nonirrad wire/"
        if not os.path.isdir(rootDir):
            print("Chech the path of %s" % rootDir)
            sys.exit()
        subdir_pattern = "*_nonirradiatedwires_%sA_10fps"  % set_current
        filename_suffix = "_MMStack_Pos0.ome.tif"
        n_wire = 2
        wires = RunWires(rootDir, subdir_pattern, filename_suffix, n_wire=n_wire)
        wires.plot_results()

    elif choice == 'LPN_20um':
        set_current = "0.14"
        rootDir = "/home/gf/Meas/Creep/CoFeB/Wires/Arianna/Ta_CoFeB_MgO_wires_LPN/20um_%sA" % set_current
        if not os.path.isdir(rootDir):
            print("Chech the path")
            sys.exit()
        subdir_pattern = "20um_%sA_10fps_*"  % set_current
        filename_suffix = "_MMStack_Pos0.ome.tif"
        n_wire = 2
        wires = RunWires(rootDir, subdir_pattern, filename_suffix, n_wire=n_wire, erase_small_events_percent=None)
        wires.plot_results()

    # As on June 26, this is the example to follow
    # Added hdf5=True
    elif choice == 'IEF_old':
        # example
        # run run_visualBarkh_on_wires IEF_old 20um 0.145 2
        width, set_current, n_wire = sys.argv[2:]
        rootDir = "/home/gf/Meas/Creep/CoFeB/Wires/Arianna/Ta_CoFeB_MgO_wires_IEF_{0}/{1}/{1}_{2}A".format(choice, width, set_current)
        if not os.path.isdir(rootDir):
            print("Chech the path")
            sys.exit()
        subdir_pattern = "%s_%sA_10fps_*"  % (width, set_current)

    elif choice == 'LPN':
        # example
        # run run_visualBarkh_on_wires IEF_old 20um 0.145 2
        width, set_current, n_wire = sys.argv[2:]
        rootDir = "/home/gf/Meas/Creep/CoFeB/Wires/Arianna/Ta_CoFeB_MgO_wires_{0}/{1}/{1}_{2}A".format(choice, width, set_current)
        if not os.path.isdir(rootDir):
            print("Chech the path")
            sys.exit()
        subdir_pattern = "%s_%sA_10fps_*"  % (width, set_current)

    elif choice == 'IEF_new':
        # example
        # run run_visualBarkh_on_wires IEF_new 20um 0.14 2
        width, set_current, n_wire = sys.argv[2:]
        rootDir = "/home/gf/Meas/Creep/CoFeB/Wires/Arianna/Ta_CoFeB_MgO_wires_{0}/{1}/{1}_{2}A".format(choice, width, set_current)
        if not os.path.isdir(rootDir):
            print("Chech the path")
            sys.exit()
        subdir_pattern = "%s_%sA_10fps_*"  % (width, set_current)

    else:
        print("Check the path!")
        sys.exit()
    n_wire = np.int(n_wire)
    wires = RunWires(rootDir, subdir_pattern, n_wire=n_wire, erase_small_events_percent=None)
    wires.plot_results()
    wires.save_hdf5()
    wires.save_figs()
        
        

        # imParameters['subDirs'] = [rootDir, "", "", "", ""]
        # imArray = bk.StackImages(**imParameters)
        # imArray.width='all'
        # imArray.useKernel = 'both'
        # imArray.kernelSign = -1
        # #imArray.useKernel = 'zero'
        # imArray.imageDir = "Left_to_right"
        # imArray.boundary = None
        
        # imArray.structure = np.ones((NN,NN))
        # imArray.showColorImage(threshold,palette='pastel',plot_contours=False)
        # #imArray.find_contours(lines_color='k',remove_bordering=True, plot_centers_of_mass=True,reference=None)
        #     #'center_of_mass')

