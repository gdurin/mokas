from __future__ import print_function
import sys
import os
import glob, re
import numpy as np
import matplotlib.pyplot as plt
import mokas_wires as mkwires
from mokas_colors import get_colors


p2p = 3 # Pixel to pixel (linear) distance for cluster detection
NN = 2*p2p + 1
structure = np.ones((NN,NN))


class RunWires:
    def __init__(self, rootDir, subdir_pattern, n_wire=1, erase_small_events_percent=None):
        self.rootDir = rootDir
        # Get the initial parameters
        wire_ini = mkwires.Wires_ini(rootDir, n_wire)
        self.imParameters = wire_ini.imParameters
        self.experiments = wire_ini.experiments
        self.threshold = wire_ini.threshold
        self.motion = wire_ini.motion
        self.erase_small_events_percent = erase_small_events_percent
        # Get the directories based on the pattern
        sub_dirs = np.array(sorted(glob.glob1(rootDir, subdir_pattern)))
        # Prepare to find a match
        q = subdir_pattern.replace("*", "(.*)")
        all_experiments = [np.int(re.search(q, sd).group(1)) for sd in sub_dirs]
        jj = [x in self.experiments for x in all_experiments]
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
        if self.motion in ['leftward', 'rightward']:
            rows1, cols1 = self.n_experiments, 1
        elif self.motion in ['downward', 'upward']:
            rows1, cols1 = 1, self.n_experiments
        self.fig1, self.axs1 = plt.subplots(rows1, cols1, sharex=True, sharey=True) # ColorImages of the wires
        self.figs.append(self.fig1)
        self.fig2, self.axs2 = plt.subplots(self.n_experiments, 1, sharey=True) # Histograms
        self.figs.append(self.fig2)
        self.fig3, self.axs3 = plt.subplots(rows1, cols1, sharex=True, sharey=True) # Pinning Centers
        self.figs.append(self.fig3)
        self.fig4, self.axs4 = plt.subplots(1, 2, sharex=True, sharey=True) # Sizes vs lenghts
        self.figs.append(self.fig4)

        #for n in range(self.n_experiments):
        for n,experiment in enumerate(self.experiments):
            sub_dir = self.sub_dirs[n]
            #trial = sub_dir[:2]
            title = str(experiment).rjust(2, "0")
            self.imParameters['subDirs'] = [self.rootDir, sub_dir, "", "", ""]
            filename = self.filenames[n]
            self.imParameters['pattern'] = filename
            print(experiment, filename)
            #imArray = StackImages(**self.imParameters)
            imArray = mkwires.Wires(self.motion, **self.imParameters)
            if n == 0:
                nImages, rows, cols = imArray.shape
                pColor = get_colors(nImages,'pastel',norm=True)
            self.imArray_collector[experiment] = imArray
            #imArray.useKernel = 'step'
            #imArray.kernelSign = -1
            #imArray.boundary = None
            #imArray.structure = structure
            imArray.showColorImage(self.threshold, palette=pColor, plot_contours=True, plotHist=None, 
                erase_small_events_percent=self.erase_small_events_percent, 
                fig=self.fig1, ax=self.axs1[n], title=title, noSwitchColor='black')
            imArray.plotHistogram(imArray._switchTimesOverThreshold,
                                    fig=self.fig2,ax=self.axs2[n],title=title,ylabel=None)
            # imArray.find_contours(lines_color='k', remove_bordering=True, plot_centers_of_mass=False,
            #                          invert_y_axis=False, plot_rays=False,
            #                          fig=self.fig3, ax=self.axs3[n], title=title)
            imArray.get_stats_prop()
            if n == 0:
                self.sizes = imArray.stats_prop['sizes']
                self.lenghts_initial = imArray.stats_prop['lenghts_initial']
                self.lenghts_final = imArray.stats_prop['lenghts_final']
                self.curvatures_initial = imArray.stats_prop['curvatures_initial']
                self.curvatures_final = imArray.stats_prop['curvatures_final']
            else:
                self.sizes = np.concatenate((imArray.stats_prop['sizes'], self.sizes))
                self.lenghts_initial = np.concatenate((imArray.stats_prop['lenghts_initial'], self.lenghts_initial))
                self.lenghts_final = np.concatenate((imArray.stats_prop['lenghts_final'], self.lenghts_final))
                self.curvatures_initial = np.concatenate((imArray.stats_prop['curvatures_initial'], self.curvatures_initial))
                self.curvatures_final = np.concatenate((imArray.stats_prop['curvatures_final'], self.curvatures_final))
            #self.axs3[n].imshow(imArray.stats_prop['image_corners'])
            if plot_contours:
                for switch in imArray.contours:
                    cnts = imArray.contours[switch]
                    X,Y = cnts[:,1], cnts[:,0]
                    self.axs3[n].plot(X, Y, c='k', antialiased=True, lw=1)
                self.axs3[n].invert_yaxis()

        # Out of the loop
        average_lengths_i, average_sizes = self._get_averages(self.lenghts_initial, self.sizes)
        self.axs4[0].loglog(self.lenghts_initial, self.sizes, 'bo')
        self.axs4[0].loglog(average_lengths_i, average_sizes, 'ro')
        self.axs4[0].set_xlabel("Initial length", fontsize=24)
        self.axs4[0].set_ylabel("Size", fontsize=24)
        self.axs4[0].grid(True)
        average_lengths_f, average_sizes = self._get_averages(self.lenghts_final, self.sizes)
        self.axs4[1].loglog(self.lenghts_final, self.sizes, 'bo')
        self.axs4[1].loglog(average_lengths_f, average_sizes, 'ro')
        self.axs4[1].set_xlabel("Final length", fontsize=24)
        self.axs4[1].set_ylabel("Size", fontsize=24)
        self.axs4[1].grid(True)
        suptitle = " - ".join(self.rootDir.split("/")[-2:])
        for fig in self.figs:
            fig.suptitle(suptitle, fontsize=30)
        plt.show()

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

    elif choice == 'IEF_old_20um':
        set_current = "0.145"
        rootDir = "/home/gf/Meas/Creep/CoFeB/Wires/Arianna/Ta_CoFeB_MgO_wires_IEF_old/20um_%sA" % set_current
        if not os.path.isdir(rootDir):
            print("Chech the path")
            sys.exit()
        subdir_pattern = "20um_%sA_10fps_*"  % set_current
        filename_suffix = "_MMStack_Pos0.ome.tif"
        n_wire = 2
        wires = RunWires(rootDir, subdir_pattern, filename_suffix, n_wire=n_wire, erase_small_events_percent=None)
        wires.plot_results()

    elif choice == 'IEF_old_200um':
        set_current = "0.14"
        rootDir = "/home/gf/Meas/Creep/CoFeB/Wires/Arianna/Ta_CoFeB_MgO_wires_IEF_old/200um_%sA" % set_current
        if not os.path.isdir(rootDir):
            print("Chech the path")
            sys.exit()
        subdir_pattern = "200um_%sA_10fps_*"  % set_current
        filename_suffix = "_MMStack_Pos0.ome.tif"
        n_wire = 1
        wires = RunWires(rootDir, subdir_pattern, filename_suffix, n_wire=n_wire, erase_small_events_percent=None)
        wires.plot_results()

    elif choice == 'IEF_new_20um':
        set_current = "0.11"
        rootDir = "/home/gf/Meas/Creep/CoFeB/Wires/Arianna/Ta_CoFeB_MgO_wires_IEF_new/20um_%sA" % set_current
        if not os.path.isdir(rootDir):
            print("Chech the path")
            sys.exit()
        subdir_pattern = "20um_%sA_10fps_*"  % set_current
        n_wire = 1
        wires = RunWires(rootDir, subdir_pattern, n_wire=n_wire, erase_small_events_percent=None)
        wires.plot_results()


    elif choice == 'irr':
        set_current = ["0.19","0.21","0.23"][0]
        rootDir = "/home/gf/Meas/Creep/CoFeB/Wires/Irradiated/run1_2/"
        subdir_pattern = "*_irradiatedwires_%sA_10fps"  % set_current

        #imParameters['imCrop'] = (0,1392,0,1040)
        #imParameters['imCrop'] = (876,1117,0,1040)
        crop_upper_left_pixel = (880,0)
        crop_lower_right_pixel = (1115,1040)
        imParameters['imCrop'] = [crop_upper_left_pixel, crop_lower_right_pixel]
        #imParameters['imCrop'] = None
        filename_suffix = "_MMStack_Pos0.ome.tif"
        #imParameters['pattern'] = "01_irradiatedwires_%sA_10fps_MMStack_Pos0.ome.tif" % set_current
        imParameters['firstIm'] = 1
        imParameters['lastIm'] = 240
        imParameters['filtering'] = 'gauss'
        #imParameters['filtering'] = None
        imParameters['sigma'] = 1.5
        threshold = 10
        wires = RunWires(rootDir, subdir_pattern, filename_suffix, imParameters, threshold, experiments=range(2))
        wires.plot_results()


    else:
        print("Check the path!")
        sys.exit()

        
        

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

