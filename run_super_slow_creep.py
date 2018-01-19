import sys, os
import numpy as np
import matplotlib.pyplot as plt
import visualBarkh as bk
import cPickle as pickle
import mokas_bubbles as mkb
from natsort import natsorted
import glob, re
from mokas_colors import get_colors
# import deepdish as dd



class RunBubbles:
    def __init__(self, rootDir, subdir_pattern):
        self.rootDir = rootDir
        # Get the initial parameters
        bubbles_ini = mkb.Bubbles_ini(rootDir)
        self.imParameters = bubbles_ini.imParameters
        self.experiments = bubbles_ini.experiments
        self.thresholds = bubbles_ini.thresholds
        try:
            erase_small_events_percent = bubbles_ini.erase_small_events_percent
        except:
            erase_small_events_percent = None
        self.erase_small_events_percent = erase_small_events_percent

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
        self.filenames = [d + bubbles_ini.filename_suffix for d in self.sub_dirs]
        print("There are %i files to analyse" % len(self.filenames))
        if self.experiments is None:
            self.n_experiments = len(self.sub_dirs)
            self.experiments = range(self.sub_dirs)
        else:
            self.n_experiments = len(self.experiments)
        print(self.filenames[0])
        self.full_title = ", ".join(self.filenames[0].split("_")[1:4])


    def plot_results(self, plot_contours=True):
        """
        Plot the different images for creep in bubbles
        """
        # Prepare to plot
        plt.close("all")
        self.figs = []
        self.imArray_collector = {}
        # We choose to plot ALL measurements 
        rows1, cols1 = 1, self.n_experiments
        self.fig1, self.axs1 = plt.subplots(rows1, cols1, sharex=True, sharey=True, squeeze=False) # ColorImages 
        self.figs.append(self.fig1)
        self.fig2, self.axs2 = plt.subplots(self.n_experiments, 1, sharey=True, squeeze=False) # Histograms
        self.figs.append(self.fig2)
        self.fig3, self.axs3 = plt.subplots(cols1, 3*rows1, sharex=True, sharey=True, squeeze=False) # events and clusters
        self.figs.append(self.fig3)
        #for n in range(self.n_experiments):
        #allParameters = self.wireParameters.copy()
        
        for n, experiment in enumerate(self.experiments):
            sub_dir = self.sub_dirs[n]
            title = str(experiment).rjust(2, "0")
            self.imParameters['subDirs'] = [self.rootDir, sub_dir, "", "", ""]
            filename = self.filenames[n]
            self.imParameters['pattern'] = filename
            print("#" * 50)
            print("Experiment # %i: %s" % (experiment, filename))
            print("The threshold applied is %f" %(self.thresholds[n]))
            #imArray = StackImages(**self.imParameters)
            #allParameters.update(self.imParameters)
            
            imArray = mkb.Bubbles(**self.imParameters)
            if n == 0:
                nImages, rows, cols = imArray.shape
                pColor = get_colors(nImages, 'pastel', norm=True)
            self.imArray_collector[experiment] = imArray
            imArray.showColorImage(self.thresholds[n], palette= 'random', #pColor, 
                                    plot_contours=True, plotHist=None, 
                                    fig=self.fig1, ax=self.axs1[0, n], 
                                    title=title, noSwitchColor='black')
            imArray.plotHistogram(imArray._switchTimesOverThreshold,
                                    fig=self.fig2, ax=self.axs2[n, 0],
                                    title=title, ylabel=None)
            # imArray.find_contours(lines_color='k', remove_bordering=True, plot_centers_of_mass=False,
            #                          invert_y_axis=False, plot_rays=False,
            #                          fig=self.fig3, ax=self.axs3[n], title=title)
            #imArray.get_stats_prop()
            imArray.getEventsAndClusters(method='edges')
            # TO BE FIXED
            axs = self.axs3[n,0], self.axs3[n,1], self.axs3[n,2]
            imArray.plotEventsAndClustersMaps(fig=self.fig3, axs=axs)
                    
        suptitle = " - ".join(self.rootDir.split("/")[-2:])
        for fig in self.figs:
            fig.suptitle(suptitle, fontsize=30)
        plt.show()  



    def save_hdf5(self):
        for experiment in self.imArray_collector:
            bubble = self.imArray_collector[experiment]
            data = [bubble.cluster2D_start, bubble.cluster2D_end, bubble._switchTimes2D, bubble._switchSteps2D]
            labels = ['cluster2D_start', 'cluster2D_end', 'switchTimes2D', 'switchSteps2D']
            bubble.hdf5.save_data(data, labels, dtype=np.int16)
            # Save histogram
            hist = [bubble.N_hist, bubble.bins_hist]
            hist_labels = ['N_hist', 'bins_hist']
            bubble.hdf5.save_data(hist, hist_labels, dtype=np.float32)
            # Save contours
            bubble.hdf5.save_data(bubble.contours, 'contours', dtype=np.float32)
            # Save waiting time histogram
            try:
                bubble.hdf5.save_data(bubble.waiting_times_hist, 'waiting_time', dtype=np.int)
            except:
                pass



    def save_figs(self):
        res_dir = os.path.join(self.rootDir, 'Results')
        if not os.path.isdir(res_dir):
            os.mkdir(res_dir)
        out_string = "_".join([str(e) for e in self.experiments])
        filename = os.path.join(res_dir, "events_and_clusters_exp%s.png" % out_string)
        self.fig3.savefig(filename)





if __name__ == "__main__":
    plt.close("all")
    imParameters = {}
    #print(sys.argv)
    try:
        irradiation = sys.argv[1]
    except:
    	irradiation = 'Irr_800uC'
    
    if irradiation == 'Irr_800uC':
        #k = sys.argv[2]
        #print k
        #k = str(k).rjust(2,"0")
        current = "0.232"
        rootDir = "/data/Meas/Creep/CoFeB/Film/SuperSlowCreep/%s/Dec2016/%sA" % (irradiation,current)

        if not os.path.isdir(rootDir):
            print("Check the path: %s") % rootDir
            sys.exit()

        subdir_pattern = "*_%s_%sA" % (irradiation,current)
        filename_suffix = "_MMStack_Pos0.ome.tif"

        bubbles = RunBubbles(rootDir, subdir_pattern)

        bubbles.plot_results()
        for experiment in bubbles.imArray_collector:
            bubble = bubbles.imArray_collector[experiment]
            bubble.waiting_times_map(is_plot=False)
        bubbles.save_hdf5()
        #bubbles.save_figs()

        


        #############################################


        # imParameters['pattern'] = "%s_Irr800uC_0.116A_MMStack_Pos0.ome.tif" % k
        # crop_upper_left_pixel, crop_lower_right_pixel = (450,330), (860,750)
        # print(imParameters['pattern'])
        # imParameters['imCrop'] = [crop_upper_left_pixel, crop_lower_right_pixel]
        # # imParameters['imCrop'] = None
        # imParameters['firstIm'] = 0 # Use python convention: start from zero$
        # imParameters['lastIm'] = -1
        # imParameters['filtering'] = 'gauss'
        # imParameters['sigma'] = 2
        # imParameters['subtract'] = None # Subtract a reference image
        # threshold = 12
        # palette = 'random'
        # erase_small_events_percent = None
        # imParameters['resize_factor'] = None
        # # Kernel setups: do not touch
        # imParameters['kernel_half_width_of_ones'] = 10
        # #imParameters['kernel_internal_points'] = 0
        # #imParameters['kernel_switch_position'] = "end"
        # ##############################
        # imParameters['subDirs'] = [rootDir, "", "", "", ""]
        # #A possible improvement is that stackImages returns also the threshold value extracting the information when uploading the images
       
        # imArray = mkb.Bubbles(**imParameters)
        # imArray.showColorImage(threshold=threshold, palette=palette, plot_contours=True, 
        #     erase_small_events_percent=None)
        #imArray.find_contours(lines_color='k', remove_bordering=True, plot_centers_of_mass=False,
        #    plot_rays=False, reference=None,invert_y_axis=True)
        #'center_of_mass')
        #save_data = raw_input("Save the data?")

        #imArray.pickle_switchMap2D(mainDir=rootDir)    
