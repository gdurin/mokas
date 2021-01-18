import sys, os
import numpy as np
import matplotlib.pyplot as plt
import visualBarkh as bk
import pickle
import mokas_bubbles as mkb
from natsort import natsorted
import glob, re
from mokas_colors import get_colors
# import deepdish as dd
import mokas_run_bubbles as rb


if __name__ == "__main__":
    plt.close("all")
    imParameters = {}
    #print(sys.argv)
    try:
        irradiation = sys.argv[1]
    except:
    	irradiation = 'NonIrr'
    
    if irradiation == 'NonIrr':
        #k = sys.argv[2]
        #print k
        #k = str(k).rjust(2,"0")
        
        #current = "0.116"
        current = "0.192"
        #rootDir = "/data/Meas/Creep/CoFeB/Film/SuperSlowCreep/%s/Dec2016/%sA" % (irradiation,current)
        #rootDir = "/data/Meas/Creep/CoFeB/Film/SuperSlowCreep/%s/Feb2018/%sA/Set1" % (irradiation,current)
	#rootDir = "/data/Meas/Creep/CoFeB/Film/SuperSlowCreep/%s/Dec2016/%sA" % (irradiation,current)
        #rootDir = "/home/gf/Meas/Creep/CoFeB/Film/SuperSlowCreep/%s/%sA" % (irradiation,current)
        rootDir = "/data/Meas/Creep/CoFeB/Film/SuperSlowCreep/%s/Feb2018/%sA/No_set" % (irradiation,current)
        current = "0.174"
        rootDir = "/data/Meas/Creep/CoFeB/Film/SuperSlowCreep/%s/Feb2018/%sA/Set1" % (irradiation,current)
        rootDir = "/home/gf/Meas/Creep/CoFeB/Film/SuperSlowCreep/%s/Feb2018/%sA/Set1" % (irradiation,current)

        if not os.path.isdir(rootDir):
            print("Check the path: %s" % rootDir)
            sys.exit()

        subdir_pattern = "*_%s_%sA" % (irradiation,current)
        filename_suffix = "_MMStack_Pos0.ome.tif"

        bubbles = rb.RunBubbles(rootDir, subdir_pattern)

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
