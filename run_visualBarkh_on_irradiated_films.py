import sys, os
import numpy as np
import matplotlib.pyplot as plt
import visualBarkh as bk
import pickle

if __name__ == "__main__":
    plt.close("all")
    imParameters = {}
    print(sys.argv)
    try:
        choice = sys.argv[1]
    except:
        choice = 'Creep'
    if choice == "Creep":
        k = sys.argv[2]
        print k
        palette = 'pastel'
        #rootDir = "/home/gf/Meas/Creep/PtCoAu50Pt50/PtCoAuPt_2c-0d-100pOe-0.975V-1.2s"
        #rootDir = "/home/gf/Meas/Creep/PtCoAu50Pt50/Rotation/90 degree/PtCoAuPt_3_2c-90d-350pOe-0.780V-3.5s_6"
        if k == "0":
            rootDir = "/home/gf/Meas/Creep/PtCoAu50Pt50/Rotation/0 degree/PtCoAuPt_3_2c-00d2-000nOe-0.780V-5.0s_1"
            imParameters['imCrop'] = (0,510,0,672)
            imParameters['pattern'] = "filename*.png"
            imParameters['firstIm'] = 1
            imParameters['lastIm'] = 15
            imParameters['filtering'] = 'gauss'
            #imParameters['filtering'] = None
            imParameters['sigma'] = 1.1
            threshold = 50
        elif k == "1":
            rootDir = "/home/gf/Meas/Creep/PtCoAu50Pt50/Rotation/0 degree/PtCoAuPt_3_2c-00d2-350pOe-0.780V-4.3s_11"
            print(rootDir)
            imParameters['imCrop'] = (0,510,0,672)
            imParameters['pattern'] = "filename*.png"
            imParameters['firstIm'] = 1
            imParameters['lastIm'] = 21
            imParameters['filtering'] = 'gauss'
            #imParameters['filtering'] = None
            imParameters['sigma'] = 2.0
            threshold = 40
        #rootDir = "/home/gf/Meas/Creep/PtCoAu/PtCoAu_2c-0d-500pOe-1.275V-1.0s"
        elif k == "22":
            rootDir = "/home/gf/Meas/Creep/PtCoPt/M2/PtCoPt_2-2c-0d-000nOe-0.657V-40.0s_1"
            imParameters['imCrop'] = (0,510,0,672)
            imParameters['pattern'] = "filename*.png"
            imParameters['firstIm'] = 1
            imParameters['lastIm'] = 144
            imParameters['filtering'] = 'gauss'
            #imParameters['filtering'] = None
            imParameters['sigma'] = 1.5
            threshold = 25
        elif k == 23:
            rootDir = "/home/gf/Meas/Creep/PtCoPt/M2/PtCoPt_2-2c-0d-100nOe-0.657V-30.0s_18"
            imParameters['imCrop'] = (0,510,0,672)
            imParameters['pattern'] = "filename*.png"
            imParameters['firstIm'] = 1
            imParameters['lastIm'] = 89
            imParameters['filtering'] = 'gauss'
            #imParameters['filtering'] = None
            imParameters['sigma'] = 1.5
            threshold = 25
        elif k == 24:
            rootDir = "/home/gf/Meas/Creep/PtCoPt/M2/PtCoPt_2-2c-0d-100pOe-0.657V-30.0s_19"
            imParameters['imCrop'] = (0,510,0,672)
            imParameters['pattern'] = "filename*.png"
            imParameters['firstIm'] = 1
            imParameters['lastIm'] = 88
            imParameters['filtering'] = 'gauss'
            #imParameters['filtering'] = None
            imParameters['sigma'] = 1.5
            threshold = 25
        elif k == 25:
            rootDir = "/home/gf/Meas/Creep/PtCoPt/M2/PtCoPt_2-2c-0d-1000nOe-0.657V-10.0s_32"
            imParameters['imCrop'] = (0,510,0,672)
            imParameters['pattern'] = "filename*.png"
            imParameters['firstIm'] = 2
            imParameters['lastIm'] = 41
            imParameters['filtering'] = 'gauss'
            #imParameters['filtering'] = None
            imParameters['sigma'] = 2.
            threshold = 25
        elif k == 2:
            rootDir = "/home/gf/Meas/Creep/Alex/PtCoPt_simm/run6/imgs"
            imParameters['pattern'] = "img*.tif"
            imParameters['firstIm'] = 2
            imParameters['lastIm'] = 70
            imParameters['initial_domain_region'] = (639,432,658,443)
            #imParameters['imCrop'] = (0,800,0,1200)
            imParameters['imCrop'] = None
            imParameters['filtering'] = 'gauss'
            #imParameters['filtering'] = None
            imParameters['sigma'] = 2.5
            threshold = 5
        elif k == 3:
            rootDir = "/home/gf/Meas/Creep/CoFeB/Film/Non-irradiated/Half-moon/run3/02_nonirradiatedFilm_0.14A"
            #imParameters['imCrop'] = (200,1040,500,1390)
            #imParameters['imCrop'] = (0,1392,0,1040)
            crop_upper_left_pixel = (0,0)
            crop_lower_right_pixel = (1392,1040)
            imParameters['imCrop'] = [crop_upper_left_pixel, crop_lower_right_pixel]
            #imParameters['imCrop'] = None
            imParameters['pattern'] = "02_nonirradiatedFilm_0.14A_MMStack_Pos0.ome.tif"
            imParameters['firstIm'] = 1
            imParameters['lastIm'] = -1
            imParameters['filtering'] = 'gauss'
            #imParameters['filtering'] = None
            imParameters['sigma'] = 1.
            threshold = 30
        elif k == 30:
            n = "01"
            current = "16"
            rootDir = "/home/gf/Meas/Creep/CoFeB/Film/Irradiated/Irradiated_Yuting/%s_irradiatedFilm_0.%sA_10fps" % (n, current)
            #imParameters['imCrop'] = (200,1040,500,1390)
            #imParameters['imCrop'] = (270,970,200,950) # good for 01 0.16A
            imParameters['imCrop'] = None
            imParameters['pattern'] = "%s_irradiatedFilm_0.%sA_10fps_MMStack_Pos0.ome.tif" % (n,current)
            imParameters['firstIm'] = 0 # Use python convention: start from zero!
            imParameters['lastIm'] = 249
            imParameters['filtering'] = 'gauss'
            #imParameters['filtering'] = None
            imParameters['sigma'] = 2
            imParameters['subtract'] = 0 # Subtract the first image
            threshold = 8
        elif k == 31:
            n = "01"
            current = "15"
            rootDir = "/home/gf/Meas/Creep/CoFeB/Film/Non-irradiated/Moon/run6/%s_nonirradiatedFilm_0.%sA_10fps" % (n,current)
            #imParameters['imCrop'] = (200,1040,500,1390)
            #imParameters['imCrop'] = (270,970,200,950) # good for 01 0.16A
            #crop_upper_left_pixel, crop_lower_right_pixel = (270,120), (1100,920) # Good for n=03, current=15
            crop_upper_left_pixel, crop_lower_right_pixel = (200,80), (1200,1020) # Good for n=01, current=15
            imParameters['imCrop'] = [crop_upper_left_pixel, crop_lower_right_pixel]
            #imParameters['imCrop'] = None
            imParameters['pattern'] = "%s_nonirradiatedFilm_0.%sA_10fps_MMStack_Pos0.ome.tif" % (n,current)
            imParameters['firstIm'] = 0 # Use python convention: start from zero!
            imParameters['lastIm'] = -1
            imParameters['filtering'] = 'gauss'
            #imParameters['filtering'] = None
            imParameters['sigma'] = 2
            imParameters['subtract'] = 0 # Subtract a reference image
            threshold = 7
            #palette ='korean'
        elif k == 4:
            rootDir = "/home/gf/Meas/Creep/CoFeB/Wires/Irradiated/run1_2/01_irradiatedwires_0.19A_10fps"
            imParameters['imCrop'] = (0,1392,0,1040)
            imParameters['imCrop'] = (876,1117,250,1040)
            imParameters['imCrop'] = None
            imParameters['pattern'] = "01_irradiatedwires_0.19A_10fps_MMStack_Pos0.ome.tif"
            imParameters['firstIm'] = 30
            imParameters['lastIm'] = 300
            imParameters['filtering'] = 'gauss'
            #imParameters['filtering'] = None
            imParameters['sigma'] = 1.
            threshold = 20
        elif k == 61:
            rootDir = "/home/gf/Meas/Creep/CoFeB/Film/Irradiated/Irr_800He/Irr_800He+_0.12A_1fps"
            #imParameters['imCrop'] = (200,1040,500,1390)
            #imParameters['imCrop'] = (270,970,200,950) # good for 01 0.16A
            #crop_upper_left_pixel, crop_lower_right_pixel = (270,120), (1100,920) # Good for n=03, current=15
            crop_upper_left_pixel, crop_lower_right_pixel = (200,80), (1200,1020) # Good for n=01, current=15
            imParameters['imCrop'] = [crop_upper_left_pixel, crop_lower_right_pixel]
            imParameters['imCrop'] = None
            imParameters['pattern'] = "Irr_800He+_0.12A_1fps_MMStack_Pos0.ome.tif" 
            imParameters['firstIm'] = 0 # Use python convention: start from zero!
            imParameters['lastIm'] = -1
            imParameters['filtering'] = 'gauss'
            #imParameters['filtering'] = None
            imParameters['sigma'] = 2
            imParameters['subtract'] = 0 # Subtract a reference image
            threshold = 50
        elif k == "62":
            #rootDir = "/home/gf/Meas/Creep/CoFeB/Film/Irradiated/Irr_800He/Irr_800He+_0.1A_2fps"
            #rootDir = "/home/gf/Meas/Creep/CoFeB/Film/Irradiated/Irr_800He/Irr_800He+_0.1A_2fps_MMStack_Pos0.ome.tif"
            #rootDir = "/home/gf/Meas/Creep/CoFeB/Film/Irradiated/Irr_800He/Irr_400uC_8e8He+/05_Irr_8e8He+_0.1A_2fps"
            #rootDir = "/home/gf/Meas/Creep/CoFeB/Film/Irradiated/Irr_800He/Irr_400uC_8e8He+/05_Irr_8e8He+_0.1A_2fps"
            rootDir = "/home/gf/Meas/Creep/CoFeB/Film/SuperSlowCreep/Irr_400uC/05_Irr_400uC_0.2A"
            rootDir = "/home/gf/Meas/Creep/CoFeB/Film/SuperSlowCreep/Irr_400uC/01_Irr_400uC_0.1A"
            #rootDir = "/home/gf/Meas/Creep/CoFeB/Film/Irradiated/01_irradiatedFilm_0.16A_10fps"
            rootDir = "/home/gf/Meas/Creep/CoFeB/Film/SuperSlowCreep/Irr_400uC_8e8He+/07_Irr_8e8He+_0.2A_100ms"
            #rootDir = "/home/gf/Meas/Creep/CoFeB/Film/SuperSlowCreep/Irr_800uC_16e8He+/02_Irr_16e8He+_0.116A_3s"#NOT WORKING!!!
            #rootDir = "/home/gf/Meas/Creep/CoFeB/Film/SuperSlowCreep/Irr_800uC/02_Irr_800uC_0.116A" # it should not work as well
            
            #rootDir = "/home/gf/Meas/Creep/CoFeB/Film/SuperSlowCreep/NonIrr/04_NonIrr_0.15A_100ms/"
            #rootDir = "/home/gf/Meas/Creep/CoFeB/Film/SuperSlowCreep/NonIrr/01_NonIrr_0.15A_100ms/"
            #rootDir = "/home/gf/Meas/Creep/CoFeB/Film/SuperSlowCreep/Irr_800uC_16e8He+/01_Irr_16e8He+_0.232A_700ms/"
            #rootDir = "/home/gf/Meas/Creep/CoFeB/Film/SuperSlowCreep/Irr_800uC_16e8He+/04_Irr_16e8He+_0.116A_3s/"
            #rootDir = "/home/gf/Meas/Creep/CoFeB/Film/01_Irr_8e8He+_0.1A_2fps"

            # On Genesis
            # Loading ok, memory allocation problems in doing the filter (600, :, :)
            #rootDir = "/home/gf/Meas/Creep/CoFeB/Film/SuperSlowCreep/Irr_800uC/02_Irr_800uC_0.116A"
            #imParameters['pattern'] = "02_Irr_800uC_0.116A_MMStack_Pos0.ome.tif"
            # The same on wswall.isi.it
            #rootDir = "/home/gf/Meas/Creep/CoFeB/Film/SuperSlowCreep/Irr_800uC_16e8He+/02_Irr_16e8He+_0.116A_3s"
            #imParameters['pattern'] = "02_Irr_16e8He+_0.116A_3s_MMStack_Pos0.ome.tif"
            # Loading ok, beautiful growth
            # On Genesis
            #rootDir = "/home/gf/Meas/Creep/CoFeB/Film/SuperSlowCreep/Irr_800uC/02_Irr_800uC_0.232A"
            #imParameters['pattern'] = "02_Irr_800uC_0.232A_MMStack_Pos0.ome.tif"
            # On wswall
            rootDir = "/home/gf/Meas/Creep/CoFeB/Film/SuperSlowCreep/Irr_800uC_16e8He+/02_Irr_16e8He+_0.232A_700ms"
            imParameters['pattern'] = "02_Irr_16e8He+_0.232A_700ms_MMStack_Pos0.ome.tif"
            rootDir = "/home/gf/Meas/Creep/CoFeB/Film/SuperSlowCreep/Irr_800uC_16e8He+/02_Irr_16e8He+_0.116A_3s"
            imParameters['pattern'] = "02_Irr_16e8He+_0.116A_3s_MMStack_Pos0.ome.tif"
            crop_upper_left_pixel, crop_lower_right_pixel = (100,50), (1200,1020)
            #imParameters['imCrop'] = (200,1040,500,1390)
            #imParameters['imCrop'] = (270,970,200,950) # good for 01 0.16A
            #crop_upper_left_pixel, crop_lower_right_pixel = (270,120), (1100,920) # Good for n=03, current=15
            #crop_upper_left_pixel, crop_lower_right_pixel = (200,80), (1200,1020) # Good for n=01, current=15
            imParameters['imCrop'] = [crop_upper_left_pixel, crop_lower_right_pixel]
            imParameters['imCrop'] = None
            #imParameters['pattern'] = "Irr_800He+_0.1A_2fps_MMStack_Pos0.ome.tif_MMStack_Pos0.ome.tif" 
            #imParameters['pattern'] = "05_Irr_8e8He+_0.1A_2fps_MMStack_Pos0.ome.tif"
            #imParameters['pattern'] = "05_Irr_400uC_0.2A_MMStack_Pos0.ome.tif"
            #imParameters['pattern'] = "Irr_800He+_0.1A_2fps_MMStack_Pos0.ome.tif_MMStack_Pos0.ome.tif"
            # imParameters['pattern'] = "01_irradiatedFilm_0.16A_10fps_MMStack_Pos0.ome.tif"
            #imParameters['pattern'] = "07_Irr_8e8He+_0.2A_100ms_MMStack_Pos0.ome.tif"
            #imParameters['pattern'] = "02_Irr_16e8He+_0.116A_3s_MMStack_Pos0.ome.tif"#NOT WORKING!!!
            #imParameters['pattern'] = "02_Irr_800uC_0.116A_MMStack_Pos0.ome.tif"
            # imParameters['pattern'] = "04_NonIrr_0.15A_100ms_MMStack_Pos0.ome.tif"
            # imParameters['pattern'] = "01_NonIrr_0.15A_100ms_MMStack_Pos0.ome.tif"
            # imParameters['pattern'] = "01_Irr_16e8He+_0.232A_700ms_MMStack_Pos0.ome.tif"
            # imParameters['pattern'] = "04_Irr_16e8He+_0.116A_3s_MMStack_Pos0.ome.tif"
            #imParameters['pattern'] = "Irr_800He+_0.1A_2fps_MMStack_Pos0.ome.tif_MMStack_Pos0.ome.tif"
            imParameters['firstIm'] = 0 # Use python convention: start from zero!
            imParameters['lastIm'] = -1
            #imParameters['filtering'] = 'bilateral'
            imParameters['filtering'] = 'gauss'
            #imParameters['filtering'] = 'gauss'
            imParameters['sigma'] = 2
            imParameters['subtract'] = None # Subtract a reference image
            threshold = None
            palette = 'coolwarm'
        elif k[3:] == "NonIrr_0.095A_3s":
            rootDir = "/home/gf/Meas/Creep/CoFeB/Film/SuperSlowCreep/NonIrr/"
            rootDir = os.path.join(rootDir, k)
            imParameters['pattern'] = "%s_MMStack_Pos0.ome.tif" % k
            imParameters['imCrop'] = None
            imParameters['firstIm'] = 0 # Use python convention: start from zero!
            imParameters['lastIm'] = -1
            imParameters['filtering'] = 'gauss'
            imParameters['sigma'] = 2
            imParameters['subtract'] = None # Subtract a reference image
            threshold = 2
            palette = 'coolwarm'
        elif k[3:] == "Irr_400uC_0.06A":
            rootDir = "/home/gf/Meas/Creep/CoFeB/Film/SuperSlowCreep/Irr_400uC_8e8He+/"
            rootDir = os.path.join(rootDir, k)
            imParameters['pattern'] = "%s_MMStack_Pos0.ome.tif" % k
            imParameters['imCrop'] = None
            imParameters['firstIm'] = 0 # Use python convention: start from zero!
            imParameters['lastIm'] = -1
            imParameters['filtering'] = 'gauss'
            imParameters['sigma'] = 2
            imParameters['subtract'] = None # Subtract a reference image
            threshold = 2
            palette = 'coolwarm'
        elif k[3:] == "Irr_16e8He+_0.116A_3s":
            rootDir = "/home/gf/Meas/Creep/CoFeB/Film/SuperSlowCreep/Irr_800uC_16e8He+/"
            rootDir = os.path.join(rootDir, k)
            imParameters['pattern'] = "%s_MMStack_Pos0.ome.tif" % k
            imParameters['imCrop'] = None
            imParameters['firstIm'] = 0 # Use python convention: start from zero!
            imParameters['lastIm'] = -1
            imParameters['filtering'] = 'gauss'
            imParameters['sigma'] = 2
            imParameters['subtract'] = None # Subtract a reference image
            threshold = 12
            palette = 'coolwarm'      
        else:
            print("Check the path!")
            sys.exit()
        imParameters['resize_factor'] = None
        print(imParameters['pattern'])
        # Kernel setups: do not touch
        imParameters['kernel_half_width_of_ones'] = 10
        #imParameters['kernel_internal_points'] = 0
        #imParameters['kernel_switch_position'] = "end"
        ##############################
        imParameters['subDirs'] = [rootDir, "", "", "", ""]
        #A possible improvement is that stackImages returns also the threshold value extracting the information when uploading the images
        imArray = bk.StackImages(**imParameters)
        plot_contours = True
        imArray.showColorImage(threshold=threshold, palette=palette, plot_contours=plot_contours, 
            erase_small_events_percent=10)
        #imArray.find_contours(lines_color='k', remove_bordering=True, plot_centers_of_mass=False,
        #    plot_rays=False, reference=None,invert_y_axis=True)
        #'center_of_mass')
        #save_data = raw_input("Save the data?")
        save_data = "Y"
        if save_data.upper() == "Y":
            if plot_contours:
                objects_tosave = [imArray._switchTimes2D, imArray._switchSteps2D, imArray.centers_of_mass, imArray.contours]
                objects_strings = ["switchTimes2D", "switchSteps2D", "centers_of_mass", "contours"]
            else:
                objects_tosave = [imArray._switchTimes2D, imArray._switchSteps2D]
                objects_strings = ["switchTimes2D", "switchSteps2D"]
            for obj_tosave, obj_string in zip(objects_tosave, objects_strings):
                filename = "%s.pkl" % obj_string
                print("Saving ... %s" % filename)
                filename = os.path.join(rootDir, filename)
                with open(filename, 'wb') as f:
                    pickle.dump(obj_tosave, f)

    else:
        print("Sorry, nothing to do")
