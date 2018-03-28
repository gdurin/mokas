import sys
import numpy as np
import matplotlib.pyplot as plt
import visualBarkh as bk

if __name__ == "__main__":
    plt.close("all")
    imParameters = {}
    print(sys.argv)
    try:
        choice = sys.argv[1]
    except:
        choice = 'experimental'
    if choice == 'experimental':
        
        #mainDir, pattern, firstImage, lastIm = "/media/DATA/meas/MO/CoFe/50nm/20x/run5/", "Data1-*.tif", 280, 1029
        #mainDir, pattern, sequences = "/media/DATA/meas/Barkh/Films/CoFe/50nm/10x/run5/", "Data1-*.tif", [(1,1226)]
    
    
        rootDir = "/run/media/gf/DATA/meas/Barkh/Films/NiO_Fe"
        rootDir = "/media/gf/DATA/meas/Barkh/Films/NiO_Fe"
        subMat = "NiO80"
        pattern = "Data1-*.tif"
        
        #subMagn = "M20x"
        #mainDir, pattern, firstImage, lastIm = "/media/DATA/meas/Barkh/Films/NiO-Fe/NiO80/NiO80 run2 20x/", "Data4-*.tif", 70, 125
        #mainDir, pattern, firstImage, lastIm = "/media/DATA/meas/Barkh/Films/NiO-Fe/NiO80/NiO80 run4 20x 20iter/", "Data4-*.tif", 80, 140
        #mainDir, pattern, firstImage, lastIm = "/media/DATA/meas/Barkh/Films/NiO-Fe/NiO80/NiO80 run5 20x 20iter EA/", "Data1-*.tif", 60, 90
        #mainDir, pattern, firstImage, lastIm = "/media/DATA/meas/Barkh/Films/NiO-Fe/NiO80/NiO80 run10 20x 6iter 60sec/", "Data1-*.tif", 70, 300
        #mainDir, pattern, firstImage, lastIm = "/media/DATA/meas/Barkh/Films/NiO-Fe/NiO80/NiO80 run10 20x 6iter 60sec/", "Data1-*.tif", 670, 890
        #mainDir, pattern, firstImage, lastIm = "/media/DATA/meas/Barkh/Films/NiO-Fe/NiO80/NiO80 run9 20x 6iter 40sec/", "Data1-*.tif", 40, 220
        #mainDir, pattern, firstImage, lastIm = "/media/DATA/meas/Barkh/Films/NiO-Fe/NiO80/NiO80 run9 20x 6iter 40sec/", "Data1-*.tif", 400, 620
        #mainDir, pattern, firstImage, lastIm = "/media/DATA/meas/Barkh/Films/NiO-Fe/NiO80/NiO80 run11 20x/", "Data1-*.tif", 1170, 1512
        #mainDir, pattern, firstImage, lastIm = "/media/DATA/meas/Barkh/Films/NiO-Fe/NiO80/NiO80 run12 20x/", "Data1-*.tif", 50, 260
        subMagn = "M10x"
        #mainDir, pattern, firstImage, lastIm = "/media/DATA/meas/Barkh/Films/NiO-Fe/NiO80/NiO80 run13 10x/", "Data1-*.tif", 13, 641
        #mainDir, pattern, firstImage, lastIm = "/media/DATA/meas/Barkh/Films/NiO-Fe/NiO80/NiO80 run14 10x bin1/", "Data1-*.tif", 8, 90
        #mainDir, pattern, sequences = "/media/DATA/meas/Barkh/Films/NiO-Fe/NiO80/NiO80 run15 10x pinLin/", "Data1-*.tif", [(10, 436)]
        #mainDir, pattern, sequences = "/media/DATA/meas/Barkh/Films/NiO-Fe/NiO80/NiO80 run16 10x pinLin/", "Data1-*.tif", [(25, 240)]
        #mainDir, pattern = "/media/DATA/meas/Barkh/Films/NiO-Fe/NiO80/NiO80 run18 10x pinLin 10iter/", "Data1-*.tif"
        #sequences = [(32, 304), (566, 955), (1103, 1489), (1617, 2082), (2126, 2598), (2640, 3146), (3162, 3673)]
        #mainDir, pattern = "/media/DATA/meas/Barkh/Films/NiO-Fe/NiO80/NiO80 run19 10x pinLin 10iter/", "Data1-*.tif"
        #sequences = [(12,100), (160,250), (300,430)]
        #mainDir, pattern = "/media/DATA/meas/Barkh/Films/NiO-Fe/NiO80/10x/NiO80 run20 10x/", "Data1-*.tif"
        #sequences = [(8,205)]
        subRun = "run21"
        subSeq = [(4,100),(158,275),(306,440),(452,570)]    
        #subMagn = "M50x"
        
        #mainDir, pattern = "/media/DATA/meas/Barkh/Films/NiO-Fe/NiO80/50x/NiO80 run23 50x/", "Data1-*.tif"
        #sequences = [(5,120)]    
        #mainDir, pattern, firstImage, lastIm  = "/media/DATA/meas/MO/Picostar/orig", "B*.TIF", 0, 99
        
    
        imageDirections=["Top_to_bottom","Left_to_right", 
                         "Bottom_to_top","Right_to_left"]
    
    
        for seq in subSeq[1:2]:
            firstImage, lastIm = seq
            subSeq = "seq_%03i_%03i" % seq
            subDirs = [rootDir, subMat, subMagn, subRun, subSeq]
            imArray = bk.StackImages(subDirs, pattern, resize_factor=None,\
                                 filtering='gauss', sigma=1.5,\
                                 firstIm=firstImage, lastIm=lastIm)
            imArray.width='small'
            imArray.useKernel = 'step'
            imArray.imageDir = imageDirections[1]
            p2p = 5 # Pixel to pixel (linear) distance for cluster detection
            NN = 2*p2p + 1
            imArray.structure = np.ones((NN,NN))
            imArray.showColorImage(palette='ral')
            #imArray.saveHdf5()
    
            #imArray0.showMeanGreyLevels()
            #imArray.useKernel = 'zero'
            #imArray1 = bk.StackImages(mainDir, pattern, resize_factor=False,\
                                         #filtering='gauss', sigma=3.5,\
                                         #firstImage=firstImage, lastIm=lastIm)
            #imArray1.width='small'
            #imArray1.useKernel = 'step'
            #imArray1.showColorImage(palette='ral')
    
    
            #d0 = imArray0.getDistributions()
            #d1 = imArray1.getDistributions(hold=True)
            #plt.figure()
            #for d in [d0,d1]:
                #D_x, D_y = d
                #plt.loglog(D_x,D_y,'o', label='cluster')
    
            #plt.legend()
            #plt.show()    
    elif choice=='simul':
        rootDir = "/run/media/gf/DATA/meas/Simulation/Andrea/test3/Images"
        subDirs = [rootDir, "", "", "", ""]
        pattern = "Image*.jpg"
        firstImage, lastIm = (1,186)
        imArray = bk.StackImages(subDirs, pattern, resize_factor=None,\
                              filtering='binary', sigma=1.5,\
                              firstIm=firstImage, lastIm=lastIm)
        imArray.width='all'
        imArray.useKernel = 'step'
        #imArray.useKernel = 'zero'
        imArray.imageDir = "Bottom_to_top"
        imArray.boundary = 'periodic'
        p2p = 3 # Pixel to pixel (linear) distance for cluster detection
        NN = 2*p2p + 1
        imArray.structure = np.ones((NN,NN))
        imArray.showColorImage(10,palette='random')
    elif choice=='cecilia':
        rootDir = "/home/gf/Meas/sequenzePythonMOIF/199B"
        subDirs = [rootDir, "", "", "", ""]
        pattern = "*_199B.tif"
        firstImage, lastIm = (0,199)
        imArray = bk.StackImages(subDirs, pattern, resize_factor=None,\
                              filtering=None, sigma=1.5,\
                              firstIm=firstImage, lastIm=lastIm)
        imArray.width='all'
        imArray.useKernel = 'step'
        imArray.kernelSign = -1
        #imArray.useKernel = 'zero'
        imArray.imageDir = "Left_to_right"
        imArray.boundary = None
        p2p = 3 # Pixel to pixel (linear) distance for cluster detection
        NN = 2*p2p + 1
        imArray.structure = np.ones((NN,NN))
        imArray.showColorImage(10,palette='random')
    elif choice=="Creep":
        k = int(sys.argv[2])
        print k
        #palette = 'pastel'
        #rootDir = "/home/gf/Meas/Creep/PtCoAu50Pt50/PtCoAuPt_2c-0d-100pOe-0.975V-1.2s"
        #rootDir = "/home/gf/Meas/Creep/PtCoAu50Pt50/Rotation/90 degree/PtCoAuPt_3_2c-90d-350pOe-0.780V-3.5s_6"
        if k == 0:
            rootDir = "/home/gf/Meas/Creep/PtCoAu50Pt50/Rotation/0 degree/PtCoAuPt_3_2c-00d2-000nOe-0.780V-5.0s_1"
            imParameters['imCrop'] = (0,510,0,672)
            imParameters['pattern'] = "filename*.png"
            imParameters['firstIm'] = 1
            imParameters['lastIm'] = 15
            imParameters['filtering'] = 'gauss'
            #imParameters['filtering'] = None
            imParameters['sigma'] = 1.1
            threshold = 50
        elif k == 1:
            rootDir = "/home/gf/Meas/Creep/PtCoAu50Pt50/Rotation/0 degree/PtCoAuPt_3_2c-00d2-350pOe-0.780V-4.3s_11"
            print(rootDir)
            imParameters['imCrop'] = [(0,0), (510, 672)]
            imParameters['pattern'] = "filename*.png"
            imParameters['firstIm'] = 1
            imParameters['lastIm'] = 21
            imParameters['filtering'] = 'gauss'
            #imParameters['filtering'] = None
            imParameters['sigma'] = 2.0
            threshold = 40
        #rootDir = "/home/gf/Meas/Creep/PtCoAu/PtCoAu_2c-0d-500pOe-1.275V-1.0s"
        elif k == 20:
            #rootDir = "/home/gf/Meas/Creep/PtCoAu50Pt50/Rotation/0 degree/PtCoAuPt_3_2c-00d2-650nOe-0.780V-1.3s_29/"
            rootDir = "/data/Meas/Creep/WCoFeB/W-CoFeB-MgO_NonIrr/movies/15_12A"
            imParameters['imCrop'] = [(0,0),(1040,1392)]
            imParameters['imCrop'] = [(650,420),(980,650)]
            imParameters['pattern'] = "15_12A_MMStack_Pos0.ome.tif"
            imParameters['firstIm'] = 1
            imParameters['lastIm'] = 26
            imParameters['filtering'] = 'gauss'
            #imParameters['filtering'] = None
            imParameters['sigma'] = 2.
            threshold = 20
        elif k == 21:
            rootDir = "/home/gf/Meas/Creep/PtCoAu50Pt50/Rotation/0 degree/PtCoAuPt_3_2c-00d2-650nOe-0.780V-1.3s_29/"
            imParameters['imCrop'] = [(0,0),(554,672)]
            imParameters['imCrop'] = [(270,90),(550,410)]
            imParameters['pattern'] = "filename*.png"
            imParameters['firstIm'] = 1
            imParameters['lastIm'] = 12
            imParameters['filtering'] = 'gauss'
            #imParameters['filtering'] = None
            imParameters['sigma'] = 1.5
            threshold = 20
        elif k == 22:
            rootDir = "/home/gf/Meas/Creep/PtCoPt/M2/PtCoPt_2-2c-0d-000nOe-0.657V-40.0s_1"
            imParameters['imCrop'] = [(0,0), (510, 672)]
            imParameters['pattern'] = "filename*.png"
            imParameters['firstIm'] = 1
            imParameters['lastIm'] = 144
            imParameters['filtering'] = 'gauss'
            #imParameters['filtering'] = None
            imParameters['sigma'] = 1.5
            threshold = 25
        elif k == 23:
            rootDir = "/home/gf/Meas/Creep/PtCoPt/M2/PtCoPt_2-2c-0d-100nOe-0.657V-30.0s_18"
            imParameters['imCrop'] = [(0,0), (510, 672)]
            imParameters['pattern'] = "filename*.png"
            imParameters['firstIm'] = 1
            imParameters['lastIm'] = 89
            imParameters['filtering'] = 'gauss'
            #imParameters['filtering'] = None
            imParameters['sigma'] = 1.5
            threshold = 25
        elif k == 24:
            rootDir = "/home/gf/Meas/Creep/PtCoPt/M2/PtCoPt_2-2c-0d-100pOe-0.657V-30.0s_19"
            #imParameters['imCrop'] = (0,510,0,672)
            imParameters['imCrop'] = [(0,0), (510, 672)]
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
            rootDir = "/home/gf/Meas/Creep/CoFeB/Film/Irradiated/%s_irradiatedFilm_0.%sA_10fps" % (n, current)
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
            n = "05"
            current = "15"
            rootDir = "/home/gf/Meas/Creep/CoFeB/Film/Non-irradiated/Moon/run6/%s_nonirradiatedFilm_0.%sA_10fps" % (n,current)
            #imParameters['imCrop'] = (200,1040,500,1390)
            #imParameters['imCrop'] = (270,970,200,950) # good for 01 0.16A
            #crop_upper_left_pixel, crop_lower_right_pixel = (270,120), (1100,920) # Good for n=03, current=15
            crop_upper_left_pixel, crop_lower_right_pixel = (200,80), (1200,1020) # Good for n=01, current=15
            imParameters['imCrop'] = [crop_upper_left_pixel, crop_lower_right_pixel]
            imParameters['imCrop'] = None
            imParameters['pattern'] = "%s_nonirradiatedFilm_0.%sA_10fps_MMStack_Pos0.ome.tif" % (n,current)
            imParameters['firstIm'] = 0 # Use python convention: start from zero!
            imParameters['lastIm'] = 50
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
            imParameters['pattern'] = "01_irradiatedwires_0.19A_10fps_MMStack_Pos0.ome.tif"
            imParameters['firstIm'] = 30
            imParameters['lastIm'] = 300
            imParameters['filtering'] = 'gauss'
            #imParameters['filtering'] = None
            imParameters['sigma'] = 1.
            threshold = 20
        elif k == 6:
            rootDir = "/home/gf/Meas/Creep/WCoFeB/super_slow_creep_90mV_dc_2hours_242images"
            #imParameters['imCrop'] = (200,1040,500,1390)
            #imParameters['imCrop'] = (270,970,200,950) # good for 01 0.16A
            #crop_upper_left_pixel, crop_lower_right_pixel = (270,120), (1100,920) # Good for n=03, current=15
            crop_upper_left_pixel, crop_lower_right_pixel = (200,80), (1200,1020) # Good for n=01, current=15
            imParameters['imCrop'] = [crop_upper_left_pixel, crop_lower_right_pixel]
            imParameters['imCrop'] = None
            imParameters['pattern'] = "super_slow_creep_90mV_dc_2hours_242images_MMStack_Pos0.ome.tif" 
            imParameters['firstIm'] = 0 # Use python convention: start from zero!
            imParameters['lastIm'] = 241
            imParameters['filtering'] = 'gauss'
            #imParameters['filtering'] = None
            imParameters['sigma'] = 3
            imParameters['subtract'] = None # Subtract a reference image
            threshold = 120
            #palette ='korean'  
        elif k == 61:
            rootDir = "/home/gf/Meas/Creep/CoFeB/Film/Irradiated/Irr_800He_0.12A_1fps"
            #imParameters['imCrop'] = (200,1040,500,1390)
            #imParameters['imCrop'] = (270,970,200,950) # good for 01 0.16A
            #crop_upper_left_pixel, crop_lower_right_pixel = (270,120), (1100,920) # Good for n=03, current=15
            crop_upper_left_pixel, crop_lower_right_pixel = (200,80), (1200,1020) # Good for n=01, current=15
            imParameters['imCrop'] = [crop_upper_left_pixel, crop_lower_right_pixel]
            imParameters['imCrop'] = None
            imParameters['pattern'] = "Irr_800He_0.12A_1fps_MMStack_Pos0.ome.tif" 
            imParameters['firstIm'] = 0 # Use python convention: start from zero!
            imParameters['lastIm'] = 300
            imParameters['filtering'] = 'gauss'
            #imParameters['filtering'] = None
            imParameters['sigma'] = 3
            imParameters['subtract'] = 0 # Subtract a reference image
            threshold = 20
            #palette ='korean'  
        elif k == 7:
            rootDir = "/home/gf/Meas/Creep/WCoFeB/super_slow_creep_90mV_dc_3hours_362images_MMStack_Pos0.ome.tif"
            #imParameters['imCrop'] = (200,1040,500,1390)
            #imParameters['imCrop'] = (270,970,200,950) # good for 01 0.16A
            #crop_upper_left_pixel, crop_lower_right_pixel = (270,120), (1100,920) # Good for n=03, current=15
            crop_upper_left_pixel, crop_lower_right_pixel = (200,80), (1200,1020) # Good for n=01, current=15
            imParameters['imCrop'] = [crop_upper_left_pixel, crop_lower_right_pixel]
            imParameters['imCrop'] = None
            imParameters['pattern'] = "super_slow_creep_90mV_dc_3hours_362images_MMStack_Pos0.ome.tif" 
            imParameters['firstIm'] = 0 # Use python convention: start from zero!
            imParameters['lastIm'] = 361
            imParameters['filtering'] = 'gauss'
            #imParameters['filtering'] = None
            imParameters['sigma'] = 1
            imParameters['subtract'] = None # Subtract a reference image
            threshold = 7
            #palette ='korean'  
        elif k == 8:
            # Samridth data
            rootDir = "/home/gf/Meas/Creep/WCoFeB/Const_InPl_Vary_OOP"
            crop_upper_left_pixel, crop_lower_right_pixel = (100,100), (600,450)
            imParameters['imCrop'] = [crop_upper_left_pixel, crop_lower_right_pixel]
            #imParameters['imCrop'] = None
            imParameters['pattern'] = "exp_70mV_1s_17.avi" 
            imParameters['firstIm'] = 15 # Use python convention: start from zero!
            imParameters['lastIm'] = 36
            imParameters['filtering'] = 'gauss'
            imParameters['filtering'] = None
            imParameters['sigma'] = 1
            imParameters['subtract'] = None # Subtract a reference image
            threshold = 20
        elif k == 9:
            # Samridth data of labyrinth
            rootDir = "/home/gf/isiCloud/Data share/Samridh/labyrinth"
            imParameters['imCrop'] = None
            imParameters['pattern'] = "oop avg_1.avi" 
            imParameters['firstIm'] = 0 # Use python convention: start from zero!
            imParameters['lastIm'] = -1
            imParameters['filtering'] = 'gauss'
            #imParameters['filtering'] = None
            imParameters['sigma'] = 1
            imParameters['subtract'] = None # Subtract a reference image
            imParameters['use_max_criterium'] = False
            threshold = 20
        else:
            print("Check the path!")
            sys.exit()
        imParameters['resize_factor'] = None
        print(imParameters['pattern'])
        #imParameters['imCrop'] = (70,460,210,580)
        #imParameters['imCrop'] = (60,460,162,512)
        #imParameters['imCrop'] = (0,510,0,672)
        # Kernel setups
        imParameters['kernel_half_width_of_ones'] = 15
        imParameters['kernel_half_width_of_ones'] = 5
        #imParameters['kernel_internal_points'] = 0
        #imParameters['kernel_switch_position'] = "end"
        ##############################
        imParameters['subDirs'] = [rootDir, "", "", "", ""]
        imArray = bk.StackImages(**imParameters)
        #palette = 'coolwarm'
        palette = 'random'
        imArray.showColorImage(threshold, palette=palette, plot_contours=False)
        #imArray.find_contours(lines_color='k', remove_bordering=True, plot_centers_of_mass=False,
        #    plot_rays=False, reference=None,invert_y_axis=True)
        #'center_of_mass')

    else:
        print("Sorry, nothing to do")
