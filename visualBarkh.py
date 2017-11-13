import os, sys, glob, time, re
import scipy
import scipy.ndimage as nd
import scipy.signal as signal
import scipy.stats.stats
import numpy as np
import numpy.ma as ma
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from colorsys import hsv_to_rgb
from gpuSwitchtime import get_gpuSwitchTime #gpuSwitchtimeBackup
from PIL import Image
import tifffile
import getLogDistributions as gLD
import getAxyLabels as gal
import rof
import mahotas
#import h5py
import tables
import polar
import collect_images
from mokas_colors import get_cmap, getKoreanColors, getPalette
import mokas_gpu as mkGpu
from mokas_domains import Domains
import cPickle as pickle
from skimage import measure


# Check if pycuda is available
try:
    import pycuda.driver as driver
    isPyCuda = True
    #free_mem_gpu, total_mem_gpu = driver.mem_get_info()
except:
    isPyCuda = False
    print("Please install PyCuda")
    sys.exit()

# Load scikits modules if available
try:
    from skimage.filters import tv_denoise
    isTv_denoise = True
    filters['tv'] = tv_denoise
except:
    isTv_denoise = False

try:
    import skimage.io as im_io
    from skimage import filters as skfilters
    from skimage import measure
    isScikits = True
except:
    isScikits = False
    print("*********** There is no Scikits-image installed")
    sys.exit()

plugins = im_io.available_plugins
keys = plugins.keys()
mySeq = ['gtk', 'pil', 'matplotlib', 'qt']
for plugin in mySeq:
    if plugin in keys:
        use = plugin
try:
    im_io.use_plugin('pil', 'imread')
except:
    print("No plugin available between %s" % str(mySeq))


# Adjust the interpolation scheme to show the images
mpl.rcParams['image.interpolation'] = 'nearest'



def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

# ###################################################################


class StackImages:
    """
    Load and analyze a sequence of images
    as a multi-dimensional scipy 3D array.
    The k-th element of the array (i.e. myArray[k])
    is the k-th image of the sequence.

    Parameters:
    ----------------
    mainDir : string
        Directory of the image files

    pattern : string
        Pattern of the input image files,
        as for instance "Data1-*.tif"

    firstImage, lastIm : int, opt
       first and last image (included) to be loaded
       These numbers refer to the numbers of the filenames
    
    subDirs : list
       list of subdirectories to save the data into (hdf5 style)

    resize_factor : int, opt
       Reducing factor for the size of raw images

    filtering : string, opt
        Apply a filter to the raw images between the following:
        'gauss': nd.gaussian_filter (Default)
        'fouriergauss': nd.fourier_gaussian,
        'median': nd.median_filter,
        'tv': tv_denoise,
        'wiener': signal.wiener

    sigma : scalar or sequence of scalars, required with filtering
       for 'gauss': standard deviation for Gaussian kernel.
       for 'fouriergauss': The sigma of the Gaussian kernel.
       for 'median': the size of the filter
       for 'tv': denoising weight
       for 'wiener': A scalar or an N-length list as size of the Wiener filter
       window in each dimension.

    imCrop : 4-element tuple
       Crop the image

    hdf5_signature : dict
        A dictionary containing the essential data to identify a measurement
        if None, the code does not save data on a hdf5
    """

    def __init__(self, subDirs, pattern, resize_factor=None,
                 firstIm=None, lastIm=None, 
                 filtering=None, sigma=None, 
                 kernel_half_width_of_ones = 10, 
                 #kernel_internal_points = 0,
                 #kernel_switch_position = 'center',
                 boundary=None, imCrop=False, 
                 initial_domain_region=None, subtract=None,
                 exclude_switches_from_central_domain=True,
                 exclude_switches_out_of_final_domain=True,
                 rotation=None, fillValue=-1, 
                 hdf5_use=False, 
                 hdf5_signature=None):
        """
        Initialized the class
        """
        # Initialize variables
        self._subDirs = subDirs
        rootDir, subMat, subMagn, subRun, seq = subDirs
        self._mainDir = os.path.join(rootDir, subMat, subMagn, subRun)
        self._colorImage = None
        self._koreanPalette = None
        self._isColorImage = False
        self._isSwitchAndStepsDone = False
        self._isDistributions = False
        self._switchTimes = None
        self._threshold = 0
        self._figTimeSeq = None
        self.figDiffs = None
        self._figHistogram = None
        self.is_histogram = False
        self._figColorImage = None
        self._figColorImage2 = None
        self.isConnectionRawImage = False
        self.figRawAndCalc = None
        self.imagesRange = (firstIm, lastIm)
        self.imageDir = None
        self.initial_domain_region = initial_domain_region
        self.is_find_contours = False
        self.isTwoImages = False
        self.is_minmax_switches = False
        self.pattern = pattern
        self.fillValue = fillValue
        #self.NNstructure = np.asanyarray([[0,1,0],[1,1,1],[0,1,0]])
        NN = 3
        self.NNstructure = np.ones((NN,NN))
        if boundary == 'periodic':
            self.boundary = 'periodic'
        else:
            self.boundary = None
        if not lastIm:
            lastIm = -1
        #self.exclude_switches_from_central_domain = exclude_switches_from_central_domain
        self.exclude_switches_out_of_final_domain = exclude_switches_out_of_final_domain

        # Check paths
        if not os.path.isdir(self._mainDir):
            print("Please check you dir %s" % self._mainDir)
            print("Path not found")
            sys.exit()

        #############################################################################
        out = collect_images.images2array(self._mainDir, pattern, firstIm, lastIm, imCrop, 
                                    rotation, filtering, sigma, subtract=subtract, 
                                    hdf5_use=hdf5_use, hdf5_signature=hdf5_signature)
        if hdf5_use:
            self.Array, self.imageNumbers, self.hdf5 = out
        else:
            self.Array, self.imageNumbers = out
        ##############################################################################
        self.shape = self.Array.shape
        self.n_images, self.dimX, self.dimY = self.shape
        print("%i image(s) loaded, of %i x %i pixels" % (self.n_images, self.dimX, self.dimY))

        self.multiplier = self._get_multiplier('bubble')
        self.convolSize = kernel_half_width_of_ones
        # Make a kernel as a step-function
        self.kernel_half_width_of_ones = kernel_half_width_of_ones
        

    def _get_multiplier(self, method='average_gray'):
        """
        multiplier is +1 if -1 -> +1
        multiplier is -1 if +1 -> -1
        The method below is only valid for bubbles
        TODO: make it as a specific method of the subclasses (bubbles, wires, labyrinths)
        """
        if method == 'average_gray':
            grey_first_image = np.mean([scipy.mean(self.Array[k,:,:].flatten()) for k in range(4)])
            grey_last_image = np.mean([scipy.mean(self.Array[-k,:,:].flatten()) for k in range(1,5)])
            print("grey scale: %i, %i" % (grey_first_image, grey_last_image))
            if grey_first_image > grey_last_image:
                return 1.
            else:
                return -1.
        elif method == 'bubble':
            #im = self.Array[-1]
            return -1

    def __get__(self):
        """
        redifine the get
        """
        return self.Array

    def __getitem__(self,n):
        """Get the n-th image"""
        index = self._getImageIndex(n)
        if index is not None:
            return self.Array[index]

    def _getImageIndex(self,n):
        """
        check if image number n has been loaded
        and return the index of it in the Array
        """
        ns = list(self.imageNumbers)
        try:
            if n < 0:
                n = self.imageNumbers[n]
            return ns.index(n)
        except:
            print("Image number %i is out of the range (%i,%i)" % (n, ns[0], ns[-1]))
            return None

    def showMeanGreyLevels(self):
        indexImages = np.array(self.imageNumbers) - self.imageNumbers[0]
        plt.figure()
        for n in indexImages:
            meanGrey = np.mean(self.Array[n,:,:].flatten())
            plt.plot(n, meanGrey, 'bo')
        return


    def showRawImage(self, imageNumber, plugin='mpl'):
        """
        showImage(imageNumber)

        Show the n-th image where n = image_number

        Parameters:
        ---------------
        imageNumber : int
            Number of the image to be shown.

        plugin : str, optional
        Use a plugin to show an image (default: matplotlib)
        """
        n = self._getImageIndex(imageNumber)
        if n is not None:
            im = self[imageNumber]
            if plugin == 'mpl':
                plt.imshow(im, plt.cm.gray)
                plt.show()
            else:
                im_io.imshow(self[imageNumber])

    def _getWidth(self):
        try:
            width = self.width
        except:
            self.width = 'all'
            print("Warning: the levels are calculated over all the points of the sequence")
        return self.width


    def _pixel2rowcol(self, pixel, closest_integer=False):
        """
        Transforn from pixel values to (row, col)
        calculating the integer values if needed
        """
        if closest_integer:
            pixel = int(round(pixel[0])), int(round(pixel[1]))
        return pixel[1], pixel[0]


    def pixelTimeSequence(self,pixel=(0,0)):
        """
        pixelTimeSequence(pixel)

        Returns the temporal sequence of the gray level of a pixel

        Parameters:
        ---------------
        pixel : tuple
           The (x,y) pixel of the image, as (row, column)
        """
        x,y = self._pixel2rowcol(pixel)
        return self.Array[:,x,y]

    def showPixelTimeSequence(self, pixel=(0,0), newPlot=False, show_kernel=True):
        """
        pixelTimeSequenceShow(pixel)

        Plot the temporal sequence of the gray levels of a pixel;

        Parameters:
        ---------------
        pixel : tuple
            The (x,y) pixel of the image
        newPlot : bool
            Option to open a new frame or use the last one
        show_kernel : bool
            Show the kernel (step function) in the plot
        """
        if not self._figTimeSeq or newPlot==True:
            self._figTimeSeq = plt.figure()
        else:
            plt.figure(self._figTimeSeq.number)
        ax = plt.gca()
        # Plot the temporal sequence first
        pxt = self.pixelTimeSequence(pixel)
        label_pixel = "(%i, %i)" % (pixel[0], pixel[1])
        ax.plot(self.imageNumbers,pxt,'-o', label=label_pixel)
        # Add the calculus of GPU
        row, col = self._pixel2rowcol(pixel)
        switch = self._switchTimes2D[row, col]
        step = self._switchSteps2D[row, col]
        # Plot the step-like function
        l0 = self.kernel_half_width_of_ones
        pxt_average = np.mean(pxt[switch - l0/2:switch + l0 + 1]) # to check
        print("switch at %i, gray level change = %i" % (switch, step))
        kernel0 = -self.multiplier * np.ones(self.kernel_half_width_of_ones)
        kernel0 = np.concatenate((kernel0, -kernel0))
        kernel = kernel0 * step / 2 + pxt_average
        # This is valid for the step kernel ONLY
        x =  np.arange(switch - l0, switch + l0)
        ax.plot(x, kernel, '-o')
        plt.legend()
        plt.show()

    def getSwitchTime(self, pixel=(0,0), useKernel='step', method='convolve1d'):
        """
        getSwitchTime(pixel, useKernel='step', method="convolve1d")

        Return the position of a step in a sequence
        and the left and the right values of the gray level (as a tuple)

        Parameters:
        ---------------
        pixel : tuple
            The (x,y) pixel of the image, as (row, column).
        useKernel : string
            step = [1]*5 +[-1]*5
            zero = [1]*5 +[0] + [-1]*5
            both = step & zero, the one with the highest convolution is chosen
        method : string
            For the moment, only the 1D convolution calculation
            with scipy.ndimage.convolve1d is available
        """
        pxTimeSeq = self.pixelTimeSequence(pixel)
        if method == "convolve1d":
            if useKernel == 'step' or useKernel == 'both':
                convolution_of_stepKernel = nd.convolve1d(pxTimeSeq,self.kernel)
                minStepKernel = convolution_of_stepKernel.min()
                switchStepKernel = convolution_of_stepKernel.argmin() +1
                switch = switchStepKernel
                kernel_to_use = 'step'
            if useKernel == 'zero' or useKernel == 'both':
                convolution_of_zeroKernel = nd.convolve1d(pxTimeSeq,self.kernel0)
                minZeroKernel = convolution_of_zeroKernel.min()
                switchZeroKernel = convolution_of_zeroKernel.argmin() + 1
                switch = switchZeroKernel
                kernel_to_use = 'zero'
            if useKernel == 'both':
                if minStepKernel <= minZeroKernel:
                    switch = switchStepKernel
                    kernel_to_use = 'step'
                else:
                    switch = switchZeroKernel
                    kernel_to_use = 'zero'
        else:
            raise RuntimeError("Method not yet implemented")
        levels = self._getLevels(pxTimeSeq, switch, kernel_to_use)
        # Now redefine the switch using the correct image number
        switch = self.imageNumbers[switch]
        return switch, levels

    def _imDiff(self, imNumbers, invert=False):
        """Properly rescaled difference between images

        Parameters:
        ---------------
        imNumbers : tuple
        the numbers the images to subtract
        invert : bool
        Invert black and white grey levels
        TODO: do it properly!!!!
        """
        i, j = imNumbers
        try:
            im = self[i]-self[j]
        except:
            return
        if invert:
            im = 255 - im
        imMin = scipy.amin(im)
        imMax = scipy.amax(im)
        im = scipy.absolute(im-imMin)/float(imMax-imMin)*255
        return scipy.array(im,dtype='int16')

    def showTwoImageDifference(self, imNumbers, invert=False):
        """Show the output of self._imDiff

        Parameters:
        ---------------
        imNumbers : tuple
        the numbers of the two images to subtract

        invert : bool, opt
        Invert the gray level black <-> white
        """
        if type(invert).__name__ == 'int':
            imNumbers = imNumbers, invert
            print("Warning: you should use a tuple as image Numbers")

        try:
            plt.imshow(self._imDiff(imNumbers, invert), plt.cm.gray)
            plt.show()
        except:
            return

    def imDiffSave(self,imNumbers='all', invert=False, mainDir=None):
        """
        Save the difference(s) between a series of images

        Parameters:
        ---------------
        imNumbers : tuple or string
        the numbers of the images to subtract
        * when 'all' the whole sequence of differences is saved
        * when a tuple of two number (i.e., (i, j),
        all the differences of the images between i and j (included)
        are saved
        """
        if mainDir == None:
            mainDir = self._mainDir
        dirSeq = os.path.join(mainDir,"Diff")
        if not os.path.isdir(dirSeq):
            os.mkdir(dirSeq)
        if imNumbers == 'all':
            imRange = self.imageNumbers[:-1]
        else:
            im0, imLast = imNumbers
            imRange = range(im0, imLast)
            if im0 >= imLast:
                print("Error: sequence not valid")
                return
        for i in imRange:
            im = self._imDiff((i+1,i))
            imPIL = scipy.misc.toimage(im)
            fileName = "imDiff_%i_%i.tif" % (i+1,i)
            print(fileName)
            imageFileName = os.path.join(dirSeq, fileName)
            imPIL.save(imageFileName)

    def getSwitchTimesAndSteps(self, isCuda=True, kernel=None, device=0):
        """
        Calculate the switch times and the gray level changes
        for each pixel in the image sequence.
        It calculates:
        self._switchTimes
        self._switchSteps

        Calculus of switch times.
        Assume a kernel from +1 to -1, for reference
        if kernel is a step function, switch is the last point at +1
        so we need to add 1 to switch
        If kernel has points between +1 and -1,
        i.e. self.kernel_internal_points is not 0
        we need to add int(self.kernel_internal_points/2) + 1
        """
        #if kernel is None:
        #    kernel = self.kernel
        startTime = time.time()
        # ####################
        if isPyCuda and isCuda:
            #kernel32 = np.asarray(kernel, dtype=np.int32)
            stack32 = np.asarray(self.Array, dtype=np.int32)
            need_mem = 2 * stack32.nbytes +  2 * stack32[0].nbytes
            print("Total memory to be used: %.2f GB" % (need_mem/1e9))
            current_dev, ctx, (free_mem_gpu, total_mem_gpu) = mkGpu.gpu_init(device)
            if need_mem < free_mem_gpu:
                switchTimes, switchSteps = get_gpuSwitchTime(stack32, self.convolSize,self.multiplier, current_dev, ctx)
            else:
                nsplit = int(float(need_mem)/free_mem_gpu) + 1
                print("Splitting images in %d parts..." % nsplit)
                stack32s = np.array_split(stack32, nsplit, 1)
                print("Done")
                switchTimes = np.array([])
                switchSteps = np.array([])
                for k, stack32 in enumerate(stack32s):
                    print("Calculation split %i" % k)
                    #a = stack32.astype(np.int32)
                    switch, step = get_gpuSwitchTime(stack32, self.convolSize,self.multiplier, current_dev, ctx)
                    if not k:
                        switchTimes = switch
                        switchSteps = step
                    else:
                        switchTimes = np.vstack((switchTimes, switch))
                        switchSteps = np.vstack((switchSteps, step))
            self._switchSteps = switchSteps.flatten()
            # Add the value of the first image
            self._switchTimes = self.imageNumbers[0] + switchTimes.flatten()
            # Close device properly
            success = mkGpu.gpu_deinit(current_dev, ctx)
            if not success:
                print("There is a problem with the device %i" % device)
            print('Analysing done in %f seconds' % (time.time()-startTime))
        else:
            # DO NOT USE!
            switchTimes = []
            switchSteps = []
            for x in range(self.dimX):
                # Print current row
                if not (x+1)%10:
                    strOut = 'Analysing row:  %i/%i on %f seconds\r' % (x+1, self.dimX, time.time()-startTime)
                    sys.stderr.write(strOut)
                    #sys.stdout.flush()
                for y in range(self.dimY):
                    switch, levels = self.getSwitchTime((x,y))
                    grayChange = np.abs(levels[0]- levels[1])
                    if switch == 0: # TODO: how to deal with steps at zero time
                        print(x,y)
                    switchTimes.append(switch)
                    switchSteps.append(grayChange)
            print("\n")
            # Note that the _swithchSteps can be zero and 
            # _swithTimes set to the first image
            # This problem must be solved out this method
            self._switchTimes = np.asarray(switchTimes)
            self._switchSteps = np.asarray(switchSteps)
        self._isColorImage = True
        self._isSwitchAndStepsDone = True
        return

    def _getSwitchTimesOverThreshold(self, isFirstSwitchZero=False, fillValue=-1):
        """
        _getSwitchTimesOverThreshold()
        Returns the array of the switch times
        considering a self._threshold in the gray level change at the switch

        Parameters:
        ----------------
        isFirstSwitchZero : bool
            Put the first switch equal to zero, useful to set the colors
            in a long sequence of images where the first avalanche
            occurs after many frames
        fillValue : number, int
            The value to set in the array for the non-switching pixel (below the threshold)
            -1 is used as the last value of array when used as index (i.e. with colors)
        """
        # self.isPixelSwitched = (self._switchSteps >= self._threshold)
        ### 
        #get sigma from hist of images and use it as threshold
        ###
        if self._threshold is None:
            self._threshold = int(np.std(self.Array.flatten())*0.1)
            print("estimated threshold = %d"%self._threshold)

        self.isPixelSwitched = (self._switchSteps >= self._threshold) & (self._switchTimes > self.kernel_half_width_of_ones)
        maskedSwitchTimes = ma.array(self._switchTimes, mask = ~self.isPixelSwitched)
        # Move to the first switch time if required
        self._switchTimesOverThreshold = maskedSwitchTimes.compressed()
        if isFirstSwitchZero:
            maskedSwitchTimes = maskedSwitchTimes - self.min_switch
        # Set the non-switched pixels to use the last value of the pColor array, i.e. noSwitchColorValue
        switchTimesWithFillValue = maskedSwitchTimes.filled(fillValue) # Isn't it fantastic?
        return switchTimesWithFillValue


    def _isColorImageDone(self,ask=True):
        print("You must first run the getSwitchTimesAndSteps script: I'll do that for you")
        if ask:
            yes_no = raw_input("Do you want me to run the script for you (y/N)?")
            yes_no = yes_no.upper()
            if yes_no != "Y":
                return
        self.getSwitchTimesAndSteps()
        return

    def _getColorImage(self, palette, noSwitchColor='black', erase_small_events_percent=None):
        """
        Calculate the color Image using the output of getSwitchTimesAndSteps

        Parameters:
        ---------------
        threshold: int, opt
        Set the minimim value of the gray level change at the switch to
        consider the pixel as 'switched', i.e. belonging to an avalanche
        This value is set as a class variable from here on.
        Rerun self._getColorImage to change it

        Results:
        ----------
        self._switchTimes2D as a 2D array of the switchTime
        with steps >= threshold, and first image number set to 0
        """

        if not self._isSwitchAndStepsDone:
            self._isColorImageDone(ask=False)

        self.min_switch = np.min(self._switchTimes)
        self.max_switch = np.max(self._switchTimes)
        print("Avalanches occur between frame %i and %i" % (self.min_switch, self.max_switch))
        self._nImagesWithSwitch = self.max_switch - self.min_switch + 1
        print("Gray changes are between %s and %s" % (min(self._switchSteps), max(self._switchSteps)))

        # Calculate the colours, considering the range of the switch values obtained
        self._pColors = getPalette(self._nImagesWithSwitch, palette, noSwitchColor)
        self._colorMap = mpl.colors.ListedColormap(self._pColors, 'pColorMap')
        central_points = np.arange(self.min_switch, self.max_switch, dtype=float)
        # Calculate the switch time Array (2D) considering the threshold and the start from zero
        self._switchTimes2D = self._getSwitchTimesOverThreshold(False, self.fillValue).reshape(self.dimX, self.dimY)
        self._switchSteps2D = self._switchSteps.reshape(self.dimX, self.dimY)
        self._switchTimes2D_original = np.copy(self._switchTimes2D)
        # Now check if getting rid of the wrong switches
        if self.exclude_switches_out_of_final_domain:
            q = self._switchTimes2D != self.fillValue
            clusters, n_cluster = mahotas.label(q, self.NNstructure)
            sizes = mahotas.labeled.labeled_size(clusters)
            max_size = np.max(sizes[1:])
            too_small = np.where(sizes < max_size)
            # Get the largest final domain only
            final_domain = mahotas.labeled.remove_regions(clusters, too_small).astype('bool')
            self._switchTimes2D[~final_domain] = self.fillValue
        if erase_small_events_percent:
            percentage = erase_small_events_percent/100.
            # This gets rid of the wrong switchesf
            # Using the cluster max sizes for the switches
            # It redefines self._switchTimes2D
            #self.final_domain = self._find_final_domain(self._switchTimes2D, fillValue)
            #self._switchTimes2D[self.final_domain == False] = -1
            # Erase small event
            #im, n_cluster = mahotas.label(~self.final_domain, self.NNstructure)
            q = np.copy(self._switchTimes2D)
            q[q == -1] = 0
            im, n_cluster = mahotas.label(q, self.NNstructure)
            im = mahotas.labeled.remove_bordering(im)
            im, n_cluster = mahotas.labeled.relabel(im)
            sizes = mahotas.labeled.labeled_size(im)
            too_small = np.where(sizes < percentage * np.max(sizes[1:]))
            im = mahotas.labeled.remove_regions(im, too_small)
            print("Small events erased")
            #index_max_size = sizes.argmax()
            #self.initial_domain = im == index_max_size + 1
            # Erase the small switches
            self._switchTimes2D[im == 0] = self.fillValue
        if self.fillValue in self._switchTimes2D:
            self._switchTimesUnique = np.unique(self._switchTimes2D)[1:] + self.min_switch
        else:
            self._switchTimesUnique = np.unique(self._switchTimes2D) + self.min_switch

        return


    def showColorImage(self, threshold=None, data=None, palette='random', erase_small_events_percent=None,
                        plotHist=False, plot_contours=False, 
                        noSwitchColor='black', ask=False, fig=None, ax=None, title=None, figsize=(8,7)):
        """
        Show the calculated color Image of the avalanches.
        Run getSwitchTimesAndSteps if not done before.

        Parameters
        ---------------
        threshold: integer, optional
            Defines if the pixel switches when gray_level_change >= threshold
        palette: string, required, default = 'random'
            Choose a palette between 'korean', 'randomKorean', 'random', 'hue', 'randomHue', 'ral'
            'randomKorean' is a random permutation of the korean palette
            'random' : calculated on the fly, so each call of the method gives different colors
            'hue' : equally spaced colors in the HUE weel
            'randomHue' : random of above
            'ral': ral colors
            'coolwarm': from Red to Blue
            'pastel': add white to random colors
        erase_small_events_percent: int, optional
            Erase events smaller than a percentage of the largest one 
        noSwithColor: string, optional, default = 'black'
            background color for pixels having gray_level_change below the threshold
        """
        # set the threshold
        estimated_threshold = int(np.std(self.Array.flatten())*0.1)
        print("Estimated threshold = %d" % estimated_threshold)
        if not threshold:
            self._threshold = estimated_threshold
        else:
            self._threshold = threshold
        # Calculate the Color Image
        self._getColorImage(palette, noSwitchColor, erase_small_events_percent)
        # Prepare to plot
        if data is None:
            data = self._switchTimes2D
        if fig is None:
            self._figColorImage = self._plotColorImage(data, 
                self._colorMap, self._figColorImage, title=title, figsize=figsize)
            fig = self._figColorImage
        else:
            fig = self._plotColorImage(data, colorMap=self._colorMap, fig=fig, ax=ax, title=title)
        if plot_contours:
            if ax is None:
                ax = fig.gca()
            self.find_contours(lines_color='k', remove_bordering=True, invert_y_axis=False, fig=fig, ax=ax)
        if plotHist:
            # Plot the histogram
            self.plotHistogram(self._switchTimesOverThreshold, ylabel="Avalanche size (pixels)")
        # Count the number of the switched pixels
        switchPixels = np.sum(self.isPixelSwitched)
        totNumPixels = self.dimX * self.dimY
        noSwitchPixels = totNumPixels - switchPixels
        swPrint = (switchPixels, switchPixels/float(totNumPixels)*100., 
                   noSwitchPixels, noSwitchPixels/float(totNumPixels)*100.)
        print("There are %d (%.2f %%) switched and %d (%.2f %%) not-switched pixels" % swPrint)
        plt.show()

    def _call_pixel_switch(self, p0, p1):
        x, y = self._pixel2rowcol((p0,p1), closest_integer=True) 
        if x >= 0 and x < self.dimX and y >= 0 and y < self.dimY:
            index = x * self.dimY + y
            s = "pixel (%i,%i) - switch at: %i, gray step: %i" % \
                (p0, p1, self._switchTimes2D[x,y], self._switchSteps2D[x,y])
            return s
        else:
            return ""

    def _call_pixel_time_sequence(self, event):
        if event.dblclick:
            pixel = int(round(event.xdata)), int(round(event.ydata))
            print("Pixel: (%i, %i)" % (pixel[0],pixel[1]))
            self.showPixelTimeSequence(pixel, newPlot=True)
            #plt.show()

    def _plotColorImage(self, data, colorMap, fig=None, ax=None, title=None, figsize=(8,7)):
        """
        if the caption is not shown, just enlarge the image
        as it depends on the length of the string retured by
        _call_pixel_switch
        """
        if not fig:
            fig = plt.figure()
            fig.set_size_inches(*figsize, forward=True)
            ax = fig.add_subplot(1, 1, 1)
        else:
            fig = plt.figure(fig.number)
            if ax is None:
                ax = fig.gca()
        ax.format_coord = lambda p0, p1: self._call_pixel_switch(p0, p1)
        # Sets the limits of the image
        extent = 0, self.dimY, self.dimX, 0
        bounds = np.unique(data)
        bounds = np.append(bounds, bounds[-1]+1)-0.5
        self.norm = mpl.colors.BoundaryNorm(bounds, len(bounds)-1)
        #ax.pcolormesh(data,cmap=colorMap,norm=self.norm)
        ax.imshow(data, cmap=colorMap, norm=self.norm)
        ax.axis(extent)
        if title is None:
            first, last = self.imagesRange
            #title = "DW motion from image %i to image %i" % (first, last)
            title = self.pattern
        ax.title.set_text(title)
        cid = fig.canvas.mpl_connect('button_press_event', self._call_pixel_time_sequence)
        return fig

    def plotHistogram(self, data, fig=None, ax=None, title=None, ylabel=None):
        #image_numbers = np.unique(data)
        #i0, i1 = image_numbers[0], image_numbers[-1] + 1
        #central_points = np.arange(i0, i1)
        central_points = self.imageNumbers
        rng = np.append(central_points, central_points[-1] + 1) - 0.5
        if fig:
            plt.figure(fig.number)
            if ax is None:
                ax = plt.gca()
        else:
            if self._figHistogram is None:
                self._figHistogram = plt.figure()
            else:
                plt.figure(self._figHistogram.number)
                plt.clf()
            ax = plt.gca()
        self.N_hist, self.bins_hist, patches = ax.hist(data, rng)       
        ax.format_coord =  lambda x,y : "image Number: %i, avalanche size (pixels): %i" % (int(x+0.5), int(y+0.5))
        ax.set_xlabel("image number")
        if title:
            ax.set_title(title)
        if ylabel:
           ax.set_ylabel(ylabel)
        ax.set_xlim(0, ax.get_xlim()[1])
        # Set the colors
        for thisfrac, thispatch in zip(central_points, patches):
            c = self._colorMap(self.norm(thisfrac))
            thispatch.set_facecolor(c)
        self.is_histogram = True
        return None


    def saveColorImage(self,fileName,threshold=None, palette='random',noSwitchColor='black'):
        """
        saveColorImage(fileName, threshold=None, 
                       palette='korean',noSwitchColor='black')

        makes color image and saves
        """
        self._colorImage = self._getColorImage(palette,noSwitchColor)
        imOut = scipy.misc.toimage(self._colorImage)
        imOut.save(fileName)

    def saveImage(self, figureNumber):
        """
        generic method to save an image or a plot in the figure(figureNumber)
        """
        filename = raw_input("file name (.png for images)? ")
        fig = plt.figure(figureNumber)
        ax = fig.gca()
        if len(ax.get_images()):
            im = ax.get_images()[-1]
            name, ext = os.path.splitext(filename)
            if ext != ".png":
                filename = name +".png"
            filename = os.path.join(self._mainDir, filename)
            im.write_png(filename)
        else:
            filename = os.path.join(self._mainDir, filename)
            fig.save(filename)

    def imDiffCalculated(self, imageNum, haveColors=True):
        """
        Get the difference in BW between two images imageNum and imageNum+1
        as calculated by the self._colorImage
        """
        if not self._isColorImage:
            self._isColorImageDone(ask=False)
        imDC = (self._switchTimes2D == imageNum) * 1
        if haveColors:
            imDC = scipy.array(imDC, dtype='int16')
            l, n = self.label(imDC, self.NNstructure)
            im_io.imshow(l, plt.cm.prism)
        else:
            # Normalize to a BW image
            self.imDiffCalcArray = imDC * 255
            scipy.misc.toimage(self.imDiffCalcArray).show()
        return None

    def _show_next_raw_image(self, event, showRawAndCalcImages=None):
        """
        show the next row image
        calling showRawAndCaldImages with the
        next available image
        """
        dobj = {'event':self.showRawAndCalcEvents, 'cluster': self.showRawAndCalcClusters}
        key = str(event.key).lower()

        self.ax0_axis = self.figRawAndCalc.axes[0].axis()
        if key != 'n' and key != 'p' and key!= "a":
            return
        if self.data_type == 'event':
            sw = self._switchTimesUnique   
        elif self.data_type == 'cluster':
            sw = self.sw_clusters_start
        step = 1*(key=='n') -1*(key=='p')

        if key == "a":
            outDir = os.path.join(self._mainDir, "Results")
            if not os.path.isdir(outDir):
                os.mkdir(outDir)
            fname = "Raw_and_calc_%s.pdf" % self.nImage
            fname = os.path.join(outDir, fname)
            self.figRawAndCalc.savefig(fname)
            return
        if self.nImage == sw[-1] and key == 'n':
            print("No more images available")
            return
        elif self.nImage == sw[0] and key == 'p':
            print("No previous images available")
            return
        i, = np.nonzero(sw == self.nImage)
        n = sw[i[0] + step]
        dobj[self.data_type](n, isTwoImages=self.isTwoImages, subtract_first_image=self.subtract_first_image)
        return
    
    def saveRawAndCalcImages(self, frmt='pdf', isTwoImages=False):
        """
        frmt: string
        Output format (def. pdf)
        """
        outDir = os.path.join(self._mainDir, "Results")
        if not os.path.isdir(outDir):
            os.mkdir(outDir)
        # Reset the images
        self.figRawAndCalc = None
        for image in self._switchTimesUnique:
            self.showRawAndCalcImages(image, isTwoImages=isTwoImages)
            fname = "Raw_and_calc_%05d.%s" % (image, frmt)
            fname = os.path.join(outDir, fname)
            self.figRawAndCalc.show()
            if isTwoImages:
                fc = 'k'
            else:
                fc = 'w'
            self.figRawAndCalc.savefig(fname, facecolor=fc)        
            print("Saving image %i" % image)
        return

    def label(self, image):
        """
        calculate the clusters using
        the mahotas code
        http://mahotas.readthedocs.org/en/latest/labeled.html
        """
        edges = [('0001', '0010'), ('0100', '1000'), ('0111', '1000'), ('0111','1001')]
        im, n_cluster = mahotas.label(image, self.NNstructure)
        if self.boundary == 'periodic':
            dic_labels = {}
            # First find the image of no_touching clusters
            no_touching = mahotas.labeled.remove_bordering(im)
            # Then find the image of touching clusters
            im_touching = im - no_touching
            labels_touching = np.unique(im_touching)[1:]
            for label in labels_touching:
                ltype, = gal.getAxyLabels(im_touching==label, self.imageDir, 1)
                dic_labels[ltype] = dic_labels.get(ltype, []) + [label]
            for edge in edges:
                e0, e1 = edge
                if e0 in dic_labels.keys() and e1 in dic_labels.keys():
                    for up_label in dic_labels[e0]:
                        for down_label in dic_labels[e1]:
                            # Check the n. of clusters
                            imUp, imDown = im_touching==up_label, im_touching==down_label
                            imPlus = imUp + imDown
                            im0, n0 = mahotas.label(imPlus, self.NNstructure)
                            # Join the two images
                            if e0 == '0001':
                                imDouble = np.vstack((imPlus, imPlus))
                            elif e0 == '0100':
                                imDouble = np.hstack((imPlus, imPlus))
                                # Check if there is only a single avalanche
                            imOut, nclus = mahotas.label(imDouble, structure)
                            if nclus == 3:
                                im[im==down_label] = up_label
                                n_cluster -= 1
                                 # TODO renumber the clusters
                
            im, n_cluster = mahotas.labeled.relabel(im)
        return im, n_cluster


    def showRawAndCalcClusters(self, nImage=None, preAvalanches=True, \
        isTwoImages=False, subtract_first_image=False, autoscale=False):
        # Check if cluster2D exists
        try:
            q = isinstance(self.cluster2D_start, object)
        except AttributeError:
            print("self.cluster2D does non exists yet. Run the appropriate code")
            return
        self.sw_clusters_start = np.unique(self.cluster2D_start)[1:]
        self.sw_clusters_end = np.unique(self.cluster2D_end)[1:]
        if nImage is None:
            self.nImage = self.sw_clusters_start[0]
            print("Starting from image %i" % self.nImage)
        elif nImage not in self.sw_clusters_start:
            print("No switch there")
            i = np.argmin(abs(nImage - self.sw_clusters_start))
            print("Closest values: %s " % " ".join([str(p) for p in self.sw_clusters_start[i-1:i+2]]))
            return
        else:
            self.nImage = nImage
        self.showRawAndCalcImages('cluster', nImage, preAvalanches, isTwoImages,
                            subtract_first_image, autoscale)


    def showRawAndCalcEvents(self, nImage=None, preAvalanches=True, \
        isTwoImages=False, subtract_first_image=False, autoscale=False):
        if self._switchTimes is None:
            print("Need to calculate the color image first")
            return
        if nImage is None:
            self.nImage = self._switchTimesUnique[0]
            print("Starting from image %i" % self.nImage)
        elif nImage not in self._switchTimesUnique:
            print("No switch there")
            return
        else:
            self.nImage = nImage
        self.showRawAndCalcImages('event', nImage, preAvalanches, isTwoImages,
                            subtract_first_image, autoscale)

    def showRawAndCalcImages(self, data_type, nImage, preAvalanches=True, \
        isTwoImages=False, subtract_first_image=False, autoscale=False):

        """
        show the Raw and the Calculated image n
        Automatically increases the values of the image
        pressing "n" (next) or "p" (previous)
        Uses the measure.find_contours script

        Parameters
        ---------------
        nImage: integer, optional
            Defines the image to display
            
        preAvalanches: bool
        show the previous calculated avalanches in white
        
        isTwoImages: bool
        if True shows only two images, the raw+1 and the calculated one
        """
        gray, black, white = (0.75,0.75,0.75), (0,0,0), (1,1,1)
        self.data_type = data_type
        if data_type == 'event':
            data = self._switchTimes2D
            step_image = 1
        elif data_type == 'cluster':
            data = self.cluster2D_start
            i, = np.where(self.sw_clusters_start==self.nImage)
            im = data == self.nImage
            step_image = np.max(self._switchTimes2D[im].flatten()) - (self.nImage - 1)
            #step_image = self.sw_clusters_end[i[0]] - (self.nImage - 1)

        # Subtract first image
        if subtract_first_image:
            self.subtract_first_image = True
        else:
            self.subtract_first_image = False

        if isTwoImages:
            rows, cols = 1,2
        else:
            rows, cols = 2,3

        if self.figRawAndCalc and self.isTwoImages==isTwoImages:
            plt.figure(self.figRawAndCalc.number)
            axs = self.figRawAndCalc.get_axes()
            axs = np.array(axs).reshape((rows,cols))
            for i in range(rows):
                for j in range(cols):
                    axs[i,j].clear()
        else:
            #self.figRawAndCalc = plt.figure()
            self.figRawAndCalc,axs =  plt.subplots(rows,cols,sharex=True, sharey=True)
            self.figRawAndCalc.set_size_inches((10*rows,5*rows))
            if isTwoImages:
                self.figRawAndCalc.set_facecolor('black')
                self.isTwoImages = isTwoImages
            self.ax0_axis = None
        # Prepare the color map of calculated avalanches
        switchTimes_images = data == self.nImage      
        contours = measure.find_contours(switchTimes_images, 0.5)
        cl = self._pColors[self.nImage - self.min_switch]
        #myMap = mpl.colors.ListedColormap([(0,0,0),cl],'mymap',2)
        #mapGreyandBlack = mpl.colors.ListedColormap([(0.75,0.75,0.75),(0,0,0)],'mymap',2) # in grey and black      
        mapGreyandBlack = mpl.colors.ListedColormap([white, gray],'mymap',2) # in grey and black      




        im, n_clusters = self.label(switchTimes_images)
        sizes = mahotas.labeled.labeled_size(im)[1:]
                
        self.figRawAndCalc.suptitle('Number of clusters = %i ; Biggest cluster size = %i pixels' % (n_clusters, np.max(sizes)), fontsize=18)
            




        # Plot the two raw images first
        if isTwoImages:
            #ax1 = plt.subplot(1,2,1)
            ax = axs[0,0]
            if self.subtract_first_image:
                im = self._imDiff((self.nImage,self.imageNumbers[0]))
            else:
                im = self[self.nImage]
            ax.imshow(im, plt.cm.gray)
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
        else:
            for i in range(2):
                ax = axs[0,i]
                n0 = self.nImage - 1 + i * step_image
                if self.subtract_first_image:
                    im = self._imDiff((n0,self.imageNumbers[0]))
                else:
                    im = self[n0]
                ax.imshow(im, plt.cm.gray)
                ax.set_title("Raw Image %i" % n0)
                ax.grid(color='blue', ls="-")
        if autoscale:
            ax.axis((0, self.dimY, self.dimX, 0))
        elif self.ax0_axis is not None:
            ax.axis(self.ax0_axis)
        # Add the calculated avalanches to the first raw image
        if not isTwoImages:
            #plt.subplot(2,3,3)
            ax = axs[0,2]
            if self.subtract_first_image:
                im = self._imDiff((self.nImage-1,self.imageNumbers[0]))
            else:
                im = self[self.nImage-1]
            ax.imshow(im, plt.cm.gray)
            # Use find_contours
            for n, contour in enumerate(contours):
                ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color='k')
            if data_type == 'cluster':
                for sw in range(self.nImage, self.nImage+step_image+1):
                    im = np.logical_and((self._switchTimes2D == sw), (data == self.nImage))
                    sub_cnts = measure.find_contours(im, 0.5)
                    for sub_cnt in sub_cnts:
                        ax.plot(sub_cnt[:, 1], sub_cnt[:, 0], linewidth=1)
            ax.axis((0, self.dimY, self.dimX, 0))        
            ax.grid(color='blue',ls="-")
            ax.set_title("Image %s + Calculated %s" % (self.nImage-1, data_type))        
            # Show the raw image difference
            ax = axs[1,0]
            n0 = self.nImage - 1 + step_image
            ax.imshow(self._imDiff((n0,self.nImage-1)), plt.cm.gray)
            ax.set_title("Diff. Image %s and Image %s" % (n0, self.nImage-1))
            ax.grid(color='blue', ls="-")
            if autoscale:
                ax.axis((0, self.dimY, self.dimX, 0))
            elif self.ax0_axis is not None:
                ax.axis(self.ax0_axis)
            
            # Show the raw difference images and the calculated avalanches
            ax1 = axs[1,1]
            #plt.subplot(2,3,5)
            ax1.imshow(self._imDiff((n0,self.nImage-1)), plt.cm.gray)
            # Use find_contours
            for contour in contours:
                ax1.plot(contour[:, 1], contour[:, 0], linewidth=2, color='r')

            if autoscale:
                ax1.axis((0, self.dimY, self.dimX, 0))
            elif self.ax0_axis is not None:
                ax1.axis(self.ax0_axis)
        
            ax1.set_title("Raw Diff and Calculated %s" % data_type)
            ax1.grid(color='blue',ls="-")
        
        # Show the calculated avalanches only
        if isTwoImages:
            ax2 = axs[0,1]
        else:
            ax2 = axs[1,2] 
        # if data_type == 'event':
        #     imOut, n_clusters = self.label(switchTimes_images)
        # elif data_type == 'cluster':
        #     im = np.logical_and((data >= self.nImage), (data <= self.nImage + step_image))
        #     imOut, n_clusters = self.label(im)
        #     print(n_clusters)
        imOut, n_clusters = self.label(switchTimes_images)
        # Prepare the palette
        myPalette_background = [(0.9,0.9,0.9)]
        myPalette = myPalette_background + [hsv_to_rgb(j/float(n_clusters),1,1)
                                          for j in np.random.permutation(range(n_clusters))]
        if preAvalanches:
            #w = self._getSwitchTimesOverThreshold(False, fillValue=self.fillValue).reshape(self.dimX, self.dimY) 
            _pre = np.logical_and((data < self.nImage), (data > 0))
            imOut = imOut + _pre * (max(imOut.flatten()) + 1)
            myPalette = myPalette + [gray] # Add gray to the palette
            
        ax2.imshow(imOut, mpl.colors.ListedColormap(myPalette))
        #plt.imshow(switchTimes_images, mapGreyandBlack)
        if not isTwoImages:
            ax2.grid(color='blue',ls="-")
            ax2.set_title("Calculated %s" % data_type)
        else:
            ax2.xaxis.set_visible(False)
            ax2.yaxis.set_visible(False)
        if autoscale:
            ax2.axis((0, self.dimY, self.dimX, 0))
        elif self.ax0_axis is not None:
            ax2.axis(self.ax0_axis)
        
        plt.draw()
        plt.show()
        print("Press: Next, Previous, sAve")
        size = np.sum(switchTimes_images)
        msg = "%s %i: %i pixels (%.2f %% of the image)" % \
              (data_type.capitalize(), self.nImage, size, float(size)/(self.dimX*self.dimY)*100)
        print(msg)
        sub_cluster_sizes = sorted([np.sum(im==i) for i in range(1, n_clusters+1)], reverse=True)
        print("Cluster sizes: " + ", ".join([str(size) for size in sub_cluster_sizes if size>1]))
        if not self.isConnectionRawImage:
            self.figRawAndCalc.canvas.mpl_connect('key_press_event', self._show_next_raw_image)
            self.isConnectionRawImage = True
        return

    def ghostbusters(self, clusterThreshold = 15, showImages=False, imageNumber=None):
        """
        Find the presence of 'ghost' images,
        given by not fully-resolved avalanches.
        Calculates the number of clusters for each avalanche,
        and see if it larger than the threshold.
        *** Automatic check
        If it is, checks if also the following image has an avalanche
        with a large number of cluster
        In positive, join the two avalanches, and check the number
        of clusters again. If it smaller than the threshold,
        the switch time is updated.
        *** Manual check
        If showImages is enabled, the user must manually set the images to join
        *** If the imageNumber is given,
        the method works only on that frame and the following one.
        This is used to manually check two frames and joint them

        Parameters
        ---------------
        clusterThreshold : int
        Minimum number of clusters to consider the avalanche as 'spongy'

        showImages : bool
        If True, show all the 'spongy' images and their joint one
        """
        joinImages = False
        if not self._isColorImage:
            print("This is available only after the color image is done")
            return
        if showImages:
            figCluster = plt.figure()
            figCluster.set_size_inches(12, 8, forward=True)
            i0 = [0,1,1] # Index of the first raw image
            i1 = [-1,0,-1] # Index of the second raw image
        n_of_images_with_ghosts = []
        images_with_ghosts = {}
        # ##############################
        # TODO
        # The script below needs fixing
        # ##############################
        if imageNumber:
            iterator = np.asarray([imageNumber, imageNumber+1]) - self.min_switch
            clusterThreshold = 0
        else:
            iterator = np.unique(self._switchTimes2D)
        # Calculates the set of switches
        for imageNumber0 in iterator:
            im0 = (self._switchTimes2D==imageNumber0)*1
            im0 = scipy.array(im0, dtype="int16")
            array_labels, n_clusters = self.label(im0, self.NNstructure)
            if n_clusters >= clusterThreshold:
                imageNumber = imageNumber0 + self.min_switch
                n_of_images_with_ghosts.append(imageNumber)
                images_with_ghosts[imageNumber] = array_labels, n_clusters
        # Now evaluate spongy avalanches and check if number of clusters is reduced
        # Let us do it first on consecutive images belonging to n_of_images_with_ghosts
        gh = scipy.asarray(n_of_images_with_ghosts)
        # Consider consecutive images only
        ghosts_images = gh[gh[1:] == gh[:-1]+1]
        if len(ghosts_images) == 0:
            print("Warning, no images to consider")
            return
        for ghi in ghosts_images:
            image1, n1 = images_with_ghosts[ghi]
            image2, n2 = images_with_ghosts[ghi+1]
            new_array = scipy.array(image1+image2, dtype="int16")
            image3, n3 = self.label(new_array, self.NNstructure)
            if showImages:
                for i, results in enumerate(zip([image1, image2, image3],[n1, n2, n3])):
                    im, clusters = results
                    plt.subplot(2, 3, i+1)
                    # Prepare the palette, from red to magenta (see hue weel for details)
                    myPalette = [(0,0,0)] + [hsv_to_rgb(j/float(clusters),1,1) 
                                             for j in range(clusters)]
                    plt.imshow(im, mpl.colors.ListedColormap(myPalette))
                    imageNum = str(ghi+i)*(i<2) + (i==2)*"joint"
                    plt.title("Image: %s, N. clusters: %i" % (imageNum, clusters))
                    plt.subplot(2, 3, i+4)
                    plt.imshow(self._imDiff((ghi+i0[i],ghi+i1[i])), plt.cm.gray)
                y_n = raw_input("Join these avalanches from image %i and %i? (y/N)" % (ghi, ghi+1))
                y_n = y_n.upper()
                if y_n in ["Y", "YES"]:
                    joinImages = True
                else:
                    joinImages = False
            if (n3 < clusterThreshold and not showImages) or (showImages and joinImages):
                print("Joining images %i and %i" % (ghi, ghi + 1))
                whereChange = self._switchTimes==ghi
                # Update the 'untouched' array of switch times
                self._switchTimes[whereChange] = ghi + 1
                # Update the array with threshold and zero time at the beginning
                self._switchTimes2D[whereChange.reshape(self.dimX, self.dimY)] \
                    = ghi + 1 - self.min_switch
        # Add the image without ghosts to the original one
        self._figColorImage = self._plotColorImage(self._switchTimes2D, 
                                                   self._colorMap, fig=self._figColorImage)
        self.plotHistogram(self._switchTimes)

    def manualGhostbuster(self):
        """
        Manually adjust spongy avalanches by looking the color image
        The raw and the calculated image are presented
        """
        if not self._isColorImage:
            print("This is available only after the color image is done")
            return
        while True:
            imageNumber = raw_input("Number of the image to join with its next (Return to exit): ")
            if imageNumber is not "":
                imageNumber = int(imageNumber)
                self.ghostbusters(0, True, imageNumber)
            else:
                return

    def _getImageDirection(self, threshold=None):
        """
        _getImageDirection(threshold=None)

        Returns the direction of the sequence of avalanches as:
        "Top_to_bottom","Left_to_right", "Bottom_to_top","Right_to_left"

        Parameters:
        ----------------
        threshold : int
            Minimum value of the gray level change to conisider
            a pixel as part of an avalanche (i.e. it is switched)
        """
        # Top, left, bottom, rigth
        imageDirections=["Top_to_bottom","Left_to_right", 
                         "Bottom_to_top","Right_to_left"]
        # Check if color Image is available
        if not self._isColorImage:
            self._isColorImageDone(ask=False)
        switchTimesMasked = self._switchTimes2D
        pixelsUnderMasks = []
        # first identify first 10 avalanches of whole image
        firstAvsList = np.unique(self._switchTimes2D)[:11]
        # Excluding non-switching pixels
        firstAvsList = np.compress(firstAvsList>0, firstAvsList)
        # Prepare the mask
        m = np.ones((self.dimX, self.dimY))
        # Top mask
        mask = np.rot90(np.triu(m)) * np.triu(m)
        top = switchTimesMasked * mask
        pixelsUnderMasks.append(sum([np.sum(top==elem) 
        for elem in firstAvsList]))
        # Now we need to rotate the mask
        for i in range(3):
            mask = np.rot90(mask)
            top = switchTimesMasked * mask
            pixelsUnderMasks.append(sum([np.sum(top==elem) 
            for elem in firstAvsList]))
        max_in_mask = scipy.array(pixelsUnderMasks).argmax()
        return imageDirections[max_in_mask]

    def getDistributions(self, log_step=0.2, edgeThickness=1, fraction=0.01):
        """
        Calculates the distribution of avalanches and clusters

        Parameters:
        ---------------
        log_step : float
        The step in log scale between points in the log-log distribution.
        For instance, 0.2 means 5 points/decade

        edgeThickness : int
        No of pixels for each edge to consider as the frame of the image

        fraction : float
        This is the minimum fraction of the size of the avalanche/cluster inside
        an edge (of thickness edgeThickness) with sets the avalanche/cluster
        as touching
        """          
        # Check if analysis of avalanches has been performed
        if not self._isColorImage:
            self._isColorImageDone(ask=False)
        # Initialize variables
        self.D_avalanches = []
        self.D_cluster = scipy.array([], dtype='int32')
        self.N_cluster = []
        self.dictAxy = {}
        self.dictAxy['aval'] = {}
        self.dictAxy['clus'] = {}
        a0 = scipy.array([],dtype='int32')
        #Define the number of nearest neighbourg
        # This has to be defined elsewhere
        # as self.structure
        # Find the direction of the avalanches (left <-> right, top <-> bottom)
        if not self.imageDir:
            self.imageDir = self._getImageDirection(self._threshold)
        #
        # Make a loop to calculate avalanche and clusters for each image
        #
        for imageNum in self._switchTimesUnique:
            strOut = 'Analysing image n:  %i\r' % (imageNum)
            sys.stderr.write(strOut)
            #sys.stdout.flush()
            # Select the pixel flipped at the imageNum
            imageNum0 = imageNum - self.min_switch
            im0 = (self._switchTimes2D == imageNum0) * 1
            im0 = scipy.array(im0, dtype="int16")
            # Update the list of sizes of the global avalanche (i.e. for the entire image imageNum)
            avalanche_size = scipy.sum(im0)
            self.D_avalanches.append(avalanche_size)
            # Find how many edges this avalanche touches
            Axy = gal.getAxyLabels(im0, self.imageDir, edgeThickness)
            Axy = Axy[0] # There is only one value for the whole image
            # Update the dictionary of the avalanches
            self.dictAxy['aval'][Axy] = scipy.concatenate((self.dictAxy['aval'].get(Axy,a0), [avalanche_size]))
            #
            # Now move to cluster distributions
            #
            # Detect local clusters 
            array_labels, n_labels = self.label(im0, self.NNstructure)
            # Make a list the sizes of the clustersgetAxyLabels
            list_clusters_sizes = nd.sum(im0, array_labels, 
                                         range(1, n_labels+1))
            array_cluster_sizes = np.array(list_clusters_sizes, 
                                              dtype='int32')                        
            # Update the distributions
            self.D_cluster = scipy.concatenate((self.D_cluster, 
                                                list_clusters_sizes))
            #self.N_cluster[avalanche_size] = scipy.concatenate((self.N_cluster.get(avalanche_size, a0), [n_labels]))
            self.N_cluster.append(n_labels)
            # Now find the Axy distributions (A00, A10, etc)
            # First make an array of the edges each cluster touches
            array_Axy = gal.getAxyLabels(array_labels, self.imageDir, 
                                         edgeThickness)
            # Note: we can restrict the choice to left and right edges (case of strip) using:
            # array_Axy = [s[:2] for s in array_Axy]
            # Now select each type of cluster ('0000', '0010', etc), make the S*P(S), and calculate the distribution
            for Axy in np.unique(array_Axy):
                sizes = array_cluster_sizes[array_Axy==Axy] # Not bad...
                self.dictAxy['clus'][Axy] = scipy.concatenate((self.dictAxy['clus'].get(Axy,a0), sizes))
        print("Done")
        # Calculate and plot the distributions of clusters and avalanches
        D_x, D_y, D_yerr = gLD.logDistribution(self.D_cluster, log_step=log_step, 
                                       first_point=1., normed=True)
        P_x, P_y, P_yerr = gLD.logDistribution(self.D_avalanches, log_step=log_step, 
                                       first_point=1., normed=True)
        # Plots of the distributions
        plt.figure()
        plt.loglog(D_x,D_y,'o',label='cluster')
        plt.errorbar(D_x,D_y,D_yerr, fmt=None)
        plt.loglog(P_x,P_y,'v',label='avalanches')
        plt.errorbar(P_x, P_y, P_yerr,fmt=None)
        plt.legend()
        plt.show()
        # Show the N_clusters vs. size_of_avalanche
        plt.figure()
        clusterArray = np.array(zip(self.D_avalanches, self.N_cluster))
        sizeCluster, nClusters = gLD.averageLogDistribution(clusterArray, 
                                                            log_step=log_step, first_point=1.)
        plt.loglog(sizeCluster, nClusters,'o')
        plt.xlabel("Avalanche size")
        plt.ylabel("N. of clusters")
        plt.show()
        self._isDistributions = True
        return


    def _max_switch_not_touching_edges(self, sw):
        """
        Calculated the max switch with a fully internal domain 
        It is used to calculate the initial nucleated domain 
        This is too slow
        """
        q = np.copy(self._switchTimes2D)
        # The code below is too slow
        # switch0 = sw[0]
        # for switch in sw:
        #     im, n_cluster = mahotas.label(q==switch, self.NNstructure)
        #     if '0000' not in [gal.getAxyLabels(im==n) for n in range(1,n_cluster+1)]:
        #         return switch0
        #     else:
        #         switch0 = switch
        # return sw[-1]
        # Replace with this
        im, n_cluster = mahotas.label(q!=-1, self.NNstructure)
        max_size = mahotas.labeled.labeled_size(im)[1:]
        index_max_size = max_size.argmax()
    
    def minmax_switches(self):
        self.sw = np.unique(self._switchTimes2D)
        if self.sw[0] == -1:
            self.n_first = 1
        else:
            self.n_first = 0
        self.firstSw = self.sw[self.n_first]
        self.max_switch = self._max_switch_not_touching_edges(self.sw[self.n_first:])
        self.is_minmax_switches = True


    def _find_final_domain(self, data):
        """
        find the final domain
        if use_max_criterium:
            get all the pixel switches and get the max cluster only
        """
        if self.use_max_criterium:
            self.sw = np.unique(data)
            is_touching = True
            k = -1
            while (is_touching == True):
                im, n_cluster = mahotas.label(data!=self.fillValue, self.NNstructure)
                sizes = mahotas.labeled.labeled_size(im)[1:]
                index_max_size = sizes.argmax()
                final_domain = im == index_max_size + 1
                # Check if the final domain touches the edges
                im, n_cluster = mahotas.labeled.filter_labeled(final_domain, remove_bordering=True)
                if n_cluster == 1:
                    is_touching = False
                    self.max_switch = self.sw[k]
                    if self.sw[0] == -1:
                        self.n_first = 1
                    else:
                        self.n_first = 0
                    self.firstSw = self.sw[self.n_first]
                    self.is_minmax_switches = True
                else:
                    data[data == self.sw[k]] = self.fillValue
                    k -= 1

        else:
            print("Not implemented yet")
            return None
        return final_domain

    def find_central_domain(self, initial_domain_region=None):
        # check if initial_domain_region is set
        if initial_domain_region == None:
            initial_domain_region = self.initial_domain_region
        # Find the first switch time
        
        # Find the initial nucleated domain
        # Panic!
        # If the internal domain touches two edges, the label calculation
        # fails, as the backgroub is split into two clusters
        if initial_domain_region == None:
            if not self.is_minmax_switches:
                self.minmax_switches()
            q = np.copy(self._switchTimes2D)
            # Set the switched pixels as the backgroud
            q[(self._switchTimes2D >= self.firstSw) & (self._switchTimes2D <= self.max_switch)] = 0
            q[self._switchTimes2D == -1] = 1
            if self.max_switch != self.sw[-1]:
                q[self._switchTimes2D >= self.max_switch] = 1
            im, n_cluster = mahotas.label(q, self.NNstructure)
            # Find the nucleated domain
            # It assumes it does not touches the edges
            im = mahotas.labeled.remove_bordering(im)
            # It is better to relabel
            im, n_cluster = mahotas.label(im, self.NNstructure)
            # find the largest cluster
            size_clusters = mahotas.labeled.labeled_size(im)[1:]
            index_central_domain = size_clusters.argmax()
            size_central_domain = size_clusters[index_central_domain]
            central_domain = im == index_central_domain + 1
            # Check if the central domain is compact
            # and exclude the holes from the switched pixels
            if self.exclude_switches_from_central_domain:
                central_domain = self._exclude_holes_from_central_domain(central_domain)
            yc,xc = nd.measurements.center_of_mass(central_domain)
        else:
            # The code below is clearly buggy
            xmin, ymin, xmax, ymax = initial_domain_region
            for switch in self.sw[self.n_first:]:
                q = self._switchTimes2D == switch
                im, n_cluster = mahotas.label(q, self.NNstructure)
                for i in range(1, n_cluster+1):
                    central_domain = im == i
                    yc,xc = nd.measurements.center_of_mass(central_domain)
                    if xc > xmin and xc < xmax and yc > ymin and yc < ymax:
                        print("Central domain found at switch %i" % switch)
                        size_central_domain = np.sum(im)
                        central_domain = self._exclude_holes_from_central_domain(central_domain)
                        return (yc,xc), central_domain, size_central_domain
            print("Sorry, I could not find any central domain within (%i,%i,%i,%i)" % initial_domain_region)
            central_domain = None
            xc, yc = None, None
        return (yc,xc), central_domain, size_central_domain

    def _exclude_holes_from_central_domain(self, central_domain):
        # Check if the central domain is compact
        # and exclude the holes from the switched pixels
        not_central_domain = ~central_domain
        not_central_domain, n_cluster = mahotas.label(not_central_domain, self.NNstructure)
        if n_cluster:
            not_central_domain = mahotas.labeled.remove_bordering(not_central_domain)
            # Add the clusters to the central domain
            clusters = not_central_domain > 0
            central_domain = central_domain + clusters
        # Exclude them all from the switched points
        self._switchTimes2D[clusters] = -1
        return central_domain

    def _get_contours(self, domain, longest=False, 
                    connectivity='high', threshold=0.5):
        """
        find the contour of a generic domain
        Parameters:
            domain: bool of the domain to find the contours of
            longest: bool
                if True, returns the longest, otherwise all
            connectivity: high, low
        """
        cnts = measure.find_contours(domain, threshold, connectivity)
        if not longest:
            return cnts
        else:
            l = [len(cnt) for cnt in cnts]
            i = np.argmax(l)
            return cnts[i]

    def find_contours(self, lines_color=None, invert_y_axis=True, step_image=1,
                        consider_events_around_a_central_domain=True,
                        initial_domain_region=None, remove_bordering=False,
                        plot_centers_of_mass = False, reference=None,
                        rescale_area=False, plot_rays=False,
                        fig=None, ax=None, title=None):
        """
        Find the contours of the sequence of DW displacements
        This is suitable for DW bubble expansion experiments
        where the difference between images is saved in the measurement

        Parameters:
        -----------
        lines_color : string
           Color of the lines
        step_image : int
            step in the image sequence to calculate the contour
        remove_bordering : bool
            exclude the domain touching one the edges
        reference : str
            None or 'center_of_mass'
        """
        if fig is None:
            fig = plt.figure(figsize=self._figColorImage.get_size_inches())
            ax = fig.gca()
        else:
            plt.figure(fig.number)
            if ax is None:
                ax = fig.gca()
        self.contours = {}
        self.bubbles = {}
        self.centers_of_mass = {}
        # find the initial domain
        self.domain = Domains(self._switchTimes2D)
        central_domain = self.domain.get_initial_domain(is_remove_small_holes=False)
        size_central_domain = np.sum(central_domain)
        yc, xc = nd.measurements.center_of_mass(central_domain)
        #cnts0 = measure.find_contours(central_domain, 0.5, 'high')[0]
        cnts0 = self._get_contours(central_domain, longest=True)
        self.contours[0] = cnts0
        self.bubbles[0] = central_domain
        self.centers_of_mass[0] = (yc, xc)
        # Rescale the area if needed
        if rescale_area:
            scaling = size_central_domain**0.5
        else:
            scaling = 1.
        # Plot the central domain
        X, Y = cnts0[:, 1], cnts0[:, 0]
        # Plot the center of mass of the nucleated domain
        if reference == 'center_of_mass':
            X, Y = (X - xc) / scaling, (Y - yc) / scaling
            ax.plot(0, 0, 'o', color=lines_color)
        else:
            X, Y = X / scaling, Y / scaling
            ax.plot(xc, yc, 'o', color=lines_color)
        # The nucleated domain is always black
        ax.plot(X, Y, 'k', antialiased=True, lw=2)
        self.sw = self.domain.sw[:self.domain.max_switch]
        for k, switch in enumerate(self.sw):
            #print(switch)
            q = self._switchTimes2D == switch
            q = q + central_domain
            labeled, n_cluster = mahotas.label(q, self.NNstructure)
            im, n_cluster = mahotas.labeled.filter_labeled(labeled,
                            remove_bordering=remove_bordering, min_size=size_central_domain)
            if n_cluster > 1:
                print("switch %i has %i clusters" % (switch, n_cluster))
                # If there are many clusters, take the larger one
                # TODO: this is not very general
                size_clusters = mahotas.labeled.labeled_size(im)[1:]
                #print(size_clusters)
                index_central_domain = size_clusters.argmax()
                size_central_domain = size_clusters[index_central_domain]
                central_domain = im == index_central_domain + 1
            else:
                central_domain = im
                size_central_domain = np.sum(central_domain)
            # Get the properties
            try:
                properties = measure.regionprops(central_domain*1)[0]
            except:
                print(properties)
                print("There is a problem with the central domain: properties not avaliable")
                self.central_domain = central_domain
                self.im = im
                print("switch: {}, n_cluster: {}".format(switch,n_cluster))
                break
            # Eccentricity to be checked
            # if k==0:
            #     ecc = properties.eccentricity
            # else:
            #     # TODO: check if the eccentricity gets too different
            #     # Need to check when also grow horizontally
            #     if ecc/properties.eccentricity > 1.5:
            #         print("there is a problem with the contour, please check it")
            #         ecc = None
            #     else:
            #         ecc = properties.eccentricity
            #print(ecc)
            y,x = nd.measurements.center_of_mass(central_domain)
            self.bubbles[switch] = central_domain
            self.centers_of_mass[switch] = (y, x)
            n_images = len(self.sw)
            n = float(switch - self.sw[0])
            clr = getKoreanColors(n, n_images)
            #print(n, clr)
            clr = tuple([c / 255. for c in clr])
            if plot_centers_of_mass and reference is None:
                ax.plot(x,y,'o',c=clr)
            #plt.plot(x,y,'o')
            try:
                #cnts = measure.find_contours(central_domain*1,.5, 'high')[0]
                cnts = self._get_contours(central_domain, longest=True)
            except:
                self.im = im
                print("There is a problem with the contour of image n. {}".format(switch))
                break
            # TODO: the contour changes drastically when two walls merge. How to fix it?
            self.contours[switch] = cnts 
            if not k%step_image:
                if rescale_area:
                    scaling = size_central_domain**0.5
                    lw = 1.
                else:
                    scaling = 1.
                    lw = 0.5
                X,Y = cnts[:,1], cnts[:,0]
                if reference == 'center_of_mass':
                    X, Y = (X-x)/scaling, (Y-y)/scaling
                else:
                    X, Y = X/scaling, Y/scaling
                if lines_color is not None:
                    ax.plot(X,Y,lines_color,antialiased=True,lw=lw)
                else:
                    ax.plot(X,Y,c=clr,antialiased=True,lw=lw)
        if plot_rays:
            # Plot the refence lines
            alpha = np.pi/10
            #ax.axis(axs)
            if reference == 'center_of_mass':
                axs = (-xc/scaling,(self.dimY-xc)/scaling,(self.dimX-yc)/scaling,-yc/scaling)
                polar.plot_rays(center=(0,0),step_angle=alpha,ax=ax,axis_limits=axs)    
            else:
                axs = (0,self.dimY/scaling,self.dimX/scaling,0)
                polar.plot_rays(center=(xc,yc),step_angle=alpha,ax=ax,axis_limits=axs)
        if invert_y_axis:
            ax.invert_yaxis()
            #ax.set_aspect('equal')
        if title:
            ax.set_title(title)
        ax.axis('equal')
        plt.show()
        self.is_find_contours = True
        return
        
    def rescale_contours(self,invert_y_axis=True,fig=None,ax=None):
        if not self.is_find_contours:
            print("Please, run find_contours first")
            return
        if fig is None:
            fig = plt.figure(figsize=self._figColorImage.get_size_inches())
            ax = fig.gca()
        else:
            plt.figure(fig.number)
            if ax is None:
                ax = fig.gca()
        switches = sorted(self.bubbles.keys())
        rescale_factors = {}
        area0 = float(np.sum(self.bubbles[switches[-1]]))
        print(area0)
        for switch in switches[:-1]:
            resize_factor = (np.sum(self.bubbles[switch])/area0)**0.5
            print(resize_factor)
            yc,xc = self.centers_of_mass[switch]
            axs = (-xc,self.dimY-xc,self.dimX-yc,-yc)
            cnts = self.contours[switch]
            X,Y = (cnts[:,1]-xc)/resize_factor, (cnts[:,0]-yc)/resize_factor
            ax.plot(X,Y,antialiased=True,lw=1,label=switch)
        if invert_y_axis:
            ax.invert_yaxis()
            ax.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
        return


    def waiting_times_map(self, is_plot=True, log_norm=True):
        """
        calculate and plot the waiting time matrix
        """
        from matplotlib.colors import LogNorm
        if not self.is_find_contours:
            print("You have to run find_contours first")
        x_points = np.array([])
        y_points = np.array([])
        # Collect the x and y of the contours
        for k in self.contours:
            cnt = self.contours[k]
            x, y = cnt[:,0], cnt[:,1]
            x_points = np.append(x, x_points)
            y_points = np.append(y, y_points)
        n_images, rows, cols = self.shape
        bins = (np.arange(0,rows,.5), np.arange(0,cols,.5))
        waiting_times_hist, xedges, yedges = np.histogram2d(x_points, y_points, bins=bins)
        wt_masked = np.ma.masked_where(waiting_times_hist==0, waiting_times_hist)
        if is_plot:
            if log_norm:
                norm = LogNorm()
            else:
                norm = 'None'
            fig1 = plt.figure()
            plt.imshow(waiting_times_hist,extent=[yedges[-1], yedges[0], xedges[0], xedges[-1]], 
                norm=norm, interpolation='nearest')
            fig2 = plt.figure()
            plt.imshow(wt_masked,extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                norm=norm, interpolation='nearest')
        self.waiting_times_hist = waiting_times_hist
        return


    def pickle_switchMap2D(self, data=None, mainDir=None):
        """
        save the array ._switchTimes2D in pickle format in the same 
        folder where the images are taken from
        """
        if data is None:
            data = self._switchTimes2D
        
        filename = os.path.join(self._mainDir,'switchMap2D.pkl')
        with open (filename, 'wb') as f:
            pickle.dump(data, f)
        return      