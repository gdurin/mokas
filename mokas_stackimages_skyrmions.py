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
from gpuSkyrmions import gpuSkyrmions #gpuSwitchtimeBackup
from PIL import Image
import tifffile
import getLogDistributions as gLD
import getAxyLabels as gal
import rof
import mahotas
#import h5py
import tables
import mokas_polar as polar
import mokas_collect_images as collect_images
from mokas_colors import get_cmap, getKoreanColors, getPalette
import mokas_gpu as mkGpu
from mokas_domains import Domains
import pickle
from skimage import measure
import warnings

try:
    #from bokeh.models import (BasicTicker, Circle, ColumnDataSource, DataRange1d,
    #                      Grid, LinearAxis, PanTool, Plot, WheelZoomTool,)
    import bokeh.models as bkm
    import bokeh.plotting as plk
    from bokeh.models import Label, DataRange1d, Range1d
    from bokeh.models.annotations import Title
    import bokeh.palettes as palettes
    from bokeh.transform import factor_cmap, linear_cmap
    import colorcet as cc
    is_bokeh = True
except:
    is_bokeh = False

SPACE = "###"

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


class StackImagesSkyrmions:
    """
    Load and analyze a sequence of images
    as a multi-dimensional scipy 3D array.
    The k-th element of the array (i.e. myArray[k])
    is the k-th image of the sequence.
    This class is adapted from mokas_stackimages.py

    Parameters:
    ----------------
    direc : string
        Directory of the image files

    pattern : string
        Pattern of the input image files,
        as for instance "Data1-*.tif"

    firstImage, lastIm : int, opt
       first and last image (included) to be loaded
       These numbers refer to the numbers of the filenames

    hdf5_signature : dict
        A dictionary containing the essential data to identify a measurement
        if None, the code does not save data on a hdf5

    visualization_library : str
        'mpl': matplotlib (default)
        'bokeh' : Bokeh (https://docs.bokeh.org/en/latest/index.html)
    """

    def __init__(self, direc, pattern,
                 firstIm=None, lastIm=None, 
                 kernel_half_width_of_ones = 10,
                 fillValue=-1, 
                 hdf5_use=False, 
                 hdf5_signature=None,
                 visualization_library = 'mpl'):
        """
        Initialized the class
        """
        # Initialize variables
        self._dir = direc
        self._switches = None
        self._switched = None # Contains all the pixel positions (as tuple) where a switch happens somewhere in the video
        self._threshold = None
        self.imagesRange = (firstIm, lastIm)
        self.imageDir = None
        self.pattern = pattern
        self.fillValue = fillValue
        if not lastIm:
            lastIm = -1
        self.visualization_library = visualization_library
        
        # Check path
        if not os.path.isdir(self._dir):
            print("Please check you dir %s" % self._mainDir)
            print("Path not found")
            sys.exit()

        #############################################################################
        out = collect_images.images2array(self._dir, pattern, firstIm, lastIm, 
                                    hdf5_use=hdf5_use, hdf5_signature=hdf5_signature)
        if hdf5_use:
            self.Array, self.imageNumbers, self.hdf5 = out
        else:
            self.Array, self.imageNumbers = out
        ##############################################################################
        self.shape = self.Array.shape
        self.n_images, self.dimX, self.dimY = self.shape
        print("%i image(s) loaded, of %i x %i pixels" % (self.n_images, self.dimX, self.dimY))

        self.convolSize = kernel_half_width_of_ones
        # Make a kernel as a step-function
        self.kernel_half_width_of_ones = kernel_half_width_of_ones
        self.visualization_library = visualization_library

    def __get__(self):
        """
        redefine the get
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

        Show the n-th image where n = imageNumber

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

    def pixelSwitchesSequence(self,pixel=(0,0)):
        """
        pixelSwitchesSequence(pixel)

        Returns the temporal sequence of the switches of a pixel

        Parameters:
        ---------------
        pixel : tuple
           The (x,y) pixel of the image, as (row, column)
        """
        x,y = self._pixel2rowcol(pixel)
        return self._switches[:,x,y]

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
            Show the kernels (step functions) in the plot
        """
        
        figTimeSeq = plt.figure()
        ax = plt.gca()
        # Plot the temporal sequence first
        pxt = self.pixelTimeSequence(pixel)
        label_pixel = "(%i, %i)" % (pixel[0], pixel[1])
        ax.plot(self.imageNumbers,pxt,'-o', label=label_pixel)
        # Add the calculus of GPU
        row, col = self._pixel2rowcol(pixel)
        for switch, step in enumerate(self._switches[:, row, col]):
            if step != 0 : # for each detected step
                # Plot the step-like function
                l0 = self.kernel_half_width_of_ones
                pxt_average = np.mean(pxt[switch - l0:switch + l0]) # to check
                print("switch at %i, gray level change = %i" % (switch, step))
                kernel0 = np.ones(self.kernel_half_width_of_ones)
                kernel0 = np.concatenate((-kernel0, kernel0))
                kernel = kernel0 * step / 2 + pxt_average
                # This is valid for the step kernel ONLY
                x =  np.arange(switch - l0, switch + l0)
                ax.plot(x, kernel, '-o')
        plt.legend()
        plt.show()

    def getSwitches(self, isCuda=True, kernel=None, device=0, threshold=None, showHist=True):
        """
        Calculate the switches (time given by the position
        in the array, level given by the value)
        for each pixel in the image sequence.
        It calculates self._switches

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
                mySkyrmion = gpuSkyrmions(stack32, convolSize=self.convolSize, current_dev=current_dev, ctx=ctx)
                switches = mySkyrmion.getSwitchesSkyrmions(current_dev=current_dev, ctx=ctx, threshold=threshold)
            else:
                nsplit = int(float(need_mem)/free_mem_gpu) + 1
                print("Splitting images in %d parts..." % nsplit)
                stack32s = np.array_split(stack32, nsplit, 1)
                print("Done")
                switches = np.array([])
                for k, stack32 in enumerate(stack32s):
                    print("Calculation split %i" % k)
                    #a = stack32.astype(np.int32)
                    mySkyrmion = gpuSkyrmions(stack32, convolSize=self.convolSize, current_dev=current_dev, ctx=ctx)
                    switches_tmp = mySkyrmion.getSwitchesSkyrmions(current_dev=current_dev, ctx=ctx, threshold=threshold)
                    if not k:
                        switches = switches_tmp
                    else:
                        switches = np.vstack((switches, switches_tmp))

            # Close device properly
            success = mkGpu.gpu_deinit(current_dev, ctx)
            if not success:
                print("There is a problem with the device %i" % device)
            print('Analysing done in %f seconds' % (time.time()-startTime))


            if showHist:
                plt.hist(switches[switches!=0], bins=100)
                plt.show()

            self._switches = switches

        else:
            print("Error with PyCuda")

        return

    def computeSwitched(self):
        self._switched = []
        for i in range(self.dimX):
            for j in range(self.dimY):
                if np.count_nonzero(self._switches[:, i, j]) != 0:
                    self._switched.append((i, j))

    def getSwitched(self):
        """
        Calculates self._switched which is an
        array containing all the pixel positions
        (as tuples) where a shitch happens at
        anytime in the video
        """
        if self._switched is None:
            self.computeSwitched()

        return self._switched

    def plotSwitched(self):
        sw = [[len(self._switches[:, i, j].nonzero()[0]) == 0 for j in range(self.dimY)] for i in range(self.dimX)]

        plt.imshow(sw)
        plt.show()