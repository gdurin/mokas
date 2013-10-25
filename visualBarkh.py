import os, sys, glob, time, re
import scipy
import scipy.ndimage as nd
import scipy.signal as signal
import scipy.stats.stats
import numpy as np
import numpy.ma as ma
import matplotlib as mpl
import matplotlib.pyplot as plt
from gpuSwitchtime import get_gpuSwitchTime
from colorsys import hsv_to_rgb
import Image
import getLogDistributions as gLD
reload(gLD)
import getAxyLabels as gal
reload(gal)
import rof
import mahotas
#import h5py
import tables

def denoise(im):
    U,T = rof.denoise(im,im)
    return np.asarray(U, dtype='int32')

filters = {'gauss': nd.gaussian_filter, 'fouriergauss': nd.fourier_gaussian,
           'median': nd.median_filter, 'wiener': signal.wiener, 'rof':None}


# Check if pycuda is available
try:
    import pycuda.driver as driver
    isPyCuda = True
    free_mem_gpu, total_mem_gpu = driver.mem_get_info()
except:
    isPyCuda = False

# Load scikits modules if available
try:
    from skimage.filter import tv_denoise
    isTv_denoise = True
    filters['tv'] = tv_denoise
except:
    isTv_denoise = False

try:
    import skimage.io as im_io
    from skimage import filter, measure
    isScikits = True
except:
    isScikits = False
    print "*********** There is no Scikits installed"
    sys.exit()

if isScikits:    
    class Imread_convert():
        """
        Class to read images with PIL
        with check in the available image mode
        """
        def __init__(self, mode, resize_factor=None):
            self.mode = mode
            self.resize_factor = resize_factor

        def __call__(self, f):
            #if self.mode != "I;16":
                #return im_io.imread(f).astype(np.int16)
            #else:
            im = Image.open(f)
            sizeX, sizeY = im.size
            if self.resize_factor:
                sizeX, sizeY = sizeX/self.resize_factor, sizeY/self.resize_factor
                im = im.resize((sizeX, sizeY), Image.NEAREST)
            imageList = list(im.getdata())
            return np.asanyarray(imageList).reshape(sizeY, sizeX)

    plugins = im_io.plugins()
    keys = plugins.keys()
    mySeq = ['gtk', 'pil', 'matplotlib', 'qt']
    for plugin in mySeq:
        if plugin in keys:
            use = plugin

    try:
        im_io.use_plugin('pil', 'imread')
    except:
        print("No plugin available between %s" % str(mySeq))
else:
    print "Scikits.image not available"



# Adjust the interpolation scheme to show the images
mpl.rcParams['image.interpolation'] = 'nearest'

# Load ral colors
# See http://www.ralcolor.com/
ral_colors = []
f = open("ral_color_selected2.txt")
rals = f.readlines()
f.close()
ral_colors = [r.split()[-3:] for r in rals]
ral_colors = [[int(rgb) for rgb in r.split()[-3:]] for r in rals]
ral_colors = np.array(ral_colors)
ral_colors = np.concatenate((ral_colors, ral_colors / 2))
ral_colors = np.concatenate((ral_colors, ral_colors))


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
       list of subdirectories so save the data into (hdf5 style)

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
    """

    def __init__(self, subDirs, pattern, resize_factor=None,
                 firstIm=None, lastIm=None, 
                 filtering=None, sigma=None, halfWidthKrn=8):
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
        self._figColorImage = None
        self._figColorImage2 = None
        self.isConnectionRawImage = False
        self.figRawAndCalc = None
        self.imagesRange = (firstImage, lastIm)
        self.imageDir = None
        if not lastIm:
            lastIm = -1
        # Make a kernel as a step-function
        # Good for Black_to_White change of grey scale
        self.kernel = np.array([-1] * halfWidthKrn + [1] * halfWidthKrn)
        # Good for White_to_Black change of grey scale
        self.kernel0 = np.array([-1] * halfWidthKrn + [0] + [1] * halfWidthKrn)
        
        if not os.path.isdir(self._mainDir):
            print("Please check you dir %s" % self._mainDir)
            print("Path not found")
            sys.exit()
        # Collect the list of images in mainDir
        s = "(%s|%s)" % tuple(pattern.split("*"))
        patternCompiled = re.compile(s)
        # Load all the image filenames
        imageFileNames = glob.glob1(self._mainDir, pattern)
        # Sort it with natural keys
        imageFileNames.sort(key=natural_key)

        if not len(imageFileNames):
            print "ERROR, no images in %s" % self._mainDir
            sys.exit()
        else:
            print "Found %d images in %s" % (len(imageFileNames), self._mainDir)
        # Search the number of all the images given the pattern above
        _imNumbs = [int(patternCompiled.sub("", fn)) for fn in imageFileNames]
        # Search the indexes of the first and the last images to load
        if firstIm is None:
            firstIm = _imNumbs[0]
        if lastIm < 0:
            lastIm = len(_imNumbs) + lastIm + firstIm
        try:
            iFirst, iLast = _imNumbs.index(firstIm), _imNumbs.index(lastIm)
        except:
            i0, i1 = _imNumbs[0], _imNumbs[-1]
            out = (i0, i1, firstIm, lastIm)
            print("Error: range of the images is %s-%s (%s-%s chosen)" % out)
            sys.exit()
        # Save the list of numbers of the images to be loaded
        self.imageNumbers = _imNumbs[iFirst:iLast + 1]
        self.imageIndex = []
        print "First image: %s" % imageFileNames[iFirst]
        print "Last image: %s" % imageFileNames[iLast]
        # Check the mode of the images
        fname = os.path.join(self._mainDir, imageFileNames[iFirst])
        imageOpen = Image.open(fname)
        imageMode = imageOpen.mode
        imageSizeX, imageSizeY = imageOpen.size
        imread_convert = Imread_convert(imageMode, resize_factor)
        # Load the images
        print "Loading images: "
        load_pattern = [os.path.join(self._mainDir,ifn) for 
                        ifn in imageFileNames[iFirst:iLast + 1]]

        if isScikits:
            imageCollection = im_io.ImageCollection(load_pattern, 
                                                    load_func=imread_convert)
        else:
            sys.exit()
        if filtering:
            filtering = filtering.lower()
            if filtering not in filters:
                print "Filter not available"
                sys.exit()
            else:
                print "Filter: %s" % filtering
                # Renormalize the value of the sigma
                # Reference value : 1.5 for 500x500 pixels image
                sigma = sigma * imageSizeX/500 # Good for squared images
                if filtering == 'wiener':
                    sigma = [sigma, sigma]
                #self.Array = np.dstack([np.int16(filters[filtering](im,sigma)) for im in imageCollection])
                if filtering == 'rof':
                    self.Array = np.array(tuple([denoise(im) for im in imageCollection]))
                else:
                    self.Array = np.array(tuple([np.int16(filters[filtering](im,sigma)) for im in imageCollection]))
        else:
            self.Array = np.array(tuple([im for im in imageCollection]))
        self.shape = self.Array.shape
        self.n_images, self.dimX, self.dimY = self.shape
        print "%i image(s) loaded, of %i x %i pixels" % (self.n_images, self.dimX, self.dimY)

        # Check for the grey direction
        grey_first_image = np.mean([scipy.mean(self.Array[k,:,:].flatten()) for k in range(4)])
        grey_last_image = np.mean([scipy.mean(self.Array[-k,:,:].flatten()) for k in range(1,5)])
        print "grey scale: %i, %i" % (grey_first_image, grey_last_image)
        if grey_first_image > grey_last_image:
            self.kernel = -self.kernel
            self.kernel0 = -self.kernel0

    def __get__(self):
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
        ns = self.imageNumbers
        try:
            return ns.index(n)
        except:
            print "Image number %i is out of the range (%i,%i)" % (n, ns[0], ns[-1])
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

    def _getLevels(self, pxTimeSeq, switch, kernel='step'):
        """
        _getLevels(pxTimeSeq, switch, kernel='step')

        Internal function to calculate the gray level before and
        after the switch of a sequence, using the kernel

        Parameters:
        ---------------
        pxTimeSeq : list
            The sequence of the gray level for a given pixel.
        switch : number, int
            the position of the switch as calculated by getSwitchTime
        kernel : 'step' or 'zero'
           the kernel of the step function

        Returns:
        -----------
        levels : tuple
           Left and right levels around the switch position
        """
        width = self._getWidth()

        # Get points before the switch
        if width == 'small':
            halfWidth = len(self.kernel)/2
            lowPoint = switch - halfWidth - 1*(kernel=='zero')
            if lowPoint < 0:
                lowPoint = 0
            highPoint = switch + halfWidth
            if highPoint > len(pxTimeSeq):
                highPoint = len(pxTimeSeq)
        elif width == 'all':
            lowPoint, highPoint = 0, len(pxTimeSeq)
        else:
            print 'Method not implement yet'
            return None
        leftLevel = np.int(np.mean(pxTimeSeq[lowPoint:switch - 1*(kernel=='zero')])+0.5)
        rigthLevel = np.int(np.mean(pxTimeSeq[switch:highPoint])+0.5)
        levels = leftLevel, rigthLevel
        return levels


    def pixelTimeSequence(self,pixel=(0,0)):
        """
        pixelTimeSequence(pixel)

        Returns the temporal sequence of the gray level of a pixel

        Parameters:
        ---------------
        pixel : tuple
           The (x,y) pixel of the image, as (row, column)
        """
        x,y = pixel
        return self.Array[:,x,y]

    def showPixelTimeSequence(self,pixel=(0,0),newPlot=False):
        """
        pixelTimeSequenceShow(pixel)

        Plot the temporal sequence of the gray levels of a pixel;

        Parameters:
        ---------------
        pixel : tuple
            The (x,y) pixel of the image, as (row, column)
        newPlot : bool
            Option to open a new frame or use the last one
        """
        width = self._getWidth()
        # Plot the temporal sequence first
        pxt = self.pixelTimeSequence(pixel)
        if not self._figTimeSeq or newPlot==True:
            self._figTimeSeq = plt.figure()
        else:
            self._figTimeSeq
        plt.plot(self.imageNumbers,pxt,'-o')
        # Add the two kernels function
        kernels = [self.kernel, self.kernel0]
        for k,kernel in enumerate(['step','zero']):	
            switch, (value_left, value_right) = self.getSwitchTime(pixel, useKernel=kernel)
            print "switch %s, Kernel = %s" % (kernel, switch)
            print ("gray level change at switch = %s") % abs(value_left-value_right)
            if width == 'small':
                halfWidth = len(kernels[k])/2
                x0,x1 = switch - halfWidth - 1*(k==1), switch + halfWidth
                x = range(x0,x1)
                n_points_left = halfWidth
                n_points_rigth = halfWidth
            elif width=='all':
                #x = range(len(pxt))
                x = self.imageNumbers
                n_points_left = switch - 1 * (k==1)
                n_points_rigth = len(pxt) - switch
            y = n_points_left * [value_left] + [(value_left+value_right)/2.] * (k==1) + n_points_rigth * [value_right]

            plt.plot(x,y)
        plt.draw()
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
                    #leftLevel = np.int(np.mean(pxTimeSeq[0:switch])+0.5)
                    #rightLevel = np.int(np.mean(pxTimeSeq[switch+1:])+0.5)
                    #middle = (leftLevel+rightLevel)/2
                    #rightLevelStep = np.int(np.mean(pxTimeSeq[switchStepKernel+1:])+0.5)
                    #if abs(pxTimeSeq[switch]-middle)>abs(pxTimeSeq[switch]-rightLevelStep):
                        #switch = switchStepKernel
                    #switch = (switch-1)*(pxTimeSeq[switch]<middle)+switch*(pxTimeSeq[switch]>=middle)
                #switch = switchStepKernel * (minStepKernel<=minZeroKernel/1.1) + switchZeroKernel * (minStepKernel >minZeroKernel/1.1)
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
            plt.imshow(self._imDiff(imNumbers, invert),plt.cm.gray)
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
                print "Error: sequence not valid"
                return
        for i in imRange:
            im = self._imDiff((i+1,i))
            imPIL = scipy.misc.toimage(im)
            fileName = "imDiff_%i_%i.tif" % (i+1,i)
            print fileName
            imageFileName = os.path.join(dirSeq, fileName)
            imPIL.save(imageFileName)

    def getSwitchTimesAndSteps(self, isCuda=True):
        """
        Calculate the switch times and the gray level changes
        for each pixel in the image sequence.
        It calculates:
        self._switchTimes
        self._switchSteps
        """
        startTime = time.time()
        # ####################
        if isPyCuda and isCuda:
            kernel32 = np.asarray(self.kernel, dtype=np.int32)
            stack32 = np.asarray(self.Array, dtype=np.int32)
            need_mem = 2 * stack32.nbytes
            free_mem_gpu, total_mem_gpu = driver.mem_get_info()
            free_mem_gpu = 0.9*free_mem_gpu
            if need_mem < free_mem_gpu:
                switchTimes, switchSteps = get_gpuSwitchTime(stack32, kernel32, device=1)
            else:
                nsplit = int(float(need_mem)/free_mem_gpu) + 1
                print "Splitting images in %d parts..." % nsplit
                stack32s = np.array_split(stack32, nsplit, 1)
                switchTimes = np.array([])
                switchSteps = np.array([])
                for k, stack32 in enumerate(stack32s):
                    a = stack32.astype(np.int32)
                    switch, step = get_gpuSwitchTime(a, kernel32, device=1)
                    if not k:
                        switchTimes = switch
                        switchSteps = step
                    else:
                        switchTimes = np.vstack((switchTimes, switch))
                        switchSteps = np.vstack((switchSteps, step))
            self._switchSteps = switchSteps.flatten()
            self._switchTimes = self.imageNumbers[0] + switchTimes.flatten() + 1
            print 'Analysing done in %f seconds' % (time.time()-startTime)
        else:
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
                        print x,y
                    switchTimes.append(switch)
                    switchSteps.append(grayChange)
            print "\n"
            self._switchTimes = np.asarray(switchTimes)
            self._switchSteps = np.asarray(switchSteps)
        self._isColorImage = True
        self._isSwitchAndStepsDone = True
        return

    def _getSwitchTimesArray(self, isFirstSwitchZero=False, fillValue=-1):
        """
        _getSwitchTimesArray()

        Returns the array of the switch times
        considering a threshold in the gray level change at the switch

        Parameters:
        ----------------
        threshold : int
            The miminum value of the gray level change at the switch
        isFirstSwitchZero : bool
            Put the first switch equal to zero, useful to set the colors
            in a long sequence of images where the first avalanche
            occurs after many frames
        fillValue : number, int
            The value to set in the array for the non-switching pixel (below the threshold)
            -1 is use as the last value of array when used as index (i.e. with colors)
        """

        self.isPixelSwitched = self._switchSteps >= self._threshold
        maskedSwitchTimes = ma.array(self._switchTimes, mask = ~self.isPixelSwitched)
        # Move to the first switch time if required
        if isFirstSwitchZero:
            maskedSwitchTimes = maskedSwitchTimes - self.min_switch
        # Set the non-switched pixels to use the last value of the pColor array, i.e. noSwitchColorValue
        switchTimes = maskedSwitchTimes.filled(fillValue) # Isn't it fantastic?
        return switchTimes

    def _getKoreanColors(self, switchTime, n_images=None):
        """
        Make a palette in the korean style
        """
        if not n_images:
            n_images = self.n_images
        n = float(switchTime)/float(n_images)*3.
        R = (n<=1.)+ (2.-n)*(n>1.)*(n<=2.)
        G = n*(n<=1.)+ (n>1.)*(n<=2.)+(3.-n)*(n>2.)
        B = (n-1.)*(n>=1.)*(n<2.)+(n>=2.)
        R, G, B = [int(i*255) for i in [R,G,B]]
        return R,G,B

    def _isColorImageDone(self,ask=True):
        print "You must first run the getSwitchTimesAndSteps script: I'll do that for you"
        if ask:
            yes_no = raw_input("Do you want me to run the script for you (y/N)?")
            yes_no = yes_no.upper()
            if yes_no != "Y":
                return
        self.getSwitchTimesAndSteps()
        return

    def _getPalette(self, palette='ral', noSwitchColor='black'):
        """
        get the color palette
        """           
        if self._koreanPalette is None:
            # Prepare the Korean Palette
            self._koreanPalette = np.array([self._getKoreanColors(i, self._nImagesWithSwitch) 
                                            for i in range(self._nImagesWithSwitch)])

        if palette == 'korean':
            pColor = self._koreanPalette
        elif palette == 'randomKorean':
            pColor = np.random.permutation(self._koreanPalette)
        elif palette == 'random':
            pColor = np.random.randint(0, 256, self._koreanPalette.shape)
        elif palette == 'randomHue':
            # Use equally spaced colors in the HUE weel, and
            # then randomize
            pColor = [hsv_to_rgb(j/float(self._nImagesWithSwitch),1, 
                                 np.random.uniform(0.75,1)) 
                                 for j in range(self._nImagesWithSwitch)]
            pColor = np.random.permutation(pColor)
        elif palette == 'ral':
            pColor = np.random.permutation(ral_colors)[:self._nImagesWithSwitch]
        if noSwitchColor == 'black':
            noSwitchColorValue = 3*[0]
        elif noSwitchColor == 'white':
            noSwitchColorValue = 3*[255]
        return np.concatenate(([noSwitchColorValue], pColor))/255.        

    def _getColorImage(self, palette='korean', noSwitchColor='black'):
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
        print "Avalanches occur between frame %i and %i" % (self.min_switch, self.max_switch)
        self._nImagesWithSwitch = self.max_switch - self.min_switch + 1
        print "Gray changes are between %s and %s" % (min(self._switchSteps), max(self._switchSteps))

        # Calculate the colours, considering the range of the switch values obtained
        self._pColors = self._getPalette(palette, noSwitchColor)
        self._colorMap = mpl.colors.ListedColormap(self._pColors, 'pColorMap')
        # Calculate the switch time Array (2D) considering the threshold and the start from zero
        self._switchTimes2D = self._getSwitchTimesArray(True, -1).reshape(self.dimX, self.dimY)
        self._switchTimesUnique = np.unique(self._switchTimes2D) + self.min_switch
        return


    def showColorImage(self, threshold=0, palette='random', 
                       noSwitchColor='black', ask=False):
        """
        showColorImage([threshold=0, palette='random', noSwitchColor='black', ask=False])

        Show the calculated color Image of the avalanches.
        Run getSwitchTimesAndSteps if not done before.

        Parameters
        ---------------
        threshold: integer, optional
            Defines if the pixel switches when gray_level_change >= threshold
        palette: string, required, default = 'korean'
            Choose a palette between 'korean', 'randomKorean', and 'random'
            'randomKorean' is a random permutation of the korean palette
            'random' is calculated on the fly, so each call of the method gives different colors
        noSwithColor: string, optional, default = 'black'
            background color for pixels having gray_level_change below the threshold
        """
        # set the threshold
        self._threshold = threshold
        # Calculate the Color Image
        self._getColorImage(palette, noSwitchColor)
        # Prepare to plot
        self._figColorImage = self._plotColorImage(self._switchTimes2D, 
                                                   self._colorMap, 
                                                   self._figColorImage)
        # Plot the histogram
        self._plotHistogram(self._switchTimes)
        # Count the number of the switched pixels
        switchPixels = np.sum(self.isPixelSwitched)
        totNumPixels = self.dimX * self.dimY
        noSwitchPixels = totNumPixels - switchPixels
        swPrint = (switchPixels, switchPixels/float(totNumPixels)*100., 
                   noSwitchPixels, noSwitchPixels/float(totNumPixels)*100.)
        print "There are %d (%.2f %%) switched and %d (%.2f %%) \
                not-switched pixels" % swPrint
        #yes_no = raw_input("Do you want to save the image (y/N)?")
        #yes_no = yes_no.upper()
        #if yes_no == "Y":
            #fileName = raw_input("Filename (ext=png): ")
            #if len(fileName.split("."))==1:
                #fileName = fileName+".png"
            #fileName = os.path.join(self._mainDir,fileName)
            #imOut = scipy.misc.toimage(self._colorImage)
            #imOut.save(fileName)

    def _call_pixel_switch(self,x,y):
        x, y = int(y+0.5), int(x+.5)
        if x >= 0 and x < self.dimX and y >= 0 and y < self.dimY:
            index = x * self.dimY + y
            s = "pixel (%i,%i) - switch at: %i, gray step: %i" % \
                (x, y, self._switchTimes[index], self._switchSteps[index])
            return s
        else:
            return None

    def _plotColorImage(self, data, colorMap, fig=None):
        """
        if the caption is not shown, just enlarge the image
        as it depends on the length of the string retured by
        _call_pixel_switch
        """
        if fig == None:
            fig = plt.figure()
            fig.set_size_inches(8,7,forward=True)
            ax = fig.add_subplot(1,1,1)
        else:
            fig = plt.figure(fig.number)
            ax = fig.gca()
        ax.format_coord = lambda x,y: self._call_pixel_switch(x, y)
        #ax.set_title(title)
        # Sets the limits of the image
        extent = 0, self.dimY, self.dimX, 0
        plt.imshow(data, colorMap, norm=mpl.colors.NoNorm(), extent=extent)
        title = "Avalanches from image %i to image %i" % self.imagesRange
        plt.title(title)
        plt.show()
        return fig

    def _plotHistogram(self, data):
        rng = np.arange(self.min_switch-0.5, self.max_switch+0.5)
        if self._figHistogram is None:
            self._figHistogram = plt.figure()
            #ax = self._figHistogram.add_subplot(1,1,1)
        else:
            plt.figure(self._figHistogram.number)
            plt.clf()
        self.N_hist, self.bins_hist, patches = plt.hist(data, rng)
        ax = plt.gca()
        ax.format_coord =  lambda x,y : "image Number: %i,\
                    avalanche size (pixels): %i" % (int(x+0.5), int(y+0.5))
        plt.xlabel("image number")
        plt.ylabel("Avalanche size (pixels)")
        for i in range(len(self.N_hist)):
            patches[i].set_color((tuple(self._pColors[i])))
        plt.show()
        return None

    def saveColorImage(self,fileName,threshold=None, 
                       palette='korean',noSwitchColor='black'):
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


    def imDiffCalculated(self,imageNum,haveColors=True):
        """
        Get the difference in BW between two images imageNum and imageNum+1
        as calculated by the self._colorImage
        """
        if not self._isColorImage:
            self._isColorImageDone(ask=False)
        imDC = (self._switchTimes2D==imageNum)*1
        if haveColors:
            imDC = scipy.array(imDC,dtype='int16')
            structure = [[0, 1, 0], [1,1,1], [0,1,0]]
            l, n = nd.label(imDC,structure)
            im_io.imshow(l,plt.cm.prism)
        else:
            # Normalize to a BW image
            self.imDiffCalcArray = imDC*255
            scipy.misc.toimage(self.imDiffCalcArray).show()
        return None

    def _show_next_raw_image(self, event):
        """
        show the next row image
        calling showRawAndCaldImages with the
        next available image
        """
        key = str(event.key).lower()
        if key != 'n' and key != 'p' and key!= "a":
            return
        if key == "a":
            outDir = os.path.join(imArray._mainDir, "Results")
            if not os.path.isdir(outDir):
                os.mkdir(outDir)
            fname = "Raw_and_calc_%s.pdf" % self.nImage
            fname = os.path.join(outDir, fname)
            self.figRawAndCalc.savefig(fname)
            return
        if self.nImage == self._switchTimesUnique[-1] and key == 'n':
            print "No more images available"
            return
        elif self.nImage == self._switchTimesUnique[0] and key == 'p':
            print "No previous images available"
            return
        step = 1*(key=='n') -1*(key=='p')
        n = self._switchTimesUnique[np.nonzero(self._switchTimesUnique == self.nImage)[0][0] + step]
        self.showRawAndCalcImages(n, isTwoImages=self.isTwoImages)
        return
    
    def saveRawAndCalcImages(self, frmt='pdf', isTwoImages=False):
        """
        frmt: string
        Output format (def. pdf)
        """
        outDir = os.path.join(imArray._mainDir, "Results")
        if not os.path.isdir(outDir):
            os.mkdir(outDir)
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
            print "Saving image %i" % image
        return

    def label(self, images):
        """
        calculate the clusters using
        the mahotas code
        """
        if self.structure == None:
            self.structure = np.asanyarray([[0, 1, 0], [1,1,1], [0,1,0]])
        return mahotas.label(images, self.structure)
        


    def showRawAndCalcImages(self, nImage=None, preAvalanches=True, isTwoImages=False):
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
        show the previous calculated avalanche in white
        
        isTwoImages: bool
        if True shows only two images, the raw+1 and the calculated one
        """
        if self._switchTimes == None:
            print("Need to calculate the color image first")
            return
        if nImage == None:
            self.nImage = self._switchTimesUnique[0]
            print "Starting from image %i" % self.nImage
        elif nImage not in self._switchTimesUnique:
            print "No switch there"
            return
        else:
            self.nImage = nImage

        self.isTwoImages = isTwoImages

        if self.figRawAndCalc:
            plt.figure(self.figRawAndCalc.number)
        else:
            self.figRawAndCalc = plt.figure()
            if isTwoImages:
                self.figRawAndCalc.set_size_inches((10,5),forward=True)
                self.figRawAndCalc.set_facecolor('black')
            else:
                self.figRawAndCalc.set_size_inches((20,10),forward=True)
        # Prepare the color map of calculated avalanches
        switchTimes_images = self._getSwitchTimesArray(self._threshold, fillValue=0).\
            reshape(self.dimX, self.dimY) == self.nImage
        contours = measure.find_contours(switchTimes_images*100, 1)
        cl = self._pColors[self.nImage - self.min_switch]
        #myMap = mpl.colors.ListedColormap([(0,0,0),cl],'mymap',2)
        mapGreyandBlack = mpl.colors.ListedColormap([(0.75,0.75,0.75),(0,0,0)],'mymap',2) # in grey and black      
        plt.clf()
        # Plot the two raw images first
        if isTwoImages:
            ax1 = plt.subplot(1,2,1)
            plt.imshow(self[self.nImage], plt.cm.gray)
            ax1.xaxis.set_visible(False)
            ax1.yaxis.set_visible(False)
        else:
            for i in range(2):
                plt.subplot(2,3,1+i)
                plt.imshow(self[self.nImage-1+i], plt.cm.gray)
                plt.title("Raw Image %i" % (self.nImage-1+i))
                plt.grid(color='blue', ls="-")
        plt.axis((0, self.dimX, self.dimY, 0))
        # Add the calculated avalanches to the first raw image
        if not isTwoImages:
            plt.subplot(2,3,3)
            plt.imshow(self[self.nImage-1], plt.cm.gray)
            # Use find_contours
            for n, contour in enumerate(contours):
                plt.plot(contour[:, 1], contour[:, 0], linewidth=1, color='r')
            plt.axis((0, self.dimX, self.dimY, 0))        
            plt.grid(color='blue',ls="-")
            plt.title("Raw Image %s + Calculated Aval." % (self.nImage-1))        
            # Show the raw image difference
            plt.subplot(2,3,4)
            plt.imshow(self._imDiff((self.nImage,self.nImage-1)), plt.cm.gray)
            plt.title("Diff. Image %s and Image %s" % (self.nImage, self.nImage-1))
            plt.grid(color='blue', ls="-")
            plt.axis((0, self.dimX, self.dimY, 0))
            
            # Show the raw difference images and the calculated avalanches
            ax1 = plt.subplot(2,3,5)
            plt.imshow(self._imDiff((self.nImage,self.nImage-1)), plt.cm.gray)
            # Use find_contours
            for n, contour in enumerate(contours):
                plt.plot(contour[:, 1], contour[:, 0], linewidth=1, color='r')
            plt.axis((0, self.dimX, self.dimY, 0))
            # Use canny filter
            #edges1 = filter.canny(switchTimes_images*1000, 1)
            #myMap = mpl.colors.ListedColormap([(1,1,1),(1,1,0)],'mymap',2)
            #plt.imshow(edges1, hold=True)
            plt.title("Raw Diff and Calculated Aval.")
            plt.grid(color='blue',ls="-")
        
        # Show the calculated avalanches only
        if isTwoImages:
            ax2 = plt.subplot(1,2,2)
        else:
            plt.subplot(2,3,6)
        im, n_clusters = self.label(switchTimes_images)
        myPalette = [(0.25,0.25,0.25)] + [hsv_to_rgb(j/float(n_clusters),1,1)
                                          for j in np.random.permutation(range(n_clusters))]
        if preAvalanches:
            white = self._getSwitchTimesArray(self._threshold, fillValue=0).\
                        reshape(self.dimX, self.dimY) < self.nImage
            im = im + white * max(im.flatten())
            myPalette = myPalette + [(1,1,1)] # Add white to the palette
        plt.imshow(im, mpl.colors.ListedColormap(myPalette))      
        #plt.imshow(switchTimes_images, mapGreyandBlack)
        if not isTwoImages:
            plt.grid(color='blue',ls="-")
            plt.title("Calculated Aval.")
        else:
            ax2.xaxis.set_visible(False)
            ax2.yaxis.set_visible(False)
        plt.axis((0, self.dimX, self.dimY, 0))

        plt.draw()
        print("Press: Next, Previous, sAve")
        size = np.sum(switchTimes_images)
        print "Avalanche %i: %i pixels (%.2f %% of the image)" % \
              (self.nImage, size, float(size)/(self.dimX*self.dimY)*100)
        if not self.isConnectionRawImage:
            plt.connect('key_press_event', self._show_next_raw_image)
            self.isConnectionRawImage = True
        return

    def ghostbusters(self, clusterThreshold = 15, 
                     showImages=False, imageNumber=None):
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
        # The scipt below needs fixing
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
            array_labels, n_clusters = self.label(im0)
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
            image3, n3 = self.label(new_array)
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
        self._plotHistogram(self._switchTimes)

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


    def getDistributions(self, log_step=0.2, edgeThickness=1, 
                         fraction=0.01, hold=False):
        """
        Calculates the distribution of avalanches and clusters

        Parameters:
        ---------------
        NN : int
        No of Nearest Neighbours around a pixel to consider two clusters
        as touching or not

        log_step: float
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
        #self.N_cluster = {}
        self.N_cluster = []
        self.dictAxy = {}
        self.dictAxy['aval'] = {}
        self.dictAxy['clus'] = {}
        a0 = scipy.array([],dtype='int32')
        #Define the number of nearest neighbourg

        # Find the direction of the avalanches (left <-> right, top <-> bottom)
        if not self.imageDir:
            self.imageDir = self._getImageDirection(self._threshold)
        print self.imageDir
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
            # Detect local clusters using scipy.ndimage method
            array_labels, n_labels = self.label(im0)
            # Make a list the sizes of the clusters
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
        D_x, D_y = gLD.logDistribution(self.D_cluster, log_step=log_step, 
                                       first_point=1., normed=True)
        P_x, P_y = gLD.logDistribution(self.D_avalanches, log_step=log_step, 
                                       first_point=1., normed=True)
        # Plots of the distributions
        plt.figure()
        plt.loglog(D_x,D_y,'o',label='cluster')
        plt.loglog(P_x,P_y,'v',label='avalanches')
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

    def _groupHdf5(self, fileHdf5, parent, group):
        """
        get the group or create if not exists
        parent can be a string or a group
        """
        if type(parent) == tables.group.Group:
            if parent.__contains__(group):
                gr = parent._f_getChild(group)
            else:
                gr = fileHdf5.createGroup(parent, group, group)
        else: 
            _group = os.path.join(parent, group)
            if not fileHdf5.__contains__(_group):
                print "Create group: %s" % _group
                gr = fileHdf5.createGroup(parent, group, group)
            else:
                gr = fileHdf5.getNode(_group)
        return gr
    
    def _nodeHdf5(self, fileHdf5, parent, node, data, data_type=None):
        """
        write or rewrite data in a node
        """
        # Check if data already exist
        if not type(parent) == tables.group.Group:
            pp = fileHdf5.getNode(parent)
        else:
            pp = parent
        if pp.__contains__(node):
            pp._g_remove(node)
        arr = fileHdf5.createArray(pp, node, data)
        if data_type == 'IMAGE':
            arr._v_attrs['CLASS'] = data_type
            arr.attrs['IMAGE_SUBCLASS'] = 'IMAGE_TRUECOLOR'
        return
    
    def _getMeasureData(self):
        measure_file = open(os.path.join(self._mainDir, "measure.txt"))
        lines = measure_file.readlines()
        lns = [line.strip() for line in lines if "[X]" in line]
        for i, camera in enumerate(['Zeiss', "Hamamatsu", "PicoStar"]):
            for j, line in enumerate(lns):
                if camera in line:
                    camera_n = i
                    camera_line = j
                    break
        sizes = re.findall("\d+x\d+", lns[camera_line+1])
        size = sizes[camera_n]
        return size

                   
    def saveHdf5(self):
        """
        Prepare a hdf5 file to
        save sequences of data with labels ('0000', '1000', etc)
        save images
        save experimental details
        """
        if not self._isDistributions:
            self.getDistributions()
        # Prepare the hdf5 file for saving results
        outDir = os.path.join(*self._subDirs[:1])
        mainMat = imArray._subDirs[0].split("/")[-1]
        f5filename = mainMat + "_results.h5"
        f5filename = os.path.join(outDir, f5filename)
        #f5 = h5py.File(f5filename, mode="a", title=mainMat+" results")
        if os.path.isfile(f5filename):
            f5 = tables.openFile(f5filename, mode="a")
        else:
            f5 = tables.openFile(f5filename, mode="a", title=mainMat+" results")
        # Prepare the subdir to welcome data
        fullPath = f5.rootUEP
        for subDir in self._subDirs[1:]:
            #gr = gr.create_group(subDir)
            fullPath = self._groupHdf5(f5, fullPath, subDir)
        # Add the measurements data to the seq group
        size = self._getMeasureData()
        fullPath._v_attrs['IMAGE_SIZE'] = size
        # Save the data
        for label in self.dictAxy:
            # create the 'clus' and 'aval' groups
            labelGroup = self._groupHdf5(f5, fullPath, label)
            for Axy in self.dictAxy[label]:
                data = self.dictAxy[label][Axy]
                #subGroup.create_dataset(Axy, data=data)
                aval_type = "A%s" % Axy
                # Check if data already exist
                self._nodeHdf5(f5, labelGroup, aval_type, data)
        # Save images as 24 bit images
        imagesGroup = self._groupHdf5(f5, fullPath, "images")
        colors = np.array(self._pColors*255, dtype=np.int16)
        imColor = colors[self._switchTimes2D]
        self._nodeHdf5(f5, imagesGroup, "raw_data", self._switchTimes2D)
        self._nodeHdf5(f5, imagesGroup, "lastmage", imColor, "IMAGE")
        palettes = ['korean', 'randomKorean', 'ral', 'random', 'randomHue']
        for palette in palettes:
            colors = self._getPalette(palette, 'black')
            imColor = colors[self._switchTimes2D]
            self._nodeHdf5(f5, imagesGroup, palette, imColor, "IMAGE")
        # Save the histogram
        hist = zip(self.bins_hist, self.N_hist)
        self._nodeHdf5(f5, imagesGroup, "hist", hist)
        f5.flush()
        f5.close()
        return

if __name__ == "__main__":
    if False:
        
        #mainDir, pattern, firstImage, lastIm = "/media/DATA/meas/MO/CoFe/50nm/20x/run5/", "Data1-*.tif", 280, 1029
        #mainDir, pattern, sequences = "/media/DATA/meas/Barkh/Films/CoFe/50nm/10x/run5/", "Data1-*.tif", [(1,1226)]
    
    
        rootDir = "/run/media/gf/DATA/meas/Barkh/Films/NiO_Fe"
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
            imArray = StackImages(subDirs, pattern, resize_factor=None,\
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
            #imArray1 = StackImages(mainDir, pattern, resize_factor=False,\
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
    elif True:
        rootDir = "/run/media/gf/DATA/meas/Simulation/Andrea/frames/"    
        subDirs = [rootDir, "", "", "", ""]
        pattern = "frame*.jpg"
        firstImage, lastIm = (500,1000)
        imArray = StackImages(subDirs, pattern, resize_factor=2,\
                              filtering='gauss', sigma=1.5,\
                              firstIm=firstImage, lastIm=lastIm)
        imArray.width='small'
        imArray.useKernel = 'step'
        imArray.imageDir = "Bottom_to_top"
        p2p = 5 # Pixel to pixel (linear) distance for cluster detection
        NN = 2*p2p + 1
        imArray.structure = np.ones((NN,NN))
        imArray.showColorImage(5,palette='ral')        




