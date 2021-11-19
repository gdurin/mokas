import sys, os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mokas_stackimages_skyrmions import StackImagesSkyrmions
import mahotas
from skimage import measure
import getLogDistributions as gLD
import mokas_events as mke
import mokas_cluster_methods as mcm
import mokas_cluster_distributions as mcd
import configparser
import numpy.fft as fft


class Skyrmions(StackImagesSkyrmions):
    """
    define a proper class to handle
    the sequence of images
    taken from skyrmions
    """
    def getPeriod(self, fromVid=True, positions=None, n_px=-1, showFFT=False, thresholdAmp=0.01):
        """
        Calculates the period (in number of images)
        of the breathing of the skyrmion

        Parameters:
        ----------------
        fromVid : boolean
            If True, get the period with the video directly, by FFT
            Otherwise, get the period with the interval between two consecutive positive steps in self._switches

        positions : array of tuples [(x1, y1), (x2, y2), ], optionnal
            Positions of the pixels where the period is calculated. The median of the calculated periods is returned.
            If None, we take into account n pixels from self._switched for the calculation, where n=n_px

        n_px : int, opt
            Numper of pixels to take into account for the calculation. If negative or 0, all the pixels are taken into
            account (form positions, if given, or from self._switched otherwise)

        showFFT : boolean, opt
            If True, plots the FFT (does nothing if fromVid=False)

        thresholdAmp : float between 0 and 1, opt
            When using the video (fromVid=True), we take the first peak (with non-zero frequency) of the FFT as the skyrmion frequency.
            This amplitude threshold is the threshold above which we consider that the signal is not noise.
            Note : 0.01 means "at least 1% of the highest peak of the FFT"

        """
        if(self._switches is None):
            print("You have to run getSwitches before...")
            return

        #get the positions where to compute the frequency
        if positions is None:
            positions = self.getSwitched()

        if 0 < n_px < len(positions):
            step = len(positions)/n_px
            positions = [positions[int((k+0.5)*step)] for k in range(n_px)] # Take into account n_px from the list of positions (do not take the very first and very last of positions (thanks to "+0.5") to avoid edges)

        if fromVid: #Uses FFT
            estimatedFreqs = []
            for x, y in positions:
                # Computes FFT
                spectrum = abs(fft.fft(self.Array[:, x, y]))
                freq = fft.fftfreq(len(spectrum))
                if showFFT:
                    plt.plot(freq, spectrum)
                    plt.show()

                #Selects main peaks
                threshold = thresholdAmp*max(spectrum)
                mask = spectrum > threshold
                peaks = freq[mask]
                #print(peaks)
                if len(peaks)>=2 :
                    estimatedFreqs.append(peaks[1]) #Take the first peak (peaks[0] is the continuous component, we want the fundamental)

            period = 1/np.median(estimatedFreqs) # We consider the median : on most of the pixels, we (hope to) get the right frequency, but some errors happens : the average would be impacted, not the median
            print("Estimated period :", period, "images")
            return period

        else:
            """
            Counts the number of images between two consecutives steps with the same sign.
            It assumes that "mod2_unique" was used in "gpuSkyrmions.py" (so two consecutives steps have necessary opposite signs)
            """
            periods = []
            for x, y in positions:
                indexSwitches = self._switches[:, x, y].nonzero()[0]
                periodsTmp = []
                for i in range(2, len(indexSwitches)):
                    periodsTmp.append(indexSwitches[i]-indexSwitches[i-2])
                if len(periodsTmp) > 0:
                    periods.append(np.mean(periodsTmp)) #For one pixel, we take the mean of the periods (more precise)

            period = np.median(periods) #For the final result, we take the median (if some errors happens, the average would be impacted, not the median)
            print("Estimated period :", period, "images")
            return period

if __name__ == "__main__":
    stack = Skyrmions("/home/mokas/Meas/Creep/PtCoPt/M2 modified for skyrmions/Renamed", "filename*.png")
    stack.getSwitches(threshold=120, showHist=False)
    stack.showPixelTimeSequence(pixel=(400, 250))
    stack.getPeriod(n_px = 5, showFFT=True)
    stack.getPeriod(fromVid=False, n_px = 5)
    stack.plotSwitched()