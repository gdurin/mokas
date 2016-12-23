import numpy as np
import scipy.ndimage as nd
#import pycuda.autoinit
from pycuda.gpuarray import to_gpu
from pycuda.compiler import SourceModule
import mokas_gpu as mkGpu

def find_avalanches(image):
	