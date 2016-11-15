# Routine to calculate the kernel step-like function
# to make the convolution with the gray level sequence
from __future__ import division
import sys
import numpy as np

def get_kernel(kernel_half_width_of_ones = 3, kernel_internal_points = 0, start = -1.):
    """
    Kernel: is the (step-like) function to calculate the correlation 
    with the sequence of grey levels. 
    
    The simple form is a step function, where:
    kernel_half_width_of_ones (int) is the number of points at left (-1) and at the right (+1)
    kernel_internal_points is the number of point between the -1 and -1;
    kernel_internal_points = 0 (default) : step function
    kernel_internal_points = 1 : [-1, -1, -1, 0, +1, +1, +1]
    kernel_internal_points = 2 : [-1, -1, -1, -0.33, +0.33, +1, +1, +1]
    kernel_internal_points = 3 : [-1, -1, -1, -0.5, 0, +0.5, +1, +1, +1]
    kernel_internal_points = 4 : [-1, -1, -1, -0.6, -0.2, +0.2, +0.6, +1, +1, +1]
    kernel_internal_points = 5 : [-1, -1, -1, -0.66, -0.33, 0, +0.33, 0.66, +1, +1, +1]
    """
    step = -2*start/ (1 + kernel_internal_points)
    stop = -start + step
    seq = np.arange(start, stop, step)
    seq_left = start * np.ones(kernel_half_width_of_ones - 1)
    seq = np.append(seq_left, seq)
    seq = np.append(seq, -seq_left)
    return seq

if __name__ == "__main__":
    for i in range(6):
        print(get_kernel(kernel_internal_points=i, start=-2))