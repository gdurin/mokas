import numpy as np
import scipy.ndimage as nd
#import pycuda.autoinit
from pycuda.gpuarray import to_gpu
from pycuda.compiler import SourceModule
import mokas_gpu as gpu

class gpuSkyrmions:

    def __init__(self, stackImages, convolSize=10, current_dev=None, ctx=None, block_size=(256,1), verbose=False):
        #Init GPU
        deinitNeeded = False
        if(current_dev is None):
            current_dev, ctx, (free, total) = gpu.gpu_init(0)
            deinitNeeded = True

        #Calculates the convolution
        self.convolStack = self._get_gpuConvolSkyrmions(stackImages, convolSize=convolSize, current_dev=current_dev, ctx=ctx, block_size=block_size, verbose=verbose)

        if(deinitNeeded):
            gpu.gpu_deinit(current_dev, ctx)


    def getSwitchesSkyrmions(self, current_dev=None, ctx=None, block_size=(256,1), threshold=30, verbose=False):
        """
        Calculates and returns the switches (3D array). A threshold is needed from now on (see _get_gpuSwitchesSkyrmions below)
        """
        #init GPU
        deinitNeeded = False
        if(current_dev is None):
            current_dev, ctx, (free, total) = gpu.gpu_init(0)
            deinitNeeded = True

        #Calculates the switches
        switches = self._get_gpuSwitchesSkyrmions(current_dev=current_dev, ctx=ctx, block_size=block_size, threshold=threshold, verbose=verbose)

        if(deinitNeeded):
            gpu.gpu_deinit(current_dev, ctx)

        return switches

    def _get_gpuConvolSkyrmions(self, stackImages, convolSize=10, current_dev=None, ctx=None, block_size=(256,1), verbose=False):
        """
        Return a 3D-array of the convolution
        Same as gpuSwitchtims.py
        Parameters:
        ---------------
        """

        # =========================================
        # Set the card to work with: DONE EXTERNALLY
        # =========================================
        if verbose:
            print("working on card %s" % current_dev.name())
        used_device = ctx.get_device()
        # Convert to int32
        dim_z, dim_y, dim_x = stackImages.shape
        dim_Z, dim_Y, dim_X = np.int32(stackImages.shape)
        block_X, block_Y = block_size
        grid_X, grid_Y = dim_x*dim_y*dim_z / block_X if (dim_x*dim_y*dim_z % block_X)==0 else dim_x*dim_y*dim_z / block_X +1 , 1
        grid_X2, grid_Y2 = dim_x / block_X + 1, dim_y/ block_Y + 1
        grid_X = int(grid_X)
        grid_Y = int(grid_Y)
        grid_X2 = int(grid_X2)
        grid_Y2 = int(grid_Y2)
        if verbose:
            print("Print grid dimensions: ", grid_X, grid_Y)
        convolStack = np.zeros((dim_z , dim_y, dim_x), dtype=np.float32)
        convolSize32 = np.int32(convolSize)
        #Host to Device copy
        stack_gpu = to_gpu(stackImages)
        if verbose:
            print("Stack_gpu copied")
        convolStack_gpu = to_gpu(convolStack)
        if verbose:
            print("convolStack_gpu copied")
            print("Data transfered to GPU")
            print("Tokenizing 1")
        # contracts the kernel size when approaching edges
        mod1_a = SourceModule("""
        __global__ void findconvolve1d(int *stack_gpu, float *convolStack_gpu, int dim_x, int dim_y, int dim_z, int convolSize)
        {
            int idx = threadIdx.x + blockIdx.x * blockDim.x; 
            if ( idx > dim_x*dim_y*dim_z)
                return;
            else{
                int k = idx/(dim_x*dim_y);
                int convolSizeReal = ( ( convolSize + k >= dim_z || -convolSize + k < 0 ) ? min(k,abs(dim_z-1-k)) : convolSize);
                float partialSum=0;     
                int multiplierInit=-1;
                int multiplier=multiplierInit; 
                for (int r0=-convolSizeReal; r0<convolSizeReal; r0++){
                    int r=r0;
                    if( r0 >= 0 ) 
                        multiplier=-multiplierInit;
                    partialSum+=(multiplier*stack_gpu[idx+r*dim_x*dim_y]);
                    }
                convolStack_gpu[idx]=partialSum/convolSizeReal;
            }
        }
        """)
        # keeps constant value
        mod1_b = SourceModule("""
        __global__ void findconvolve1d(int *stack_gpu, float *convolStack_gpu, int dim_x, int dim_y, int dim_z, int convolSize)
        {
            int idx = threadIdx.x + blockIdx.x * blockDim.x; 
            if ( idx > dim_x*dim_y*dim_z)
                return;
            else{
                int k = idx/(dim_x*dim_y);
                //int i = idx%(dim_x*dim_y)/dim_y;
                //int j = idx % (dim_x*dim_y) % dim_y; 

                float partialSum=0;     
                int multiplierInit=-1;
                int multiplier=multiplierInit; 
                for (int r0=-convolSize; r0<convolSize; r0++){
                    int r=r0;
                    if( r0 >= 0 ) 
                        multiplier=-multiplierInit;
                    if( r0+k >= dim_z ) r=dim_z-1;
                    if( r0+k < 0 )  r=0;
                    partialSum+=(multiplier*stack_gpu[idx+r*dim_x*dim_y]);
                    }
                convolStack_gpu[idx]=partialSum/convolSize;
            }
        }
        """)
        #Keeps constant at beginning and mirrors at end of sequence
        mod1_c = SourceModule("""
        __global__ void findconvolve1d(int *stack_gpu, float *convolStack_gpu, int dim_x, int dim_y, int dim_z, int convolSize)
        {
            int idx = threadIdx.x + blockIdx.x * blockDim.x; 
            if ( idx > dim_x*dim_y*dim_z)
                return;
            else{
                int k = idx/(dim_x*dim_y);
                float partialSum=0; 
                int multiplierInit=-1;
                int multiplier=multiplierInit; 
                int r;
                for (int r0=-convolSize; r0<convolSize; r0++){
                    if( r0 >= 0 ) 
                        multiplier=-multiplierInit;
                    if( r0+k >= dim_z)
                        r=-r0;
                    else if ( r0+k < 0 )
                        r=0;
                    else r=r0;
                        
                    partialSum+=(multiplier*stack_gpu[idx+r*dim_x*dim_y]);
                    }
                convolStack_gpu[idx]=partialSum/convolSize;
            }
        }
        """)

        # mirrors values out of bound
        mod1 = SourceModule("""
        __global__ void findconvolve1d(int *stack_gpu, float *convolStack_gpu, int dim_x, int dim_y, int dim_z, int convolSize)
        {
            int idx = threadIdx.x + blockIdx.x * blockDim.x; 
            if ( idx > dim_x*dim_y*dim_z)
                return;
            else{
                int k = idx/(dim_x*dim_y);
                //int i = idx%(dim_x*dim_y)/dim_y;
                //int j = idx % (dim_x*dim_y) % dim_y; 

                float partialSum=0;     
                int multiplierInit=-1;
                int multiplier=multiplierInit; 
                for (int r0=-convolSize; r0<convolSize; r0++){
                    int r=r0;
                    if( r0 >= 0 ) 
                        multiplier=-multiplierInit;
                    if( r0+k >= dim_z || r0+k < 0 )
                        r=-r0;
                    partialSum+=(multiplier*stack_gpu[idx+r*dim_x*dim_y]);
                    }
                convolStack_gpu[idx]=partialSum/convolSize;
            }
        }
        """)


        if verbose:
            print("Defining kernel convolve")
        func_findconvolve1d = mod1_c.get_function("findconvolve1d")
        # Get the array with the switching time

        #Function calls
        if verbose:
            print("Ready to calculate the convolution")
        func_findconvolve1d(stack_gpu, convolStack_gpu, dim_X, dim_Y, dim_Z,convolSize32, block=(block_X, block_Y, 1),
             grid=(grid_X, grid_Y))
        if verbose:
            print("Done.")
            #Device to host copy  
            print("Copy to Host switchtimes")
        convolStack = convolStack_gpu.get()
        if verbose:
            print("Done")
        # As an alternative
        #driver.memcpy_dtoh(switch, switch_gpu)
        #driver.memcpy_dtoh(levels, levels_gpu)

        #Free GPU memory
        if verbose:
            print("Clearing memory of GPU")
        stack_gpu.gpudata.free()
        convolStack_gpu.gpudata.free()

        return convolStack


    def _get_gpuSwitchesSkyrmions(self, current_dev=None, ctx=None, block_size=(256,1), threshold=30, verbose=False):
        """
        Return a 3D-array with the positions and height of the steps in a sequence for each pixel (+ for step "up", - for step "down", 0 for no step)
        For example : switch[:, x, y] = [0   0   +50   0   0  -20   0   0   0   0   0   0   0   +100]

        Parameters:
        ---------------
        """

        #Note : Only mod2_unique is commented : the two othe versions are easier

        # =========================================
        # Set the card to work with: DONE EXTERNALLY
        # =========================================
        if verbose:
            print("working on card %s" % current_dev.name())
        used_device = ctx.get_device()
        # Convert to int32
        dim_z, dim_y, dim_x = self.convolStack.shape
        dim_Z, dim_Y, dim_X = np.int32(self.convolStack.shape)
        block_X, block_Y = block_size
        grid_X, grid_Y = dim_x*dim_y*dim_z / block_X if (dim_x*dim_y*dim_z % block_X)==0 else dim_x*dim_y*dim_z / block_X +1 , 1
        grid_X2, grid_Y2 = dim_x / block_X + 1, dim_y/ block_Y + 1
        grid_X = int(grid_X)
        grid_Y = int(grid_Y)
        grid_X2 = int(grid_X2)
        grid_Y2 = int(grid_Y2)
        if verbose:
            print("Print grid dimensions: ", grid_X, grid_Y)
        switch = np.zeros((dim_z, dim_y, dim_x), dtype=np.int32)
        threshold32 = np.int32(threshold)
        #Host to Device copy
        switch_gpu = to_gpu(switch)
        if verbose:
            print("Switch_gpu copied")
        convolStack_gpu = to_gpu(self.convolStack)
        if verbose:
            print("convolStack_gpu copied")
            print("Data transfered to GPU")
            print("Tokenizing 2")

        #Finds *all* the steps : there can be two "+1" or two "-1" following each
        # (1) : 0  0 +1 +1 +1 +1  0  0   POSSIBLE
        # (2) : 0  0 +1  0  0  0 +1  0   POSSIBLE
        # (3) : 0  0 +1  0  0  0 -1  0   POSSIBLE
        # (4) : 0  0 +1 -1  0  0  0  0   POSSIBLE
        mod2 = SourceModule("""
        __global__ void findswitches(float *convolStack_gpu, int *switch_gpu, int dim_x, int dim_y, int dim_z, int threshold)
        {
            int idx = threadIdx.x + blockIdx.x * blockDim.x; 
            int idy = threadIdx.y + blockIdx.y * blockDim.y; 
            if (idx >= dim_x || idy >= dim_y)
                return;
            for(int idz = 0; idz <dim_z; idz++)
            {
                int flat_id = idx + dim_x * idy + (dim_x * dim_y) * idz;      
                if(abs(convolStack_gpu[flat_id])>threshold)
                {
                    switch_gpu[flat_id]=convolStack_gpu[flat_id]; //(convolStack_gpu[flat_id]>0)-(convolStack_gpu[flat_id]<0); //Sign of convolStack_gpu[flat_id]
                }
            }
        }
        """)

        #Finds steps without repetition : there cannot be two "+1" or two "-1" following each other in
        #the result switch_gpu (1) BUT there still can be two "separated" identical steps (2) AND there
        #can be a "+1" followed by a "-1" (4):
        # (1) : 0  0 +1 +1 +1 +1  0  0   IMPOSSIBLE (we will consider only one "+1" of the group)
        # (2) : 0  0 +1  0  0  0 +1  0   POSSIBLE
        # (3) : 0  0 +1  0  0  0 -1  0   POSSIBLE
        # (4) : 0  0 +1 -1  0  0  0  0   POSSIBLE
        mod2_no_repeat = SourceModule("""
        __global__ void findswitches(float *convolStack_gpu, int *switch_gpu, int dim_x, int dim_y, int dim_z, int threshold)
        {
            int idx = threadIdx.x + blockIdx.x * blockDim.x; 
            int idy = threadIdx.y + blockIdx.y * blockDim.y; 
            if (idx >= dim_x || idy >= dim_y)
                return;

            //Because of the convolution, there can be multiple "switch on" (or "switch off") side by side
            //The goal of the next variables is to select the best one from them
            int localSwitchSign = 0;
            int tmpMax = 0;
            int idMax = -1;

            for(int idz = 0; idz <dim_z; idz++)
            {
                int flat_id = idx + dim_x * idy + (dim_x * dim_y) * idz;      
                if(abs(convolStack_gpu[flat_id])>threshold)
                {
                    int actualSign = (convolStack_gpu[flat_id]>0)-(convolStack_gpu[flat_id]<0);
                    
                    if(localSwitchSign != 0)
                    {
                        if(localSwitchSign == actualSign && abs(convolStack_gpu[flat_id]) > tmpMax)
                        {
                            tmpMax = abs(convolStack_gpu[flat_id]);
                            idMax = flat_id;
                        }
                        else if(localSwitchSign != actualSign)
                        {
                            switch_gpu[idMax] = localSwitchSign*tmpMax;

                            localSwitchSign = actualSign;
                            tmpMax = abs(convolStack_gpu[flat_id]);
                            idMax = flat_id;
                        }
                    }
                    else
                    {
                        localSwitchSign = actualSign;
                        tmpMax = abs(convolStack_gpu[flat_id]);
                        idMax = flat_id;
                    }
                }
                else if(localSwitchSign != 0)
                {
                    switch_gpu[idMax] = localSwitchSign*tmpMax;

                    localSwitchSign = 0;
                    tmpMax = 0;
                    idMax = -1;
                }
            }
        }
        """)

        #Finds *unique* steps : there cannot be two "+1" or two "-1" following each other in
        #the result switch_gpu (1) AND there cannot be two "separated" identical steps (2) BUT
        #a "+1" followed by a "-1" is always possible :
        # (1) : 0  0 +1 +1 +1 +1  0  0   IMPOSSIBLE (we will consider only one "+1" of the group)
        # (2) : 0  0 +1  0  0  0 +1  0   IMPOSSIBLE (we will consider only the first one)
        # (3) : 0  0 +1  0  0  0 -1  0   POSSIBLE
        # (4) : 0  0 +1 -1  0  0  0  0   POSSIBLE
        mod2_unique = SourceModule("""
        __global__ void findswitches(float *convolStack_gpu, int *switch_gpu, int dim_x, int dim_y, int dim_z, int threshold)
        {
            int idx = threadIdx.x + blockIdx.x * blockDim.x; 
            int idy = threadIdx.y + blockIdx.y * blockDim.y; 
            if (idx >= dim_x || idy >= dim_y)
                return;

            //Because of the convolution, there can be multiple "switch on" (or "switch off") side by side
            //The goal of the next variables is to select the best one from them

            int localSwitchSign = 0;    // Sign of the convolution at the previous image
            int lastSwitchSign = 0;     // Sign of the convolution at the previous switch (allow to delect following steps with same sign (e.g.: 0  0  0 +1  0  0 +1  0  0), and avoid the second one to be taken into account)
            int tmpMax = 0;             // Saves the actual local maximum to store it (with the right sign) in "switch" *after* returning under the threshold
            int idMax = -1;             // Saves the flat_id of the actual local maximum to store it...

            for(int idz = 0; idz <dim_z; idz++)
            {
                int flat_id = idx + dim_x * idy + (dim_x * dim_y) * idz;      
                if(abs(convolStack_gpu[flat_id])>threshold)
                {
                    int actualSign = (convolStack_gpu[flat_id]>0)-(convolStack_gpu[flat_id]<0);
                    
                    if(actualSign == lastSwitchSign && localSwitchSign == 0)        // Avoid two identical steps : 0  0  0 +1  0  0 +1  0  0
                        continue;

                    if(localSwitchSign != 0)                                // If we were already above the threshold, we have to compare the actual value to the maximum
                    {
                        if(localSwitchSign == actualSign && abs(convolStack_gpu[flat_id]) > tmpMax)
                        {
                            tmpMax = abs(convolStack_gpu[flat_id]);
                            idMax = flat_id;
                        }
                        else if(localSwitchSign != actualSign)              // This is tricky : we can stay above the threshold (in absolute value) but with a different sign (e.g.: [0, 0, +255, -255, 0, 0]). Must be rare, but possible.
                        {
                            switch_gpu[idMax] = localSwitchSign*tmpMax;

                            lastSwitchSign = localSwitchSign;
                            localSwitchSign = actualSign;
                            tmpMax = abs(convolStack_gpu[flat_id]);
                            idMax = flat_id;
                        }
                    }
                    else                                                    // If we have just passed above the threshold, we directly set the variables (max, id_max, signs...) : no comparison to the max is needed
                    {
                        localSwitchSign = actualSign;
                        tmpMax = abs(convolStack_gpu[flat_id]);
                        idMax = flat_id;
                    }
                }
                else if(localSwitchSign != 0)                               // If we are below the threshold but we were above just before, we need to set the local maximum to the right position to indicate a step (with the right sign)
                {
                    switch_gpu[idMax] = localSwitchSign*tmpMax;

                    lastSwitchSign = localSwitchSign;
                    localSwitchSign = 0;
                    tmpMax = 0;
                    idMax = -1;
                }
            }
        }
        """)

        # Get the array with the switching time
        if verbose:
            print("Defining kernel findswitches")
        func_findswitches = mod2_unique.get_function("findswitches")

        if verbose:
            print("Ready to find the switches")
        func_findswitches(convolStack_gpu, switch_gpu, dim_X, dim_Y, dim_Z, threshold32,  block=(block_X, block_Y, 1), grid=(grid_X2, grid_Y2))

        if verbose:
            print("Done")
            #Device to host copy  
            print("Copy to Host switchtimes")

        switch = switch_gpu.get()

        if verbose:
            print("Done")
        # As an alternative
        #driver.memcpy_dtoh(switch, switch_gpu)
        #driver.memcpy_dtoh(levels, levels_gpu)

        #Free GPU memory
        if verbose:
            print("Clearing memory of GPU")

        switch_gpu.gpudata.free()
        convolStack_gpu.gpudata.free()

        return switch


if __name__ == "__main__":
    import time
    import mokas_gpu as gpu

    current_dev, ctx, (free, total) = gpu.gpu_init(0)
    # Prepare a 3D array of random data as int32
    dim_x = 5
    dim_y = 5 
    dim_z = 15
    a = np.array( [[[255, 0  , 0  , 0  , 210],
                    [0  , 255, 0  , 0  , 0  ],
                    [0  , 0  , 0  , 0  , 0  ],
                    [0  , 0  , 0  , 255, 0  ],
                    [0  , 0  , 0  , 0  , 0  ]],

                   [[255, 0  , 0  , 0  , 210],
                    [0  , 255, 0  , 0  , 0  ],
                    [0  , 0  , 0  , 0  , 0  ],
                    [0  , 0  , 0  , 255, 0  ],
                    [0  , 0  , 0  , 0  , 255]],

                   [[255, 0  , 0  , 0  , 210],
                    [0  , 255, 0  , 0  , 0  ],
                    [0  , 0  , 0  , 0  , 0  ],
                    [0  , 0  , 0  , 0  , 0  ],
                    [0  , 0  , 0  , 0  , 0  ]],

                   [[255, 0  , 0  , 0  , 105],
                    [0  , 0  , 0  , 0  , 0  ],
                    [0  , 0  , 0  , 0  , 0  ],
                    [0  , 0  , 0  , 0  , 0  ],
                    [105, 0  , 0  , 0  , 255]],

                   [[0  , 0  , 0  , 0  , 105],
                    [0  , 0  , 0  , 0  , 0  ],
                    [0  , 0  , 0  , 0  , 0  ],
                    [0  , 0  , 0  , 255, 0  ],
                    [105, 0  , 0  , 0  , 0  ]],

                   [[0  , 0  , 0  , 0  , 105],
                    [0  , 0  , 0  , 0  , 0  ],
                    [0  , 0  , 255, 0  , 0  ],
                    [0  , 0  , 0  , 255, 0  ],
                    [105, 0  , 0  , 0  , 255]],

                   [[0  , 0  , 0  , 0  , 0  ],
                    [0  , 255, 0  , 0  , 0  ],
                    [0  , 0  , 255, 0  , 0  ],
                    [0  , 0  , 0  , 0  , 0  ],
                    [210, 0  , 0  , 0  , 0  ]],

                   [[0  , 0  , 0  , 0  , 0  ],
                    [0  , 255, 0  , 0  , 0  ],
                    [0  , 0  , 255, 0  , 0  ],
                    [0  , 0  , 0  , 0  , 0  ],
                    [210, 0  , 0  , 0  , 255]],

                   [[255, 0  , 0  , 0  , 0  ],
                    [0  , 255, 0  , 0  , 0  ],
                    [0  , 0  , 255, 0  , 0  ],
                    [0  , 0  , 0  , 255, 0  ],
                    [210, 0  , 0  , 0  , 0  ]],

                   [[255, 0  , 0  , 0  , 105],
                    [0  , 0  , 0  , 0  , 0  ],
                    [0  , 0  , 255, 0  , 0  ],
                    [0  , 0  , 0  , 255, 0  ],
                    [105, 0  , 0  , 0  , 255]],

                   [[255, 0  , 0  , 0  , 105],
                    [0  , 0  , 0  , 0  , 0  ],
                    [0  , 0  , 0  , 0  , 0  ],
                    [0  , 0  , 0  , 0  , 0  ],
                    [105, 0  , 0  , 0  , 0  ]],

                   [[255, 0  , 0  , 0  , 105],
                    [0  , 0  , 0  , 0  , 0  ],
                    [0  , 0  , 0  , 0  , 0  ],
                    [0  , 0  , 0  , 0  , 0  ],
                    [105, 0  , 0  , 0  , 255]],

                   [[0  , 0  , 0  , 0  , 210],
                    [0  , 255, 0  , 0  , 0  ],
                    [0  , 0  , 0  , 0  , 0  ],
                    [0  , 0  , 0  , 255, 0  ],
                    [0  , 0  , 0  , 0  , 0  ]],

                   [[0  , 0  , 0  , 0  , 210],
                    [0  , 255, 0  , 0  , 0  ],
                    [0  , 0  , 0  , 0  , 0  ],
                    [0  , 0  , 0  , 255, 0  ],
                    [0  , 0  , 0  , 0  , 255]],

                   [[0  , 0  , 0  , 0  , 210],
                    [0  , 255, 0  , 0  , 0  ],
                    [0  , 0  , 0  , 0  , 0  ],
                    [0  , 0  , 0  , 0  , 0  ],
                    [0  , 0  , 0  , 0  , 0  ]]])

    a = a.astype(np.int32)
    print("Loading %.2f MB of data" % (a.nbytes/1e6))

    monSkyrmion = gpuSkyrmions(a, convolSize=1, current_dev=current_dev, ctx=ctx, block_size=(5,5))
    print(monSkyrmion.getSwitchesSkyrmions(current_dev=current_dev, ctx=ctx, block_size=(5,5), threshold=50))
    print(monSkyrmion.getSwitchesSkyrmions(current_dev=current_dev, ctx=ctx, block_size=(5,5), threshold=100))
    print(monSkyrmion.getSwitchesSkyrmions(current_dev=current_dev, ctx=ctx, block_size=(5,5), threshold=200))

    gpu.gpu_deinit(current_dev, ctx)