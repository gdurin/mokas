import numpy as np
import scipy.ndimage as nd
import pycuda.driver as driver
import pycuda.tools as tools
import pycuda.autoinit
from pycuda.gpuarray import to_gpu
from pycuda.compiler import SourceModule

def get_gpuSwitchTime(stackImages, convolSize=10, multiplier=1, device=0):
    """
    Return a matrix with the positions of a step in a sequence for each pixel

    Parameters:
    ---------------
    stackImages: int32 : 3D Array of images

    useKernel : string
    	step = [-1]*5 +[1]*5
    	zero = [-1]*5 +[0] + [1]*5

    device: Set the GPU device to use (numbered from 0). Default is 0
    """

    ngpus = driver.Device.count()
    if device >= ngpus :
        device = 0
    # Set the card to work with
    ctx = driver.Device(device).make_context()
    print("working on card %s"%driver.Device(device).name())
    used_device = ctx.get_device()
    max_threads = tools.DeviceData(used_device).max_threads
    # Convert to int32
    dim_z, dim_y, dim_x = stackImages.shape
    dim_Z, dim_Y, dim_X = np.int32(stackImages.shape)
    block_X = 256
    block_Y = 1
    grid_X, grid_Y = dim_x*dim_y*dim_z / block_X if (dim_x*dim_y*dim_z / block_X)%2==0 else dim_x*dim_y*dim_z / block_X +1 , 1
    grid_X2, grid_Y2 = dim_x / block_X + 1, dim_y/ block_Y + 1
    print("Print grid dimensions: ", grid_X, grid_Y)
    convolStack = np.zeros((dim_z , dim_y, dim_x), dtype=np.int32)
    switch = np.zeros((dim_y,dim_x), dtype=np.int32)
    levels = np.zeros((dim_y,dim_x), dtype=np.int32)
    convolSize32 = np.int32(convolSize)
    multiplier32 = np.int32(multiplier)
    #Host to Device copy
    stack_gpu = to_gpu(stackImages)
    print("Stack_gpu copied")
    switch_gpu = to_gpu(switch)
    print("Switch_gpu copied")
    levels_gpu = to_gpu(levels)
    print("Level_gpu copied")
    convolStack_gpu = to_gpu(convolStack)
    print("convolStack_gpu copied")
    print("Data transfered to GPU")
    # As an alternatice (longer)
    #stack_gpu = driver.mem_alloc(stackImages.nbytes)
    #kernel_gpu = driver.mem_alloc(kernel2.nbytes)
    #switch_gpu = driver.mem_alloc(switch.nbytes)
    #levels_gpu = driver.mem_alloc(levels.nbytes)
    #aMod_gpu = driver.mem_alloc(aMod.nbytes)
    
    #driver.memcpy_htod(stack_gpu, stackImages)
    #driver.memcpy_htod(kernel_gpu, kernel2)
    #driver.memcpy_htod(switch_gpu, switch)
    #driver.memcpy_htod(levels_gpu, levels)
    #driver.memcpy_htod(aMod_gpu, aMod)
    
    #int idx = threadIdx.x + blockIdx.x * blockDim.x; 
     #  int idy = threadIdx.y + blockIdx.y * blockDim.y; 
     #  if (idx >= dim_x || idy >= dim_y)
     #    return;
     #  int j,idz,id,id1;

        # //Copy the elements of the second stack to mirror the first and the last elements
        # for(idz=0;idz<dim_z;idz++)
        # {
        #   int flat_id = idx + dim_x * idy + (dim_x * dim_y) * idz ;
        #   int flat_id2= idx + dim_x * idy + (dim_x * dim_y) * (idz+len_kernel/2+origine);
        #   amod[flat_id2]=stack_gpu[flat_id];
        # }

        # //Mirror of the first elements
        # int idz2=0;//len_kernel/2-1+origine; //no mirroring
        # for(id=0;id<len_kernel/2+origine;id++)
        # {

        #   int flat_id = idx + dim_x * idy + (dim_x * dim_y) * idz2;       
        #   int flat_id3= idx + dim_x * idy + (dim_x * dim_y) * id; 
        #   amod[flat_id3]=stack_gpu[flat_id];
        #   //idz2--; //no mirroring
        # }
        # //Mirror of the last elements
        # int idz3=dim_z-1;
        # for(id1=dim_z+len_kernel/2+origine;id1<dim_z+len_kernel-1;id1++)
        # {
        #   int flat_id4= idx + dim_x * idy + (dim_x * dim_y) * id1;
        #   int flat_id5= idx + dim_x * idy + (dim_x * dim_y) * idz3;   
        #   amod[flat_id4]=stack_gpu[flat_id5];
        #   //idz3--; //no mirroring

        # }



        # for(idz = 0; idz <dim_z; idz++)
        # {
        #   int flat_id8 = idx + dim_x * idy + (dim_x * dim_y) * idz; 
        #   stack_gpu[flat_id8]=0;
        # } 



        # for(idz=len_kernel/2+origine;idz<dim_z+len_kernel/2+origine;idz++)
        # {

        #   int flat_id6 = idx + dim_x * idy + (dim_x * dim_y) * (idz-len_kernel/2-origine);    

        #   for(j=0;j<len_kernel;j++)
        #   {
        #       int flat_id7 = idx + dim_x * idy + (dim_x * dim_y) * (idz-len_kernel/2-origine+j); 
        #       stack_gpu[flat_id6]+=amod[flat_id7]*kernel_gpu[j];
        #   }
        # } 
    # mod3 = SourceModule("""
    # __global__ void findconvolve1d(int *stack_gpu, int *kernel_gpu, int *amod, int dim_x, int dim_y, int dim_z, int len_kernel, int origine)
    # {
    #     int idx = threadIdx.x + blockIdx.x * blockDim.x; 
    #     if ( idx > dim_x*dim_y*dim_z)
    #         return;
    #     else{
    #         //int i = idx/(dim_y*dim_z);
    #         //int j = idx%(dim_y*dim_z)/dim_z;
    #         int k = idx % (dim_y*dim_z) % dim_z;     
    #         int partialSum=0;     
    #         int multiplier=1;  
    #         for (int r=1; r<16; r++){
    #             int a = ( k-r < 0 ? k+r : k-r );
    #             int b = ( k+r >= dim_z ? k-r : k+r );
    #             amod[idx]+=(stack_gpu[a]-stack_gpu[b]);
                
    #         }

    #     }
    # }
    # """)
    print("Tokenizing 1")
    mod1 = SourceModule("""
	__global__ void findconvolve1d(int *stack_gpu, int *convolStack_gpu, int dim_x, int dim_y, int dim_z, int convolSize,int multiplier0)
	{
        int idx = threadIdx.x + blockIdx.x * blockDim.x; 
        if ( idx > dim_x*dim_y*dim_z)
            return;
        else{
            int k = idx/(dim_x*dim_y);
            //int i = idx%(dim_x*dim_y)/dim_y;
            //int j = idx % (dim_x*dim_y) % dim_y; 

            int partialSum=0;     
            int multiplier=multiplier0; 
            for (int r0=-convolSize; r0<convolSize; r0++){
                int r=r0;
                if( r0 >= 0 ) 
                    multiplier=-multiplier0;
                if(r0+k>=dim_z || r0+k<0)
                    r=-r0;
                partialSum+=(multiplier*stack_gpu[idx+r*dim_x*dim_y]);
                }
            convolStack_gpu[idx]=partialSum;
        }
 	}
    """)
    print("Tokenizing 2")
    mod2 = SourceModule("""
    __global__ void findmin(int *convolStack_gpu, int *switch_gpu, int *levels_gpu, int dim_x, int dim_y, int dim_z)
    {
    int len_kernel_half = 15;
    int idx = threadIdx.x + blockIdx.x * blockDim.x; 
    int idy = threadIdx.y + blockIdx.y * blockDim.y; 
      if (idx >= dim_x || idy >= dim_y)
        return;
    int flat_id1 = idx + dim_x * idy ;
    int min=4294967295;
    for(int idz = 0; idz <dim_z; idz++)
      {
        int flat_id = idx + dim_x * idy + (dim_x * dim_y) * idz;      
        if(convolStack_gpu[flat_id]<min)
        {
        min=convolStack_gpu[flat_id];
        switch_gpu[flat_id1]=idz;
        }
      }
        levels_gpu[flat_id1]=abs(min)/len_kernel_half;
    }
    """)
 #    print("Tokenizing 2")
 #    mod2 = SourceModule("""
 #    __global__ void findmin(int *switch_gpu, int *levels_gpu, int *convolStack_gpu, int dim_x, int dim_y, int dim_z, const int block_size)
	# {
 #        int __shared__ sdata[block_size];
	#     int idx = threadIdx.x + blockIdx.x * blockDim.x; 
 #        int i = idx/(dim_y*dim_z);
 #        int j = idx%(dim_y*dim_z)/dim_z;
 #        int k = idx % (dim_y*dim_z) % dim_z; 
 #        int x=0;

 #        if(idx<dim_x*dim_y*dim_z)
 #            x=convolStack_gpu[idx];
 #        __syncthreads();
 #        sdata[k]=x;
 #        for ( int offset=dim_z / 2; offset > 0 ; offset>>1){
 #            if (k < offset) {
 #                sdata[k]= ( sdata[k+offset*dim_y*dim_z+dim_z*j+i]<sdata[threadIdx.x] ? sdata[k+offset*dim_y*dim_z+dim_z*j+i] : sdata[k]);
 #            }
 #            __syncthreads();
 #        }
 #        if (k==0){
 #            switch_gpu[i+dim_x*j]=sdata[0];
 #        }

	# }
	# """)

    print("working on card %s"%driver.Device(device).name())

    print("Defining kernel convolve")
    func_findconvolve1d = mod1.get_function("findconvolve1d")
    # Get the array with the switching time
    print("Defining kernel findmin")
    func_findmin = mod2.get_function("findmin")

    #Function calls
    print("Ready to calculate the convolution")
    func_findconvolve1d(stack_gpu, convolStack_gpu, dim_X, dim_Y, dim_Z,convolSize32,multiplier32, block=(block_X, block_Y, 1),
         grid=(grid_X, grid_Y))
    print("Done.")
    print("Ready to find the minimum of convolution")
    func_findmin(convolStack_gpu, switch_gpu, levels_gpu,  dim_X, dim_Y, dim_Z,  block=(block_X, block_Y, 1),
          grid=(grid_X2, grid_Y2))
    print("Done")
    #Device to host copy
    print("working on card %s"%driver.Device(device).name())
    
    print("Copy to Host switchtimes")
    switch = switch_gpu.get()
    print("Copy to Host levels")
    levels = levels_gpu.get()
    print("Done")
    # As an alternative
    #driver.memcpy_dtoh(switch, switch_gpu)
    #driver.memcpy_dtoh(levels, levels_gpu)

    #Free GPU memory
    print("Clearing memory of GPU")
    stack_gpu.gpudata.free()
    switch_gpu.gpudata.free()
    convolStack_gpu.gpudata.free()
    levels_gpu.gpudata.free()
    pycuda.driver.Context.pop()
    # Close device properly
    ctx.pop()

    return switch, levels


if __name__ == "__main__":
    import time
    # Prepare a 3D array of random data as int32
    dim_x = 650
    dim_y = 650 
    dim_z = 80
    a = np.random.randn(dim_z,dim_y,dim_x)
    a = a.astype(np.int32)
    print("Loading %.2f MB of data" % (a.nbytes/1e6))
    # Call the GPU kernel
    kernel = np.array([-1]*15+[1]*15)
    t0 = time.time()
    gpuswitch, gpulevels = get_gpuSwitchTime(a, kernel, device=1)	
    timeGpu = time.time() - t0
    # Make the same calculation on the CPU
    step = kernel
    cpuswitch=np.zeros((dim_y,dim_x),dtype=np.int32)
    cpulevels=np.zeros((dim_y,dim_x),dtype=np.int32)
    print("Loading %.2f MB of data" % (2*cpuswitch.nbytes/1e6))
    t3=time.time()
    for i in range(0,dim_x):
        for j in range(0,dim_y):
            indice=(nd.convolve1d(a[:,j,i],step,mode='reflect')).argmin()      
            cpuswitch[j,i]=indice
    timeCpu = time.time()-t3
    print("GPU calculus done = %.4f s" %timeGpu)
    print("CPU calculus done = %.4f s" %timeCpu)
    print("Difference on switch : \n")
    print(gpuswitch-cpuswitch)
    print("\nGPU is %d times faster than CPU " %(timeCpu/timeGpu))