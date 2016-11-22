import numpy as np
import scipy.ndimage as nd
import pycuda.driver as driver
import pycuda.tools as tools
import pycuda.autoinit
from pycuda.gpuarray import to_gpu
from pycuda.compiler import SourceModule

def get_gpuSwitchTime(stackImages, kernel, device=0):
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
    used_device = ctx.get_device()
    max_threads = tools.DeviceData(used_device).max_threads  
    block_size = int(max_threads**0.5)/2
        
    kernel2 = np.array(-kernel, dtype=np.int32)
    if (len(kernel)%2==0):
        origin=-1
    else:
        origin=0
    len_kernel = len(kernel)

    # Convert to int32
    origin = np.int32(origin)
    len_kernel = np.int32(len_kernel)
    dim_z, dim_y, dim_x = stackImages.shape
    dim_Z, dim_Y, dim_X = np.int32(stackImages.shape)
    block_X = 256
    block_Y = 1
    grid_X, grid_Y = (dim_x - 1) / block_X + 1, (dim_y - 1) / block_Y + 1
    
    
    aMod = np.zeros(((dim_z + len_kernel -  1), dim_y, dim_x), dtype=np.int32)
    switch = np.zeros((dim_y,dim_x), dtype=np.int32)
    levels = np.zeros((dim_y,dim_x), dtype=np.int32)
    
    #Host to Device copy
    stack_gpu = to_gpu(stackImages)
    kernel_gpu = to_gpu(kernel2)
    switch_gpu = to_gpu(switch)
    levels_gpu = to_gpu(levels)
    aMod_gpu = to_gpu(aMod)    

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
    
    
    
    mod = SourceModule("""
	__global__ void findconvolve1d(int *stack_gpu,int *kernel_gpu ,int *amod,int dim_x, int dim_y, int dim_z,int len_kernel,int origine)
	{
	  int idx = threadIdx.x + blockIdx.x * blockDim.x; 
	  int idy = threadIdx.y + blockIdx.y * blockDim.y; 
	  if (idx >= dim_x || idy >= dim_y)
	    return;
	  int j,idz,id,id1;

		//Copy the elements of the second stack to mirror the first and the last elements
		for(idz=0;idz<dim_z;idz++)
		{
			int flat_id = idx + dim_x * idy + (dim_x * dim_y) * idz	;
			int flat_id2= idx + dim_x * idy + (dim_x * dim_y) * (idz+len_kernel/2+origine);
			amod[flat_id2]=stack_gpu[flat_id];
		}

		//Mirror of the first elements
		int idz2=len_kernel/2-1+origine;
		for(id=0;id<len_kernel/2+origine;id++)
		{

			int flat_id = idx + dim_x * idy + (dim_x * dim_y) * idz2;		
			int flat_id3= idx + dim_x * idy + (dim_x * dim_y) * id;	
			amod[flat_id3]=stack_gpu[flat_id];
			idz2--;
		}
		//Mirror of the last elements
		int idz3=dim_z-1;
		for(id1=dim_z+len_kernel/2+origine;id1<dim_z+len_kernel-1;id1++)
		{
			int flat_id4= idx + dim_x * idy + (dim_x * dim_y) * id1;
			int flat_id5= idx + dim_x * idy + (dim_x * dim_y) * idz3;	
			amod[flat_id4]=stack_gpu[flat_id5];
			idz3--;

		}



		for(idz = 0; idz <dim_z; idz++)
		{
			int flat_id8 = idx + dim_x * idy + (dim_x * dim_y) * idz; 
			stack_gpu[flat_id8]=0;
		}	



		for(idz=len_kernel/2+origine;idz<dim_z+len_kernel/2+origine;idz++)
		{

			int flat_id6 = idx + dim_x * idy + (dim_x * dim_y) * (idz-len_kernel/2-origine);  	

			for(j=0;j<len_kernel;j++)
			{
				int flat_id7 = idx + dim_x * idy + (dim_x * dim_y) * (idz-len_kernel/2-origine+j); 
				stack_gpu[flat_id6]+=amod[flat_id7]*kernel_gpu[j];
			}
		} 



	}
	__global__ void findmin(int *stack_gpu,int *switch_gpu,int *levels_gpu, int dim_x, int dim_y, int dim_z, int len_kernel)
	{
	int len_kernel_half = len_kernel/2;
	int idx = threadIdx.x + blockIdx.x * blockDim.x; 
	int idy = threadIdx.y + blockIdx.y * blockDim.y; 
	  if (idx >= dim_x || idy >= dim_y)
	    return;
	int flat_id1 = idx + dim_x * idy ;
	int min=4294967295;
	for(int idz = 0; idz <dim_z; idz++)
	  {
		int flat_id = idx + dim_x * idy + (dim_x * dim_y) * idz;      
		if(stack_gpu[flat_id]<min)
		{
		min=stack_gpu[flat_id];
		switch_gpu[flat_id1]=idz;
		}
	  }
        levels_gpu[flat_id1]=abs(min)/len_kernel_half;
	}
	""")
    func_findconvolve1d = mod.get_function("findconvolve1d")
    # Get the array with the switching time
    func_findmin = mod.get_function("findmin")

    #Function calls
    func_findconvolve1d(stack_gpu, kernel_gpu, aMod_gpu, dim_X, dim_Y, dim_Z, len_kernel, origin, block=(block_X, block_Y, 1),
         grid=(grid_X, grid_Y))

    func_findmin(stack_gpu, switch_gpu, levels_gpu, dim_X, dim_Y, dim_Z, len_kernel, block=(block_X, block_Y, 1),
          grid=(grid_X, grid_Y))

    #Device to host copy
    switch = switch_gpu.get()
    levels = levels_gpu.get()

    # As an alternative
    #driver.memcpy_dtoh(switch, switch_gpu)
    #driver.memcpy_dtoh(levels, levels_gpu)
    
    #Free GPU memory
    stack_gpu.gpudata.free()
    switch_gpu.gpudata.free()
    aMod_gpu.gpudata.free()
    levels_gpu.gpudata.free()
    pycuda.driver.Context.pop()

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
    kernel = np.array([-1]*5+[1]*5)
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