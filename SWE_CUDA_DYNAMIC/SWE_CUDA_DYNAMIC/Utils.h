#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <assert.h>

#define cucheck_dev(call) \
{\
	cudaError_t res = (call);\
	if(res != cudaSuccess) {\
		const char* err_str = cudaGetErrorString(res);\
		printf("%s (%d): %s (%d) in %s", __FILE__, __LINE__, err_str, res, #call);\
		assert(0);\
	}\
}

//linear index helper function
inline __device__ __host__ unsigned int li(unsigned int width, unsigned int x, unsigned int y)
{
	return y * width + x;
}

inline __device__ __host__ unsigned int divUp(const unsigned int i, const unsigned int d)
{
	return ((i + d - 1) / d);
}

inline __device__ __host__ unsigned int getCellExt(unsigned int refinementBase, unsigned int recursionLevel)
{
	return (unsigned int)pow((double)refinementBase, (double)recursionLevel);
}

__device__ __forceinline__ unsigned int getLaneId()
{
	unsigned int id;
	asm("mov.u32 %0, %%laneid;" : "=r"(id));
	return id;
}

__device__ static float atomicMax(float* address, float val)
{
	int* address_as_i = (int*)address;
	int old = *address_as_i, assumed;
	do {
		assumed = old;
		old = ::atomicCAS(address_as_i, assumed,
			__float_as_int(::fmaxf(val, __int_as_float(assumed))));
	} while (assumed != old);
	return __int_as_float(old);
}

__inline__ __device__ float warpReduceMax(float val)
{
	for (int offset = warpSize / 2; offset > 0; offset /= 2)
		val = fmaxf(val, __shfl_down(val, offset));
	return val;
}

__inline__ __device__
float warpReduceSum(float val)
{
	for (int offset = warpSize / 2; offset > 0; offset /= 2)
		val += __shfl_down(val, offset);
	return val;
}

template<typename T>
__device__ void fillRectLoop(T* data, int width, int height, int startX, int startY, int extX, int extY, T val)
{
	for (int x = startX; x < min(width, startX + extX); x++)
	{
		for (int y = startY; y < min(height, startY + extY); y++)
		{
			data[li(width, x, y)] = val;
		}
	}
}

template<typename T>
__global__ void fillRectDynamic_child(T* data, int width, int height, int startX, int startY, int extX, int extY, T val)
{
	int xoff = threadIdx.x + blockIdx.x * blockDim.x;
	int yoff = threadIdx.y + blockIdx.y * blockDim.y;

	if (xoff >= extX || yoff >= extY || startX + xoff >= width || startY + yoff >= height)
		return;

	data[li(width, startX + xoff, startY + yoff)] = val;
}

template<typename T>
__device__ void fillRectDynamic(T* data, int width, int height, int startX, int startY, int extX, int extY, T val, int blockX = 16, int blockY = 16)
{
	dim3 block = dim3(blockX, blockY);
	dim3 grid(divUp(min(extX, width - startX), block.x), divUp(min(extY, height - startY), block.y));
	fillRectDynamic_child<T> << <grid, block >> >(data, width, height, startX, startY, extX, extY, val);
}

//--------------------------------------------------------------------------------------------------

template<typename T>
__device__ T sumRectLoop(T* data, int width, int height, int startX, int startY, int extX, int extY)
{
	T sum = 0;
	for (int x = startX; x < min(width, startX + extX); x++)
	{
		for (int y = startY; y < min(height, startY + extY); y++)
		{
			sum += data[li(width, x, y)];
		}
	}
	return sum;
}

template<typename T>
__global__ void sumRectDynamic_child(T* data, int width, int height, int startX, int startY, int extX, int extY, T* sum)
{
	int xoff = threadIdx.x + blockIdx.x * blockDim.x;
	int yoff = threadIdx.y + blockIdx.y * blockDim.y;

	T psum = 0;

	if (xoff < extX && yoff < extY && startX + xoff < width && startX + xoff < height)
		psum = data[li(width, startX + xoff, startY + yoff)]; //read value and participate with it in reduction, if we are in range
	
	//reduce
	psum = warpReduceSum(psum);
	//write it back to global sum
	if (getLaneId() == 0)
		atomicAdd(sum, psum);
}

template<typename T>
__device__ T sumRectDynamic(T* data, int width, int height, int startX, int startY, int extX, int extY, int blockX = 16, int blockY = 16)
{
	T* sum = new T;
	*sum = 0;
	dim3 block = dim3(blockX, blockY);
	dim3 grid(divUp(min(extX, width - startX), block.x), divUp(min(extY, height - startY), block.y));
	sumRectDynamic_child<T> << <grid, block >> >(data, width, height, startX, startY, extX, extY, sum);
	cudaDeviceSynchronize();
	T ret = *sum;
	delete sum;
	return ret;
}

//--------------------------------------------------------------------------------------------------------------------
// summation of border values (\sum ([startX, startX + d), startY))
__device__ float sumLine(float* data, int width, int height, int startX, int startY, int d, bool horizontal)
{
	int tid = threadIdx.x + threadIdx.y * blockDim.x;
	float val = 0.0f;

	if (horizontal)
		for (int x = tid + startX; x < startX + d; x += blockDim.x * blockDim.y)
			val += data[li(width, x, startY)];
	else
		for (int y = tid + startY; y < startY + d; y += blockDim.x * blockDim.y)
			val += data[li(width, startX, y)];

	//reduce val over block
	__shared__ float arr[BX * BY];
	int nt = min(d, BX * BY);
	if (tid < nt)
		arr[tid] = val;

	__syncthreads();
	for (; nt > 1; nt /= 2)
	{
		if (tid < nt / 2)
			arr[tid] += arr[tid + nt / 2];
		__syncthreads();
	}
	return arr[0];
}