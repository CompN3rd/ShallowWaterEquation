#include "SWE.h"


SWE::SWE(int _nx, int _ny, float _dx, float _dy, float _g, int blockX, int blockY)
{
	nx = _nx;
	ny = _ny;
	dx = _dx;
	dy = _dy;
	g = _g;

	left = WALL;
	right = WALL;
	top = WALL;
	bottom = WALL;

	blockSize = dim3(blockX, blockY);

	const int variableSize = (nx + 2) * (ny + 2);
	h = new float[variableSize];
	hu = new float[variableSize];
	hv = new float[variableSize];
	b = new float[variableSize];

	//device arrays
	checkCudaErrors(cudaMalloc(&hd, variableSize * sizeof(float)));
	checkCudaErrors(cudaMalloc(&hud, variableSize * sizeof(float)));
	checkCudaErrors(cudaMalloc(&hvd, variableSize * sizeof(float)));
	checkCudaErrors(cudaMalloc(&bd, variableSize * sizeof(float)));
	checkCudaErrors(cudaMalloc(&Bud, variableSize * sizeof(float)));
	checkCudaErrors(cudaMalloc(&Bvd, variableSize * sizeof(float)));

	const int flowSize = (nx + 1) * (ny + 1);
	checkCudaErrors(cudaMalloc(&Fhd, flowSize * sizeof(float)));
	checkCudaErrors(cudaMalloc(&Fhud, flowSize * sizeof(float)));
	checkCudaErrors(cudaMalloc(&Fhvd, flowSize * sizeof(float)));
	checkCudaErrors(cudaMalloc(&Ghd, flowSize * sizeof(float)));
	checkCudaErrors(cudaMalloc(&Ghud, flowSize * sizeof(float)));
	checkCudaErrors(cudaMalloc(&Ghvd, flowSize * sizeof(float)));
}


SWE::~SWE()
{
	delete[] h;
	delete[] hu;
	delete[] hv;
	delete[] b;
	checkCudaErrors(cudaFree(hd));
	checkCudaErrors(cudaFree(hud));
	checkCudaErrors(cudaFree(hvd));
	checkCudaErrors(cudaFree(bd));
	checkCudaErrors(cudaFree(Bud));
	checkCudaErrors(cudaFree(Bvd));
	checkCudaErrors(cudaFree(Fhd));
	checkCudaErrors(cudaFree(Fhud));
	checkCudaErrors(cudaFree(Fhvd));
	checkCudaErrors(cudaFree(Ghd));
	checkCudaErrors(cudaFree(Ghud));
	checkCudaErrors(cudaFree(Ghvd));
}

void SWE::uploadSolution()
{
	int numElem = (nx + 2) * (ny + 2);
	checkCudaErrors(cudaMemcpyAsync(hd, h, numElem * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyAsync(hud, hu, numElem * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(hvd, hu, numElem * sizeof(float), cudaMemcpyHostToDevice));
}

void SWE::downloadSolution()
{
	int numElem = (nx + 2) * (ny + 2);
	checkCudaErrors(cudaMemcpyAsync(h, hd, numElem * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpyAsync(hu, hud, numElem * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(hv, hud, numElem * sizeof(float), cudaMemcpyDeviceToHost));
}

void SWE::setInitialValues(float _h, float _u, float _v)
{
	for (int i = 0; i < nx + 2; i++)
	{
		for (int j = 0; j < ny + 2; j++)
		{
			h[li(nx + 2, i, j)] = _h;
			hu[li(nx + 2, i, j)] = _h * _u;
			hv[li(nx + 2, i, j)] = _h * _v;
		}
	}

	uploadSolution();
}

void SWE::setInitialValues(float(*_h)(float, float), float _u, float _v)
{
	for (int i = 0; i < nx + 2; i++)
	{
		for (int j = 0; j < ny + 2; j++)
		{
			h[li(nx + 2, i, j)] = _h((i - 0.5f) * dx, (j - 0.5f) * dy);
			hu[li(nx + 2, i, j)] = h[li(nx + 2, i, j)] * _u;
			hv[li(nx + 2, i, j)] = h[li(nx + 2, i, j)] * _v;
		}
	}

	uploadSolution();
}

void SWE::setBathymetry(float(*_b)(float, float))
{
	for (int i = 0; i < nx + 2; i++)
		for (int j = 0; j < nx + 2; j++)
			b[li(nx + 2, i, j)] = _b((i - 0.5f) * dx, (j - 0.5f) * dy);

	checkCudaErrors(cudaMemcpy(bd, b, (nx + 2) * (ny + 2) * sizeof(float), cudaMemcpyHostToDevice));
	computeBathymetrySources();
}

void SWE::setBathymetry(float _b)
{
	for (int i = 0; i < nx + 2; i++)
		for (int j = 0; j < nx + 2; j++)
			b[li(nx + 2, i, j)] = _b;

	checkCudaErrors(cudaMemcpy(bd, b, (nx + 2) * (ny + 2) * sizeof(float), cudaMemcpyHostToDevice));
	computeBathymetrySources();
}

void SWE::setBoundaryType(BoundaryType left, BoundaryType right, BoundaryType bottom, BoundaryType top)
{
	this->left = left;
	this->right = right;
	this->bottom = bottom;
	this->top = top;
}

__global__ void setVerticalBoundaryLayer(float* hd, float* hud, float* hvd, int width, int height, BoundaryType left, BoundaryType right)
{
	int j = threadIdx.x + blockIdx.x * blockDim.x;

	if (j >= height)
		return;

	if (left == CONNECT)
	{
		hd[li(width, 0, j)] = hd[li(width, width - 2, j)];
		hud[li(width, 0, j)] = hud[li(width, width - 2, j)];
		hvd[li(width, 0, j)] = hvd[li(width, width - 2, j)];
	}
	else
	{
		hd[li(width, 0, j)] = hd[li(width, 1, j)];
		hud[li(width, 0, j)] = (left == WALL) ? -hud[li(width, 1, j)] : hud[li(width, 1, j)];
		hvd[li(width, 0, j)] = hvd[li(width, 1, j)];
	}
	if (right == CONNECT)
	{
		hd[li(width, width - 1, j)] = hd[li(width, 1, j)];
		hud[li(width, width - 1, j)] = hud[li(width, 1, j)];
		hvd[li(width, width - 1, j)] = hvd[li(width, 1, j)];
	}
	else
	{
		hd[li(width, width - 1, j)] = hd[li(width, width - 2, j)];
		hud[li(width, width - 1, j)] = (right == WALL) ? -hud[li(width, width - 2, j)] : hud[li(width, width - 2, j)];
		hvd[li(width, width - 1, j)] = hvd[li(width, width - 2, j)];
	}
}

__global__ void setHorizontalBoundaryLayer(float* hd, float* hud, float* hvd, int width, int height, BoundaryType bottom, BoundaryType top)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (i >= width)
		return;

	if (bottom == CONNECT)
	{
		hd[li(width, i, 0)] = hd[li(width, i, height - 2)];
		hud[li(width, i, 0)] = hud[li(width, i, height - 2)];
		hvd[li(width, i, 0)] = hvd[li(width, i, height - 2)];
	}
	else
	{
		hd[li(width, i, 0)] = hd[li(width, i, 1)];
		hud[li(width, i, 0)] = hud[li(width, i, 1)];
		hvd[li(width, i, 0)] = (bottom == WALL) ? -hvd[li(width, i, 1)] : hvd[li(width, i, 1)];
	}
	if (top == CONNECT)
	{
		hd[li(width, i, height - 1)] = hd[li(width, i, 1)];
		hud[li(width, i, height - 1)] = hud[li(width, i, 1)];
		hvd[li(width, i, height - 1)] = hvd[li(width, i, 1)];
	}
	else
	{
		hd[li(width, i, height - 1)] = hd[li(width, i, height - 2)];
		hud[li(width, i, height - 1)] = hud[li(width, i, height - 2)];
		hvd[li(width, i, height - 1)] = (top == WALL) ? -hvd[li(width, i, height - 2)] : hvd[li(width, i, height - 2)];
	}
}

void SWE::setBoundaryLayer()
{
	dim3 verticalBlock = dim3(blockSize.y * blockSize.y);
	dim3 verticalGrid = dim3(divUp(ny + 2, verticalBlock.x));

	setVerticalBoundaryLayer << <verticalGrid, verticalBlock >> >(hd, hud, hvd, nx + 2, ny + 2, left, right);

	dim3 horizontalBlock = dim3(blockSize.x * blockSize.x);
	dim3 horizontalGrid = dim3(divUp(nx + 2, verticalBlock.x));

	setHorizontalBoundaryLayer << <horizontalGrid, horizontalBlock >> >(hd, hud, hvd, nx + 2, ny + 2, bottom, top);
}

__device__ float computeFlux(float fLow, float fHigh, float xiLow, float xiHigh, float llf)
{
	return 0.5f * (fLow + fHigh) - 0.5f * llf * (xiHigh - xiLow);
}

__device__ float computeLocalSV(float* hd, float* hud, float* hvd, int i, int j, int width, float g, char dir)
{
	float sv1, sv2;
	if (dir == 'x')
	{
		sv1 = fabsf(hud[li(width, i, j)] / hd[li(width, i, j)]) + sqrtf(g * hd[li(width, i, j)]);
		sv2 = fabsf(hud[li(width, i + 1, j)] / hd[li(width, i + 1, j)]) + sqrtf(g * hd[li(width, i + 1, j)]);
	}
	else
	{
		sv1 = fabsf(hvd[li(width, i, j)] / hd[li(width, i, j)]) + sqrtf(g * hd[li(width, i, j)]);
		sv2 = fabsf(hvd[li(width, i, j + 1)] / hd[li(width, i, j + 1)]) + sqrtf(g * hd[li(width, i, j + 1)]);
	}
	return (sv1 > sv2) ? sv1 : sv2;
}

//__global__ void computeHorizontalFluxes(float* hd, float* hud, float* hvd, float* Fhd, float* Fhud, float* Fhvd, int width, int height, float g)
//{
//	int i = threadIdx.x + blockIdx.x * blockDim.x;
//	int j = threadIdx.y + blockIdx.y * blockDim.y;
//
//	if (i >= width - 1 || j >= height - 1)
//		return;
//
//	float llf = computeLocalSV(hd, hud, hvd, i, j, width, g, 'x');
//
//	Fhd[li(width - 1, i, j)] = computeFlux(hud[li(width, i, j)], hud[li(width, i + 1, j)], hd[li(width, i, j)], hd[li(width, i + 1, j)], llf);
//
//	Fhud[li(width - 1, i, j)] = computeFlux(hud[li(width, i, j)] * hud[li(width, i, j)] / hd[li(width, i, j)] + 0.5f * g * hd[li(width, i, j)] * hd[li(width, i, j)],
//		hud[li(width, i + 1, j)] * hud[li(width, i + 1, j)] / hd[li(width, i + 1, j)] + 0.5f * g * hd[li(width, i + 1, j)] * hd[li(width, i + 1, j)],
//		hud[li(width, i, j)],
//		hud[li(width, i + 1, j)],
//		llf);
//
//	Fhvd[li(width - 1, i, j)] = computeFlux(hud[li(width, i, j)] * hvd[li(width, i, j)] / hd[li(width, i, j)], hud[li(width, i + 1, j)] * hvd[li(width, i + 1, j)] / hd[li(width, i + 1, j)], hvd[li(width, i, j)], hvd[li(width, i + 1, j)], llf);
//}
//
//__global__ void computeVerticalFluxes(float* hd, float* hud, float* hvd, float* Ghd, float* Ghud, float* Ghvd, int width, int height, float g)
//{
//	int i = threadIdx.x + blockIdx.x * blockDim.x;
//	int j = threadIdx.y + blockIdx.y * blockDim.y;
//
//	if (i >= width - 1 || j >= height - 1)
//		return;
//
//	float llf = computeLocalSV(hd, hud, hvd, i, j, width, g, 'y');
//
//	Ghd[li(width - 1, i, j)] = computeFlux(hvd[li(width, i, j)], hvd[li(width, i, j + 1)], hd[li(width, i, j)], hd[li(width, i, j + 1)], llf);
//
//	Ghud[li(width - 1, i, j)] = computeFlux(hud[li(width, i, j)] * hvd[li(width, i, j)] / hd[li(width, i, j)], hud[li(width, i, j + 1)] * hvd[li(width, i, j + 1)] / hd[li(width, i, j + 1)], hud[li(width, i, j)], hud[li(width, i, j + 1)], llf);
//
//	Ghvd[li(width - 1, i, j)] = computeFlux(hvd[li(width, i, j)] * hvd[li(width, i, j)] / hd[li(width, i, j)] + 0.5f * g * hd[li(width, i, j)] * hd[li(width, i, j)],
//		hvd[li(width, i, j + 1)] * hvd[li(width, i, j + 1)] / hd[li(width, i, j + 1)] + 0.5f * g * hd[li(width, i, j + 1)] * hd[li(width, i, j + 1)],
//		hvd[li(width, i, j)],
//		hvd[li(width, i, j + 1)],
//		llf);
//}

__global__ void computeFluxes_kernel(float* hd, float* hud, float* hvd, float* Fhd, float* Fhud, float* Fhvd, float* Ghd, float* Ghud, float* Ghvd, int width, int height, float g)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if (i >= width - 1 || j >= height - 1)
		return;

	const float llfx = computeLocalSV(hd, hud, hvd, i, j, width, g, 'x');
	const float llfy = computeLocalSV(hd, hud, hvd, i, j, width, g, 'y');

	const int outIndex = li(width - 1, i, j);

	const float hd_curr = hd[li(width, i, j)];
	const float hd_right = hd[li(width, i + 1, j)];
	const float hd_top = hd[li(width, i, j + 1)];

	const float hud_curr = hud[li(width, i, j)];
	const float hud_right = hud[li(width, i + 1, j)];
	const float hud_top = hud[li(width, i, j + 1)];

	const float hvd_curr = hvd[li(width, i, j)];
	const float hvd_right = hvd[li(width, i + 1, j)];
	const float hvd_top = hvd[li(width, i, j + 1)];

	Fhd[outIndex] = computeFlux(hud_curr, hud_right, hd_curr, hd_right, llfx);

	Fhud[outIndex] = computeFlux(hud_curr * hud_curr / hd_curr + 0.5f * g * hd_curr * hd_curr,
		hud_right * hud_right / hd_right + 0.5f * g * hd_right * hd_right,
		hud_curr,
		hud_right,
		llfx);

	Fhvd[outIndex] = computeFlux(hud_curr * hvd_curr / hd_curr, hud_right * hvd_right / hd_right, hvd_curr, hvd_right, llfx);

	Ghd[outIndex] = computeFlux(hvd_curr, hvd_top, hd_curr, hd_top, llfy);

	Ghud[outIndex] = computeFlux(hud_curr * hvd_curr / hd_curr, hud_top * hvd_top / hd_top, hud_curr, hud_top, llfy);

	Ghvd[outIndex] = computeFlux(hvd_curr * hvd_curr / hd_curr + 0.5f * g * hd_curr * hd_curr,
		hvd_top * hvd_top / hd_top + 0.5f * g * hd_top * hd_top,
		hvd_curr,
		hvd_top,
		llfy);
}

void SWE::computeFluxes()
{
	dim3 gridSize = dim3(divUp(nx + 1, blockSize.x), divUp(ny + 1, blockSize.y));
	computeFluxes_kernel << <gridSize, blockSize >> >(hd, hud, hvd, Fhd, Fhud, Fhvd, Ghd, Ghud, Ghvd, nx + 2, ny + 2, g);
}

__global__ void computeBathymetrySources_kernel(float* hd, float* bd, float* Bud, float* Bvd, int width, int height, float g)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
	int j = threadIdx.y + blockIdx.y * blockDim.y + 1;

	if (i >= width - 1 || j >= height - 1)
		return;

	Bud[li(width, i, j)] = g * 0.5f * (hd[li(width, i, j)] + hd[li(width, i - 1, j)]) * (bd[li(width, i, j)] - bd[li(width, i - 1, j)]);
	Bvd[li(width, i, j)] = g * 0.5f * (hd[li(width, i, j)] + hd[li(width, i, j - 1)]) * (bd[li(width, i, j)] - bd[li(width, i, j - 1)]);
}

void SWE::computeBathymetrySources()
{
	dim3 gridSize = dim3(divUp(nx, blockSize.x), divUp(ny, blockSize.y));
	computeBathymetrySources_kernel << <gridSize, blockSize >> >(hd, bd, Bud, Bvd, nx + 2, ny + 2, g);
}


__global__ void eulerTimestep_kernel(float* hd, float* hud, float* hvd, float* Fhd, float* Fhud, float* Fhvd, float* Ghd, float* Ghud, float* Ghvd, float* Bud, float* Bvd, int width, int height, float dt, float dx, float dy)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
	int j = threadIdx.y + blockIdx.y * blockDim.y + 1;

	if (i >= width - 1 || j >= height - 1)
		return;

	const int currentIndexH = li(width, i, j);
	const int currentIndex = li(width - 1, i, j);
	const int leftIndex = li(width - 1, i - 1, j);
	const int bottomIndex = li(width - 1, i, j - 1);

	hd[currentIndexH] -= dt*((Fhd[currentIndex] - Fhd[leftIndex]) / dx + (Ghd[currentIndex] - Ghd[bottomIndex]) / dy);
	hud[currentIndexH] -= dt*((Fhud[currentIndex] - Fhud[leftIndex]) / dx + (Ghud[currentIndex] - Ghud[bottomIndex]) / dy + Bud[currentIndexH] / dx);
	hvd[currentIndexH] -= dt*((Fhvd[currentIndex] - Fhvd[leftIndex]) / dx + (Ghvd[currentIndex] - Ghvd[bottomIndex]) / dy + Bvd[currentIndexH] / dy);
}

float SWE::eulerTimestep()
{
	computeFluxes();

	dim3 gridSize = dim3(divUp(nx, blockSize.x), divUp(ny, blockSize.y));
	eulerTimestep_kernel << <gridSize, blockSize >> >(hd, hud, hvd, Fhd, Fhud, Fhvd, Ghd, Ghud, Ghvd, Bud, Bvd, nx + 2, ny + 2, dt, dx, dy);

	return dt;
}

inline __device__ float warpReduceMax(float val)
{
	for (int offset = warpSize / 2; offset > 0; offset /= 2)
		val = fmaxf(val, __shfl_down(val, offset));
	return val;
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

__global__ void getMaximumWaveSpeed(float* hd, float* hud, float* hvd, int width, int height, float* maxWaveSpeed, float g)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
	int j = threadIdx.y + blockIdx.y * blockDim.y + 1;

	if (i >= width - 1 || j >= height - 1)
		return;

	float localMaxWaveSpeed = 0.0f;
	if (hd[li(width, i, j)] > 1e-5f)
	{
		float momentum = fmaxf(fabsf(hud[li(width, i, j)]), fabsf(hvd[li(width, i, j)]));
		float particleVelocity = momentum / hd[li(width, i, j)];

		localMaxWaveSpeed = particleVelocity + sqrtf(g * hd[li(width, i, j)]);
	}

	//do the reduction
	float warpMaxWaveSpeed = warpReduceMax(localMaxWaveSpeed);

	//use atomics to determine max speed
	if (getLaneId() == 0)
	{
		atomicMax(maxWaveSpeed, warpMaxWaveSpeed);
	}
}

float SWE::getMaxTimestep(float cfl_number)
{
	float maximumWaveSpeed = 0.0f;
	float* maximumWaveSpeed_device;
	checkCudaErrors(cudaMalloc(&maximumWaveSpeed_device, sizeof(float)));
	checkCudaErrors(cudaMemset(maximumWaveSpeed_device, 0, sizeof(float)));

	dim3 gridSize = dim3(divUp(nx, blockSize.x), divUp(ny, blockSize.y));
	getMaximumWaveSpeed << <gridSize, blockSize >> >(hd, hud, hvd, nx + 2, ny + 2, maximumWaveSpeed_device, g);

	checkCudaErrors(cudaMemcpy(&maximumWaveSpeed, maximumWaveSpeed_device, sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(maximumWaveSpeed_device));

	return cfl_number * fminf(dx, dy) / maximumWaveSpeed;
}


float SWE::simulate(float tStart, float tEnd)
{
	static int iter = 0;

	float t = tStart;
	do
	{
		float tMax = getMaxTimestep();
		setTimestep(tMax);
		setBoundaryLayer();
		computeBathymetrySources();
		t += eulerTimestep();
		iter++;
	} while (t < tEnd);

	return t;
}

void SWE::writeVTKFile(string FileName)
{
	downloadSolution();
	// VTK HEADER
	Vtk_file.open(FileName.c_str());
	Vtk_file << "# vtk DataFile Version 2.0" << endl;
	Vtk_file << "HPC Tutorials: Michael Bader, Kaveh Rahnema, Oliver Meister" << endl;
	Vtk_file << "ASCII" << endl;
	Vtk_file << "DATASET RECTILINEAR_GRID" << endl;
	Vtk_file << "DIMENSIONS " << nx + 1 << " " << ny + 1 << " " << "1" << endl;
	Vtk_file << "X_COORDINATES " << nx + 1 << " float" << endl;
	//GITTER PUNKTE
	for (int i = 0; i < nx + 1; i++)
		Vtk_file << i*dx << endl;
	Vtk_file << "Y_COORDINATES " << ny + 1 << " float" << endl;
	//GITTER PUNKTE
	for (int i = 0; i < ny + 1; i++)
		Vtk_file << i*dy << endl;
	Vtk_file << "Z_COORDINATES 1 float" << endl;
	Vtk_file << "0" << endl;
	Vtk_file << "CELL_DATA " << ny*nx << endl;
	Vtk_file << "SCALARS H float 1" << endl;
	Vtk_file << "LOOKUP_TABLE default" << endl;
	//DOFS
	for (int j = 1; j <= ny; j++)
		for (int i = 1; i <= nx; i++)
			Vtk_file << h[li(nx + 2, i, j)] /* + b[li(nx + 2, i, j)]*/ << endl;
	Vtk_file << "SCALARS HU float 1" << endl;
	Vtk_file << "LOOKUP_TABLE default" << endl;
	for (int j = 1; j <= ny; j++)
		for (int i = 1; i <= nx; i++)
			Vtk_file << hu[li(nx + 2, i, j)] /*/ h[li(nx + 2, i, j)]*/ << endl;
	Vtk_file << "SCALARS HV float 1" << endl;
	Vtk_file << "LOOKUP_TABLE default" << endl;
	for (int j = 1; j <= ny; j++)
		for (int i = 1; i <= nx; i++)
			Vtk_file << hv[li(nx + 2, i, j)] /*/ h[li(nx + 2, i, j)]*/ << endl;
	Vtk_file << "SCALARS B float 1" << endl;
	Vtk_file << "LOOKUP_TABLE default" << endl;
	for (int j = 1; j <= ny; j++)
		for (int i = 1; i <= nx; i++)
			Vtk_file << b[li(nx + 2, i, j)] << endl;
	Vtk_file << "SCALARS WATER_HEIGHT float 1" << endl;
	Vtk_file << "LOOKUP_TABLE default" << endl;
	for (int j = 1; j <= ny; j++)
		for (int i = 1; i <= nx; i++)
			Vtk_file << h[li(nx + 2, i, j)] + b[li(nx + 2, i, j)] << endl;
	Vtk_file.close();
}