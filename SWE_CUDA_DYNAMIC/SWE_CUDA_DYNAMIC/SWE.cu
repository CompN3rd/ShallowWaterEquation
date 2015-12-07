#include "SWE.h"
#include "Utils.h"
#include <algorithm>

SWE::SWE(int _nx, int _ny, float _dx, float _dy, float _g, int maxRecursion, int blockX, int blockY)
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
	this->maxRecursion = maxRecursion;
	//if (getCellExt(blockX, maxRecursion) != nx || getCellExt(blockY, maxRecursion) != ny)
	//{
	//	cerr << "Block size and computation area do not match" << endl;
	//	exit(1);
	//}

	//cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, maxRecursion + 1);
	//cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, nx > ny ? nx : ny);

	const int variableSize = (nx + 2) * (ny + 2);
	h = new float[variableSize];
	hu = new float[variableSize];
	hv = new float[variableSize];
	t = new int[variableSize];
	b = new float[variableSize];

	//device arrays
	checkCudaErrors(cudaMalloc(&td, variableSize * sizeof(int)));
	setTree(MAX_DEPTH); //set the tree to the finest recursion

	checkCudaErrors(cudaMalloc(&hd, variableSize * sizeof(float)));
	checkCudaErrors(cudaMalloc(&nghd, variableSize * sizeof(float)));
	checkCudaErrors(cudaMalloc(&hud, variableSize * sizeof(float)));
	checkCudaErrors(cudaMalloc(&hvd, variableSize * sizeof(float)));
	checkCudaErrors(cudaMalloc(&bd, variableSize * sizeof(float)));
	checkCudaErrors(cudaMalloc(&Bud, variableSize * sizeof(float)));
	checkCudaErrors(cudaMalloc(&Bvd, variableSize * sizeof(float)));
	computeRefinement(); //set the tree and solution vector to the desired levels

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
	delete[] t;
	delete[] b;
	checkCudaErrors(cudaFree(td));
	checkCudaErrors(cudaFree(hd));
	checkCudaErrors(cudaFree(nghd));
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

__global__ void setTree_kernel(int* td, int width, int height, int layer)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= width || y >= height)
		return;

	td[li(width, x, y)] = layer;
}

void SWE::setTree(int layer)
{
	dim3 grid(divUp(nx + 2, blockSize.x), divUp(ny + 2, blockSize.y));
	setTree_kernel << <grid, blockSize >> >(td, nx + 2, ny + 2, layer);
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
	checkCudaErrors(cudaMemcpyAsync(hv, hud, numElem * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(t, td, numElem * sizeof(float), cudaMemcpyDeviceToHost));
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

//-----------------------------------------------------------------------------------------------------------------------------
__global__ void eulerTimestepPixel_kernel(int xOff, int yOff, int d, float* hd, float* hud, float* hvd, float* Fhd, float* Fhud, float* Fhvd, float* Ghd, float* Ghud, float* Ghvd, float* Bud, float* Bvd, int width, int height, float dt, float dx, float dy)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if (i >= d || j >= d)
		return;

	i += xOff;
	j += yOff;

	if (i >= width - 1 || j >= width - 1)
		return;

	const int currentIndexH = li(width, i, j);
	const int currentIndex = li(width - 1, i, j);
	const int leftIndex = li(width - 1, i - 1, j);
	const int bottomIndex = li(width - 1, i, j - 1);

	hd[currentIndexH] -= dt*((Fhd[currentIndex] - Fhd[leftIndex]) / dx + (Ghd[currentIndex] - Ghd[bottomIndex]) / dy);
	hud[currentIndexH] -= dt*((Fhud[currentIndex] - Fhud[leftIndex]) / dx + (Ghud[currentIndex] - Ghud[bottomIndex]) / dy + Bud[currentIndexH] / dx);
	hvd[currentIndexH] -= dt*((Fhvd[currentIndex] - Fhvd[leftIndex]) / dx + (Ghvd[currentIndex] - Ghvd[bottomIndex]) / dy + Bvd[currentIndexH] / dy);
}

__global__ void eulerTimestep_kernel(int xOff, int yOff, int d, int depth, 
	int width, int height, 
	int* td, float* hd, float* hud, float* hvd, 
	float* Fhd, float* Fhud, float* Fhvd, 
	float* Ghd, float* Ghud, float* Ghvd, 
	float* Bud, float* Bvd, 
	float dt, float dx, float dy)
{
	xOff += blockIdx.x * d;
	yOff += blockIdx.y * d;
	int treeVal = td[li(width, xOff, yOff)];

	float Fhd_right, Fhd_left, Ghd_top, Ghd_bottom,
		Fhud_right, Fhud_left, Ghud_top, Ghud_bottom,
		Fhvd_right, Fhvd_left, Ghvd_top, Ghvd_bottom;

	if (treeVal == depth)
	{
		Fhd_right = sumLine(Fhd, width - 1, height - 1, xOff + d, yOff, d, false);
		Fhd_left = sumLine(Fhd, width - 1, height - 1, xOff - 1, yOff, d, false);
		Ghd_top = sumLine(Ghd, width - 1, height - 1, xOff, yOff + d, d, true);
		Ghd_bottom = sumLine(Ghd, width - 1, height - 1, xOff, yOff - 1, d, true);

		Fhud_right = sumLine(Fhud, width - 1, height - 1, xOff + d, yOff, d, false);
		Fhud_left = sumLine(Fhud, width - 1, height - 1, xOff - 1, yOff, d, false);
		Ghud_top = sumLine(Ghud, width - 1, height - 1, xOff, yOff + d, d, true);
		Ghud_bottom = sumLine(Ghud, width - 1, height - 1, xOff, yOff - 1, d, true);

		Fhvd_right = sumLine(Fhvd, width - 1, height - 1, xOff + d, yOff, d, false);
		Fhvd_left = sumLine(Fhvd, width - 1, height - 1, xOff - 1, yOff, d, false);
		Ghvd_top = sumLine(Ghvd, width - 1, height - 1, xOff, yOff + d, d, true);
		Ghvd_bottom = sumLine(Ghvd, width - 1, height - 1, xOff, yOff - 1, d, true);
	}

	if (threadIdx.x == 0 && threadIdx.y == 0)
	{
		if (treeVal == depth)
		{
			float currentH = hd[li(width, xOff, yOff)];
			float currentHu = hud[li(width, xOff, yOff)];
			float currentHv = hvd[li(width, xOff, yOff)];
			
			currentH -= dt * ((Fhd_right - Fhd_left) / dx + (Ghd_top - Ghd_bottom) / dy);
			currentHu -= dt * ((Fhud_right - Fhud_left) / dx + (Ghud_top - Ghud_bottom) / dy); //TODO: add bathymetry
			currentHv -= dt * ((Fhvd_right - Fhvd_left) / dx + (Ghvd_top - Ghvd_bottom) / dy);

			//fill cell with calculated updates
			fillRectDynamic(hd, width, height, xOff, yOff, d, d, currentH);
			fillRectDynamic(hud, width, height, xOff, yOff, d, d, currentHu);
			fillRectDynamic(hvd, width, height, xOff, yOff, d, d, currentHv);
		}
		else if (depth + 1 < MAX_DEPTH)
		{
			//subdivide recursively
			dim3 bs(BX, BY), grid(SUBDIV, SUBDIV);
			eulerTimestep_kernel << < grid, bs >> > (xOff, yOff, d / SUBDIV, depth + 1, width, height, td, hd, hud, hvd, Fhd, Fhud, Fhvd, Ghd, Ghud, Ghvd, Bud, Bvd, dt, dx / SUBDIV, dy / SUBDIV);
		}
		else
		{
			//leaf, per pixel kernel
			dim3 bs(BX, BY), grid(divUp(d, BX), divUp(d, BY));
			eulerTimestepPixel_kernel << <grid, bs >> >(xOff, yOff, d, hd, hud, hvd, Fhd, Fhud, Fhvd, Ghd, Ghud, Ghvd, Bud, Bvd, width, height, dt, dx / SUBDIV, dy / SUBDIV);
		}
	}
}

float SWE::eulerTimestep()
{
	computeFluxes();

	dim3 grid(INIT_SUBDIV, INIT_SUBDIV);
	dim3 block(BX, BY);
	eulerTimestep_kernel << <grid, block >> >(1, 1, nx / INIT_SUBDIV, 0, 
		nx + 2, ny + 2, 
		td, hd, hud, hvd, 
		Fhd, Fhud, Fhvd, 
		Ghd, Ghud, Ghvd, 
		Bud, Bvd, 
		dt, dx * nx / INIT_SUBDIV, dy * nx / INIT_SUBDIV);

	return dt;
}

//-----------------------------------------------------------------------------------------------------------------------------
__global__ void getMaximumWaveSpeed(float* hd, float* hud, float* hvd, int width, int height, float* maxWaveSpeed, float g)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
	int j = threadIdx.y + blockIdx.y * blockDim.y + 1;

	float localMaxWaveSpeed = 0.0f;
	float currH = 0.0f;

	if (i < width - 1 || j < height - 1)
	{
		currH = hd[li(width, i, j)];
		if (currH > 1e-5f)
		{
			float momentum = fmaxf(fabsf(hud[li(width, i, j)]), fabsf(hvd[li(width, i, j)]));
			float particleVelocity = momentum / currH;

			localMaxWaveSpeed = particleVelocity + sqrtf(g * currH);
		}
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

__global__ void computeRefinementFirstRecursionKernel(float* hd, float* hud, float* hvd, float* normGradH, int* td, int* d_levels, int d, float theta_cor, int width, int height)
{
	//TODO: compute gradient of b + h and not only of h!

	//compute global starting point
	const int i = threadIdx.x * SUBDIV + blockIdx.x * d + 1;
	const int j = threadIdx.y * SUBDIV + blockIdx.y * d + 1;
	const int tid = threadIdx.x + threadIdx.y * blockDim.x;

	__shared__ float gradHNorm[BX * BY];
	__shared__ float avgH[BX * BY];
	__shared__ float avgHu[BX * BY];
	__shared__ float avgHv[BX * BY];

	float gradNorm = 0.0f;
	float aH = 0.0f;
	float aHu = 0.0f;
	float aHv = 0.0f;
	for (int xInd = i; xInd < i + SUBDIV; xInd++)
	{
		for (int yInd = j; yInd < j + SUBDIV; yInd++)
		{
			gradNorm += normGradH[li(width, xInd, yInd)];

			aH += hd[li(width, xInd, yInd)];
			aHu += hud[li(width, xInd, yInd)];
			aHv += hvd[li(width, xInd, yInd)];
		}
	}
	gradHNorm[tid] = gradNorm;
	avgH[tid] = aH;
	avgHu[tid] = aHu;
	avgHv[tid] = aHv;
	__syncthreads();

	//block reduce the gradHNorm and the average cell values:
	for (int nt = BX * BY; nt > 1; nt /= 2)
	{
		if (tid < nt / 2)
		{
			gradHNorm[tid] += gradHNorm[tid + nt / 2];
			avgH[tid] += avgH[tid + nt / 2];
			avgHu[tid] += avgHu[tid + nt / 2];
			avgHv[tid] += avgHv[tid + nt / 2];
		}
		__syncthreads();
	}

	gradHNorm[tid] /= (BX * BY * SUBDIV * SUBDIV);
	avgH[tid] /= (BX * BY * SUBDIV * SUBDIV);
	avgHu[tid] /= (BX * BY * SUBDIV * SUBDIV);
	avgHv[tid] /= (BX * BY * SUBDIV * SUBDIV);

	//thread with tid 0 now holds the average water gradient and averaged cell values
	if (tid == 0)
	{
		printf("norm: %f\n", gradHNorm[tid]);
		//decide the content of the tree and the solution vector
		if (gradHNorm[tid] >= theta_cor)
		{
			//no recoarsening, leave everything as is and write max_depth to tree
			fillRectDynamic(td, width, height, i, j, d, d, MAX_DEPTH);
		}
		else
		{
			//coarsen the grid, write max_depth - 1 to tree and fill solution vector with averaged values
			atomicAdd(d_levels + (MAX_DEPTH - 1), 1); //increase counter for this cell level
			fillRectDynamic(td, width, height, i, j, d, d, MAX_DEPTH - 1);
			fillRectDynamic(hd, width, height, i, j, d, d, avgH[tid]);
			fillRectDynamic(hud, width, height, i, j, d, d, avgHu[tid]);
			fillRectDynamic(hvd, width, height, i, j, d, d, avgHv[tid]);
		}
	}
}

__global__ void computeHigherRecursionKernel(float* hd, float* hud, float* hvd, float* normGradH, int* td, int* d_levels, int d, int level, int width, int height)
{
	//TODO: compute gradient of b + h and not only of h!
	int i = d * (threadIdx.x + blockIdx.x * blockDim.x) + 1;
	int j = d * (threadIdx.y + blockIdx.y * blockDim.y) + 1;

	//check if all cells in a subdiv x subdiv grid are on the same level as this thread
	for (int xOff = 0; xOff < SUBDIV; xOff++)
	{
		for (int yOff = 0; yOff < SUBDIV; yOff++)
		{
			//one subcell is on a too fine level to coarsen the grid
			if (td[li(width, i + xOff * d / SUBDIV, j + yOff * d / SUBDIV)] > level)
				return;
		}
	}
}

__global__ void computeNormOfGradient(float* hd, float* normGradH, int width, int height)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.x + blockIdx.x * blockDim.x;

	if (i >= width || j >= height)
		return;

	//using central differences
	float dhdx = 0.5f * (hd[li(width, min(i + 1, width - 1), j)] - hd[li(width, max(i - 1, 0), j)]);
	float dhdy = 0.5f * (hd[li(width, i, min(j + 1, height - 1))] - hd[li(width, i, max(j - 1, 0))]);

	normGradH[li(width, i, j)] = sqrtf(dhdx * dhdx + dhdy * dhdy);
}

void SWE::computeRefinement()
{
	int levelCount[MAX_DEPTH]; //number of cells for each level < maxLevel
	int* d_levelCount;
	checkCudaErrors(cudaMalloc(&d_levelCount, MAX_DEPTH * sizeof(int)));
	checkCudaErrors(cudaMemset(d_levelCount, 0, MAX_DEPTH * sizeof(int)));

	dim3 normBlock(BX, BY);
	dim3 normGrid(divUp(nx + 2, normBlock.x), divUp(ny + 2, normBlock.y));
	computeNormOfGradient << <normGrid, normBlock >> >(hd, nghd, nx + 2, ny + 2);

	//each block computes a cell of size BX * SUBDIV
	dim3 grid(nx / (BX * SUBDIV), ny / (BY * SUBDIV));
	dim3 block(BX, BY);
	int d = BX * SUBDIV;
	cout << "cellLength for level " << MAX_DEPTH - 1 << " : " << d << endl;
	computeRefinementFirstRecursionKernel << <grid, block >> >(hd, hud, hvd, nghd, td, d_levelCount, d, 1e-20f, nx + 2, ny + 2);

	//higher recursive levels
	for (int level = MAX_DEPTH - 1; level > 0; level--)
	{
	//	//TODO: compute new nghd
	//	grid = dim3(grid.x / SUBDIV, grid.y / SUBDIV);
		d *= SUBDIV;
		cout << "cellLength for level " << level - 1 << " : " << d << endl;
	//	//each thread computes a cell of size SUBDIV
	//	dim3 gr(grid.x / block.x, grid.y / block.y);
	//	computeHigherRecursionKernel << <gr, block >> >(hd, hud, hvd, td, d_levelCount, d, level, nx + 2, ny + 2);
	}

	checkCudaErrors(cudaMemcpy(levelCount, d_levelCount, MAX_DEPTH * sizeof(int), cudaMemcpyDeviceToHost));

	int pixelCount = 0;
	for (int l = 0; l < MAX_DEPTH; l++)
	{
		cout << "Num Cells at level " << l << " : " << levelCount[l] << endl;
		pixelCount += levelCount[l] * d * d;
		d /= SUBDIV;
	}
	cout << "Num Cells at finest level: " << (nx * ny) - pixelCount << endl;

	checkCudaErrors(cudaFree(d_levelCount));
}

float SWE::simulate(float tStart, float tEnd)
{
	static int iter = 0;

	float t = tStart;
	do
	{
		float tMax = getMaxTimestep();
		setTimestep(tMax);
		cout << "Iteration: " << iter << ", Timestep: " << tMax << endl;
		setBoundaryLayer();
		computeBathymetrySources();
		t += eulerTimestep();
		computeRefinement();
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
	Vtk_file << "SCALARS TREE int 1" << endl;
	Vtk_file << "LOOKUP_TABLE default" << endl;
	for (int j = 1; j <= ny; j++)
		for (int i = 1; i <= nx; i++)
			Vtk_file << t[li(nx + 2, i, j)] << endl;
	Vtk_file.close();
}