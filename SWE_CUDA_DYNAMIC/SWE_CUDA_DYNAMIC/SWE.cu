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
	ngh = new float[variableSize];
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
	computeInitialRefinement(); //set the tree and solution vector to the desired levels
}

void SWE::downloadSolution()
{
	int numElem = (nx + 2) * (ny + 2);
	checkCudaErrors(cudaMemcpyAsync(h, hd, numElem * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpyAsync(ngh, nghd, numElem * sizeof(float), cudaMemcpyDeviceToHost));
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

__global__ void expandSolutionKernel(float* hd, float* hud, float* hvd, int xOff, int yOff, int d, int width, int height)
{
	const int writeX = xOff + threadIdx.x + blockIdx.x * blockDim.x;
	const int writeY = yOff + threadIdx.y + blockIdx.y * blockDim.y;

	const int readX = xOff + (threadIdx.x + blockIdx.x * blockDim.x) / d; //round down
	const int readY = yOff + (threadIdx.y + blockIdx.y * blockDim.y) / d;

	hd[li(width, writeX, writeY)] = hd[li(width, readX, readY)];
	hud[li(width, writeX, writeY)] = hud[li(width, readX, readY)];
	hvd[li(width, writeX, writeY)] = hvd[li(width, readX, readY)];
}

__global__ void eulerTimestepKernel(int xOff, int yOff, int d, int depth, 
	int width, int height, 
	int* td, float* hd, float* hud, float* hvd, 
	float* Fhd, float* Fhud, float* Fhvd, 
	float* Ghd, float* Ghud, float* Ghvd, 
	float* Bud, float* Bvd, 
	float dt, float dx, float dy)
{
	//compute global starting point of cell
	xOff += (threadIdx.x + blockIdx.x * blockDim.x) * d;
	yOff += (threadIdx.y + blockIdx.y * blockDim.y) * d;
	const int tid = li(blockDim.x, threadIdx.x, threadIdx.y);

	__shared__ int levelSum[BX * BY];

	//are we a leaf?
	if (depth == MAX_DEPTH)
	{
		//minimal cell-length (pixel)
		if (xOff >= width - 1 || yOff >= height - 1)
			return;

		const int currentIndexH = li(width, xOff, yOff);
		const int currentIndex = li(width - 1, xOff, yOff);
		const int leftIndex = li(width - 1, xOff - 1, yOff);
		const int bottomIndex = li(width - 1, xOff, yOff - 1);

		hd[currentIndexH] -= dt * ((Fhd[currentIndex] - Fhd[leftIndex]) / dx + (Ghd[currentIndex] - Ghd[bottomIndex]) / dy);
		hud[currentIndexH] -= dt * ((Fhud[currentIndex] - Fhud[leftIndex]) / dx + (Ghud[currentIndex] - Ghud[bottomIndex]) / dy + Bud[currentIndexH] / dx);
		hvd[currentIndexH] -= dt * ((Fhvd[currentIndex] - Fhvd[leftIndex]) / dx + (Ghvd[currentIndex] - Ghvd[bottomIndex]) / dy + Bvd[currentIndexH] / dy);
	}
	else
	{
		//each thread reads it's level into shared memory
		levelSum[tid] = td[li(width, xOff, yOff)];
		__syncthreads();
		//reduce
		for (int nt = BX * BY; nt > 1; nt /= 2)
		{
			if (tid < nt / 2)
			{
				levelSum[tid] += levelSum[tid + nt / 2]; //TODO: maximum here
			}
			__syncthreads();
		}
		
		//each thread reads the result
		if (levelSum[0] != BX * BY * depth) //TODO: check logic in these if-cases
		{
			//we have to refine this block, thread(0, 0) will do this
			if (tid == 0)
			{
				dim3 block(BX, BY);
				dim3 grid(SUBDIV, SUBDIV);
				eulerTimestepKernel << <grid, block >> >(xOff, yOff, d / SUBDIV, depth + 1, width, height, td, hd, hud, hvd, Fhd, Fhud, Fhvd, Ghd, Ghud, Ghvd, Bud, Bvd, dt, dx / SUBDIV, dy / SUBDIV);
			}
		}
		else
		{
			float currentH = hd[li(width, xOff, yOff)];
			float currentHu = hud[li(width, xOff, yOff)];
			float currentHv = hvd[li(width, xOff, yOff)];

			//we can use this depth for the whole thread-block
			//note that we know, that all cells (threads) in this block have the same depth as we do and use this for faster calculation of fluxes
			float  Fhd_right, Fhd_left, Ghd_top, Ghd_bottom,
				Fhud_right, Fhud_left, Ghud_top, Ghud_bottom,
				Fhvd_right, Fhvd_left, Ghvd_top, Ghvd_bottom;

			Fhd_left = d * Fhd[li(width - 1, xOff - 1, yOff)];
			Fhd_right = d * Fhd[li(width - 1, xOff + d - 1, yOff)];
			Fhud_left = d * Fhud[li(width - 1, xOff - 1, yOff)];
			Fhud_right = d * Fhud[li(width - 1, xOff + d - 1, yOff)];
			Fhvd_left = d * Fhvd[li(width - 1, xOff - 1, yOff)];
			Fhvd_right = d * Fhvd[li(width - 1, xOff + d - 1, yOff)];
			Ghd_bottom = d * Ghd[li(width - 1, xOff, yOff - 1)];
			Ghd_top = d * Ghd[li(width - 1, xOff, yOff + d - 1)];
			Ghud_bottom = d * Ghud[li(width - 1, xOff, yOff - 1)];
			Ghud_top = d * Ghud[li(width - 1, xOff, yOff + d - 1)];
			Ghvd_bottom = d * Ghvd[li(width - 1, xOff, yOff - 1)];
			Ghvd_top = d * Ghvd[li(width - 1, xOff, yOff + d - 1)];


			//overwrite with specific values:
			if (threadIdx.x == 0) //left border
			{
				Fhd_left = 0.0f;
				Fhud_left = 0.0f;
				Fhvd_left = 0.0f;

				for (int i = 0; i < SUBDIV; i++)
				{
					Fhd_left += d / SUBDIV * Fhd[li(width - 1, xOff - 1, yOff + i * d / SUBDIV)];
					Fhud_left += d / SUBDIV * Fhud[li(width - 1, xOff - 1, yOff + i * d / SUBDIV)];
					Fhvd_left += d / SUBDIV * Fhvd[li(width - 1, xOff - 1, yOff + i * d / SUBDIV)];
				}
			}
			if (threadIdx.x == blockDim.x - 1) //right border
			{
				Fhd_right = 0.0f;
				Fhud_right = 0.0f;
				Fhvd_right = 0.0f;
				
				for (int i = 0; i < SUBDIV; i++)
				{
					Fhd_right += d / SUBDIV * Fhd[li(width - 1, xOff + d - 1, yOff + i * d / SUBDIV)];
					Fhud_right += d / SUBDIV * Fhud[li(width - 1, xOff + d - 1, yOff + i * d / SUBDIV)];
					Fhvd_right += d / SUBDIV * Fhvd[li(width - 1, xOff + d - 1, yOff + i * d / SUBDIV)];
				}
			}

			if (threadIdx.y == 0) //bottom border
			{
				Ghd_bottom = 0.0f;
				Ghud_bottom = 0.0f;
				Ghvd_bottom = 0.0f;

				for (int i = 0; i < SUBDIV; i++)
				{
					Ghd_bottom += d / SUBDIV * Ghd[li(width - 1, xOff + i * d / SUBDIV, yOff - 1)];
					Ghud_bottom += d / SUBDIV * Ghud[li(width - 1, xOff + i * d / SUBDIV, yOff - 1)];
					Ghvd_bottom += d / SUBDIV * Ghvd[li(width - 1, xOff + i * d / SUBDIV, yOff - 1)];
				}
			}
			if (threadIdx.y == blockDim.y - 1) //top border
			{
				Ghd_top = 0.0f;
				Ghud_top = 0.0f;
				Ghvd_top = 0.0f;

				for (int i = 0; i < SUBDIV; i++)
				{
					Ghd_top += d / SUBDIV * Ghd[li(width - 1, xOff + i * d / SUBDIV, yOff + d - 1)];
					Ghud_top += d / SUBDIV * Ghud[li(width - 1, xOff + i * d / SUBDIV, yOff + d - 1)];
					Ghvd_top += d / SUBDIV * Ghvd[li(width - 1, xOff + i * d / SUBDIV, yOff + d - 1)];
				}
			}
			
			//use these values to update current solution
			currentH -= dt * ((Fhd_right - Fhd_left) / dx + (Ghd_top - Ghd_bottom) / dy);
			currentHu -= dt * ((Fhud_right - Fhud_left) / dx + (Ghud_top - Ghud_bottom) / dy /*+ Bud[currentIndexH] / dx*/); //TODO: add bathymetry term
			currentHv -= dt * ((Fhvd_right - Fhvd_left) / dx + (Ghvd_top - Ghvd_bottom) / dy /*+ Bvd[currentIndexH] / dy*/);

			//write solution back
			hd[li(width, xOff, yOff)] = currentH;
			hud[li(width, xOff, yOff)] = currentHu;
			hvd[li(width, xOff, yOff)] = currentHv;

			//important! synchronize global writes
			__syncthreads();

			//expand solution to whole region
			if (tid == 0)
			{
				dim3 block(BX, BY);
				dim3 grid(d, d); 
				expandSolutionKernel << <grid, block >> >(hd, hud, hvd, xOff, yOff, d, width, height);
			}
		}
	}
}

float SWE::eulerTimestep()
{
	computeFluxes();

	dim3 grid(INIT_SUBDIV, INIT_SUBDIV);
	dim3 block(BX, BY);
	eulerTimestepKernel << <grid, block >> >(1, 1, nx / (INIT_SUBDIV * BX), 0, 
		nx + 2, ny + 2, 
		td, hd, hud, hvd, 
		Fhd, Fhud, Fhvd, 
		Ghd, Ghud, Ghvd, 
		Bud, Bvd, 
		dt, dx * nx / (INIT_SUBDIV * BX), dy * ny / (INIT_SUBDIV * BY));

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

__global__ void setSolutionAndTreeKernel(int* td, float* hd, float* hud, float* hvd, int tVal, float hVal, float huVal, float hvVal, int width, int height, int startX, int startY, int d)
{
	int xoff = threadIdx.x + blockIdx.x * blockDim.x;
	int yoff = threadIdx.y + blockIdx.y * blockDim.y;

	if (xoff >= d || yoff >= d || startX + xoff >= width || startY + yoff >= height)
		return;

	td[li(width, startX + xoff, startY + yoff)] = tVal;
	hd[li(width, startX + xoff, startY + yoff)] = hVal;
	hud[li(width, startX + xoff, startY + yoff)] = huVal;
	hvd[li(width, startX + xoff, startY + yoff)] = hvVal;
}

__global__ void computeInitialRefinementKernel(float* hd, float* hud, float* hvd, float* normGradH, int* td, int d, int* d_levels, int depth, float theta_cor, int width, int height)
{
	int i = (threadIdx.x + blockIdx.x * blockDim.x) * d + 1;
	int j = (threadIdx.y + blockIdx.y * blockDim.y) * d + 1;

	//TODO: use grad(b+h)

	float ngH = 0.0f;
	float cH = 0.0f;
	float cHu = 0.0f;
	float cHv = 0.0f;
	int levels = 0;
	//also capture the neighbors:
	int nLevels = 0;

	for (int xOff = 0; xOff < SUBDIV; xOff++)
	{
		for (int yOff = 0; yOff < SUBDIV; yOff++)
		{
			levels += td[li(width, i + xOff * d / SUBDIV, j + yOff * d / SUBDIV)];
			ngH += normGradH[li(width, i + xOff * d / SUBDIV, j + yOff * d / SUBDIV)];
			cH += hd[li(width, i + xOff * d / SUBDIV, j + yOff * d / SUBDIV)];
			cHu += hud[li(width, i + xOff * d / SUBDIV, j + yOff * d / SUBDIV)];
			cHv += hvd[li(width, i + xOff * d / SUBDIV, j + yOff * d / SUBDIV)];

			if (xOff == 0) //left neighbor
				nLevels += td[li(width, i - 1, j + yOff * d / SUBDIV)];
			if (xOff == SUBDIV - 1) //right neighbor
				nLevels += td[li(width, i + d, j + yOff * d / SUBDIV)];
			if (yOff == 0) //bottom neighbor
				nLevels += td[li(width, i + xOff * d / SUBDIV, j - 1)];
			if (yOff == SUBDIV - 1) //top neighbor
				nLevels += td[li(width, i + xOff * d / SUBDIV, j + d)];
		}
	}

	//if all are on the same level and the neighbors are also on our level (or already on a higher level) and the averaged norm is below the threshold
	if (levels <= SUBDIV * SUBDIV * (depth + 1) && nLevels <= 4 * SUBDIV * (depth + 1) && ngH / (SUBDIV * SUBDIV) <= theta_cor)
	{
		//average solution and write depth to tree
		fillRectLoop(hd, width, height, i, j, d, d, cH / (SUBDIV * SUBDIV));
		fillRectLoop(hud, width, height, i, j, d, d, cHu / (SUBDIV * SUBDIV));
		fillRectLoop(hvd, width, height, i, j, d, d, cHv / (SUBDIV * SUBDIV));

		//also capture boundary values
		if (i == 1 && j == 1)
			fillRectLoop(td, width, height, i - 1, j - 1, d + 1, d + 1, depth);
		else if (i == 1 && j == height - d - 1)
			fillRectLoop(td, width, height, i - 1, j, d + 1, d + 1, depth);
		else if (i == width - d - 1 && j == 1)
			fillRectLoop(td, width, height, i, j - 1, d + 1, d + 1, depth);
		else if (i == width - d - 1 && j == height - d - 1)
			fillRectLoop(td, width, height, i, j, d + 1, d + 1, depth);
		else if (i == 1)
			fillRectLoop(td, width, height, i - 1, j, d + 1, d, depth);
		else if (i == width - d - 1)
			fillRectLoop(td, width, height, i, j, d + 1, d, depth);
		else if (j == 1)
			fillRectLoop(td, width, height, i, j - 1, d, d + 1, depth);
		else if (j == height - d - 1)
			fillRectLoop(td, width, height, i, j, d, d + 1, depth);
		else
			fillRectLoop(td, width, height, i, j, d, d, depth);

		//update statistic counter
		atomicAdd(d_levels + (depth + 1), -SUBDIV * SUBDIV); //subtract subdiv^2 cells from smaller level
		atomicAdd(d_levels + depth, 1); //add 1 to current cell level
	}
	//else do nothing
}

__global__ void computeNormOfGradient(float* hd, float* normGradH, int width, int height)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if (i >= width || j >= height)
		return;

	//using central differences
	float dhdx = 0.5f * (hd[li(width, min(i + 1, width - 1), j)] - hd[li(width, max(i - 1, 0), j)]);
	float dhdy = 0.5f * (hd[li(width, i, min(j + 1, height - 1))] - hd[li(width, i, max(j - 1, 0))]);

	normGradH[li(width, i, j)] = sqrtf(dhdx * dhdx + dhdy * dhdy);
}

void SWE::computeInitialRefinement()
{
	setTree(MAX_DEPTH); //reset the tree to finest resolution
	int levelCount[MAX_DEPTH]; //number of cells for each level < maxLevel
	int* d_levelCount;
	checkCudaErrors(cudaMalloc(&d_levelCount, MAX_DEPTH * sizeof(int)));
	checkCudaErrors(cudaMemset(d_levelCount, 0, MAX_DEPTH * sizeof(int)));

	dim3 normBlock(BX, BY);
	dim3 normGrid(divUp(nx + 2, normBlock.x), divUp(ny + 2, normBlock.y));
	computeNormOfGradient << <normGrid, normBlock >> >(hd, nghd, nx + 2, ny + 2);

	dim3 grid(nx / (SUBDIV * BX), ny / (SUBDIV * BY));
	dim3 block(BX, BY);
	int d = SUBDIV;
	computeInitialRefinementKernel << <grid, block >> >(hd, hud, hvd, nghd, td, d, d_levelCount, MAX_DEPTH - 1, 0.0f, nx + 2, ny + 2);

	d *= SUBDIV;
	grid.x /= SUBDIV;
	grid.y /= SUBDIV;
	computeInitialRefinementKernel << <grid, block >> >(hd, hud, hvd, nghd, td, d, d_levelCount, MAX_DEPTH - 2, 0.0f, nx + 2, ny + 2);

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
		computeInitialRefinement();
		iter++;

		//debug output
		//cout << "Writing file for iteration: " << iter << endl;
		//writeVTKFile(generateFileName("iter", iter));
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
	Vtk_file << "SCALARS NORM_GRAD_H float 1" << endl;
	Vtk_file << "LOOKUP_TABLE default" << endl;
	for (int j = 1; j <= ny; j++)
		for (int i = 1; i <= nx; i++)
			Vtk_file << ngh[li(nx + 2, i, j)] << endl;
	Vtk_file << "SCALARS TREE int 1" << endl;
	Vtk_file << "LOOKUP_TABLE default" << endl;
	for (int j = 1; j <= ny; j++)
		for (int i = 1; i <= nx; i++)
			Vtk_file << t[li(nx + 2, i, j)] << endl;
	Vtk_file.close();
}