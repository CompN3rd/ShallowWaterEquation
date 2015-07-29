#include "SWE.h"
#include "Utils.h"


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
	if (getCellExt(blockX, maxRecursion) != nx || getCellExt(blockY, maxRecursion) != ny)
	{
		cerr << "Block size and computation area do not match" << endl;
		exit(1);
	}

	//cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, maxRecursion + 1);
	//cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, nx > ny ? nx : ny);

	const int variableSize = (nx + 2) * (ny + 2);
	h = new float[variableSize];
	hu = new float[variableSize];
	hv = new float[variableSize];
	b = new float[variableSize];

	//device arrays
	checkCudaErrors(cudaMalloc(&td, variableSize * sizeof(int)));
	setTree(maxRecursion); //set the tree to the finest recursion

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
	checkCudaErrors(cudaFree(td));
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

__global__ void setTree_kernel(int* td, int width, int height, int layer)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= width || y >= height)
		return;

	td[li(width, x, y)] = layer;
	if(x >= width || y >= height)
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
//__global__ void eulerTimestep_kernel(float* hd, float* hud, float* hvd, float* Fhd, float* Fhud, float* Fhvd, float* Ghd, float* Ghud, float* Ghvd, float* Bud, float* Bvd, int width, int height, float dt, float dx, float dy)
//{
//	int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
//	int j = threadIdx.y + blockIdx.y * blockDim.y + 1;
//
//	if (i >= width - 1 || j >= height - 1)
//		return;
//
//	const int currentIndexH = li(width, i, j);
//	const int currentIndex = li(width - 1, i, j);
//	const int leftIndex = li(width - 1, i - 1, j);
//	const int bottomIndex = li(width - 1, i, j - 1);
//
//	hd[currentIndexH] -= dt*((Fhd[currentIndex] - Fhd[leftIndex]) / dx + (Ghd[currentIndex] - Ghd[bottomIndex]) / dy);
//	hud[currentIndexH] -= dt*((Fhud[currentIndex] - Fhud[leftIndex]) / dx + (Ghud[currentIndex] - Ghud[bottomIndex]) / dy + Bud[currentIndexH] / dx);
//	hvd[currentIndexH] -= dt*((Fhvd[currentIndex] - Fhvd[leftIndex]) / dx + (Ghvd[currentIndex] - Ghvd[bottomIndex]) / dy + Bvd[currentIndexH] / dy);
//}

__global__ void eulerTimestep_child_kernel(int* td, int recLevel, int maxRecursion, int cellStartX, int cellStartY, float* hd, float* hud, float* hvd, float* Fhd, float* Fhud, float* Fhvd, float* Ghd, float* Ghud, float* Ghvd, float* Bud, float* Bvd, int width, int height, float dt, float dx, float dy)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	const int haloOffset = 1;
	
	//are we inside the child cell?
	const int ccellExtX = getCellExt(gridDim.x * blockDim.x, maxRecursion - recLevel);
	const int ccellExtY = getCellExt(gridDim.y * blockDim.y, maxRecursion - recLevel);
	if (i * ccellExtX >= getCellExt(gridDim.x * blockDim.x, maxRecursion - recLevel + 1) || j * ccellExtY >= getCellExt(gridDim.y * blockDim.y, maxRecursion - recLevel + 1))
		return;

	unsigned int ccellStartX = i * ccellExtX + cellStartX;
	unsigned int ccellStartY = j * ccellExtY + cellStartY;

	//are we inside the computation area?
	if (ccellStartX >= width - haloOffset || ccellStartY >= height - haloOffset)
		return;

	//do we need to refine?
	if (td[li(width, ccellStartX, ccellStartY)] > recLevel)
	{
		//we need to refine
		//dim3 gridSize(1, 1);
		eulerTimestep_child_kernel << <gridDim, blockDim >> >(td, recLevel + 1, maxRecursion, ccellStartX, ccellStartY, hd, hud, hvd, Fhd, Fhud, Fhvd, Ghd, Ghud, Ghvd, Bud, Bvd, width, height, dt, dx, dy);
	}
	else
	{
		//compute net updates
		const int currentIndexH = li(width, ccellStartX, ccellStartY);

		//read out current values:
		float currentH = hd[currentIndexH];
		float currentHu = hud[currentIndexH];
		float currentHv = hvd[currentIndexH];
		
		//update with the bathymetry source term: TODO look into this
		currentHu -= dt * Bud[currentIndexH] / (dx * ccellExtX);
		currentHv -= dt * Bvd[currentIndexH] / (dy * ccellExtY);

		//update with horizontal fluxes Fh(u/v)
		const float Fh_right = sumRectLoop<float>(Fhd, width - 1, height - 1, ccellStartX + ccellExtX - 1, ccellStartY, 1, ccellExtY);
		const float Fh_left = sumRectLoop<float>(Fhd, width - 1, height - 1, ccellStartX - 1, ccellStartY, 1, ccellExtY);
		const float Fhu_right = sumRectLoop<float>(Fhud, width - 1, height - 1, ccellStartX + ccellExtX - 1, ccellStartY, 1, ccellExtY);
		const float Fhu_left = sumRectLoop<float>(Fhud, width - 1, height - 1, ccellStartX - 1, ccellStartY, 1, ccellExtY);
		const float Fhv_right = sumRectLoop<float>(Fhvd, width - 1, height - 1, ccellStartX + ccellExtX - 1, ccellStartY, 1, ccellExtY);
		const float Fhv_left = sumRectLoop<float>(Fhvd, width - 1, height - 1, ccellStartX - 1, ccellStartY, 1, ccellExtY);

		currentH -= dt * (Fh_right - Fh_left) / (dx * ccellExtX); //TODO: really multiply with ccellExtX?
		currentHu -= dt * (Fhu_right - Fhu_left) / (dx * ccellExtX);
		currentHv -= dt * (Fhv_right - Fhv_left) / (dx * ccellExtX);
		
		//update with vertical fluxes Gh(u/v)
		const float Gh_top = sumRectLoop<float>(Ghd, width - 1, height - 1, ccellStartX, ccellStartY + ccellExtY - 1, ccellExtX, 1);
		const float Gh_bottom = sumRectLoop<float>(Ghd, width - 1, height - 1, ccellStartX, ccellStartY - 1, ccellExtX, 1);
		const float Ghu_top = sumRectLoop<float>(Ghud, width - 1, height - 1, ccellStartX, ccellStartY + ccellExtY - 1, ccellExtX, 1);
		const float Ghu_bottom = sumRectLoop<float>(Ghud, width - 1, height - 1, ccellStartX, ccellStartY - 1, ccellExtX, 1);
		const float Ghv_top = sumRectLoop<float>(Ghvd, width - 1, height - 1, ccellStartX, ccellStartY + ccellExtY - 1, ccellExtX, 1);
		const float Ghv_bottom = sumRectLoop<float>(Ghvd, width - 1, height - 1, ccellStartX, ccellStartY - 1, ccellExtX, 1);

		currentH -= dt * (Gh_top - Gh_bottom) / (dy * ccellExtY);
		currentHu -= dt * (Ghu_top - Ghu_bottom) / (dy * ccellExtY);
		currentHv -= dt * (Ghv_top - Ghv_bottom) / (dy * ccellExtY);

		//fill whole child rect with new values
		fillRectLoop<float>(hd, width, height, ccellStartX, ccellStartY, ccellExtX, ccellExtY, currentH);
		fillRectLoop<float>(hud, width, height, ccellStartX, ccellStartY, ccellExtX, ccellExtY, currentHu);
		fillRectLoop<float>(hvd, width, height, ccellStartX, ccellStartY, ccellExtX, ccellExtY, currentHv);
	}
}

__global__ void eulerTimestep_parent_kernel(int* td, int maxRecursion, float* hd, float* hud, float* hvd, float* Fhd, float* Fhud, float* Fhvd, float* Ghd, float* Ghud, float* Ghvd, float* Bud, float* Bvd, int width, int height, float dt, float dx, float dy)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	const int haloOffset = 1;

	unsigned int cellStartX = i * getCellExt(gridDim.x * blockDim.x, maxRecursion - 1) + haloOffset;
	unsigned int cellStartY = j * getCellExt(gridDim.y * blockDim.y, maxRecursion - 1) + haloOffset;

	if (cellStartX >= width - haloOffset || cellStartX >= height - haloOffset)
		return;

	const int cellExtX = getCellExt(gridDim.x * blockDim.x, maxRecursion - 1);
	const int cellExtY = getCellExt(gridDim.y * blockDim.y, maxRecursion - 1);
	//do we need to refine?
	if (td[li(width, cellStartX, cellStartY)] > 1)
	{
		//we need to refine
		//dim3 gridSize(1, 1);
		eulerTimestep_child_kernel << <gridDim, blockDim >> >(td, 2, maxRecursion, cellStartX, cellStartY, hd, hud, hvd, Fhd, Fhud, Fhvd, Ghd, Ghud, Ghvd, Bud, Bvd, width, height, dt, dx, dy);
	}
	else
	{
		//compute net updates
		const int currentIndexH = li(width, cellStartX, cellStartY);

		//read out current values:
		float currentH = hd[currentIndexH];
		float currentHu = hud[currentIndexH];
		float currentHv = hvd[currentIndexH];
		
		//update with the bathymetry source term: TODO look into this
		currentHu -= dt * Bud[currentIndexH] / (dx * cellExtX);
		currentHv -= dt * Bvd[currentIndexH] / (dy * cellExtY);

		//update with horizontal fluxes Fh(u/v)
		const float Fh_right = sumRectLoop<float>(Fhd, width - 1, height - 1, cellStartX + cellExtX - 1, cellStartY, 1, cellExtY);
		const float Fh_left = sumRectLoop<float>(Fhd, width - 1, height - 1, cellStartX - 1, cellStartY, 1, cellExtY);
		const float Fhu_right = sumRectLoop<float>(Fhud, width - 1, height - 1, cellStartX + cellExtX - 1, cellStartY, 1, cellExtY);
		const float Fhu_left = sumRectLoop<float>(Fhud, width - 1, height - 1, cellStartX - 1, cellStartY, 1, cellExtY);
		const float Fhv_right = sumRectLoop<float>(Fhvd, width - 1, height - 1, cellStartX + cellExtX - 1, cellStartY, 1, cellExtY);
		const float Fhv_left = sumRectLoop<float>(Fhvd, width - 1, height - 1, cellStartX - 1, cellStartY, 1, cellExtY);

		currentH -= dt * (Fh_right - Fh_left) / (dx * cellExtX); //TODO: really multiply with ccellExtX?
		currentHu -= dt * (Fhu_right - Fhu_left) / (dx * cellExtX);
		currentHv -= dt * (Fhv_right - Fhv_left) / (dx * cellExtX);
		
		//update with vertical fluxes Gh(u/v)
		const float Gh_top = sumRectLoop<float>(Ghd, width - 1, height - 1, cellStartX, cellStartY + cellExtY - 1, cellExtX, 1);
		const float Gh_bottom = sumRectLoop<float>(Ghd, width - 1, height - 1, cellStartX, cellStartY - 1, cellExtX, 1);
		const float Ghu_top = sumRectLoop<float>(Ghud, width - 1, height - 1, cellStartX, cellStartY + cellExtY - 1, cellExtX, 1);
		const float Ghu_bottom = sumRectLoop<float>(Ghud, width - 1, height - 1, cellStartX, cellStartY - 1, cellExtX, 1);
		const float Ghv_top = sumRectLoop<float>(Ghvd, width - 1, height - 1, cellStartX, cellStartY + cellExtY - 1, cellExtX, 1);
		const float Ghv_bottom = sumRectLoop<float>(Ghvd, width - 1, height - 1, cellStartX, cellStartY - 1, cellExtX, 1);

		currentH -= dt * (Gh_top - Gh_bottom) / (dy * cellExtY);
		currentHu -= dt * (Ghu_top - Ghu_bottom) / (dy * cellExtY);
		currentHv -= dt * (Ghv_top - Ghv_bottom) / (dy * cellExtY);

		//fill whole child rect with new values
		fillRectLoop<float>(hd, width, height, cellStartX, cellStartY, cellExtX, cellExtY, currentH);
		fillRectLoop<float>(hud, width, height, cellStartX, cellStartY, cellExtX, cellExtY, currentHu);
		fillRectLoop<float>(hvd, width, height, cellStartX, cellStartY, cellExtX, cellExtY, currentHv);
	}
}

float SWE::eulerTimestep()
{
	computeFluxes();

	//dim3 gridSize = dim3(divUp(nx, blockSize.x), divUp(ny, blockSize.y));
	//eulerTimestep_kernel << <gridSize, blockSize >> >(hd, hud, hvd, Fhd, Fhud, Fhvd, Ghd, Ghud, Ghvd, Bud, Bvd, nx + 2, ny + 2, dt, dx, dy);
	//dim3 gridSize = dim3(divUp(nx, getCellExt(blockSize.x, maxRecursion)), divUp(ny, getCellExt(blockSize.y, maxRecursion)));
	dim3 block(8, 8);
	dim3 grid(2, 2);
	eulerTimestep_parent_kernel << <grid, block >> >(td, maxRecursion, hd, hud, hvd, Fhd, Fhud, Fhvd, Ghd, Ghud, Ghvd, Bud, Bvd, nx + 2, ny + 2, dt, dx, dy);

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