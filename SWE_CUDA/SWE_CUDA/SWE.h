#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <limits>

using namespace std;


typedef enum BoundaryType 
{
	OUTFLOW, WALL, CONNECT
} BoundaryType;

class SWE
{
public:
	SWE(int _nx, int _ny, float _dx, float _dy, float _g = 9.81, int blockX = 16, int blockY = 16);
	~SWE();

	void setInitialValues(float _h, float _u, float _v);
	void setInitialValues(float(*_h)(float, float), float _u, float _v);
	void setBathymetry(float(*_b)(float, float));
	void setBathymetry(float _b);
	void setTimestep(float _dt){ dt = _dt; }

	void setBoundaryType(BoundaryType left, BoundaryType right, BoundaryType bottom, BoundaryType top);
	float simulate(float tStart, float tEnd);

	void writeVTKFile(string filename);

	static inline std::string generateFileName(std::string baseName, int timeStep) {

		std::ostringstream FileName;
		FileName << baseName << timeStep << ".vtk";
		return FileName.str();
	};

private:
	//input/output arrays on host
	float* h;
	float* hu;
	float* hv;
	float* b;

	//parameter arrays on device
	float* hd;
	float* hud;
	float* hvd;
	float* bd;

	float* Fhd;
	float* Fhud;
	float* Fhvd;
	float* Ghd;
	float* Ghud;
	float* Ghvd;
	float* Bud;
	float* Bvd;

	BoundaryType left, right, top, bottom;

	dim3 blockSize;

	void uploadSolution();
	void downloadSolution();
	void setBoundaryLayer();
	void computeFluxes();
	void computeBathymetrySources();
	float eulerTimestep();
	float getMaxTimestep(float cfl_number = 0.5f);

	int nx, ny;
	float dx, dy, dt, g;

	ofstream Vtk_file;
};


//linear index helper function
inline __device__ __host__ unsigned int li(unsigned int width, unsigned int x, unsigned int y)
{
	return y * width + x;
}

inline __device__ __host__ unsigned int divUp(const unsigned int i, const unsigned int d)
{
	return ((i + d - 1) / d);
}