#pragma once

#include <omp.h>
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
	SWE(int _nx, int _ny, float _dx, float _dy, float _g = 9.81);
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
	float* h;
	float* hu;
	float* hv;
	float* b;

	float* Fh;
	float* Fhu;
	float* Fhv;
	float* Gh;
	float* Ghu;
	float* Ghv;
	float* Bu;
	float* Bv;

	BoundaryType left, right, top, bottom;

	inline unsigned int li(unsigned int width, unsigned int x, unsigned int y)
	{
		return y * width + x;
	}

	void setBoundaryLayer();
	float computeLocalSV(int i, int j, char dir);
	float computeFlux(float fLow, float fHigh, float xiLow, float xiHigh, float llf);
	void computeFluxes();
	void computeBathymetrySources();
	float eulerTimestep();
	float getMaxTimestep(float cfl_number = 0.5f);

	int nx, ny;
	float dx, dy, dt, g;

	ofstream Vtk_file;
};

