#include "SWE.h"


SWE::SWE(int _nx, int _ny, float _dx, float _dy, float _g)
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

	h = new float[(nx + 2) * (ny + 2)];
	hu = new float[(nx + 2) * (ny + 2)];
	hv = new float[(nx + 2) * (ny + 2)];
	b = new float[(nx + 2) * (ny + 2)];
	Bu = new float[(nx + 2) * (ny + 2)];
	Bv = new float[(nx + 2) * (ny + 2)];

	Fh = new float[(nx + 1) * (ny + 1)];
	Fhu = new float[(nx + 1) * (ny + 1)];
	Fhv = new float[(nx + 1) * (ny + 1)];
	Gh = new float[(nx + 1) * (ny + 1)];
	Ghu = new float[(nx + 1) * (ny + 1)];
	Ghv = new float[(nx + 1) * (ny + 1)];
}


SWE::~SWE()
{
	delete[] h;
	delete[] hu;
	delete[] hv;
	delete[] b;
	delete[] Bu;
	delete[] Bv;
	delete[] Fh;
	delete[] Fhu;
	delete[] Fhv;
	delete[] Gh;
	delete[] Ghu;
	delete[] Ghv;
}

void SWE::setInitialValues(float _h, float _u, float _v)
{
#pragma omp parallel for
	for (int i = 0; i < nx + 2; i++)
	{
		for (int j = 0; j < ny + 2; j++)
		{
			h[li(nx + 2, i, j)] = _h;
			hu[li(nx + 2, i, j)] = _h * _u;
			hv[li(nx + 2, i, j)] = _h * _v;
		}
	}
}

void SWE::setInitialValues(float(*_h)(float, float), float _u, float _v)
{
#pragma omp parallel for
	for (int i = 0; i < nx + 2; i++)
	{
		for (int j = 0; j < ny + 2; j++)
		{
			h[li(nx + 2, i, j)] = _h((i - 0.5f) * dx, (j - 0.5f) * dy);
			hu[li(nx + 2, i, j)] = h[li(nx + 2, i, j)] * _u;
			hv[li(nx + 2, i, j)] = h[li(nx + 2, i, j)] * _v;
		}
	}
}

void SWE::setBathymetry(float(*_b)(float, float))
{
#pragma omp parallel for
	for (int i = 0; i < nx + 2; i++)
		for (int j = 0; j < nx + 2; j++)
			b[li(nx + 2, i, j)] = _b((i - 0.5f) * dx, (j - 0.5f) * dy);

	computeBathymetrySources();
}

void SWE::setBathymetry(float _b)
{
#pragma omp parallel for
	for (int i = 0; i < nx + 2; i++)
		for (int j = 0; j < nx + 2; j++)
			b[li(nx + 2, i, j)] = _b;

	computeBathymetrySources();
}

void SWE::setBoundaryType(BoundaryType left, BoundaryType right, BoundaryType bottom, BoundaryType top)
{
	this->left = left;
	this->right = right;
	this->bottom = bottom;
	this->top = top;
}

void SWE::setBoundaryLayer()
{
	if (left == CONNECT)
	{
#pragma omp parallel for
		for (int j = 0; j < ny + 2; j++)
		{
			h[li(nx + 2, 0, j)] = h[li(nx + 2, nx, j)];
			hu[li(nx + 2, 0, j)] = hu[li(nx + 2, nx, j)];
			hv[li(nx + 2, 0, j)] = hv[li(nx + 2, nx, j)];
		}
	}
	else
	{
#pragma omp parallel for
		for (int j = 0; j < ny + 2; j++)
		{
			h[li(nx + 2, 0, j)] = h[li(nx + 2, 1, j)];
			hu[li(nx + 2, 0, j)] = (left == WALL) ? -hu[li(nx + 2, 1, j)] : hu[li(nx + 2, 1, j)];
			hv[li(nx + 2, 0, j)] = hv[li(nx + 2, 1, j)];
		}
	}
	if (right == CONNECT)
	{
#pragma omp parallel for
		for (int j = 0; j < ny + 2; j++)
		{
			h[li(nx + 2, nx + 1, j)] = h[li(nx + 2, 1, j)];
			hu[li(nx + 2, nx + 1, j)] = hu[li(nx + 2, 1, j)];
			hv[li(nx + 2, nx + 1, j)] = hv[li(nx + 2, 1, j)];
		}
	}
	else
	{
#pragma omp parallel for
		for (int j = 0; j < ny + 2; j++)
		{
			h[li(nx + 2, nx + 1, j)] = h[li(nx + 2, nx, j)];
			hu[li(nx + 2, nx + 1, j)] = (right == WALL) ? -hu[li(nx + 2, nx, j)] : hu[li(nx + 2, nx, j)];
			hv[li(nx + 2, nx + 1, j)] = hv[li(nx + 2, nx, j)];
		}
	}
	if (bottom == CONNECT)
	{
#pragma omp parallel for
		for (int i = 0; i < nx + 2; i++)
		{
			h[li(nx + 2, i, 0)] = h[li(nx + 2, i, ny)];
			hu[li(nx + 2, i, 0)] = hu[li(nx + 2, i, ny)];
			hv[li(nx + 2, i, 0)] = hv[li(nx + 2, i, ny)];
		}
	}
	else
	{
#pragma omp parallel for
		for (int i = 0; i < nx + 2; i++)
		{
			h[li(nx + 2, i, 0)] = h[li(nx + 2, i, 1)];
			hu[li(nx + 2, i, 0)] = hu[li(nx + 2, i, 1)];
			hv[li(nx + 2, i, 0)] = (bottom == WALL) ? -hv[li(nx + 2, i, 1)] : hv[li(nx + 2, i, 1)];
		}
	}
	if (top == CONNECT)
	{
#pragma omp parallel for
		for (int i = 0; i < nx + 2; i++)
		{
			h[li(nx + 2, i, ny + 1)] = h[li(nx + 2, i, 1)];
			hu[li(nx + 2, i, ny + 1)] = hu[li(nx + 2, i, 1)];
			hv[li(nx + 2, i, ny + 1)] = hv[li(nx + 2, i, 1)];
		}
	}
	else
	{
#pragma omp parallel for
		for (int i = 0; i < nx + 2; i++)
		{
			h[li(nx + 2, i, ny + 1)] = h[li(nx + 2, i, ny)];
			hu[li(nx + 2, i, ny + 1)] = hu[li(nx + 2, i, ny)];
			hv[li(nx + 2, i, ny + 1)] = (top == WALL) ? -hv[li(nx + 2, i, ny)] : hv[li(nx + 2, i, ny)];
		}
	}
}

float SWE::computeFlux(float fLow, float fHigh, float xiLow, float xiHigh, float llf)
{
	return 0.5f * (fLow + fHigh) - 0.5f * llf * (xiHigh - xiLow);
}

float SWE::computeLocalSV(int i, int j, char dir)
{
	float sv1, sv2;
	if (dir == 'x')
	{
		sv1 = fabsf(hu[li(nx + 2, i, j)] / h[li(nx + 2, i, j)]) + sqrtf(g * h[li(nx + 2, i, j)]);
		sv2 = fabsf(hu[li(nx + 2, i + 1, j)] / h[li(nx + 2, i + 1, j)]) + sqrtf(g * h[li(nx + 2, i + 1, j)]);
	}
	else
	{
		sv1 = fabsf(hv[li(nx + 2, i, j)] / h[li(nx + 2, i, j)]) + sqrtf(g * h[li(nx + 2, i, j)]);
		sv2 = fabsf(hv[li(nx + 2, i, j + 1)] / h[li(nx + 2, i, j + 1)]) + sqrtf(g * h[li(nx + 2, i, j + 1)]);
	}
	return (sv1 > sv2) ? sv1 : sv2;
}

void SWE::computeFluxes()
{
	//fluxes in x direction:
#pragma omp parallel for
	for (int i = 0; i <= nx; i++)
	{
		for (int j = 0; j <= ny; j++)
		{
			float llf = computeLocalSV(i, j, 'x');
			Fh[li(nx + 1, i, j)] = computeFlux(h[li(nx + 2, i, j)] * hu[li(nx + 2, i, j)], h[li(nx + 2, i + 1, j)] * hu[li(nx + 2, i + 1, j)], h[li(nx + 2, i, j)], h[li(nx + 2, i + 1, j)], llf);
			Fhu[li(nx + 1, i, j)] = computeFlux(hu[li(nx + 2, i, j)] * hu[li(nx + 2, i, j)] + 0.5f * g * h[li(nx + 2, i, j)],
				hu[li(nx + 2, i + 1, j)] * hu[li(nx + 2, i + 1, j)] + 0.5f * g * h[li(nx + 2, i + 1, j)],
				hu[li(nx + 2, i, j)],
				hu[li(nx + 2, i + 1, j)],
				llf);
			Fhv[li(nx + 1, i, j)] = computeFlux(hu[li(nx + 2, i, j)] * hv[li(nx + 2, i, j)], hu[li(nx + 2, i + 1, j)] * hv[li(nx + 2, i + 1, j)], hv[li(nx + 2, i, j)], hv[li(nx + 2, i + 1, j)], llf);
		}
	}

	//fluxes in y direction
#pragma omp parallel for
	for (int j = 0; j <= ny; j++)
	{
		for (int i = 0; i <= nx; i++)
		{
			float llf = computeLocalSV(i, j, 'y');
			Gh[li(nx + 1, i, j)] = computeFlux(h[li(nx + 2, i, j)] * hv[li(nx + 2, i, j)], h[li(nx + 2, i, j + 1)] * hv[li(nx + 2, i, j + 1)], h[li(nx + 2, i, j)], h[li(nx + 2, i, j + 1)], llf);
			Ghu[li(nx + 1, i, j)] = computeFlux(hu[li(nx + 2, i, j)] * hv[li(nx + 2, i, j)], hu[li(nx + 2, i, j + 1)] * hv[li(nx + 2, i, j + 1)], hu[li(nx + 2, i, j)], hu[li(nx + 2, i, j + 1)], llf);
			Ghv[li(nx + 1, i, j)] = computeFlux(hv[li(nx + 2, i, j)] * hv[li(nx + 2, i, j)] + 0.5f * g * h[li(nx + 2, i, j)],
				hv[li(nx + 2, i, j + 1)] * hv[li(nx + 2, i, j + 1)] + 0.5f * g * h[li(nx + 2, i, j + 1)],
				hv[li(nx + 2, i, j)],
				hv[li(nx + 2, i, j + 1)],
				llf);
		}
	}

}

void SWE::computeBathymetrySources()
{
#pragma omp parallel for
	for (int i = 1; i <= nx; i++)
	{
		for (int j = 1; j <= ny; j++)
		{
			Bu[li(nx + 2, i, j)] = g*(h[li(nx + 2, i, j)] * b[li(nx + 2, i, j)] - h[li(nx + 2, i - 1, j)] * b[li(nx + 2, i - 1, j)]);
			Bv[li(nx + 2, i, j)] = g*(h[li(nx + 2, i, j)] * b[li(nx + 2, i, j)] - h[li(nx + 2, i, j - 1)] * b[li(nx + 2, i, j - 1)]);
		}
	}

}

float SWE::eulerTimestep()
{
	computeFluxes();

#pragma omp parallel for
	for (int i = 1; i <= nx; i++)
	{
		for (int j = 1; j <= ny; j++)
		{
			h[li(nx + 2, i, j)] -= dt*((Fh[li(nx + 1, i, j)] - Fh[li(nx + 1, i - 1, j)]) / dx + (Gh[li(nx + 1, i, j)] - Gh[li(nx + 1, i, j - 1)]) / dy);
			hu[li(nx + 2, i, j)] -= dt*((Fhu[li(nx + 1, i, j)] - Fhu[li(nx + 1, i - 1, j)]) / dx + (Ghu[li(nx + 1, i, j)] - Ghu[li(nx + 1, i, j - 1)]) / dy + Bu[li(nx + 2, i, j)] / dx);
			hv[li(nx + 2, i, j)] -= dt*((Fhv[li(nx + 1, i, j)] - Fhv[li(nx + 1, i - 1, j)]) / dx + (Ghv[li(nx + 1, i, j)] - Ghv[li(nx + 1, i, j - 1)]) / dy + Bv[li(nx + 2, i, j)] / dy);
		}
	}

	return dt;
}


float SWE::getMaxTimestep(float cfl_number)
{
	float hmax = numeric_limits<float>::min();
	float vmax = numeric_limits<float>::min();
	float meshSize = (dx < dy) ? dx : dy;

	for (int i = 1; i <= nx; i++)
	{
		for (int j = 1; j <= ny; j++)
		{
			if (h[li(nx + 2, i, j)] > hmax) hmax = h[li(nx + 2, i, j)];
			if (fabsf(hu[li(nx + 2, i, j)]) > vmax) vmax = fabsf(hu[li(nx + 2, i, j)]);
			if (fabsf(hv[li(nx + 2, i, j)]) > vmax) vmax = fabsf(hv[li(nx + 2, i, j)]);
		}
	}

	cout << "hmax " << hmax << endl << flush;
	cout << "vmax " << vmax << endl << flush;
	//cout << "dt " << meshSize / (sqrt(g*hmax) + vmax) << endl;

	return cfl_number * meshSize / (sqrtf(g*hmax) + vmax);
}


float SWE::simulate(float tStart, float tEnd)
{
	static int iter = 0;

	float t = tStart;
	do
	{
		//do debug output
		writeVTKFile(generateFileName("detailed_out/out", iter));
		setBoundaryLayer();
		computeBathymetrySources();
		t += eulerTimestep();
		//float tMax = getMaxTimestep();
		//setTimestep(tMax);
		iter++;
	} while (t < tEnd);

	return t;
}

void SWE::writeVTKFile(string FileName)
{
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
	for (int j = 1; j < ny + 1; j++)
		for (int i = 1; i < nx + 1; i++)
			Vtk_file << (h[li(nx + 2, i, j)] + b[li(nx + 2, i, j)]) << endl;
	Vtk_file << "SCALARS U float 1" << endl;
	Vtk_file << "LOOKUP_TABLE default" << endl;
	for (int j = 1; j < ny + 1; j++)
		for (int i = 1; i < nx + 1; i++)
			Vtk_file << hu[li(nx + 2, i, j)] / h[li(nx + 2, i, j)] << endl;
	Vtk_file << "SCALARS V float 1" << endl;
	Vtk_file << "LOOKUP_TABLE default" << endl;
	for (int j = 1; j < ny + 1; j++)
		for (int i = 1; i < nx + 1; i++)
			Vtk_file << hv[li(nx + 2, i, j)] / h[li(nx + 2, i, j)] << endl;
	Vtk_file << "SCALARS B float 1" << endl;
	Vtk_file << "LOOKUP_TABLE default" << endl;
	for (int j = 1; j < ny + 1; j++)
		for (int i = 1; i < nx + 1; i++)
			Vtk_file << b[li(nx + 2, i, j)] << endl;
	Vtk_file.close();
}