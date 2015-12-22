#include "SWE.h"
#include <stdio.h>
#include <omp.h>

//float splash_height(float x, float y)
//{
//	//return 100 + (1 - (x + y));
//	return 2.0f - (1.0f / 512.0f) * (x + y);
//}

float getBathymetry(float x, float y)
{
	//float r = sqrtf((x - 0.5f) * (x - 0.5f) + (y - 0.5f) * (y - 0.5f));
	//return 1.0f + 9.0f * ((r < 0.5f) ? r : 0.5f);

	return 0.0f;
}

float getWaterHeight(float x, float y)
{
	//float r = sqrtf((x - 0.5f) * (x - 0.5f) + (y - 0.5f) * (y - 0.5f));
	//float h = 4.0f - 4.5f*(r / 0.5f);

	//if (r < 0.1f) h = h + 1.0f;
	//return (h > 0.0f) ? h : 0.0f;
	return ((x - 0.5f) * (x - 0.5f) + (y - 0.5f) * (y - 0.5f) < 0.05f) ? 1.0f : 0.7f;
}

int main(int argc, char** argv)
{
//	if (argc < 4)
//	{
//		cout << "too few parameters!" << endl;
//	}

	//SWE* swe = new SWE(atoi(argv[1]), atoi(argv[2]), 1.0f/atof(argv[1]), 1.0f/atof(argv[2]));
	//cout << "nx: " << atoi(argv[1]) << "ny: " << atoi(argv[2]) << "dx: " << 1.0f / atof(argv[1]) << "dy: " << 1.0f / atof(argv[2]) << endl;
	SWE* swe = new SWE(NX, NY, 1.0f / NX, 1.0f / NY);
	swe->setInitialValues(&getWaterHeight, 0.0f, 0.0f);
	swe->setBathymetry(&getBathymetry);
	swe->setBoundaryType(OUTFLOW, OUTFLOW, OUTFLOW, OUTFLOW);

	float endSimulation = 0.001f;
	//float endSimulation = 0.2f;
	int numCheckPoints = 1;

	float* checkPt = new float[numCheckPoints + 1];
	for (int cp = 0; cp <= numCheckPoints; cp++)
		checkPt[cp] = cp*(endSimulation / numCheckPoints);

	string basename;
	basename = string(argv[1]);

	cout << "Writing output file: water level at start" << endl;
	//swe->writeVTKFile(swe->generateFileName(basename, 0));

	double simulationTime = 0.0;
	float t = 0.0f;
	for (int i = 1; i <= numCheckPoints; i++)
	{
		double t1 = omp_get_wtime();
		t = swe->simulate(t, checkPt[i]);
		checkCudaErrors(cudaDeviceSynchronize());
		double t2 = omp_get_wtime();
		simulationTime += t2 - t1;
		cout << "Writing output file: water level at time " << t << endl;
		//swe->writeVTKFile(swe->generateFileName(basename, i));
	}

	checkCudaErrors(cudaDeviceSynchronize());
	cout << "Simulation done in: " << simulationTime << "s" << endl;

	delete swe;

	checkCudaErrors(cudaDeviceReset());
}