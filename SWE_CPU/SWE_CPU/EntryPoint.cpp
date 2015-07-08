#include "SWE.h"
#include <stdio.h>

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
	return ((x - 0.5f) * (x - 0.5f) + (y - 0.5f) * (y - 0.5f) < 0.05f) ? 1 : 0.7;
}


int main(int argc, char** argv)
{
	if (argc < 4)
	{
		cout << "too few parameters!" << endl;
		exit(1);
	}

	SWE swe(atoi(argv[1]), atoi(argv[2]), 1.0f/atof(argv[1]), 1.0f/atof(argv[2]));
	cout << "nx: " << atoi(argv[1]) << "ny: " << atoi(argv[2]) << "dx: " << 1.0f / atof(argv[1]) << "dy: " << 1.0f / atof(argv[2]) << endl;
	swe.setInitialValues(&getWaterHeight, 0.0f, 0.0f);
	swe.setBathymetry(&getBathymetry);
	swe.setBoundaryType(WALL, WALL, WALL, WALL);

	float endSimulation = 1.0f;
	int numCheckPoints = 10;

	float* checkPt = new float[numCheckPoints + 1];
	for (int cp = 0; cp <= numCheckPoints; cp++)
		checkPt[cp] = cp*(endSimulation / numCheckPoints);

	string basename;
	basename = string(argv[3]);

	cout << "Write output file: water level at start" << endl;
	swe.writeVTKFile(swe.generateFileName(basename, 0));

	float t = 0.0f;
	for (int i = 1; i <= numCheckPoints; i++)
	{
		t = swe.simulate(t, checkPt[i]);
		cout << "Write output file: water level at time " << t << endl;
		swe.writeVTKFile(swe.generateFileName(basename, i));
	}
}