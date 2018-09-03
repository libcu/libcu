#include "fileutils.h"

__device__ int d_dcp_rc;
__global__ void g_dcp(char *srcName, char *destName, bool setModes)
{
	d_dcp_rc = copyFile(srcName, destName, setModes);
}
int dcp(char *str, char *str2, bool setModes)
{
	size_t strLength = strlen(str) + 1;
	size_t str2Length = strlen(str2) + 1;
	char *d_str;
	char *d_str2;
	cudaMalloc(&d_str, strLength);
	cudaMalloc(&d_str2, str2Length);
	cudaMemcpy(d_str, str, strLength, cudaMemcpyHostToDevice);
	cudaMemcpy(d_str2, str2, str2Length, cudaMemcpyHostToDevice);
	g_dcp<<<1,1>>>(d_str, d_str2, setModes);
	cudaFree(d_str);
	cudaFree(d_str2);
	int rc; cudaMemcpyFromSymbol(&rc, d_dcp_rc, sizeof(rc), 0, cudaMemcpyDeviceToHost); return rc;
}
