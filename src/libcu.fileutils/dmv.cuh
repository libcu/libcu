#include <stdiocu.h>
#include <unistdcu.h>
#include <errnocu.h>
#include "fileutils.h"

__device__ int d_dmv_rc;
__global__ void g_dmv(char *srcName, char *destName)
{
	d_dmv_rc = 0;
	if (access(srcName, 0) < 0) {
		perror(srcName);
		return;
	}
	if (rename(srcName, destName) >= 0)
		return;
	if (errno != EXDEV) {
		perror(destName);
		return;
	}
	if (!copyFile(srcName, destName, true))
		return;
	if (unlink(srcName) < 0)
		perror(srcName);
	d_dmv_rc = 1;
}
int dmv(char *str, char *str2)
{
	size_t strLength = strlen(str) + 1;
	size_t str2Length = strlen(str2) + 1;
	char *d_str;
	char *d_str2;
	cudaMalloc(&d_str, strLength);
	cudaMalloc(&d_str2, str2Length);
	cudaMemcpy(d_str, str, strLength, cudaMemcpyHostToDevice);
	cudaMemcpy(d_str2, str2, str2Length, cudaMemcpyHostToDevice);
	g_dmv<<<1,1>>>(d_str, d_str2);
	cudaFree(d_str);
	cudaFree(d_str2);
	int rc; cudaMemcpyFromSymbol(&rc, d_dmv_rc, sizeof(rc), 0, cudaMemcpyDeviceToHost); return rc;
}
