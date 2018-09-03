#include <sys/statcu.h>
#include <unistdcu.h>
#include "fileutils.h"

__device__ int d_drm_rc;
__global__ void g_drm(char *str)
{
	struct stat sbuf;
	d_drm_rc = (!LSTAT(str, &sbuf) && unlink(str));
}
int drm(char *str)
{
	size_t strLength = strlen(str) + 1;
	char *d_str;
	cudaMalloc(&d_str, strLength);
	cudaMemcpy(d_str, str, strLength, cudaMemcpyHostToDevice);
	g_drm<<<1,1>>>(d_str);
	cudaFree(d_str);
	int rc; cudaMemcpyFromSymbol(&rc, d_drm_rc, sizeof(rc), 0, cudaMemcpyDeviceToHost); return rc;
}
