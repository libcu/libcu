#include <sys/statcu.h>

__device__ int d_dmkdir_rc;
__global__ void g_dmkdir(char *str, unsigned short mode)
{
	d_dmkdir_rc = mkdir(str, mode);
}
int dmkdir(char *str, unsigned short mode)
{
	size_t strLength = strlen(str) + 1;
	char *d_str;
	cudaMalloc(&d_str, strLength);
	cudaMemcpy(d_str, str, strLength, cudaMemcpyHostToDevice);
	g_dmkdir<<<1,1>>>(d_str, mode);
	cudaFree(d_str);
	int rc; cudaMemcpyFromSymbol(&rc, d_dmkdir_rc, sizeof(rc), 0, cudaMemcpyDeviceToHost); return rc;
}
