#include <sys/statcu.h>

__device__ int d_dchmod_rc;


__global__ void g_dchmod(char *str, mode_t mode)
{
	d_dchmod_rc = (chmod(str, mode) < 0);
}

int dchmod(char *str, int mode)
{
	size_t strLength = strlen(str) + 1;
	char *d_str;
	cudaMalloc(&d_str, strLength);
	cudaMemcpy(d_str, str, strLength, cudaMemcpyHostToDevice);
	g_dchmod<<<1,1>>>(d_str, mode);
	cudaFree(d_str);
	int rc; cudaMemcpyFromSymbol(&rc, d_dchmod_rc, sizeof(rc), 0, cudaMemcpyDeviceToHost); return rc;
}
