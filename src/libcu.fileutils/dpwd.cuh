#include <unistdcu.h>

__device__ int d_dpwd_rc;
__global__ void g_dpwd(char *str)
{
	getcwd(str, MAX_PATH);
	d_dpwd_rc = 0;
}
int dpwd(char *str)
{
	char *d_str;
	cudaMalloc(&d_str, MAX_PATH);
	g_dpwd<<<1,1>>>(d_str);
	cudaMemcpy(str, d_str, MAX_PATH, cudaMemcpyDeviceToHost);
	cudaFree(d_str);
	int rc; cudaMemcpyFromSymbol(&rc, d_dpwd_rc, sizeof(rc), 0, cudaMemcpyDeviceToHost); return rc;
}