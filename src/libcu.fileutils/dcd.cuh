#include <ext/pipeline.h>
#include <unistdcu.h>

__device__ int d_dcd_rc;
__global__ void g_dcd(pipelineRedir redir, char *str) {
	d_dcd_rc = 0;
}
int dcd(pipelineRedir redir, char *str) {
	pipelineOpen(redir);
	char *d_str;
	if (str) {
		size_t strLength = strlen(str) + 1;
		cudaMalloc(&d_str, strLength);
		cudaMemcpy(d_str, str, strLength, cudaMemcpyHostToDevice);
	}
	else d_str = 0;
	g_dcd<<<1, 1>>>(redir, d_str);
	if (d_str) cudaFree(d_str);
	pipelineClose(redir);
	int rc; cudaMemcpyFromSymbol(&rc, d_dcd_rc, sizeof(rc), 0, cudaMemcpyDeviceToHost); return rc;
}