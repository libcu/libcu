#include <ext/pipeline.h>
#include <unistdcu.h>

__device__ int d_drmdir_rc;
__global__ void g_drmdir(pipelineRedir redir, char *str) {
	d_drmdir_rc = rmdir(str);
}
int drmdir(pipelineRedir redir, char *str) {
	pipelineOpen(redir);
	char *d_str;
	if (str) {
		size_t strLength = strlen(str) + 1;
		cudaMalloc(&d_str, strLength);
		cudaMemcpy(d_str, str, strLength, cudaMemcpyHostToDevice);
	}
	else d_str = 0;
	g_drmdir<<<1, 1>>>(redir, d_str);
	if (d_str) cudaFree(d_str);
	pipelineClose(redir);
	int rc; cudaMemcpyFromSymbol(&rc, d_drmdir_rc, sizeof(rc), 0, cudaMemcpyDeviceToHost); return rc;
}
