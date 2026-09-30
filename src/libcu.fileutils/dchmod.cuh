#include <ext/pipeline.h>
#include <sys/statcu.h>

__device__ int d_dchmod_rc;
__global__ void g_dchmod(pipelineRedir redir, char *str, mode_t mode) {
	d_dchmod_rc = (chmod(str, mode) < 0);
}
int dchmod(pipelineRedir redir, char *str, int mode) {
	pipelineOpen(redir);
	char *d_str;
	if (str) {
		size_t strLength = strlen(str) + 1;
		cudaMalloc(&d_str, strLength);
		cudaMemcpy(d_str, str, strLength, cudaMemcpyHostToDevice);
	}
	else d_str = 0;
	g_dchmod<<<1, 1>>>(redir, d_str, mode);
	if (d_str) cudaFree(d_str);
	pipelineClose(redir);
	int rc; cudaMemcpyFromSymbol(&rc, d_dchmod_rc, sizeof(rc), 0, cudaMemcpyDeviceToHost); return rc;
}
