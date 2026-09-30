#include <ext/pipeline.h>
#include <sys/statcu.h>

__device__ int d_dmkdir_rc;
__global__ void g_dmkdir(pipelineRedir redir, char *str, unsigned short mode) {
	//fprintf(redir.out, "z: %s\n", str);
	d_dmkdir_rc = mkdir(str, mode);
}
int dmkdir(pipelineRedir redir, char *str, unsigned short mode) {
	pipelineOpen(redir);
	char *d_str;
	if (str) {
		size_t strLength = strlen(str) + 1;
		cudaMalloc(&d_str, strLength);
		cudaMemcpy(d_str, str, strLength, cudaMemcpyHostToDevice);
	}
	else d_str = 0;
	g_dmkdir<<<1, 1>>>(redir, d_str, mode);
	if (d_str) cudaFree(d_str);
	pipelineClose(redir);
	int rc; cudaMemcpyFromSymbol(&rc, d_dmkdir_rc, sizeof(rc), 0, cudaMemcpyDeviceToHost); return rc;
}
