#include <ext/pipeline.h>
#include "fileutils.h"

__device__ int d_dcp_rc;
__global__ void g_dcp(pipelineRedir redir, char *srcName, char *destName, bool setModes) {
	d_dcp_rc = copyFile(srcName, destName, setModes);
}
int dcp(pipelineRedir redir, char *str, char *str2, bool setModes) {
	pipelineOpen(redir);
	char *d_str;
	char *d_str2;
	if (str) {
		size_t strLength = strlen(str) + 1;
		cudaMalloc(&d_str, strLength);
		cudaMemcpy(d_str, str, strLength, cudaMemcpyHostToDevice);
	}
	else d_str = 0;
	if (str2) {
		size_t str2Length = strlen(str2) + 1;
		cudaMalloc(&d_str2, str2Length);
		cudaMemcpy(d_str2, str2, str2Length, cudaMemcpyHostToDevice);
	}
	else d_str2 = 0;
	g_dcp<<<1, 1>>>(redir, d_str, d_str2, setModes);
	if (d_str) cudaFree(d_str);
	if (d_str2) cudaFree(d_str2);
	pipelineClose(redir);
	int rc; cudaMemcpyFromSymbol(&rc, d_dcp_rc, sizeof(rc), 0, cudaMemcpyDeviceToHost); return rc;
}
