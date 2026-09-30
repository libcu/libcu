#include <ext/pipeline.h>
#include <sys/statcu.h>
#include <unistdcu.h>
#include "fileutils.h"

__device__ int d_drm_rc;
__global__ void g_drm(pipelineRedir redir, char *str) {
	struct stat sbuf;
	d_drm_rc = !LSTAT(str, &sbuf) && unlink(str);
}
int drm(pipelineRedir redir, char *str) {
	pipelineOpen(redir);
	char *d_str;
	if (str) {
		size_t strLength = strlen(str) + 1;
		cudaMalloc(&d_str, strLength);
		cudaMemcpy(d_str, str, strLength, cudaMemcpyHostToDevice);
	}
	else d_str = 0;
	g_drm<<<1, 1>>>(redir, d_str);
	if (d_str) cudaFree(d_str);
	pipelineClose(redir);
	int rc; cudaMemcpyFromSymbol(&rc, d_drm_rc, sizeof(rc), 0, cudaMemcpyDeviceToHost); return rc;
}
