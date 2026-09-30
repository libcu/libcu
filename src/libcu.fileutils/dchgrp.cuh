#include <ext/pipeline.h>
#include <sys/statcu.h>
#include <unistdcu.h>

__device__ int d_dchgrp_rc;
__global__ void g_dchgrp(pipelineRedir redir, char *str, int gid) {
	struct stat	statbuf;
	d_dchgrp_rc = (stat(str, &statbuf) < 0 || chown(str, statbuf.st_uid, gid) < 0);
}
int dchgrp(pipelineRedir redir, char *str, int gid) {
	pipelineOpen(redir);
	char *d_str;
	if (str) {
		size_t strLength = strlen(str) + 1;
		cudaMalloc(&d_str, strLength);
		cudaMemcpy(d_str, str, strLength, cudaMemcpyHostToDevice);
	}
	else d_str = 0;
	g_dchgrp<<<1, 1>>>(redir, d_str, gid);
	if (d_str) cudaFree(d_str);
	pipelineClose(redir);
	int rc; cudaMemcpyFromSymbol(&rc, d_dchgrp_rc, sizeof(rc), 0, cudaMemcpyDeviceToHost); return rc;
}
