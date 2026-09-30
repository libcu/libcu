#ifndef LIBCU_LEAN_AND_MEAN
#include <ext/pipeline.h>
#include <string.h>
#include <grpcu.h>

__device__ __managed__ struct group *m_getgrnam_rc;
__global__ void g_getgrnam(pipelineRedir redir, char *name) {
	m_getgrnam_rc = getgrnam(name);
}
struct group *dchgrp_getgrnam(pipelineRedir redir, char *str) {
	pipelineOpen(redir);
	char *d_str;
	if (str) {
		size_t strLength = strlen(str) + 1;
		cudaMalloc(&d_str, strLength);
		cudaMemcpy(d_str, str, strLength, cudaMemcpyHostToDevice);
	}
	else d_str = 0;
	g_getgrnam<<<1, 1>>>(redir, d_str);
	if (d_str) cudaFree(d_str);
	pipelineClose(redir);
	return m_getgrnam_rc;
}

#endif