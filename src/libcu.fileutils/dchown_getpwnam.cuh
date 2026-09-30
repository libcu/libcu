#ifndef LIBCU_LEAN_AND_MEAN
#include <ext/pipeline.h>
#include <string.h>
#include <pwdcu.h>

__device__ __managed__ struct passwd *m_getpwnam_rc;
__global__ void g_getpwnam(pipelineRedir redir, char *name) {
	m_getpwnam_rc = getpwnam(name);
}
struct passwd *dchown_getpwnam_(pipelineRedir redir, char *str) {
	pipelineOpen(redir);
	char *d_str;
	if (str) {
		size_t strLength = strlen(str) + 1;
		cudaMalloc(&d_str, strLength);
		cudaMemcpy(d_str, str, strLength, cudaMemcpyHostToDevice);
	}
	else d_str = 0;
	g_getpwnam<<<1, 1>>>(redir, d_str);
	if (d_str) cudaFree(d_str);
	pipelineClose(redir);
	return m_getpwnam_rc;
}

#endif