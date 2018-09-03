#include <sys/statcu.h>
#include <unistdcu.h>

__device__ int d_dchgrp_rc;
__global__ void g_dchgrp(char *str, int gid)
{
	struct stat	statbuf;
	d_dchgrp_rc = (stat(str, &statbuf) < 0 || chown(str, statbuf.st_uid, gid) < 0);
}
int dchgrp(char *str, int gid)
{
	size_t strLength = strlen(str) + 1;
	char *d_str;
	cudaMalloc(&d_str, strLength);
	cudaMemcpy(d_str, str, strLength, cudaMemcpyHostToDevice);
	g_dchgrp<<<1,1>>>(d_str, gid);
	cudaFree(d_str);
	int rc; cudaMemcpyFromSymbol(&rc, d_dchgrp_rc, sizeof(rc), 0, cudaMemcpyDeviceToHost); return rc;
}
