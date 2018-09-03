#include <sys/statcu.h>
#include <unistdcu.h>

__device__ int d_dchown_rc;
__global__ void g_dchown(char *str, int uid)
{
	struct stat	statbuf;
	d_dchown_rc = (stat(str, &statbuf) < 0 || chown(str, uid, statbuf.st_gid) < 0);
}
int dchown(char *str, int uid)
{
	size_t strLength = strlen(str) + 1;
	char *d_str;
	cudaMalloc(&d_str, strLength);
	cudaMemcpy(d_str, str, strLength, cudaMemcpyHostToDevice);
	g_dchown<<<1,1>>>(d_str, uid);
	cudaFree(d_str);
	int rc; cudaMemcpyFromSymbol(&rc, d_dchown_rc, sizeof(rc), 0, cudaMemcpyDeviceToHost); return rc;
}
