#include <stdiocu.h>
#include <crtdefscu.h>
#include <sys\statcu.h>
#include <assert.h>

static __global__ void g_sys_stat_test1() {
	printf("sys_stat_test1\n");
}
cudaError_t sys_stat_test1() { g_sys_stat_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }
