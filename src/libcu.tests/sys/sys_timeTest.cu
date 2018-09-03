#include <stdiocu.h>
#include <crtdefscu.h>
#include <sys\timecu.h>
#include <assert.h>

static __global__ void g_sys_time_test1() {
	printf("sys_time_test1\n");
}
cudaError_t sys_time_test1() { g_sys_time_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }
