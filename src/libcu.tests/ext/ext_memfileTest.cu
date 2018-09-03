#include <stdiocu.h>
#include <crtdefscu.h>
#include <ext\memfile.h>
#include <assert.h>

static __global__ void g_ext_memfile_test1() {
	printf("ext_memfile_test1\n");
}
cudaError_t ext_memfile_test1() { g_ext_memfile_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }
