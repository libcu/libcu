#include <stdiocu.h>
#include <errnocu.h>
#include <assert.h>

static __global__ void g_errno_test1() {
	printf("errno_test1\n");

	//// _ERRNO, ERRNO ////
	//extern __device__ int *_errno_(void);
	//#define errno (*_errno())
	int a0 = errno; assert(!a0);

	//// _SET_ERRNO, _GET_ERRNO ////
	//extern __device__ errno_t _set_errno_(int value);
	//extern __device__ errno_t _get_errno_(int *value);
	_set_errno(3);
	int b0 = errno; assert(b0 == 3);
	int b1 = _get_errno(nullptr); assert(b1 == 3);
	int b1a, b1b = _get_errno(&b1a); assert(b1a == 3); assert(b1b == 3);
}
cudaError_t errno_test1() { g_errno_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }
