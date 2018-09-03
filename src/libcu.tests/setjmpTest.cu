#include <stdiocu.h>
#include <setjmpcu.h>
#include <assert.h>

static __global__ void g_setjmp_test1() {
	printf("setjmp_test1\n");

	//// SETJMP ////
	//extern __device__ int setjmp_(jmp_buf env);
	// a0a = setjmp - not:implemented

	//// __SIGSETJMP, _SETJMP ////
	////nosupport: extern int __sigsetjmp_(struct __jmp_buf_tag env[1], int savemask);
	////nosupport: extern int _setjmp_(struct __jmp_buf_tag env[1]);

	//// LONGJMP ////
	//extern __device__ void longjmp_(jmp_buf env, int val);
	// b0a = longjmp - not:implemented

}
cudaError_t setjmp_test1() { g_setjmp_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }
