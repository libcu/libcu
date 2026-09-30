#include <setjmpcu.h>
#ifdef LIBCU_LEAN_AND_MEAN
__device__ int setjmp_(jmp_buf env) { return 0; }
__device__ void longjmp_(jmp_buf env, int val) { }
#else
#include <stdiocu.h>

__BEGIN_DECLS;

/* Store the calling environment in ENV, also saving the signal mask. Return 0.  */
__device__ int setjmp_(jmp_buf env) {
	panic("Not Implemented");
	return 0;
}

/* Jump to the environment saved in ENV, making the `setjmp' call there return VAL, or 1 if VAL is 0.  */
__device__ void longjmp_(jmp_buf env, int val) {
	panic("Not Implemented");
}

__END_DECLS;
#endif