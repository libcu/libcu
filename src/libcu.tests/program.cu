#include <cuda_runtimecu.h>
#include <sentinel.h>
#include <stdlibcu.h>
#include <stdiocu.h>

cudaError_t crtdefs_test1();
cudaError_t ctype_test1();
cudaError_t dirent_test1();
cudaError_t errno_test1();
cudaError_t fcntl_test1();
cudaError_t fsystem_test1();
cudaError_t grp_test1();
cudaError_t host_test1();
cudaError_t pwd_test1();
cudaError_t regex_test1();
cudaError_t sentinel_test1();
cudaError_t setjmp_test1();
cudaError_t stddef_test1();
cudaError_t stdio_test1();
cudaError_t stdio_64bit();
cudaError_t stdio_ganging();
cudaError_t stdio_scanf();
cudaError_t stdlib_test1();
cudaError_t stdlib_strtol();
cudaError_t stdlib_strtoq();
cudaError_t string_test1();
cudaError_t time_test1();
cudaError_t unistd_test1();

#if _HASPAUSE
#define mainPause(fmt) { printf(fmt"\n"); char c; scanf("%c", &c); }
#else
#define mainPause(fmt) { printf(fmt"\n"); }
#endif

int main(int argc, char **argv) {
	int testId = argv[1] ? atoi(argv[1]) : 11; // 23;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaError_t cudaStatus = cudaSetDevice(gpuGetMaxGflopsDevice());
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
		goto Error;
	}
	cudaErrorCheck(cudaDeviceSetLimit(cudaLimitStackSize, 1024 * 5));
	sentinelServerInitialize();
	sentinelRegisterFileUtils();

	// Launch test
	switch (testId) {
	case 0: mainPause("Press any key to continue."); break;
	case 1: cudaStatus = crtdefs_test1(); break;
	case 2: cudaStatus = ctype_test1(); break;
	case 3: cudaStatus = dirent_test1(); break;
	case 4: cudaStatus = errno_test1(); break;
	case 5: cudaStatus = fcntl_test1(); break;
	case 6: cudaStatus = fsystem_test1(); break;
	case 7: cudaStatus = grp_test1(); break;
	case 8: cudaStatus = host_test1(); break;
	case 9: cudaStatus = pwd_test1(); break;
	case 10: cudaStatus = regex_test1(); break;
	case 11: cudaStatus = sentinel_test1(); break;
	case 12: cudaStatus = setjmp_test1(); break;
	case 13: cudaStatus = stddef_test1(); break;
	case 14: cudaStatus = stdio_test1(); break; // assert
	case 15: cudaStatus = stdio_64bit(); break;
	case 16: cudaStatus = stdio_ganging(); break;
	case 17: cudaStatus = stdio_scanf(); break;
	case 18: cudaStatus = stdlib_test1(); break;
	case 19: cudaStatus = stdlib_strtol(); break;
	case 20: cudaStatus = stdlib_strtoq(); break;
	case 21: cudaStatus = string_test1(); break;
	case 22: cudaStatus = time_test1(); break;
	case 23: cudaStatus = unistd_test1(); break; //failed --missing device, throws on fast run
		// default
	default: cudaStatus = crtdefs_test1(); break;
	}
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "failed! %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// finish
	mainPause("SUCCESS");

Error:
	sentinelServerShutdown();

	// close
	if (cudaStatus != cudaSuccess) {
		// finish
		mainPause("ERROR");
		return 1;
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!\n");
		return 1;
	}

	return 0;
}
