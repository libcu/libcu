#include <cuda_runtimecu.h>
#include <stdiocu.h>
#include <stringcu.h>
#include <assert.h>

static __global__ void g_memmove_speed();
static __global__ void g_strlen_speed();
static __global__ void g_strnlen_speed();
#define g_speed g_memmove_speed

int main() {
	float milliseconds = 0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaError_t cudaStatus = cudaSetDevice(gpuGetMaxGflopsDevice());
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
		goto Error;
	}

	// Launch test
	cudaEventRecord(start);
	for (int i = 0; i < 1; i++)
		g_speed<<<1, 32>>>();
	cudaEventRecord(stop);
	//
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "test failed! %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "test launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Effective: %fn", milliseconds / 1e6);

Error:
	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!\n");
		return 1;
	}

	// finish
	printf("\nPress any key to continue.\n");
	scanf("%c");

	return 0;
}

static __constant__ const char *_quickbrownfox =
"The quick brown fox jumped over the lazy dog. The quick brown fox jumped over the lazy dog. The quick brown fox jumped over the lazy dog. The quick brown fox jumped over the lazy dog."
"The quick brown fox jumped over the lazy dog. The quick brown fox jumped over the lazy dog. The quick brown fox jumped over the lazy dog. The quick brown fox jumped over the lazy dog."
"The quick brown fox jumped over the lazy dog. The quick brown fox jumped over the lazy dog. The quick brown fox jumped over the lazy dog. The quick brown fox jumped over the lazy dog."
"The quick brown fox jumped over the lazy dog. The quick brown fox jumped over the lazy dog. The quick brown fox jumped over the lazy dog. The quick brown fox jumped over the lazy dog."
"The quick brown fox jumped over the lazy dog. The quick brown fox jumped over the lazy dog. The quick brown fox jumped over the lazy dog. The quick brown fox jumped over the lazy dog."
"The quick brown fox jumped over the lazy dog. The quick brown fox jumped over the lazy dog. The quick brown fox jumped over the lazy dog. The quick brown fox jumped over the lazy dog."
"The quick brown fox jumped over the lazy dog. The quick brown fox jumped over the lazy dog. The quick brown fox jumped over the lazy dog. The quick brown fox jumped over the lazy dog."
"The quick brown fox jumped over the lazy dog. The quick brown fox jumped over the lazy dog. The quick brown fox jumped over the lazy dog. The quick brown fox jumped over the lazy dog."
"The quick brown fox jumped over the lazy dog. The quick brown fox jumped over the lazy dog. The quick brown fox jumped over the lazy dog. The quick brown fox jumped over the lazy dog."
"The quick brown fox jumped over the lazy dog. The quick brown fox jumped over the lazy dog. The quick brown fox jumped over the lazy dog. The quick brown fox jumped over the lazy dog."
"The quick brown fox jumped over the lazy dog. The quick brown fox jumped over the lazy dog. The quick brown fox jumped over the lazy dog. The quick brown fox jumped over the lazy dog."
"The quick brown fox jumped over the lazy dog. The quick brown fox jumped over the lazy dog. The quick brown fox jumped over the lazy dog. The quick brown fox jumped over the lazy dog.";

static __device__ char _buf0[] = "The quick brown fox jumped over the lazy dog.";
static __device__ char _buf1[50];

static __global__ void g_memmove_speed() {
	for (int i = 0; i < 1000; i++) {
		void *c = memmove(_buf1, nullptr, 0);
		assert(c == _buf1);
	}
	for (int i = 0; i < 1000; i++) {
		void *c = memmove(_buf1, _buf1, 10);
		assert(c == _buf1);
	}
	//for (int i = 0; i < 1000; i++) {
	//	void *c = memmove(_buf1, _buf0, 45);
	//}
}

static __global__ void g_strlen_speed() {
	for (int i = 0; i < 1000; i++) {
		int testLength = strlen(nullptr);
		assert(testLength == 0);
	}
	for (int i = 0; i < 1000; i++) {
		int testLength = strlen(_quickbrownfox);
		assert(testLength == 2196);
	}
}

static __global__ void g_strnlen_speed() {
	for (int i = 0; i < 1000; i++) {
		int testLength = strnlen(nullptr, 3000);
		assert(testLength == 0);
	}
	for (int i = 0; i < 1000; i++) {
		int testLength = strnlen(_quickbrownfox, 3000);
		assert(testLength == 2196);
	}
}