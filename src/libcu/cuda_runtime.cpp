#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtimecu.h>

bool gpuAssert(cudaError_t code, const char *action, const char *file, int line, bool abort) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s [%s:%d]\n", cudaGetErrorString(code), file, line);
		//getchar();
		if (abort) exit(code);
		return false;
	}
	return true;
}

__forceinline__ int __convertSMVer2Cores(int major, int minor) {
	typedef struct { // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
		int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
		int Cores;
	} SMToCores;
	SMToCores gpuArchCoresPerSM[] = {
		{ 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
		{ 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
		{ 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
		{ 0x32, 192}, // Kepler Generation (SM 3.2) GK10x class
		{ 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
		{ 0x37, 192}, // Kepler Generation (SM 3.7) GK21x class
		{ 0x50, 128}, // Maxwell Generation (SM 5.0) GM10x class
		{ 0x52, 128}, // Maxwell Generation (SM 5.2) GM20x class
		{ 0x53, 128}, // Maxwell Generation (SM 5.3) GM20x class
		{ 0x60, 64 }, // Pascal Generation (SM 6.0) GP100 class
		{ 0x61, 128}, // Pascal Generation (SM 6.1) GP10x class
		{ 0x62, 128}, // Pascal Generation (SM 6.2) GP10x class
		{ 0x70, 64 }, // Volta Generation (SM 7.0) GV10x class
		{ 0x75, 64 }, // Turing Generation (SM 7.5) GT10x class
		{   -1, -1 }
	};
	int index = 0;
	while (gpuArchCoresPerSM[index].SM != -1) {
		if (gpuArchCoresPerSM[index].SM == ((major << 4) + minor))
			return gpuArchCoresPerSM[index].Cores;
		index++;
	}
	// If we don't find the values, we default use the previous one to run properly
	printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor, gpuArchCoresPerSM[index - 1].Cores);
	return gpuArchCoresPerSM[index - 1].Cores;
}

int gpuGetMaxGflopsDevice() {
	int deviceCount = 0;
	cudaDeviceProp deviceProp;
	cudaGetDeviceCount(&deviceCount);
	if (deviceCount == 1) return 0;
	// Find the best major SM Architecture GPU device
	int bestMajor = 0;
	for (int i = 0; i < deviceCount; i++) {
		cudaGetDeviceProperties(&deviceProp, i);
		// If this GPU is not running on Compute Mode prohibited, then we can add it to the list
		if (deviceProp.computeMode != cudaComputeModeProhibited && deviceProp.major > 0 && deviceProp.major < 9999)
			bestMajor = bestMajor > deviceProp.major ? bestMajor : deviceProp.major;
	}
	// Find the best CUDA capable GPU device
	int bestDevice = 0;
	unsigned long long basePerformace = 0;
	for (int i = 0; i < deviceCount; i++) {
		cudaGetDeviceProperties(&deviceProp, i);
		// If this GPU is not running on Compute Mode prohibited, then we can add it to the list
		if (deviceProp.computeMode != cudaComputeModeProhibited) {
			int sm_per_multiproc = deviceProp.major == 9999 && deviceProp.minor == 9999 ? 1 : __convertSMVer2Cores(deviceProp.major, deviceProp.minor);
			unsigned long long performace = ((unsigned long long)deviceProp.multiProcessorCount * (unsigned long long)sm_per_multiproc * (unsigned long long)deviceProp.clockRate);
			if (performace > basePerformace) {
				basePerformace = performace;
				bestDevice = i;
			}
		}
	}
	return bestDevice;
}

char **cudaDeviceTransferStringArray(size_t length, char *const value[], cudaError_t *error) {
	size_t i;
	size_t vectorSize;
	size_t size = vectorSize = (size_t)(sizeof(char *) * length);
	for (i = 0; i < length; i++)
		size += (value[i] ? strlen(value[i]) + 1 : 0);
	char *ptr = (char *)malloc(size);
	if (!ptr) {
		printf("cudaDeviceTransferStringArray: RC_NOMEM");
		if (error) *error = cudaErrorMemoryAllocation;
		return nullptr;
	}
	memset(ptr, 0, size);
	char *h = ptr;
	char **vector = (char **)ptr;
	ptr += vectorSize;
	for (i = 0; i < length; i++) {
		if (value[i]) {
			size_t valueLength = strlen(value[i]) + 1;
			memcpy((void *)ptr, value[i], valueLength);
			ptr += valueLength;
		}
	}
	cudaErrorCheck(cudaMalloc((void **)&ptr, size));
	char *d = ptr;
	ptr += vectorSize;
	for (i = 0; i < length; i++) {
		size_t valueLength = strlen(value[i]) + 1;
		vector[i] = (value[i] ? ptr : nullptr);
		ptr += valueLength;
	}
	cudaErrorCheck(cudaMemcpy(d, h, size, cudaMemcpyHostToDevice));
	free(h);
	return (char **)d;
}