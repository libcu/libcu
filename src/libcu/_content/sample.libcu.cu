// set to 1 for examples
#if 0

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtimecu.h>
#include <sentinel.h>

void addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];

	// snprintf example
#if 0
	char buf[100];
	snprintf(buf, sizeof(buf), "%d> c[%d] = a[%d] + b[%d]\n", i, c[i], a[i], b[i]);
	printf(buf);
#endif

	// file writing example (x64)
#if 0
	if (threadIdx.x != 0) return;
	FILE *f = fopen("fopen.txt", "w");
	fprintf(f, "The quick brown fox jumps over the lazy dog");
	fclose(f);
#endif
}

int main()
{
	const int arraySize = 5;
	const int a[arraySize] = { 1, 2, 3, 4, 5 };
	const int b[arraySize] = { 10, 20, 30, 40, 50 };
	int c[arraySize] = { 0 };

	// Add vectors in parallel.
	addWithCuda(c, a, b, arraySize);

	printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
		c[0], c[1], c[2], c[3], c[4]);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaErrorCheck(cudaDeviceReset());

	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
void addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	cudaDeviceHeap deviceHeap;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaErrorCheckF(cudaSetDevice(0), goto Error);

	deviceHeap = cudaDeviceHeapCreate();
	cudaDeviceHeapSelect(deviceHeap);

	// Start Sentinel
	sentinelServerInitialize();

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaErrorCheckF(cudaMalloc((void**)&dev_c, size * sizeof(int)), goto Error);
	cudaErrorCheckF(cudaMalloc((void**)&dev_a, size * sizeof(int)), goto Error);
	cudaErrorCheckF(cudaMalloc((void**)&dev_b, size * sizeof(int)), goto Error);

	// Copy input vectors from host memory to GPU buffers.
	cudaErrorCheckF(cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice), goto Error);
	cudaErrorCheckF(cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice), goto Error);

	// Launch a kernel on the GPU with one thread for each element.
	addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

	// Check for any errors launching the kernel
	cudaErrorCheckF(cudaGetLastError(), goto Error);

	// cudaDeviceHeapSynchronize..
	cudaErrorCheckF(cudaDeviceHeapSynchronize(deviceHeap), goto Error);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaErrorCheckF(cudaDeviceSynchronize(), goto Error);

	// Copy output vector from GPU buffer to host memory.
	cudaErrorCheckF(cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost), goto Error);

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	// Shutdown Sentinel
	sentinelServerShutdown();

	cudaDeviceHeapDestroy(deviceHeap);
}

#endif
