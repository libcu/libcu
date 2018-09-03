#include <stdiocu.h>
#include <fcntlcu.h>
#include <unistdcu.h>
#include <assert.h>

#ifndef HostDir
#define HostDir "C:\\T_\\"
#endif
#ifndef DeviceDir
#define DeviceDir ":\\"
#endif

#ifndef MAKEAFILE
#define MAKEAFILE
static __device__ void makeAFile(const char *file) {
	FILE *fp = fopen(file, "w");
	//fwrite("test", 4, 1, fp);
	fprintf_(fp, "test");
	fclose(fp);
}
#endif

static __global__ void g_fcntl_test1() {
	printf("fcntl_test1\n");

	//// FCTNL ////
	//extern __device__ int vfcntl_(int fd, int cmd, va_list va) #sentinel-branch
	// NEEDED

	//// OPENV ////
	//extern __device__ int vopen_(const char *file, int oflag, va_list va) #sentinel-branch
	/* Host Absolute */
	int a0a = open(HostDir"missing.txt", O_RDONLY); assert(a0a < 0);
	makeAFile(HostDir"test.txt");
	int a1a = open(HostDir"test.txt", O_RDONLY); int a1b = close(a1a); assert(a1a && !a1b);

	/* Device Absolute */
	int b0a = open(DeviceDir"missing.txt", O_RDONLY); assert(b0a < 0);
	makeAFile(DeviceDir"test.txt");
	int b1a = open(DeviceDir"test.txt", O_RDONLY); int b1b = close(b1a); assert(b1a && !b1b);

	/* Host Relative */
	chdir(HostDir);
	int c0a = open("missing.txt", O_RDONLY); assert(c0a < 0);
	makeAFile("test.txt");
	int c1a = open("test.txt", O_RDONLY); int c1b = close(c1a); assert(c1a && !c1b);

	/* Device Relative */
	chdir(DeviceDir);
	int d0a = open("missing.txt", O_RDONLY); assert(d0a < 0);
	makeAFile("test.txt");
	int d1a = open("test.txt", O_RDONLY); int d1b = close(d1a); assert(d1a && !d1b);

	//// CREATE ////
	//#define creat(file, mode)
	/* Host Absolute */
	int e0a = creat(HostDir"test.txt", 0666); int e0b = close(e0a); assert(e0a && !e0b);

	/* Device Absolute */
	int f0a = creat(DeviceDir"test.txt", 0666); int f0b = close(f0a); assert(f0a && !f0b);

	/* Host Relative */
	chdir(HostDir);
	int g0a = creat("test.txt", 0666); int g0b = close(g0a); assert(g0a && !g0b);

	/* Device Relative */
	chdir(DeviceDir);
	int h0a = creat("test.txt", 0666); int h0b = close(h0a); assert(h0a && !h0b);
}
cudaError_t fcntl_test1() { g_fcntl_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }
