#include <stdiocu.h>
#include <stringcu.h>
#include <direntcu.h>
#include <sys/statcu.h>
#include <unistdcu.h>
#include <assert.h>

#ifndef HostDir
#define HostDir "C:\\T_\\"
#endif
#ifndef DeviceDir
#define DeviceDir ":\\"
#endif

static __device__ void testReading(DIR *d) {
	//// READDIR, REWINDDIR ////
	//extern __device__ struct dirent *readdir_(DIR *dirp);
	//extern __device__ void rewinddir_(DIR *dirp);
	struct dirent *a0 = readdir(d); assert(a0); bool a1 = !strcmp(a0->d_name, "."); assert(a1);
	struct dirent *b0 = readdir(d); assert(b0); bool b1 = !strcmp(b0->d_name, ".."); assert(b1);
	struct dirent *c0 = readdir(d); assert(c0); bool c1 = !strcmp(c0->d_name, "dir0"); assert(c1);
	struct dirent *d0 = readdir(d); assert(!d0);
	rewinddir(d);
	struct dirent *e0 = readdir(d); assert(e0); bool e1 = !strcmp(e0->d_name, "."); assert(e1);
}

static __global__ void g_dirent_test1() {
	printf("dirent_test1\n");

	//// OPENDIR, CLOSEDIR, *READDIR, *REWINDDIR ////
	//extern __device__ DIR *opendir_(const char *name);
	//extern __device__ int closedir_(DIR *dirp);
	//* Open a directory stream on NAME. Return a DIR stream on the directory, or NULL if it could not be opened. */
	//* Host Absolute */
	DIR *a0a = opendir(HostDir"missing"); int a0b = closedir(a0a); assert(!a0a && a0b == -1);
	mkdir(HostDir"test", 0); mkdir(HostDir"test\\dir0", 0);
	DIR *a1a = opendir(HostDir"test"); testReading(a1a); int a1b = closedir(a1a); assert(a1a && !a1b);

	//* Device Absolute */
	DIR *b0a = opendir(DeviceDir"missing"); int b0b = closedir(b0a); assert(!b0a && b0b == -1);
	mkdir(DeviceDir"test", 0); mkdir(DeviceDir"test\\dir0", 0);
	DIR *b1a = opendir(DeviceDir"test"); testReading(b1a); int b1b = closedir(b1a); assert(b1a && !b1b);

	//* Host Relative */
	chdir(HostDir);
	DIR *c0a = opendir("missing"); int c0b = closedir(c0a); assert(!c0a && c0b == -1);
	mkdir("test", 0); mkdir("test\\dir0", 0);
	DIR *c1a = opendir("test"); testReading(c1a); int c1b = closedir(c1a); assert(c1a && !c1b);

	//* Device Relative */
	chdir(DeviceDir);
	DIR *d0a = opendir("missing"); int d0b = closedir(d0a); assert(!d0a && d0b == -1);
	mkdir("test", 0); mkdir("test\\dir0", 0);
	DIR *d1a = opendir("test"); testReading(d1a); int d1b = closedir(d1a); assert(d1a && !d1b);

}
cudaError_t dirent_test1() { g_dirent_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }
