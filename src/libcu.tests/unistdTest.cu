#include <stdiocu.h>
#include <unistdcu.h>
#include <fcntlcu.h>
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
	fprintf_(fp, "test");
	fclose(fp);
}
#endif

static __global__ void g_unistd_test1() {
	printf("unistd_test1\n");
	char buf[19];

	//// ACCESS, LSEEK, CLOSE ////
	//extern __device__ int access_(const char *name, int type); #sentinel-branch
	//extern __device__ off_t lseek_(int fd, off_t offset, int whence); #sentinel-branch
	//extern __device__ int close_(int fd); #sentinel-branch
	/* Host Absolute */
	int a0a = access(HostDir"missing.txt", F_OK); assert(a0a < 0);
	makeAFile(HostDir"test.txt");
	int a1a = access(HostDir"test.txt", F_OK); assert(!a1a);
	int a0_fd = open(HostDir"test.txt", O_RDONLY);
	int a2a = lseek(a0_fd, 1, SEEK_SET); int a2b = read(a0_fd, buf, 1); assert(a2a > 0 && a2b == 1 && buf[0] == 'e');
	int a3a = close(a0_fd); assert(!a3a);

	/* Device Absolute */
	//int b0a = access(DeviceDir"missing.txt", F_OK); assert(b0a < 0);
	//makeAFile(DeviceDir"test.txt");
	//int b1a = access(DeviceDir"test.txt", F_OK); assert(!b1a);
	//int b0_fd = open(DeviceDir"test.txt", O_RDONLY);
	//int b2a = lseek(b0_fd, 1, SEEK_SET); int b2b = read(b0_fd, buf, 1); assert(b2a > 0 && b2b == 1 && buf[0] == 'e');
	//int b3a = close(b0_fd); assert(!b3a);

	/* Host Relative */
	chdir(HostDir);
	int c0a = access("missing.txt", F_OK); assert(c0a < 0);
	makeAFile("test.txt");
	int c1a = access("test.txt", F_OK); assert(!c1a);
	int c0_fd = open("test.txt", O_RDONLY);
	int c2a = lseek(c0_fd, 1, SEEK_SET); int c2b = read(c0_fd, buf, 1); assert(c2a > 0 && c2b == 1 && buf[0] == 'e');
	int c3a = close(c0_fd); assert(!c3a);

	/* Device Relative */
	//chdir(DeviceDir);
	//int d0a = access("missing.txt", F_OK); assert(d0a < 0);
	//makeAFile("test.txt");
	//int d1a = access("test.txt", F_OK); assert(!d1a);
	//int d0_fd = open("test.txt", O_RDONLY);
	//int d2a = lseek(d0_fd, 1, SEEK_SET); int d2b = read(d0_fd, buf, 1); assert(d2a > 0 && d2b == 1 && buf[0] == 'e');
	//int d3a = close(d0_fd); assert(!d3a);

	//// READ, WRITE ////
	//extern __device__ size_t read_(int fd, void *buf, size_t nbytes, bool wait = true); #sentinel-branch
	//extern __device__ size_t write_(int fd, void *buf, size_t nbytes, bool wait = true); #sentinel-branch
	/* Host Absolute */
	int e0a = open(HostDir"test.txt", O_WRONLY); int e0b = write(e0a, "test", 4); close(e0a); assert(e0b == 4);
	int e1a = open(HostDir"test.txt", O_RDONLY); int e1b = read(e1a, buf, 4); close(e1a); assert(e1b == 4 && !strncmp(buf, "test", 4));

	/* Device Absolute */
	//int f0a = open(DeviceDir"test.txt", O_WRONLY); int f0b = write(f0b, "test", 4); close(f0a); assert(f0b == 4);
	//int f1a = open(DeviceDir"test.txt", O_RDONLY); int f1b = read(f1b, buf, 4); close(f1a); assert(f1b == 4 && !strncmp(buf, "test", 4));

	//// PIPE, ALARM ////
	////nosupport: extern __device__ int pipe_(int pipedes[2]); #notsupported
	////nosupport: extern __device__ unsigned int alarm_(unsigned int seconds); #notsupported

	//// USLEEP, SLEEP, PAUSE ////
	//extern __device__ void usleep_(unsigned long milliseconds);
	//__forceinline__ __device__ void sleep_(unsigned int seconds) { usleep_(seconds * 1000); }
	////nosupport: extern int pause_(void); #notsupported
	usleep(1); assert(1);
	sleep(1); assert(1);

	//// CHOWN ////
	//extern __device__ int chown_(const char *file, uid_t owner, gid_t group); #sentinel-branch
	/* Host Absolute */
	int g0a = chown(HostDir"test.txt", 0, 0); assert(g0a == 0);

	/* Device Absolute */
	int h0a = chown(DeviceDir"test.txt", 0, 0); assert(h0a == 0);

	//// CHDIR, GETCWD ////
	//extern __device__ int chdir_(const char *path); #sentinel-branch
	//extern __device__ char *getcwd_(char *buf, size_t size); #sentinel-branch
	/* Host Absolute */
	int i0a = chdir(HostDir); assert(i0a == 0);

	/* Device Absolute */
	int j0a = chdir(DeviceDir); assert(j0a == 0);

	//// DUP, DUP2 ////
	//extern __device__ int dup_(int fd); #sentinel-branch
	//extern __device__ int dup2_(int fd, int fd2); #sentinel-branch
	/* Host Absolute */
	int k0_fd = open(HostDir"test.txt", O_RDONLY);
	int k0a = dup(k0_fd); int k0b = close(k0a); assert(k0a && !k0b);
	int k1a = dup2(k0_fd, 10); int k1b = close(10); assert(!k1a && !k1b);
	close(k0_fd);

	/* Device Absolute */
	//int l0_fd = open(DeviceDir"test.txt", O_RDONLY);
	//int l0a = dup(l0_fd); int l0b = close(l0a); assert(l0a && !l0b);
	//int l1a = dup2(l0_fd, 10); int l1b = close(l1a); assert(!l1a && !l1b);
	//close(l0_fd);

	//// EXIT ////
	//extern __device__ char **__environ_;
	////nosupport: extern __device__ void exit_(int status); #notsupported

	//// PATHCONF, FPATHCONF ////
	////nosupport: extern __device__ long int pathconf_(const char *path, int name);
	////nosupport: extern __device__ long int fpathconf_(int fd, int name);

	//// UNLINK ////
	//extern __device__ int unlink_(const char *filename); #sentinel-branch
	/* Host Absolute */
	int m0a = unlink(HostDir"missing.txt");
	makeAFile(HostDir"test.txt");
	int m0b = unlink(HostDir"test.txt");
	assert(m0a == -1 && !m0b);

	/* Device Absolute */
	int n0a = unlink(HostDir"missing.txt");
	makeAFile(DeviceDir"test.txt");
	int n0b = unlink(DeviceDir"test.txt");
	assert(n0a == -1 && !n0b);

	//// RMDIR ////
	//extern __device__ int rmdir_(const char *path); #sentinel-branch
	/* Host Absolute */
	int o0a = rmdir(HostDir"missing");
	mkdir(HostDir"test", 0);
	int o0b = rmdir(HostDir"test");
	assert(o0a == -1 && !o0b);

	/* Device Absolute */
	//int p0a = rmdir(DeviceDir"missing");
	//mkdir(DeviceDir"test");
	//int p0b = rmdir(DeviceDir"test");
	//assert(p0a == -1 && !p0b);
}
cudaError_t unistd_test1() { g_unistd_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }
