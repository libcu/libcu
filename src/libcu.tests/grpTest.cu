#include <stdiocu.h>
#include <stringcu.h>
#include <grpcu.h>
#include <assert.h>

static __global__ void g_grp_test1() {
	printf("grp_test1\n");

	//// GETGRGID, GETGRNAM, GETGRENT, ENDGRENT, SETGRENT ///
	//extern __device__ struct group *getgrgid_(gid_t gid);
	//extern __device__ struct group *getgrnam_(const char *name);
	//extern __device__ struct group *getgrent_();
	//extern __device__ void endgrent_();
	//#define setgrent
	struct group *a0a = getgrgid(0); struct group *a0b = getgrgid(1); bool a0c = !strcmp(a0b->gr_name, "std"); assert(!a0a && a0b && a0c);
	struct group *b0a = getgrnam(nullptr); struct group *b0b = getgrnam(""); struct group *b0c = getgrnam("abc"); struct group *b0d = getgrnam("std"); bool b0e = !strcmp(b0d->gr_name, "std"); assert(!b0a && !b0b && !b0c && b0d && b0e);
	struct group *c0a = getgrent(); assert(c0a); int c0b = 1; while ((c0a = getgrent()) != nullptr) c0b++; assert(c0b == 1);
	struct group *d0a = getgrent(); setgrent(); struct group *d0b = getgrent(); endgrent(); struct group *d0c = getgrent(); assert(!d0a && d0b && d0c);
}
cudaError_t grp_test1() { g_grp_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }
