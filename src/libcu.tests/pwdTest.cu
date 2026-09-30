#ifdef LIBCU_LEAN_AND_MEAN
cudaError_t pwd_test1() { return cudaSuccess; }
#else
#include <stdiocu.h>
#include <stringcu.h>
#include <pwdcu.h>
#include <assert.h>

static __global__ void g_pwd_test1() {
	printf("pwd_test1\n");

	//// GETPWUID, GETPWNAM, GETPWENT, ENDPWENT, SETPWENT ////
	//extern __device__ struct passwd *getpwuid_(uid_t uid);
	//extern __device__ struct passwd *getpwnam_(const char *name);
	//extern __device__ struct passwd *getpwent_();
	//extern __device__ void endpwent_();
	//#define setpwent
	struct passwd *a0a = getpwuid(0); struct passwd *a0b = getpwuid(1); bool a0c = !strcmp(a0b->pw_name, "std"); assert(!a0a && a0b && a0c);
	struct passwd *b0a = getpwnam(nullptr); struct passwd *b0b = getpwnam(""); struct passwd *b0c = getpwnam("abc"); struct passwd *b0d = getpwnam("std"); bool b0e = !strcmp(b0d->pw_name, "std"); assert(!b0a && !b0b && !b0c && b0d && b0e);
	struct passwd *c0a = getpwent(); assert(c0a); int c0b = 1; while ((c0a = getpwent()) != nullptr) c0b++; assert(c0b == 1);
	struct passwd *d0a = getpwent(); setpwent(); struct passwd *d0b = getpwent(); endpwent(); struct passwd *d0c = getpwent(); assert(!d0a && d0b && d0c);
}
cudaError_t pwd_test1() { g_pwd_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }

#endif