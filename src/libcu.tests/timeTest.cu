#ifdef LIBCU_LEAN_AND_MEAN
cudaError_t time_test1() { return cudaSuccess; }
#else
#include <stdiocu.h>
#include <timecu.h>
#include <assert.h>

static __global__ void g_time_test1() {
	printf("time_test1\n");
	time_t rawtime; time(&rawtime);
	struct tm *timeinfo = localtime(&rawtime);

	//// CLOCK, TIME, DIFFTIME, MKTIME ////
	////builtin: extern __device__ clock_t clock();
	//extern __device__ time_t time_(time_t *timer);
	//extern __device__ double difftime_(time_t time1, time_t time0);
	//extern __device__ time_t mktime_(struct tm *tp); #sentinel
	time_t a0a = time(nullptr); time_t a0b; time_t a0c = time(&a0b); assert(a0a != -1 && a0b != -1 && a0c != -1);
	double a1a = difftime(1, 2); assert(a1a == -1);

	//// STRFTIME ////
	//extern size_t strftime_(char *__restrict s, size_t maxsize, const char *__restrict format, const struct tm *__restrict tp); #sentinel
	char b0_buf[80];
	int b0a = strftime(b0_buf, sizeof(b0_buf), "Now it's %I:%M%p.", timeinfo);
	assert(!strncmp(b0_buf, "Now it's ", 9));

	//// GMTIME ////
	//extern __device__ struct tm *gmtime_(const time_t *timer);
	struct tm *c0a = gmtime(&rawtime); assert(c0a);

	//// ASCTIME, CTIME ////
	//extern __device__ char *asctime_(const struct tm *tp);
	//__forceinline__ __device__ char *ctime_(const time_t *timer);
	char *d0a = asctime(timeinfo); assert(d0a);
	char *d1a = ctime(&rawtime); assert(d1a);
}
cudaError_t time_test1() { g_time_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }

#endif