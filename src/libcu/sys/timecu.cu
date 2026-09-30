#include <sys/timecu.h>
#ifdef LIBCU_LEAN_AND_MEAN
#ifdef __CUDA_ARCH__
__device__ int gettimeofday_(struct timeval *__restrict tv, void *tz) { return 0; }
#elif defined(_MSC_VER)
int gettimeofday(struct timeval *tv, void *unused) { return 0; }
#endif
#else
#include <timecu.h>

__BEGIN_DECLS;
#ifdef __CUDA_ARCH__

// gettimeofday
__device__ int gettimeofday_(struct timeval *tp, void *tz) {
	time_t seconds = time(nullptr);
	tp->tv_usec = 0;
	tp->tv_sec = seconds;
	return 0;
	//if (tz)
	//	_abort();
	//tp->tv_usec = 0;
	//return _time(&tp->tv_sec) == (time_t)-1 ? -1 : 0;
}

#elif defined(_MSC_VER)
#include <sys/timeb.h>
int gettimeofday(struct timeval *tv, void *unused) {
	struct _timeb tb;
	_ftime(&tb);
	tv->tv_sec = (long)tb.time;
	tv->tv_usec = tb.millitm * 1000;
	return 0;
}
#endif
__END_DECLS;
#endif