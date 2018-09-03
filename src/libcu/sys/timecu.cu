#include <timecu.h>
#include <sys/timecu.h>

__BEGIN_DECLS;
#if defined(__CUDA_ARCH__)

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

#else
#ifdef _MSC_VER
#include <sys/timeb.h>
int gettimeofday(struct timeval *tv, void *unused) {
	struct _timeb tb;
	_ftime(&tb);
	tv->tv_sec = (long)tb.time;
	tv->tv_usec = tb.millitm * 1000;
	return 0;
}
#endif
#endif
__END_DECLS;