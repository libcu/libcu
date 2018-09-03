#include <timecu.h>
#include <sentinel-timemsg.h>
#include <stdiocu.h>

__BEGIN_DECLS;

/* Prototypes */
static __device__ struct tm *_t2tm(const time_t *__restrict timer, int offset, struct tm *__restrict result);

/* Global shared by gmtime() and localtime(). */
static __device__ struct tm __time_tm;

/* Return the current time and put it in *TIMER if TIMER is not NULL.  */
__device__ time_t time_(time_t *timer) {
	time_time msg; time_t time = msg.RC;
	if (timer) *timer = time;
	return time;
}

/* Return the difference between TIME1 and TIME0.  */
__device__ double difftime_(time_t time1, time_t time0) {
	return (double)time1 - (double)time0;
}

/* Return the `time_t' representation of TP and normalize TP.  */
__device__ time_t mktime_(struct tm *tp) {
	time_mktime msg(tp); return msg.RC;
}

/* Format TP into S according to FORMAT. no more than MAXSIZE characters and return the number of characters written, or 0 if it would exceed MAXSIZE.  */
__device__ size_t strftime_(char *__restrict s, size_t maxsize, const char *__restrict format, const struct tm *__restrict tp) {
	time_strftime msg(s, maxsize, format, tp); return msg.RC;
}

/* Return the `struct tm' representation of *TIMER in Universal Coordinated Time (aka Greenwich Mean Time).  */
__device__ struct tm *gmtime_(const time_t *timer) {
	register struct tm *ptm = &__time_tm;
	_t2tm(timer, 0, ptm);
	return ptm;
}

/* Return a string of the form "Day Mon dd hh:mm:ss yyyy\n" that is the representation of TP in this format.  */
static __constant__ const char *__asctime_wday_name[] = { "Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat" };
static __constant__ const char *__asctime_mon_name[] = { "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec" };
static __device__ char __asctime_buf[26];
__device__ char *asctime_(const struct tm *tp) {
	sprintf(__asctime_buf, "%.3s %.3s%3d %.2d:%.2d:%.2d %d\n",
		__asctime_wday_name[tp->tm_wday],
		__asctime_mon_name[tp->tm_mon],
		tp->tm_mday, tp->tm_hour,
		tp->tm_min, tp->tm_sec,
		1900 + tp->tm_year);
	return __asctime_buf;
}

// T2TM
#pragma region T2TM
#ifndef __isleap
#define __isleap(y) (!((y) % 4) && (((y) % 100) || !((y) % 400)))
#endif
static __constant__ const unsigned short __t2tm_vals[] = { 60, 60, 24, 7 /* special */, 36524, 1461, 365, 0 };
static __constant__ const unsigned char __t2tm_days[] = { 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31, /* non-leap */ 29, };
static __device__ tm *_t2tm(const time_t *__restrict timer, int offset, struct tm *__restrict r) {
	register int *p;
	time_t t1, t, v;
	int wday;
	{
		t = *timer;
		p = (int *)r;
		p[7] = 0;
		register const unsigned short *vp = __t2tm_vals;
		do {
			if ((v = *vp) == 7) {
				/* Valid range for t is [-784223472856L, 784223421720L]. Outside of this range, the tm_year field will overflow. */
				if ((unsigned long)(t + offset - -784223472856L) > (784223421720L - -784223472856L))
					return nullptr;
				/* We have days since the epoch, so caluclate the weekday. */
				wday = (t + 4) % (*vp);	/* t is unsigned */
				/* Set divisor to days in 400 years.  Be kind to bcc... */
				v = ((time_t)(vp[1])) << 2;
				++v;
				/* Change to days since 1/1/1601 so that for 32 bit time_t values, we'll have t >= 0.  This should be changed for
				* archs with larger time_t types. Also, correct for offset since a multiple of 7. */
				t += (135140L - 366) + offset;
			}
			if ((t -= ((t1 = t / v) * v)) < 0) {
				t += v;
				--t1;
			}
			if (*vp == 7 && t == v - 1) {
				--t;			/* Correct for 400th year leap case */
				++p[4];			/* Stash the extra day... */
			}
			if (v <= 60) {
				*p++ = t;
				t = t1;
			}
			else
				*p++ = t1;
		} while (*++vp);
	}

	if (p[-1] == 4) {
		--p[-1];
		t = 365;
	}
	*p += ((int)t);			/* r[7] .. tm_yday */
	p -= 2;						/* at r[5] */
	*p = ((((p[-2] << 2) + p[-1]) * 25 + p[0]) << 2) + (p[1] - 299); /* tm_year */
	p[1] = wday;				/* r[6] .. tm_wday */
	{
		register const unsigned char *d = __t2tm_days;
		wday = 1900 + *p;
		if (__isleap(wday))
			d += 11;
		wday = p[2] + 1;		/* r[7] .. tm_yday */
		*--p = 0;				/* at r[4] .. tm_mon */
		while (wday > *d) {
			wday -= *d;
			if (*d == 29)
				d -= 11;		/* Backup to non-leap Feb. */
			++d;
			++*p;				/* Increment tm_mon. */
		}
		p[-1] = wday;			/* r[3] .. tm_mday */
	}
	p[4] = 0;					/* r[8] .. tm_isdst */
	return r;
}
#pragma endregion

__END_DECLS;