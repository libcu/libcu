/*
time.h - Date and time
The MIT License

Copyright (c) 2016 Sky Morey

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef _TIMECU_H
#define _TIMECU_H
#include <crtdefscu.h>

#include <time.h>
#ifdef __CUDA_ARCH__
__BEGIN_DECLS;

//#ifndef _WIN64
//typedef int clock_t;
//#else
//typedef long long int clock_t;
//#endif

__BEGIN_NAMESPACE_STD;
/* Time used by the program so far (user time + system time). The result / CLOCKS_PER_SECOND is program time in seconds.  */
//builtin: extern __device__ clock_t clock();

/* Return the current time and put it in *TIMER if TIMER is not NULL.  */
extern __device__ time_t time_(time_t *timer);
#define time time_

/* Return the difference between TIME1 and TIME0.  */
extern __device__ double difftime_(time_t time1, time_t time0);
#define difftime difftime_

/* Return the `time_t' representation of TP and normalize TP.  */
extern __device__ time_t mktime_(struct tm *tp);
#define mktime mktime_

/* Format TP into S according to FORMAT. no more than MAXSIZE characters and return the number of characters written, or 0 if it would exceed MAXSIZE.  */
extern __device__ size_t strftime_(char *__restrict s, size_t maxsize, const char *__restrict format, const struct tm *__restrict tp);
#define strftime strftime_
__END_NAMESPACE_STD;

__BEGIN_NAMESPACE_STD;
/* Return the `struct tm' representation of *TIMER in Universal Coordinated Time (aka Greenwich Mean Time).  */
extern __device__ struct tm *gmtime_(const time_t *timer);
#define gmtime gmtime_

/* Return the `struct tm' representation of *TIMER in the local timezone.  */
//localtime is gmtime: extern __device__ struct tm *localtime_(const time_t *timer);
#define localtime gmtime_
__END_NAMESPACE_STD;

__BEGIN_NAMESPACE_STD;
/* Return a string of the form "Day Mon dd hh:mm:ss yyyy\n" that is the representation of TP in this format.  */
extern __device__ char *asctime_(const struct tm *tp);
#define asctime asctime_

/* Equivalent to `asctime (localtime (timer))'.  */
__forceinline__ __device__ char *ctime_(const time_t *timer) { return asctime(localtime(timer)); }
#define ctime ctime_
__END_NAMESPACE_STD;

__END_DECLS;
#else

#endif  /* __CUDA_ARCH__ */

#endif  /* _TIMECU_H */
