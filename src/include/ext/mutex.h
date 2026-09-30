/*
mutex.h - xxx
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

#ifndef _EXT_MUTEX_H
#define _EXT_MUTEX_H
#ifdef __cplusplus
extern "C" {
#endif

#define MUTEXPRED_EQ 1
#define MUTEXPRED_NE 2
#define MUTEXPRED_LT 3
#define MUTEXPRED_GT 4
#define MUTEXPRED_LTE 5
#define MUTEXPRED_GTE 6
#define MUTEXPRED_AND 7
#define MUTEXPRED_ANE 8

	typedef struct mutexSleep_t {
		float msmin;
		float msmax;
		float factor;
		float ms;
	} mutexSleep_t;

	/* Mutex default sleep */
	extern __hostb_device__ mutexSleep_t mutexDefaultSleep;

	/* Mutex with exponential back-off. */
	extern __host_device__ void mutexSpinLock(void **cancelToken, volatile long *mutex, long cmp = 0, long val = 1, char pred = 0, long predVal = 0, bool(*func)(void **) = nullptr, void **funcTag = nullptr, mutexSleep_t *ms = nullptr);

	/* Mutex set. */
	extern __host_device__ void mutexSet(volatile long *mutex, long val = 0);

	/* Mutex held. */
#define mutexHeld(mutex) (*mutex == 1)

#ifdef __cplusplus
}
#endif
#endif  /* _EXT_MUTEX_H */