/*
sentinel-timemsg.h - messages for sentinel
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

#ifndef _SENTINEL_TIMEMSG_H
#define _SENTINEL_TIMEMSG_H

#include <sentinel.h>
#include <crtdefscu.h>
#include <stringcu.h>

enum {
	TIME_TIME = 70,
	TIME_MKTIME,
	TIME_STRFTIME,
};

struct time_time {
	sentinelMessage base;
	__device__ time_time() : base(TIME_TIME, SENTINELFLOW_WAIT) { sentinelDeviceSend(&base, sizeof(time_time)); }
	time_t rc;
};

struct time_mktime {
	sentinelMessage base;
	struct tm *tp;
	__device__ time_mktime(struct tm *tp) : base(TIME_MKTIME, SENTINELFLOW_WAIT), tp(tp) { sentinelDeviceSend(&base, sizeof(time_mktime)); }
	time_t rc;
};

struct time_strftime {
	sentinelMessage base;
	const char *buf; size_t maxsize; const char *str; const struct tm tp;
	__device__ time_strftime(const char *buf, size_t maxsize, const char *str, const struct tm *tp) : base(TIME_STRFTIME, SENTINELFLOW_WAIT, SENTINEL_CHUNK), buf(buf), maxsize(maxsize), str(str), tp(*tp) { sentinelDeviceSend(&base, sizeof(time_strftime), ptrsIn, ptrsOut); }
	size_t rc;
	void *ptr;
	sentinelInPtr ptrsIn[2] = {
		{ &str, -1 },
		{ nullptr }
	};
	sentinelOutPtr ptrsOut[3] = {
		{ (void *)-1 },
		{ &ptr, &buf, -1, &rc },
		{ nullptr }
	};
};

#endif  /* _SENTINEL_TIMEMSG_H */