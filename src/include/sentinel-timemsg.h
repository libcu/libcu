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

#pragma once
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
	sentinelMessage Base;
	__device__ time_time() : Base(true, TIME_TIME) { sentinelDeviceSend(&Base, sizeof(time_time)); }
	time_t RC;
};

struct time_mktime {
	sentinelMessage Base;
	struct tm *Tp;
	__device__ time_mktime(struct tm *tp) : Base(true, TIME_MKTIME), Tp(tp) { sentinelDeviceSend(&Base, sizeof(time_mktime)); }
	time_t RC;
};

struct time_strftime {
	static __forceinline__ __device__ char *Prepare(time_strftime *t, char *data, char *dataEnd, intptr_t offset) {
		int fmtLength = t->Fmt ? (int)strlen(t->Fmt) + 1 : 0;
		char *fmt = (char *)(data += ROUND8_(sizeof(*t)));
		char *ptr = (char *)(data += fmtLength);
		char *end = (char *)(data += 1024 - fmtLength);
		if (end > dataEnd) return nullptr;
		memcpy(fmt, t->Fmt, fmtLength);
		t->Fmt = fmt + offset;
		t->Ptr = ptr + offset;
		return end;
	}
	static __forceinline__ __device__ bool Postfix(time_strftime *t, intptr_t offset) {
		char *ptr = (char *)t->Ptr - offset;
		if ((int)t->RC > 0) memcpy((void *)t->Buf, ptr, t->RC);
		return true;
	}
	sentinelMessage Base;
	const char *Buf; size_t Maxsize; const char *Fmt; const struct tm Tp;
	__device__ time_strftime(const char *buf, size_t maxsize, const char *fmt, const struct tm *tp) : Base(true, TIME_STRFTIME, 1024, SENTINELPREPARE(Prepare), SENTINELPOSTFIX(Postfix)), Buf(buf), Maxsize(maxsize), Fmt(fmt), Tp(*tp) { sentinelDeviceSend(&Base, sizeof(time_strftime)); }
	size_t RC;
	void *Ptr;
};

#endif  /* _SENTINEL_TIMEMSG_H */