/*
sentinel-stdlibmsg.h - messages for sentinel
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

#ifndef _SENTINEL_STDLIBMSG_H
#define _SENTINEL_STDLIBMSG_H

#include <sentinel.h>
#include <stringcu.h>

enum {
	STDLIB_EXIT = 30,
	STDLIB_SYSTEM,
	STDLIB_GETENV,
	STDLIB_SETENV,
	STDLIB_UNSETENV,
	STDLIB_MKTEMP,
	STDLIB_MKSTEMP,
};

struct stdlib_exit {
	sentinelMessage base;
	bool std; int status;
	__device__ stdlib_exit(bool std, int status) : base(STDLIB_EXIT, SENTINELFLOW_WAIT), std(std), status(status) { sentinelDeviceSend(&base, sizeof(stdlib_exit)); }
};

struct stdlib_system {
	sentinelMessage base;
	const char *str;
	__device__ stdlib_system(const char *str) : base(STDLIB_SYSTEM, SENTINELFLOW_WAIT, SENTINEL_CHUNK), str(str) { sentinelDeviceSend(&base, sizeof(stdlib_system), ptrsIn); }
	int rc;
	sentinelInPtr ptrsIn[2] = {
		{ &str, -1 },
		{ nullptr }
	};
};

struct stdlib_getenv {
	static __forceinline__ __host__ char *hostPrepare(stdlib_getenv *t, char *data, char *dataEnd, intptr_t offset) {
		if (!t->rc) return data;
		int ptrLength = t->rc ? (int)strlen(t->rc) + 1 : 0;
		if (ptrLength > SENTINEL_CHUNK) { ptrLength = SENTINEL_CHUNK; t->rc[ptrLength] = 0; }
		char *ptr = (char *)(data += ROUND8_(sizeof(*t)));
		char *end = (char *)(data += ptrLength);
		if (end > dataEnd) return nullptr;
		memcpy(ptr, t->rc, ptrLength);
		t->rc = (char *)(ptr - offset);
		return end;
	}
	sentinelMessage base;
	const char *str;
	__device__ stdlib_getenv(const char *str) : base(STDLIB_GETENV, SENTINELFLOW_WAIT, SENTINEL_CHUNK), str(str) { sentinelDeviceSend(&base, sizeof(stdlib_getenv), ptrsIn); }
	char *rc;
	sentinelInPtr ptrsIn[2] = {
		{ &str, -1 },
		{ nullptr }
	};
};

struct stdlib_setenv {
	sentinelMessage base;
	const char *str; const char *str2; int replace;
	__device__ stdlib_setenv(const char *str, const char *str2, int replace) : base(STDLIB_SETENV, SENTINELFLOW_WAIT, SENTINEL_CHUNK), str(str), str2(str2), replace(replace) { sentinelDeviceSend(&base, sizeof(stdlib_setenv), ptrsIn); }
	int rc;
	sentinelInPtr ptrsIn[3] = {
		{ &str, -1 },
		{ &str2, -1 },
		{ nullptr }
	};
};

struct stdlib_unsetenv {
	sentinelMessage base;
	const char *str;
	__device__ stdlib_unsetenv(const char *str) : base(STDLIB_UNSETENV, SENTINELFLOW_WAIT, SENTINEL_CHUNK), str(str) { sentinelDeviceSend(&base, sizeof(stdlib_unsetenv), ptrsIn); }
	int rc;
	sentinelInPtr ptrsIn[2] = {
		{ &str, -1 },
		{ nullptr }
	};
};

struct stdlib_mktemp {
	sentinelMessage base;
	char *str;
	__device__ stdlib_mktemp(char *str) : base(STDLIB_MKTEMP, SENTINELFLOW_WAIT, SENTINEL_CHUNK), str(str) { sentinelDeviceSend(&base, sizeof(stdlib_mktemp), ptrsIn); }
	char *rc;
	sentinelInPtr ptrsIn[2] = {
		{ &str, -1 },
		{ nullptr }
	};
};

struct stdlib_mkstemp {
	static __forceinline__ __device__ bool postfix(stdlib_mkstemp *t, intptr_t offset) {
		char *ptr = (char *)t->ptr - offset;
		if (t->str) strcpy(t->str, ptr);
		return true;
	}
	sentinelMessage base;
	char *str;
	__device__ stdlib_mkstemp(char *str) : base(STDLIB_MKSTEMP, SENTINELFLOW_WAIT, SENTINEL_CHUNK, nullptr, SENTINELPOSTFIX(postfix)), str(str) { sentinelDeviceSend(&base, sizeof(stdlib_mkstemp), ptrsIn); }
	int rc;
	void *ptr;
	sentinelInPtr ptrsIn[2] = {
		{ &str, -1 },
		{ nullptr }
	};
};

#endif  /* _SENTINEL_STDLIBMSG_H */