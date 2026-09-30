/*
sentinel-direntmsg.h - messages for sentinel
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

#ifndef _SENTINEL_DIRENTMSG_H
#define _SENTINEL_DIRENTMSG_H

#include <sentinel.h>
#include <crtdefscu.h>
#include <stringcu.h>
#include <direntcu.h>

enum {
	DIRENT_OPENDIR = 61,
	DIRENT_CLOSEDIR,
	DIRENT_READDIR,
	DIRENT_REWINDDIR,
};

struct dirent_opendir {
	sentinelMessage base;
	const char *str;
	__device__ dirent_opendir(const char *str) : base(DIRENT_OPENDIR, SENTINELFLOW_WAIT, SENTINEL_CHUNK), str(str) { sentinelDeviceSend(&base, sizeof(dirent_opendir), ptrsIn); }
	DIR *rc;
	sentinelInPtr ptrsIn[2] = {
		{ &str, -1 },
		{ nullptr }
	};
};

struct dirent_closedir {
	sentinelMessage base;
	DIR *ptr;
	__device__ dirent_closedir(DIR *ptr) : base(DIRENT_CLOSEDIR, SENTINELFLOW_WAIT), ptr(ptr) { sentinelDeviceSend(&base, sizeof(dirent_closedir)); }
	int rc;
};

struct dirent_readdir {
	static __forceinline__ __host__ char *hostPrepare(dirent_readdir *t, char *data, char *dataEnd, intptr_t offset) {
		if (!t->rc) return data;
		int ptrLength = sizeof(struct dirent);
		char *ptr = data;
		char *end = data += ptrLength;
		if (end > dataEnd) return nullptr;
		memcpy(ptr, t->rc, ptrLength);
		t->rc = (struct dirent *)(ptr - offset);
		return end;
	}
#ifdef __USE_LARGEFILE64
	static __forceinline__ __host__ char *hostPrepare64(dirent_readdir *t, char *data, char *dataEnd, intptr_t offset) {
		if (!t->rc64) return data;
		int ptrLength = sizeof(struct dirent64);
		char *ptr = data;
		char *end = data += ptrLength;
		if (end > dataEnd) return nullptr;
		memcpy(ptr, t->rc64, ptrLength);
		t->rc64 = (struct dirent64 *)(ptr - offset);
		return end;
	}
#endif
	sentinelMessage Base;
	DIR *ptr; bool bit64;
	__device__ dirent_readdir(DIR *ptr, bool bit64) : Base(DIRENT_READDIR, SENTINELFLOW_WAIT, SENTINEL_CHUNK), ptr(ptr), bit64(bit64) { sentinelDeviceSend(&Base, sizeof(dirent_readdir)); }
	struct dirent *rc;
#ifdef __USE_LARGEFILE64
	struct dirent64 *rc64;
#endif
};

struct dirent_rewinddir {
	sentinelMessage base;
	DIR *ptr;
	__device__ dirent_rewinddir(DIR *ptr) : base(DIRENT_REWINDDIR, SENTINELFLOW_WAIT), ptr(ptr) { sentinelDeviceSend(&base, sizeof(dirent_rewinddir)); }
};

#endif  /* _SENTINEL_DIRENTMSG_H */