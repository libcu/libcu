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

#pragma once
#ifndef _SENTINEL_DIRENTMSG_H
#define _SENTINEL_DIRENTMSG_H

#include <sentinel.h>
#include <crtdefscu.h>
#include <stringcu.h>
#include <direntcu.h>

enum {
	DIRENT_OPENDIR = 60,
	DIRENT_CLOSEDIR,
	DIRENT_READDIR,
	DIRENT_REWINDDIR,
};

struct dirent_opendir {
	static __forceinline__ __device__ char *Prepare(dirent_opendir *t, char *data, char *dataEnd, intptr_t offset)
	{
		int strLength = t->Str ? (int)strlen(t->Str) + 1 : 0;
		char *str = (char *)(data += ROUND8_(sizeof(*t)));
		char *end = (char *)(data += strLength);
		if (end > dataEnd) return nullptr;
		memcpy(str, t->Str, strLength);
		t->Str = str + offset;
		return end;
	}
	sentinelMessage Base;
	const char *Str;
	__device__ dirent_opendir(const char *str) : Base(true, DIRENT_OPENDIR, 1024, SENTINELPREPARE(Prepare)), Str(str) { sentinelDeviceSend(&Base, sizeof(dirent_opendir)); }
	DIR *RC;
};

struct dirent_closedir {
	sentinelMessage Base;
	DIR *Ptr;
	__device__ dirent_closedir(DIR *ptr) : Base(true, DIRENT_CLOSEDIR), Ptr(ptr) { sentinelDeviceSend(&Base, sizeof(dirent_closedir)); }
	int RC;
};

struct dirent_readdir {
	static __forceinline__ __host__ char *HostPrepare(dirent_readdir *t, char *data, char *dataEnd, intptr_t offset) {
		if (!t->RC) return data;
		int ptrLength = sizeof(struct dirent);
		char *ptr = (char *)(data += ROUND8_(sizeof(*t)));
		char *end = (char *)(data += ptrLength);
		if (end > dataEnd) return nullptr;
		memcpy(ptr, t->RC, ptrLength);
		t->RC = (struct dirent *)(ptr - offset);
		return end;
	}
#ifdef __USE_LARGEFILE64
	static __forceinline__ __host__ char *HostPrepare64(dirent_readdir *t, char *data, char *dataEnd, intptr_t offset) {
		if (!t->RC64) return data;
		int ptrLength = sizeof(struct dirent64);
		char *ptr = (char *)(data += ROUND8_(sizeof(*t)));
		char *end = (char *)(data += ptrLength);
		if (end > dataEnd) return nullptr;
		memcpy(ptr, t->RC64, ptrLength);
		t->RC64 = (struct dirent64 *)(ptr - offset);
		return end;
	}
#endif
	sentinelMessage Base;
	DIR *Ptr; bool Bit64;
	__device__ dirent_readdir(DIR *ptr, bool bit64) : Base(true, DIRENT_READDIR, 1024, nullptr), Ptr(ptr), Bit64(bit64) { sentinelDeviceSend(&Base, sizeof(dirent_readdir)); }
	struct dirent *RC;
#ifdef __USE_LARGEFILE64
	struct dirent64 *RC64;
#endif
};

struct dirent_rewinddir {
	sentinelMessage Base;
	DIR *Ptr;
	__device__ dirent_rewinddir(DIR *ptr) : Base(true, DIRENT_REWINDDIR), Ptr(ptr) { sentinelDeviceSend(&Base, sizeof(dirent_rewinddir)); }
};

#endif  /* _SENTINEL_DIRENTMSG_H */