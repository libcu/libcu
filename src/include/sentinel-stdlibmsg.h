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

#pragma once
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
};

struct stdlib_exit {
	sentinelMessage Base;
	bool Std;
	int Status;
	__device__ stdlib_exit(bool std, int status) : Base(true, STDLIB_EXIT), Std(std), Status(status) { sentinelDeviceSend(&Base, sizeof(stdlib_exit)); }
};

struct stdlib_system {
	static __forceinline__ __device__ char *Prepare(stdlib_system *t, char *data, char *dataEnd, intptr_t offset) {
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
	__device__ stdlib_system(const char *str) : Base(true, STDLIB_SYSTEM, 1024, SENTINELPREPARE(Prepare)), Str(str) { sentinelDeviceSend(&Base, sizeof(stdlib_system)); }
	int RC;
};

struct stdlib_getenv {
	static __forceinline__ __device__ char *Prepare(stdlib_getenv *t, char *data, char *dataEnd, intptr_t offset) {
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
	__device__ stdlib_getenv(const char *str) : Base(true, STDLIB_GETENV, 1024, SENTINELPREPARE(Prepare)), Str(str) { sentinelDeviceSend(&Base, sizeof(stdlib_getenv)); }
	char *RC;
};

struct stdlib_setenv {
	static __forceinline__ __device__ char *Prepare(stdlib_setenv *t, char *data, char *dataEnd, intptr_t offset) {
		int strLength = t->Str ? (int)strlen(t->Str) + 1 : 0;
		int str2Length = t->Str2 ? (int)strlen(t->Str2) + 1 : 0;
		char *str = (char *)(data += ROUND8_(sizeof(*t)));
		char *str2 = (char *)(data += strLength);
		char *end = (char *)(data += str2Length);
		if (end > dataEnd) return nullptr;
		memcpy(str, t->Str, strLength);
		memcpy(str2, t->Str2, str2Length);
		t->Str = str + offset;
		t->Str2 = str2 + offset;
		return end;
	}
	sentinelMessage Base;
	const char *Str;
	const char *Str2;
	int Replace;
	__device__ stdlib_setenv(const char *str, const char *str2, int replace) : Base(true, STDLIB_SYSTEM, 1024, SENTINELPREPARE(Prepare)), Str(str), Str2(str2), Replace(replace) { sentinelDeviceSend(&Base, sizeof(stdlib_setenv)); }
	int RC;
};

struct stdlib_unsetenv {
	static __forceinline__ __device__ char *Prepare(stdlib_unsetenv *t, char *data, char *dataEnd, intptr_t offset) {
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
	__device__ stdlib_unsetenv(const char *str) : Base(true, STDLIB_SYSTEM, 1024, SENTINELPREPARE(Prepare)), Str(str) { sentinelDeviceSend(&Base, sizeof(stdlib_unsetenv)); }
	int RC;
};

#endif  /* _SENTINEL_STDLIBMSG_H */