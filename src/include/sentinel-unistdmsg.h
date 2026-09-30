/*
sentinel-unistdmsg.h - messages for sentinel
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

#ifndef _SENTINEL_UNISTDMSG_H
#define _SENTINEL_UNISTDMSG_H

#include <sentinel.h>
#include <crtdefscu.h>
#include <stringcu.h>

enum {
	UNISTD_ACCESS = 42,
	UNISTD_LSEEK,
	UNISTD_CLOSE,
	UNISTD_READ,
	UNISTD_WRITE,
	UNISTD_CHOWN,
	UNISTD_CHDIR,
	UNISTD_GETCWD,
	UNISTD_DUP,
	UNISTD_UNLINK,
	UNISTD_RMDIR,
};

struct unistd_access {
	sentinelMessage base;
	const char *str; int type;
	__device__ unistd_access(const char *str, int type) : base(UNISTD_ACCESS, SENTINELFLOW_WAIT, SENTINEL_CHUNK), str(str), type(type) { sentinelDeviceSend(&base, sizeof(unistd_access), ptrsIn); }
	int rc;
	sentinelInPtr ptrsIn[2] = {
		{ &str, -1 },
		{ nullptr }
	};
};

struct unistd_lseek {
	sentinelMessage base;
	int handle; long long offset; int whence; bool bit64;
	__device__ unistd_lseek(int fd, long long offset, int whence, bool bit64) : base(UNISTD_LSEEK, SENTINELFLOW_WAIT), handle(fd), offset(offset), whence(whence), bit64(bit64) { sentinelDeviceSend(&base, sizeof(unistd_lseek)); }
	long long rc;
};

struct unistd_close {
	sentinelMessage base;
	int handle;
	__device__ unistd_close(int fd) : base(UNISTD_CLOSE, SENTINELFLOW_WAIT), handle(fd) { sentinelDeviceSend(&base, sizeof(unistd_close)); }
	int rc;
};

struct unistd_read {
	sentinelMessage base;
	int handle; void *buf; size_t size;
	__device__ unistd_read(bool wait, int fd, void *buf, size_t size) : base(UNISTD_READ, wait ? SENTINELFLOW_WAIT : SENTINELFLOW_NONE, SENTINEL_CHUNK), handle(fd), buf(buf), size(size) { ptrsOut[0].size = (int)size; sentinelDeviceSend(&base, sizeof(unistd_read), nullptr, ptrsOut); }
	size_t rc;
	void *ptr;
	sentinelOutPtr ptrsOut[2] = {
		{ &ptr, &buf, 0, &rc },
		{ nullptr }
	};
};

struct unistd_write {
	sentinelMessage base;
	int handle; const void *ptr; size_t size;
	__device__ unistd_write(bool wait, int fd, const void *ptr, size_t size) : base(UNISTD_WRITE, wait ? SENTINELFLOW_WAIT : SENTINELFLOW_NONE, SENTINEL_CHUNK), handle(fd), ptr(ptr), size(size) { ptrsIn[0].size = (int)size; sentinelDeviceSend(&base, sizeof(unistd_write), ptrsIn); }
	size_t rc;
	sentinelInPtr ptrsIn[2] = {
		{ &ptr, 0 },
		{ nullptr }
	};
};

struct unistd_chown {
	sentinelMessage base;
	const char *str; int owner; int group;
	__device__ unistd_chown(const char *str, int owner, int group) : base(UNISTD_CHOWN, SENTINELFLOW_WAIT, SENTINEL_CHUNK), str(str), owner(owner), group(group) { sentinelDeviceSend(&base, sizeof(unistd_chown), ptrsIn); }
	int rc;
	sentinelInPtr ptrsIn[2] = {
		{ &str, -1 },
		{ nullptr }
	};
};

struct unistd_chdir {
	sentinelMessage base;
	const char *str;
	__device__ unistd_chdir(const char *str) : base(UNISTD_CHDIR, SENTINELFLOW_WAIT, SENTINEL_CHUNK), str(str) { sentinelDeviceSend(&base, sizeof(unistd_chdir), ptrsIn); }
	int rc;
	sentinelInPtr ptrsIn[2] = {
		{ &str, -1 },
		{ nullptr }
	};
};

struct unistd_getcwd {
	static __forceinline__ __device__ bool postfix(unistd_getcwd *t, intptr_t offset) { t->rc = t->rc ? t->buf : nullptr; return true; }
	sentinelMessage base;
	char *buf; size_t size;
	__device__ unistd_getcwd(char *buf, size_t size) : base(UNISTD_GETCWD, SENTINELFLOW_WAIT, SENTINEL_CHUNK, nullptr, SENTINELPOSTFIX(postfix)), buf(buf), size(size) { ptrsOut[0].size = (int)size; sentinelDeviceSend(&base, sizeof(unistd_getcwd), nullptr, ptrsOut); }
	char *rc;
	char *ptr;
	sentinelOutPtr ptrsOut[2] = {
		{ &ptr, &buf, 0 },
		{ nullptr }
	};
};

struct unistd_dup {
	sentinelMessage base;
	int handle; int handle2; bool dup1;
	__device__ unistd_dup(int fd, int fd2, bool dup1) : base(UNISTD_DUP, SENTINELFLOW_WAIT), handle(fd), handle2(fd2), dup1(dup1) { sentinelDeviceSend(&base, sizeof(unistd_dup)); }
	int rc;
};

struct unistd_unlink {
	sentinelMessage base;
	const char *str;
	__device__ unistd_unlink(const char *str) : base(UNISTD_UNLINK, SENTINELFLOW_WAIT, SENTINEL_CHUNK), str(str) { sentinelDeviceSend(&base, sizeof(unistd_unlink), ptrsIn); }
	int rc;
	sentinelInPtr ptrsIn[2] = {
		{ &str, -1 },
		{ nullptr }
	};
};

struct unistd_rmdir {
	sentinelMessage base;
	const char *str;
	__device__ unistd_rmdir(const char *str) : base(UNISTD_RMDIR, SENTINELFLOW_WAIT, SENTINEL_CHUNK), str(str) { sentinelDeviceSend(&base, sizeof(unistd_rmdir), ptrsIn); }
	int rc;
	sentinelInPtr ptrsIn[2] = {
		{ &str, -1 },
		{ nullptr }
	};
};

#endif  /* _SENTINEL_UNISTDMSG_H */