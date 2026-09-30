/*
sentinel-fcntlmsg.h - messages for sentinel
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

#ifndef _SENTINEL_FCNTLMSG_H
#define _SENTINEL_FCNTLMSG_H

#include <sentinel.h>
#include <crtdefscu.h>
#include <stringcu.h>
#if __OS_UNIX
#define _stat64 stat64
#endif

enum {
	FCNTL_FCNTL = 53,
	FCNTL_OPEN,
	FCNTL_CLOSE,
	FCNTL_STAT,
	FCNTL_FSTAT,
	FCNTL_CHMOD,
	FCNTL_MKDIR,
	FCNTL_MKFIFO,
};

struct fcntl_fcntl {
	sentinelMessage base;
	int handle; int cmd; int p0; bool bit64;
	__device__ fcntl_fcntl(int fd, int cmd, int p0, bool bit64) : base(FCNTL_FCNTL, SENTINELFLOW_WAIT), handle(fd), cmd(cmd), p0(p0), bit64(bit64) { sentinelDeviceSend(&base, sizeof(fcntl_fcntl)); }
	int rc;
};

struct fcntl_open {
	sentinelMessage base;
	const char *str; int oflag; int p0; bool bit64;
	__device__ fcntl_open(const char *str, int oflag, int p0, bool bit64) : base(FCNTL_OPEN, SENTINELFLOW_WAIT, SENTINEL_CHUNK), str(str), oflag(oflag), p0(p0), bit64(bit64) { sentinelDeviceSend(&base, sizeof(fcntl_open), ptrsIn); }
	int rc;
	sentinelInPtr ptrsIn[2] = {
		{ &str, -1 },
		{ nullptr }
	};
};

struct fcntl_stat {
	sentinelMessage base;
	const char *str; struct stat *buf; struct _stat64 *buf64; bool bit64; bool lstat_;
	__device__ fcntl_stat(const char *str, struct stat *buf, struct _stat64 *buf64, bool bit64, bool lstat_) : base(FCNTL_STAT, SENTINELFLOW_WAIT, SENTINEL_CHUNK), str(str), buf(buf), buf64(buf64), bit64(bit64), lstat_(lstat_) { if (bit64) { ptrsOut[0].buf = &buf64; ptrsOut[0].size = sizeof(struct _stat64); } sentinelDeviceSend(&base, sizeof(fcntl_stat), ptrsIn, ptrsOut); }
	int rc;
	void *ptr;
	sentinelInPtr ptrsIn[2] = {
		{ &str, -1 },
		{ nullptr }
	};
	sentinelOutPtr ptrsOut[2] = {
		{ &ptr, &buf, sizeof(struct stat) },
		{ nullptr }
	};
};

struct fcntl_fstat {
	sentinelMessage base;
	int handle; struct stat *buf; struct _stat64 *buf64; bool bit64;
	__device__ fcntl_fstat(int fd, struct stat *buf, struct _stat64 *buf64, bool bit64) : base(FCNTL_FSTAT, SENTINELFLOW_WAIT, SENTINEL_CHUNK), handle(fd), buf(buf), buf64(buf64), bit64(bit64) { if (bit64) { ptrsOut[0].buf = &buf64; ptrsOut[0].size = sizeof(struct _stat64); } sentinelDeviceSend(&base, sizeof(fcntl_fstat), nullptr, ptrsOut); }
	int rc;
	void *ptr;
	sentinelOutPtr ptrsOut[2] = {
		{ &ptr, &buf, sizeof(struct stat) },
		{ nullptr }
	};
};

struct fcntl_chmod {
	sentinelMessage base;
	const char *str; mode_t mode;
	__device__ fcntl_chmod(const char *str, mode_t mode) : base(FCNTL_CHMOD, SENTINELFLOW_WAIT, SENTINEL_CHUNK), str(str), mode(mode) { sentinelDeviceSend(&base, sizeof(fcntl_chmod), ptrsIn); }
	int rc;
	sentinelInPtr ptrsIn[2] = {
		{ &str, -1 },
		{ nullptr }
	};
};

struct fcntl_mkdir {
	sentinelMessage base;
	const char *str; mode_t mode;
	__device__ fcntl_mkdir(const char *str, mode_t mode) : base(FCNTL_MKDIR, SENTINELFLOW_WAIT, SENTINEL_CHUNK), str(str), mode(mode) { sentinelDeviceSend(&base, sizeof(fcntl_mkdir), ptrsIn); }
	int rc;
	sentinelInPtr ptrsIn[2] = {
		{ &str, -1 },
		{ nullptr }
	};
};

struct fcntl_mkfifo {
	sentinelMessage base;
	const char *str; mode_t mode;
	__device__ fcntl_mkfifo(const char *str, mode_t mode) : base(FCNTL_MKFIFO, SENTINELFLOW_WAIT, SENTINEL_CHUNK), str(str), mode(mode) { sentinelDeviceSend(&base, sizeof(fcntl_mkfifo), ptrsIn); }
	int rc;
	sentinelInPtr ptrsIn[2] = {
		{ &str, -1 },
		{ nullptr }
	};
};

#endif  /* _SENTINEL_STATMSG_H */