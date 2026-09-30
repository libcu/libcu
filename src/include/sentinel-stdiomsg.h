/*
sentinel-stdiomsg.h - messages for sentinel
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

#ifndef _SENTINEL_STDIOMSG_H
#define _SENTINEL_STDIOMSG_H

#include <sentinel.h>
#include <crtdefscu.h>
#include <stringcu.h>
#include <stdiocu.h>

enum {
	STDIO_REMOVE = 1,
	STDIO_RENAME,
	STDIO_FCLOSE,
	STDIO_FFLUSH,
	STDIO_FREOPEN,
	STDIO_SETVBUF,
	STDIO_FGETC,
	STDIO_FPUTC,
	STDIO_FGETS,
	STDIO_FPUTS,
	STDIO_UNGETC,
	STDIO_FREAD,
	STDIO_FWRITE,
	STDIO_FSEEK,
	STDIO_FTELL,
	STDIO_REWIND,
	STDIO_FSEEKO,
	STDIO_FTELLO,
	STDIO_FGETPOS,
	STDIO_FSETPOS,
	STDIO_CLEARERR,
	STDIO_FEOF,
	STDIO_FERROR,
	STDIO_FILENO,
};

struct stdio_remove {
	sentinelMessage base;
	const char *str;
	__device__ stdio_remove(const char *str) : base(STDIO_REMOVE, SENTINELFLOW_WAIT, SENTINEL_CHUNK), str(str) { sentinelDeviceSend(&base, sizeof(stdio_remove), ptrsIn); }
	int rc;
	sentinelInPtr ptrsIn[2] = {
		{ &str, -1 },
		{ nullptr }
	};
};

struct stdio_rename {
	sentinelMessage base;
	const char *str; const char *str2;
	__device__ stdio_rename(const char *str, const char *str2) : base(STDIO_RENAME, SENTINELFLOW_WAIT, SENTINEL_CHUNK), str(str), str2(str2) { sentinelDeviceSend(&base, sizeof(stdio_rename), ptrsIn); }
	int rc;
	sentinelInPtr ptrsIn[3] = {
		{ &str, -1 },
		{ &str2, -1 },
		{ nullptr }
	};
};

struct stdio_fclose {
	sentinelMessage base;
	FILE *file;
	__device__ stdio_fclose(bool wait, FILE *file) : base(STDIO_FCLOSE, wait ? SENTINELFLOW_WAIT : SENTINELFLOW_NONE), file(file) { sentinelDeviceSend(&base, sizeof(stdio_fclose)); }
	int rc;
};

struct stdio_fflush {
	sentinelMessage base;
	FILE *file;
	__device__ stdio_fflush(bool wait, FILE *file) : base(STDIO_FFLUSH, wait ? SENTINELFLOW_WAIT : SENTINELFLOW_NONE), file(file) { sentinelDeviceSend(&base, sizeof(stdio_fflush)); }
	int rc;
};

struct stdio_freopen {
	sentinelMessage base;
	const char *str; const char *str2; FILE *stream;
	__device__ stdio_freopen(const char *str, const char *str2, FILE *stream) : base(STDIO_FREOPEN, SENTINELFLOW_WAIT, SENTINEL_CHUNK), str(str), str2(str2), stream(stream) { sentinelDeviceSend(&base, sizeof(stdio_freopen), ptrsIn); }
	FILE *rc;
	sentinelInPtr ptrsIn[3] = {
		{ &str, -1 },
		{ &str2, -1 },
		{ nullptr }
	};
};

struct stdio_setvbuf {
	sentinelMessage base;
	FILE *file; char *buf; int mode; size_t size;
	__device__ stdio_setvbuf(FILE *file, char *buf, int mode, size_t size) : base(STDIO_SETVBUF, SENTINELFLOW_WAIT, SENTINEL_CHUNK), file(file), buf(buf), mode(mode), size(size) { sentinelDeviceSend(&base, sizeof(stdio_setvbuf), ptrsIn); }
	int rc;
	sentinelInPtr ptrsIn[2] = {
		{ &buf, -1 },
		{ nullptr }
	};
};

struct stdio_fgetc {
	sentinelMessage base;
	FILE *file;
	__device__ stdio_fgetc(FILE *file) : base(STDIO_FGETC, SENTINELFLOW_WAIT), file(file) { sentinelDeviceSend(&base, sizeof(stdio_fgetc)); }
	int rc;
};

struct stdio_fputc {
	sentinelMessage base;
	int ch; FILE *file;
	__device__ stdio_fputc(bool wait, int ch, FILE *file) : base(STDIO_FPUTC, wait ? SENTINELFLOW_WAIT : SENTINELFLOW_NONE), ch(ch), file(file) { sentinelDeviceSend(&base, sizeof(stdio_fputc)); }
	int rc;
};

struct stdio_fgets {
	sentinelMessage base;
	int num; FILE *file;
	__device__ stdio_fgets(char *str, int num, FILE *file) : base(STDIO_FGETS, SENTINELFLOW_WAIT, SENTINEL_CHUNK), str(str), num(num), file(file) { ptrsOut[0].size = num; sentinelDeviceSend(&base, sizeof(stdio_fgets), nullptr, ptrsOut); }
	char *str;
	char *rc;
	sentinelOutPtr ptrsOut[2] = {
		{ &str, &str, 0 },
		{ nullptr }
	};
};

struct stdio_fputs {
	sentinelMessage base;
	const char *str; FILE *file;
	__device__ stdio_fputs(bool wait, const char *str, FILE *file) : base(STDIO_FPUTS, wait ? SENTINELFLOW_WAIT : SENTINELFLOW_NONE, SENTINEL_CHUNK), str(str), file(file) { sentinelDeviceSend(&base, sizeof(stdio_fputs), ptrsIn); }
	int rc;
	sentinelInPtr ptrsIn[2] = {
		{ &str, -1 },
		{ nullptr }
	};
};

struct stdio_ungetc {
	sentinelMessage base;
	int ch; FILE *file;
	__device__ stdio_ungetc(bool wait, int ch, FILE *file) : base(STDIO_UNGETC, wait ? SENTINELFLOW_WAIT : SENTINELFLOW_NONE), ch(ch), file(file) { sentinelDeviceSend(&base, sizeof(stdio_ungetc)); }
	int rc;
};

struct stdio_fread {
	sentinelMessage base;
	void *buf; size_t size; size_t num; FILE *file;
	__device__ stdio_fread(bool wait, void *buf, size_t size, size_t num, FILE *file) : base(STDIO_FREAD, wait ? SENTINELFLOW_WAIT : SENTINELFLOW_NONE, SENTINEL_CHUNK), buf(buf), size(size), num(num), file(file) { ptrsOut[0].size = (int)(size * num); sentinelDeviceSend(&base, sizeof(stdio_fread), nullptr, ptrsOut); }
	size_t rc;
	void *ptr;
	sentinelOutPtr ptrsOut[2] = {
		{ &ptr, &buf, 0 },
		{ nullptr }
	};
};

struct stdio_fwrite {
	sentinelMessage base;
	const void *ptr; size_t size; size_t num; FILE *file;
	__device__ stdio_fwrite(bool wait, const void *ptr, size_t size, size_t num, FILE *file) : base(STDIO_FWRITE, wait ? SENTINELFLOW_WAIT : SENTINELFLOW_NONE, SENTINEL_CHUNK), ptr(ptr), size(size), num(num), file(file) { ptrsIn[0].size = (int)(size * num); sentinelDeviceSend(&base, sizeof(stdio_fwrite), ptrsIn); }
	size_t rc;
	sentinelInPtr ptrsIn[2] = {
		{ &ptr, 0 },
		{ nullptr }
	};
};

struct stdio_fseek {
	sentinelMessage base;
	FILE *file; long int offset; int origin;
	__device__ stdio_fseek(bool wait, FILE *file, long int offset, int origin) : base(STDIO_FSEEK, wait ? SENTINELFLOW_WAIT : SENTINELFLOW_NONE), file(file), offset(offset), origin(origin) { sentinelDeviceSend(&base, sizeof(stdio_fseek)); }
	int rc;
};

struct stdio_ftell {
	sentinelMessage base;
	FILE *file;
	__device__ stdio_ftell(FILE *file) : base(STDIO_FTELL, SENTINELFLOW_WAIT), file(file) { sentinelDeviceSend(&base, sizeof(stdio_ftell)); }
	int rc;
};

struct stdio_rewind {
	sentinelMessage base;
	FILE *file;
	__device__ stdio_rewind(FILE *file) : base(STDIO_REWIND, SENTINELFLOW_WAIT), file(file) { sentinelDeviceSend(&base, sizeof(stdio_rewind)); }
};

#if defined(__USE_LARGEFILE)
#ifndef __USE_LARGEFILE64
#define __off64_t char
#endif
struct stdio_fseeko {
	sentinelMessage base;
	FILE *file; __off_t offset; __off64_t offset64; int origin; bool bit64;
	__device__ stdio_fseeko(bool wait, FILE *file, __off_t offset, __off64_t offset64, int origin, bool bit64) : base(STDIO_FSEEKO, wait ? SENTINELFLOW_WAIT : SENTINELFLOW_NONE), file(file), offset(offset), offset64(offset64), origin(origin), bit64(bit64) { sentinelDeviceSend(&base, sizeof(stdio_fseeko)); }
	int rc;
};

struct stdio_ftello {
	sentinelMessage base;
	FILE *file; bool bit64;
	__device__ stdio_ftello(FILE *file, bool bit64) : base(STDIO_FTELLO, SENTINELFLOW_WAIT), file(file), bit64(bit64) { sentinelDeviceSend(&base, sizeof(stdio_ftello)); }
	__off_t rc;
#if __USE_LARGEFILE64
	__off64_t rc64;
#endif
};
#endif

#ifndef __USE_LARGEFILE64
#define fpos64_t char
#endif
struct stdio_fgetpos {
	sentinelMessage base;
	FILE *file; bool bit64;
	__device__ stdio_fgetpos(FILE *__restrict file, bool bit64) : base(STDIO_FGETPOS, SENTINELFLOW_WAIT), file(file), bit64(bit64) { sentinelDeviceSend(&base, sizeof(stdio_fgetpos)); }
	int rc;
	fpos_t pos; fpos64_t pos64;
};

struct stdio_fsetpos {
	sentinelMessage base;
	FILE *file; const fpos_t pos; const fpos64_t pos64; bool bit64;
	__device__ stdio_fsetpos(FILE *__restrict file, const fpos_t pos, const fpos64_t pos64, bool bit64) : base(STDIO_FSETPOS, SENTINELFLOW_WAIT), file(file), pos(pos), pos64(pos64), bit64(bit64) { sentinelDeviceSend(&base, sizeof(stdio_fsetpos)); }
	int rc;
};

struct stdio_clearerr {
	sentinelMessage base;
	FILE *file;
	__device__ stdio_clearerr(FILE *file) : base(STDIO_CLEARERR, SENTINELFLOW_NONE), file(file) { sentinelDeviceSend(&base, sizeof(stdio_clearerr)); }
};

struct stdio_feof {
	sentinelMessage base;
	FILE *file;
	__device__ stdio_feof(FILE *file) : base(STDIO_FEOF, SENTINELFLOW_WAIT), file(file) { sentinelDeviceSend(&base, sizeof(stdio_feof)); }
	int rc;
};

struct stdio_ferror {
	sentinelMessage base;
	FILE *file;
	__device__ stdio_ferror(FILE *file) : base(STDIO_FERROR, SENTINELFLOW_WAIT), file(file) { sentinelDeviceSend(&base, sizeof(stdio_ferror)); }
	int rc;
};

struct stdio_fileno {
	sentinelMessage base;
	FILE *file;
	__device__ stdio_fileno(FILE *file) : base(STDIO_FILENO, SENTINELFLOW_WAIT), file(file) { sentinelDeviceSend(&base, sizeof(stdio_fileno)); }
	int rc;
};

#endif  /* _SENTINEL_STDIOMSG_H */