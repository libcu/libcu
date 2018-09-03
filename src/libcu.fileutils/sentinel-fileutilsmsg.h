/*
sentinel-fileutilsmsg.h - messages for sentinel
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

#ifndef _SENTINEL_FILEUTILSMSG_H
#define _SENTINEL_FILEUTILSMSG_H
#define HAS_GPU 0
#define HAS_HOSTSENTINEL 1
#include <sentinel.h>
#include <string.h>

enum {
	FILEUTILS_DCAT = 0,
	FILEUTILS_DCHGRP,
	FILEUTILS_DCHMOD,
	FILEUTILS_DCHOWN,
	FILEUTILS_DCMP,
	FILEUTILS_DCP,
	FILEUTILS_DGREP,
	FILEUTILS_DLS,
	FILEUTILS_DMKDIR,
	FILEUTILS_DMORE,
	FILEUTILS_DMV,
	FILEUTILS_DRM,
	FILEUTILS_DRMDIR,
	FILEUTILS_DPWD,
	FILEUTILS_DCD,
};

struct fileutils_dcat {
	static __forceinline__ char *Prepare(fileutils_dcat *t, char *data, char *dataEnd, intptr_t offset) {
		int strLength = (t->Str ? (int)strlen(t->Str) + 1 : 0);
		char *str = (char *)(data += ROUND8_(sizeof(*t)));
		char *end = (char *)(data += strLength);
		if (end > dataEnd) return nullptr;
		memcpy(str, t->Str, strLength);
		t->Str = str + offset;
		return end;
	}
	sentinelMessage Base;
	char *Str;
	fileutils_dcat(char *str) : Base(true, FILEUTILS_DCAT, 1024, SENTINELPREPARE(Prepare)), Str(str) { sentinelClientSend(&Base, sizeof(fileutils_dcat)); }
	int RC;
};

struct fileutils_dchgrp {
	static __forceinline__ char *Prepare(fileutils_dchgrp *t, char *data, char *dataEnd, intptr_t offset) {
		int strLength = (t->Str ? (int)strlen(t->Str) + 1 : 0);
		char *str = (char *)(data += ROUND8_(sizeof(*t)));
		char *end = (char *)(data += strLength);
		if (end > dataEnd) return nullptr;
		memcpy(str, t->Str, strLength);
		t->Str = str + offset;
		return end;
	}
	sentinelMessage Base;
	char *Str; int Gid;
	fileutils_dchgrp(char *str, int gid) : Base(true, FILEUTILS_DCHGRP, 1024, SENTINELPREPARE(Prepare)), Str(str), Gid(gid) { sentinelClientSend(&Base, sizeof(fileutils_dchgrp)); }
	int RC;
};

struct fileutils_dchmod {
	static __forceinline__ char *Prepare(fileutils_dchmod *t, char *data, char *dataEnd, intptr_t offset) {
		int strLength = (t->Str ? (int)strlen(t->Str) + 1 : 0);
		char *str = (char *)(data += ROUND8_(sizeof(*t)));
		char *end = (char *)(data += strLength);
		if (end > dataEnd) return nullptr;
		memcpy(str, t->Str, strLength);
		t->Str = str + offset;
		return end;
	}
	sentinelMessage Base;
	char *Str; int Mode;
	fileutils_dchmod(char *str, int mode) : Base(true, FILEUTILS_DCHMOD, 1024, SENTINELPREPARE(Prepare)), Str(str), Mode(mode) { sentinelClientSend(&Base, sizeof(fileutils_dchmod)); }
	int RC;
};

struct fileutils_dchown {
	static __forceinline__ char *Prepare(fileutils_dchown *t, char *data, char *dataEnd, intptr_t offset) {
		int strLength = (t->Str ? (int)strlen(t->Str) + 1 : 0);
		char *str = (char *)(data += ROUND8_(sizeof(*t)));
		char *end = (char *)(data += strLength);
		if (end > dataEnd) return nullptr;
		memcpy(str, t->Str, strLength);
		t->Str = str + offset;
		return end;
	}
	sentinelMessage Base;
	char *Str; int Uid;
	fileutils_dchown(char *str, int uid) : Base(true, FILEUTILS_DCHOWN, 1024, SENTINELPREPARE(Prepare)), Str(str), Uid(uid) { sentinelClientSend(&Base, sizeof(fileutils_dchown)); }
	int RC;
};

struct fileutils_dcmp {
	static __forceinline__ char *Prepare(fileutils_dcmp *t, char *data, char *dataEnd, intptr_t offset) {
		int strLength = (t->Str ? (int)strlen(t->Str) + 1 : 0);
		int str2Length = (t->Str2 ? (int)strlen(t->Str2) + 1 : 0);
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
	char *Str; char *Str2;
	fileutils_dcmp(char *str, char *str2) : Base(true, FILEUTILS_DCMP, 1024, SENTINELPREPARE(Prepare)), Str(str), Str2(str2) { sentinelClientSend(&Base, sizeof(fileutils_dcmp)); }
	int RC;
};

struct fileutils_dcp {
	static __forceinline__ char *Prepare(fileutils_dcp *t, char *data, char *dataEnd, intptr_t offset) {
		int strLength = (t->Str ? (int)strlen(t->Str) + 1 : 0);
		int str2Length = (t->Str2 ? (int)strlen(t->Str2) + 1 : 0);
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
	char *Str; char *Str2; bool SetModes;
	fileutils_dcp(char *str, char *str2, bool setModes) : Base(true, FILEUTILS_DCP, 1024, SENTINELPREPARE(Prepare)), Str(str), Str2(str2), SetModes(setModes) { sentinelClientSend(&Base, sizeof(fileutils_dcp)); }
	int RC;
};

struct fileutils_dgrep {
	static __forceinline__ char *Prepare(fileutils_dgrep *t, char *data, char *dataEnd, intptr_t offset) {
		int strLength = (t->Str ? (int)strlen(t->Str) + 1 : 0);
		int str2Length = (t->Str2 ? (int)strlen(t->Str2) + 1 : 0);
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
	char *Str; char *Str2; bool IgnoreCase; bool TellName; bool TellLine;
	fileutils_dgrep(char *str, char *str2, bool ignoreCase, bool tellName, bool tellLine) : Base(true, FILEUTILS_DGREP, 1024, SENTINELPREPARE(Prepare)), Str(str), Str2(str2), IgnoreCase(ignoreCase), TellName(tellName), TellLine(tellLine) { sentinelClientSend(&Base, sizeof(fileutils_dgrep)); }
	int RC;
};

struct fileutils_dls {
	static __forceinline__ char *Prepare(fileutils_dls *t, char *data, char *dataEnd, intptr_t offset) {
		int strLength = (t->Str ? (int)strlen(t->Str) + 1 : 0);
		char *str = (char *)(data += ROUND8_(sizeof(*t)));
		char *end = (char *)(data += strLength);
		if (end > dataEnd) return nullptr;
		memcpy(str, t->Str, strLength);
		t->Str = str + offset;
		return end;
	}
	sentinelMessage Base;
	char *Str; int Flags; bool EndSlash;
	fileutils_dls(char *str, int flags, bool endSlash) : Base(true, FILEUTILS_DLS, 1024, SENTINELPREPARE(Prepare)), Str(str), Flags(flags), EndSlash(endSlash) { sentinelClientSend(&Base, sizeof(fileutils_dls)); }
	int RC;
};

struct fileutils_dmkdir {
	static __forceinline__ char *Prepare(fileutils_dmkdir *t, char *data, char *dataEnd, intptr_t offset) {
		int strLength = (t->Str ? (int)strlen(t->Str) + 1 : 0);
		char *str = (char *)(data += ROUND8_(sizeof(*t)));
		char *end = (char *)(data += strLength);
		if (end > dataEnd) return nullptr;
		memcpy(str, t->Str, strLength);
		t->Str = str + offset;
		return end;
	}
	sentinelMessage Base;
	char *Str; unsigned short Mode;
	fileutils_dmkdir(char *str, unsigned short mode) : Base(true, FILEUTILS_DMKDIR, 1024, SENTINELPREPARE(Prepare)), Str(str), Mode(mode) { sentinelClientSend(&Base, sizeof(fileutils_dmkdir)); }
	int RC;
};

struct fileutils_dmore {
	static __forceinline__ char *Prepare(fileutils_dmore *t, char *data, char *dataEnd, intptr_t offset) {
		int strLength = (t->Str ? (int)strlen(t->Str) + 1 : 0);
		char *str = (char *)(data += ROUND8_(sizeof(*t)));
		char *end = (char *)(data += strLength);
		if (end > dataEnd) return nullptr;
		memcpy(str, t->Str, strLength);
		t->Str = str + offset;
		return end;
	}
	sentinelMessage Base;
	char *Str; int Fd;
	fileutils_dmore(char *str, int fd) : Base(true, FILEUTILS_DMORE, 1024, SENTINELPREPARE(Prepare)), Str(str), Fd(fd) { sentinelClientSend(&Base, sizeof(fileutils_dmore)); }
	int RC;
};

struct fileutils_dmv {
	static __forceinline__ char *Prepare(fileutils_dmv *t, char *data, char *dataEnd, intptr_t offset) {
		int strLength = (t->Str ? (int)strlen(t->Str) + 1 : 0);
		int str2Length = (t->Str2 ? (int)strlen(t->Str2) + 1 : 0);
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
	char *Str; char *Str2;
	fileutils_dmv(char *str, char *str2) : Base(true, FILEUTILS_DMV, 1024, SENTINELPREPARE(Prepare)), Str(str), Str2(str2) { sentinelClientSend(&Base, sizeof(fileutils_dmv)); }
	int RC;
};

struct fileutils_drm {
	static __forceinline__ char *Prepare(fileutils_drm *t, char *data, char *dataEnd, intptr_t offset) {
		int strLength = (t->Str ? (int)strlen(t->Str) + 1 : 0);
		char *str = (char *)(data += ROUND8_(sizeof(*t)));
		char *end = (char *)(data += strLength);
		if (end > dataEnd) return nullptr;
		memcpy(str, t->Str, strLength);
		t->Str = str + offset;
		return end;
	}
	sentinelMessage Base;
	char *Str;
	fileutils_drm(char *str) : Base(true, FILEUTILS_DRM, 1024, SENTINELPREPARE(Prepare)), Str(str) { sentinelClientSend(&Base, sizeof(fileutils_drm)); }
	int RC;
};

struct fileutils_drmdir {
	static __forceinline__ char *Prepare(fileutils_drmdir *t, char *data, char *dataEnd, intptr_t offset) {
		int strLength = (t->Str ? (int)strlen(t->Str) + 1 : 0);
		char *str = (char *)(data += ROUND8_(sizeof(*t)));
		char *end = (char *)(data += strLength);
		if (end > dataEnd) return nullptr;
		memcpy(str, t->Str, strLength);
		t->Str = str + offset;
		return end;
	}
	sentinelMessage Base;
	char *Str;
	fileutils_drmdir(char *str) : Base(true, FILEUTILS_DRMDIR, 1024, SENTINELPREPARE(Prepare)), Str(str) { sentinelClientSend(&Base, sizeof(fileutils_drmdir)); }
	int RC;
};

struct fileutils_dpwd {
	static __forceinline__ __device__ char *Prepare(fileutils_dpwd *t, char *data, char *dataEnd, intptr_t offset) {
		t->Ptr = (char *)(data += ROUND8_(sizeof(*t)));
		char *end = (char *)(data += 1024);
		if (end > dataEnd) return nullptr;
		return end;
	}
	sentinelMessage Base;
	fileutils_dpwd() : Base(true, FILEUTILS_DPWD, 1024, SENTINELPREPARE(Prepare)) { sentinelClientSend(&Base, sizeof(fileutils_dpwd)); }
	int RC;
	char *Ptr;
};

struct fileutils_dcd {
	static __forceinline__ char *Prepare(fileutils_dcd *t, char *data, char *dataEnd, intptr_t offset) {
		int strLength = (t->Str ? (int)strlen(t->Str) + 1 : 0);
		char *str = (char *)(data += ROUND8_(sizeof(*t)));
		char *end = (char *)(data += strLength);
		if (end > dataEnd) return nullptr;
		memcpy(str, t->Str, strLength);
		t->Str = str + offset;
		return end;
	}
	sentinelMessage Base;
	char *Str;
	fileutils_dcd(char *str) : Base(true, FILEUTILS_DCD, 1024, SENTINELPREPARE(Prepare)), Str(str) { sentinelClientSend(&Base, sizeof(fileutils_dcd)); }
	int RC;
};

#endif  /* _SENTINEL_FILEUTILSMSG_H */