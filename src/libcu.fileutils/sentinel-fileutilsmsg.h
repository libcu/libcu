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
	FILEUTILS_DCAT = 10,
	FILEUTILS_DCHGRP,
	FILEUTILS_GETGRNAM, // DCHGRP
	FILEUTILS_DCHMOD,
	FILEUTILS_DCHOWN,
	FILEUTILS_GETPWNAM, // DCHOWN
	FILEUTILS_DCMP,
	FILEUTILS_DCP,
	FILEUTILS_ISADIR, // DCP
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
	sentinelClientMessage base;
	char *str;
	fileutils_dcat(pipelineRedir redir, char *str) : base(redir, FILEUTILS_DCAT, SENTINELFLOW_WAIT, SENTINEL_CHUNK), str(str) { sentinelClientSend(&base.base, sizeof(fileutils_dcat), ptrsIn); }
	int rc;
	sentinelInPtr ptrsIn[2] = {
		{ &str, -1 },
		{ nullptr }
	};
};

struct fileutils_dchgrp {
	sentinelClientMessage base;
	char *str; int gid;
	fileutils_dchgrp(pipelineRedir redir, char *str, int gid) : base(redir, FILEUTILS_DCHGRP, SENTINELFLOW_WAIT, SENTINEL_CHUNK), str(str), gid(gid) { sentinelClientSend(&base.base, sizeof(fileutils_dchgrp), ptrsIn); }
	int rc;
	sentinelInPtr ptrsIn[2] = {
		{ &str, -1 },
		{ nullptr }
	};
};

struct fileutils_getgrnam {
	sentinelClientMessage base;
	char *str;
	fileutils_getgrnam(pipelineRedir redir, char *str) : base(redir, FILEUTILS_GETGRNAM, SENTINELFLOW_WAIT, SENTINEL_CHUNK), str(str) { sentinelClientSend(&base.base, sizeof(fileutils_getgrnam), ptrsIn); }
	struct group *rc;
	sentinelInPtr ptrsIn[2] = {
		{ &str, -1 },
		{ nullptr }
	};
};

struct fileutils_dchmod {
	sentinelClientMessage base;
	char *str; int mode;
	fileutils_dchmod(pipelineRedir redir, char *str, int mode) : base(redir, FILEUTILS_DCHMOD, SENTINELFLOW_WAIT, SENTINEL_CHUNK), str(str), mode(mode) { sentinelClientSend(&base.base, sizeof(fileutils_dchmod), ptrsIn); }
	int rc;
	sentinelInPtr ptrsIn[2] = {
		{ &str, -1 },
		{ nullptr }
	};
};

struct fileutils_dchown {
	sentinelClientMessage base;
	char *str; int uid;
	fileutils_dchown(pipelineRedir redir, char *str, int uid) : base(redir, FILEUTILS_DCHOWN, SENTINELFLOW_WAIT, SENTINEL_CHUNK), str(str), uid(uid) { sentinelClientSend(&base.base, sizeof(fileutils_dchown), ptrsIn); }
	int rc;
	sentinelInPtr ptrsIn[2] = {
		{ &str, -1 },
		{ nullptr }
	};
};

struct fileutils_getpwnam {
	sentinelClientMessage base;
	char *str;
	fileutils_getpwnam(pipelineRedir redir, char *str) : base(redir, FILEUTILS_GETPWNAM, SENTINELFLOW_WAIT, SENTINEL_CHUNK), str(str) { sentinelClientSend(&base.base, sizeof(fileutils_getpwnam), ptrsIn); }
	struct passwd *rc;
	sentinelInPtr ptrsIn[2] = {
		{ &str, -1 },
		{ nullptr }
	};
};

struct fileutils_dcmp {
	sentinelClientMessage base;
	char *str; char *str2;
	fileutils_dcmp(pipelineRedir redir, char *str, char *str2) : base(redir, FILEUTILS_DCMP, SENTINELFLOW_WAIT, SENTINEL_CHUNK), str(str), str2(str2) { sentinelClientSend(&base.base, sizeof(fileutils_dcmp), ptrsIn); }
	int rc;
	sentinelInPtr ptrsIn[3] = {
		{ &str, -1 },
		{ &str2, -1 },
		{ nullptr }
	};
};

struct fileutils_dcp {
	sentinelClientMessage base;
	char *str; char *str2; bool setModes;
	fileutils_dcp(pipelineRedir redir, char *str, char *str2, bool setModes) : base(redir, FILEUTILS_DCP, SENTINELFLOW_WAIT, SENTINEL_CHUNK), str(str), str2(str2), setModes(setModes) { sentinelClientSend(&base.base, sizeof(fileutils_dcp), ptrsIn); }
	int rc;
	sentinelInPtr ptrsIn[2] = {
		{ &str, -1 },
		{ nullptr }
	};
};

struct fileutils_isadir {
	sentinelClientMessage base;
	char *str;
	fileutils_isadir(pipelineRedir redir, char *str) : base(redir, FILEUTILS_ISADIR, SENTINELFLOW_WAIT, SENTINEL_CHUNK), str(str) { sentinelClientSend(&base.base, sizeof(fileutils_isadir), ptrsIn); }
	bool rc;
	sentinelInPtr ptrsIn[2] = {
		{ &str, -1 },
		{ nullptr }
	};
};

struct fileutils_dgrep {
	sentinelClientMessage base;
	char *str; char *str2; bool ignoreCase; bool tellName; bool tellLine;
	fileutils_dgrep(pipelineRedir redir, char *str, char *str2, bool ignoreCase, bool tellName, bool tellLine) : base(redir, FILEUTILS_DGREP, SENTINELFLOW_WAIT, SENTINEL_CHUNK), str(str), str2(str2), ignoreCase(ignoreCase), tellName(tellName), tellLine(tellLine) { sentinelClientSend(&base.base, sizeof(fileutils_dgrep), ptrsIn); }
	int rc;
	sentinelInPtr ptrsIn[3] = {
		{ &str, -1 },
		{ &str2, -1 },
		{ nullptr }
	};
};

struct fileutils_dls {
	sentinelClientMessage base;
	char *str; int flags; bool endSlash;
	fileutils_dls(pipelineRedir redir, char *str, int flags, bool endSlash) : base(redir, FILEUTILS_DLS, SENTINELFLOW_WAIT, SENTINEL_CHUNK), str(str), flags(flags), endSlash(endSlash) { sentinelClientSend(&base.base, sizeof(fileutils_dls), ptrsIn); }
	int rc;
	sentinelInPtr ptrsIn[2] = {
		{ &str, -1 },
		{ nullptr }
	};
};

struct fileutils_dmkdir {
	sentinelClientMessage base;
	char *str; unsigned short mode;
	fileutils_dmkdir(pipelineRedir redir, char *str, unsigned short mode) : base(redir, FILEUTILS_DMKDIR, SENTINELFLOW_WAIT, SENTINEL_CHUNK), str(str), mode(mode) { sentinelClientSend(&base.base, sizeof(fileutils_dmkdir), ptrsIn); }
	int rc;
	sentinelInPtr ptrsIn[2] = {
		{ &str, -1 },
		{ nullptr }
	};
};

struct fileutils_dmore {
	sentinelClientMessage base;
	char *str; int fd;
	fileutils_dmore(pipelineRedir redir, char *str, int fd) : base(redir, FILEUTILS_DMORE, SENTINELFLOW_WAIT, SENTINEL_CHUNK), str(str), fd(fd) { sentinelClientSend(&base.base, sizeof(fileutils_dmore), ptrsIn); }
	int rc;
	sentinelInPtr ptrsIn[2] = {
		{ &str, -1 },
		{ nullptr }
	};
};

struct fileutils_dmv {
	sentinelClientMessage base;
	char *str; char *str2;
	fileutils_dmv(pipelineRedir redir, char *str, char *str2) : base(redir, FILEUTILS_DMV, SENTINELFLOW_WAIT, SENTINEL_CHUNK), str(str), str2(str2) { sentinelClientSend(&base.base, sizeof(fileutils_dmv), ptrsIn); }
	int rc;
	sentinelInPtr ptrsIn[3] = {
		{ &str, -1 },
		{ &str2, -1 },
		{ nullptr }
	};
};

struct fileutils_drm {
	sentinelClientMessage base;
	char *str;
	fileutils_drm(pipelineRedir redir, char *str) : base(redir, FILEUTILS_DRM, SENTINELFLOW_WAIT, SENTINEL_CHUNK), str(str) { sentinelClientSend(&base.base, sizeof(fileutils_drm), ptrsIn); }
	int rc;
	sentinelInPtr ptrsIn[2] = {
		{ &str, -1 },
		{ nullptr }
	};
};

struct fileutils_drmdir {
	sentinelClientMessage base;
	char *str;
	fileutils_drmdir(pipelineRedir redir, char *str) : base(redir, FILEUTILS_DRMDIR, SENTINELFLOW_WAIT, SENTINEL_CHUNK), str(str) { sentinelClientSend(&base.base, sizeof(fileutils_drmdir), ptrsIn); }
	int rc;
	sentinelInPtr ptrsIn[2] = {
		{ &str, -1 },
		{ nullptr }
	};
};

struct fileutils_dpwd {
	static __forceinline__ __device__ char *prepare(fileutils_dpwd *t, char *data, char *dataEnd, intptr_t offset) {
		t->ptr = data;
		char *end = data += SENTINEL_CHUNK;
		if (end > dataEnd) return nullptr;
		return end;
	}
	sentinelClientMessage base;
	fileutils_dpwd(pipelineRedir redir) : base(redir, FILEUTILS_DPWD, SENTINELFLOW_WAIT, SENTINEL_CHUNK, SENTINELPREPARE(prepare)) { sentinelClientSend(&base.base, sizeof(fileutils_dpwd)); }
	int rc;
	char *ptr;
};

struct fileutils_dcd {
	sentinelClientMessage base;
	char *str;
	fileutils_dcd(pipelineRedir redir, char *str) : base(redir, FILEUTILS_DCD, SENTINELFLOW_WAIT, SENTINEL_CHUNK), str(str) { sentinelClientSend(&base.base, sizeof(fileutils_dcd), ptrsIn); }
	int rc;
	sentinelInPtr ptrsIn[2] = {
		{ &str, -1 },
		{ nullptr }
	};
};

#endif  /* _SENTINEL_FILEUTILSMSG_H */