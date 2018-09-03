#define _CRT_SECURE_NO_WARNINGS
#include "sentinel-fileutilsmsg.h"

int dcat(char *str);
int dchgrp(char *str, int gid);
int dchmod(char *str, int mode);
int dchown(char *str, int uid);
int dcmp(char *str, char *str2);
int dcp(char *str, char *str2, bool setModes);
int dgrep(char *str, char *str2, bool ignoreCase, bool tellName, bool tellLine);
int dls(char *str, int flags, bool endSlash);
int dmkdir(char *str, unsigned short mode);
int dmore(char *str, int fd);
int dmv(char *str, char *str2);
int drm(char *str);
int drmdir(char *str);
int dpwd(char *str);
int dcd(char *str);

extern "C" bool sentinelFileUtilsExecutor(void *tag, sentinelMessage *data, int length, char *(**hostPrepare)(void*, char*, char*, intptr_t)) {
	if (data->OP < FILEUTILS_DCAT || data->OP > FILEUTILS_DCD) return false;
	switch (data->OP) {
	case FILEUTILS_DCAT: { fileutils_dcat *msg = (fileutils_dcat *)data; msg->RC = dcat(msg->Str); return true; }
	case FILEUTILS_DCHGRP: { fileutils_dchgrp *msg = (fileutils_dchgrp *)data; msg->RC = dchgrp(msg->Str, msg->Gid); return true; }
	case FILEUTILS_DCHMOD: { fileutils_dchmod *msg = (fileutils_dchmod *)data; msg->RC = dchmod(msg->Str, msg->Mode); return true; }
	case FILEUTILS_DCHOWN: { fileutils_dchown *msg = (fileutils_dchown *)data; msg->RC = dchown(msg->Str, msg->Uid); return true; }
	case FILEUTILS_DCMP: { fileutils_dcmp *msg = (fileutils_dcmp *)data; msg->RC = dcmp(msg->Str, msg->Str2); return true; }
	case FILEUTILS_DCP: { fileutils_dcp *msg = (fileutils_dcp *)data; msg->RC = dcp(msg->Str, msg->Str2, msg->SetModes); return true; }
	case FILEUTILS_DGREP: { fileutils_dgrep *msg = (fileutils_dgrep *)data; msg->RC = dgrep(msg->Str, msg->Str2, msg->IgnoreCase, msg->TellName, msg->TellLine); return true; }
	case FILEUTILS_DLS: { fileutils_dls *msg = (fileutils_dls *)data; msg->RC = dls(msg->Str, msg->Flags, msg->EndSlash); return true; }
	case FILEUTILS_DMKDIR: { fileutils_dmkdir *msg = (fileutils_dmkdir *)data; msg->RC = dmkdir(msg->Str, msg->Mode); return true; }
	case FILEUTILS_DMORE: { fileutils_dmore *msg = (fileutils_dmore *)data; msg->RC = dmore(msg->Str, msg->Fd); return true; }
	case FILEUTILS_DMV: { fileutils_dmv *msg = (fileutils_dmv *)data; msg->RC = dmv(msg->Str, msg->Str2); return true; }
	case FILEUTILS_DRM: { fileutils_drm *msg = (fileutils_drm *)data; msg->RC = drm(msg->Str); return true; }
	case FILEUTILS_DRMDIR: { fileutils_drmdir *msg = (fileutils_drmdir *)data; msg->RC = drmdir(msg->Str); return true; }
	case FILEUTILS_DPWD: { fileutils_dpwd *msg = (fileutils_dpwd *)data; msg->RC = dpwd(msg->Ptr); return true; }
	case FILEUTILS_DCD: { fileutils_dcd *msg = (fileutils_dcd *)data; msg->RC = dcd(msg->Str); return true; }
	}
	return false;
}