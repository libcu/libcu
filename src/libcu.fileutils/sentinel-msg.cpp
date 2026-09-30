#define _CRT_SECURE_NO_WARNINGS
#include "sentinel-fileutilsmsg.h"

int dcat(pipelineRedir redir, char *str);
int dchgrp(pipelineRedir redir, char *str, int gid);
struct group *dchgrp_getgrnam(pipelineRedir redir, char *str);
int dchmod(pipelineRedir redir, char *str, int mode);
int dchown(pipelineRedir redir, char *str, int uid);
struct passwd *dchown_getpwnam_(pipelineRedir redir, char *str);
int dcmp(pipelineRedir redir, char *str, char *str2);
int dcp(pipelineRedir redir, char *str, char *str2, bool setModes);
bool dcp_isadir_(pipelineRedir redir, char *str);
int dgrep(pipelineRedir redir, char *str, char *str2, bool ignoreCase, bool tellName, bool tellLine);
int dls(pipelineRedir redir, char *str, int flags, bool endSlash);
int dmkdir(pipelineRedir redir, char *str, unsigned short mode);
int dmore(pipelineRedir redir, char *str, int fd);
int dmv(pipelineRedir redir, char *str, char *str2);
int drm(pipelineRedir redir, char *str);
int drmdir(pipelineRedir redir, char *str);
int dpwd(pipelineRedir redir, char *str);
int dcd(pipelineRedir redir, char *str);

extern "C" bool sentinelFileUtilsExecutor(void *tag, sentinelMessage *data, int length, char *(**hostPrepare)(void*, char*, char*, intptr_t)) {
	if (data->op < FILEUTILS_DCAT || data->op > FILEUTILS_DCD) return false;
	switch (data->op) {
	case FILEUTILS_DCAT: { fileutils_dcat *msg = (fileutils_dcat *)data; msg->rc = dcat(msg->base.redir, msg->str); return true; }
	case FILEUTILS_DCHGRP: { fileutils_dchgrp *msg = (fileutils_dchgrp *)data; msg->rc = dchgrp(msg->base.redir, msg->str, msg->gid); return true; }
#ifndef LIBCU_LEAN_AND_MEAN
	case FILEUTILS_GETGRNAM: { fileutils_getgrnam *msg = (fileutils_getgrnam *)data; msg->rc = dchgrp_getgrnam(msg->base.redir, msg->str); return true; }
#endif
	case FILEUTILS_DCHMOD: { fileutils_dchmod *msg = (fileutils_dchmod *)data; msg->rc = dchmod(msg->base.redir, msg->str, msg->mode); return true; }
	case FILEUTILS_DCHOWN: { fileutils_dchown *msg = (fileutils_dchown *)data; msg->rc = dchown(msg->base.redir, msg->str, msg->uid); return true; }
#ifndef LIBCU_LEAN_AND_MEAN
	case FILEUTILS_GETPWNAM: { fileutils_getpwnam *msg = (fileutils_getpwnam *)data; msg->rc = dchown_getpwnam_(msg->base.redir, msg->str); return true; }
#endif
	case FILEUTILS_DCMP: { fileutils_dcmp *msg = (fileutils_dcmp *)data; msg->rc = dcmp(msg->base.redir, msg->str, msg->str2); return true; }
	case FILEUTILS_DCP: { fileutils_dcp *msg = (fileutils_dcp *)data; msg->rc = dcp(msg->base.redir, msg->str, msg->str2, msg->setModes); return true; }
	case FILEUTILS_ISADIR: { fileutils_isadir *msg = (fileutils_isadir *)data; msg->rc = dcp_isadir_(msg->base.redir, msg->str); return true; }
	case FILEUTILS_DGREP: { fileutils_dgrep *msg = (fileutils_dgrep *)data; msg->rc = dgrep(msg->base.redir, msg->str, msg->str2, msg->ignoreCase, msg->tellName, msg->tellLine); return true; }
	case FILEUTILS_DLS: { fileutils_dls *msg = (fileutils_dls *)data; msg->rc = dls(msg->base.redir, msg->str, msg->flags, msg->endSlash); return true; }
	case FILEUTILS_DMKDIR: { fileutils_dmkdir *msg = (fileutils_dmkdir *)data; msg->rc = dmkdir(msg->base.redir, msg->str, msg->mode); return true; }
	case FILEUTILS_DMORE: { fileutils_dmore *msg = (fileutils_dmore *)data; msg->rc = dmore(msg->base.redir, msg->str, msg->fd); return true; }
	case FILEUTILS_DMV: { fileutils_dmv *msg = (fileutils_dmv *)data; msg->rc = dmv(msg->base.redir, msg->str, msg->str2); return true; }
	case FILEUTILS_DRM: { fileutils_drm *msg = (fileutils_drm *)data; msg->rc = drm(msg->base.redir, msg->str); return true; }
	case FILEUTILS_DRMDIR: { fileutils_drmdir *msg = (fileutils_drmdir *)data; msg->rc = drmdir(msg->base.redir, msg->str); return true; }
	case FILEUTILS_DPWD: { fileutils_dpwd *msg = (fileutils_dpwd *)data; msg->rc = dpwd(msg->base.redir, msg->ptr); return true; }
	case FILEUTILS_DCD: { fileutils_dcd *msg = (fileutils_dcd *)data; msg->rc = dcd(msg->base.redir, msg->str); return true; }
	}
	return false;
}