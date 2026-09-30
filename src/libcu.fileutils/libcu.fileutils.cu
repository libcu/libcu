#include <sentinel.h>
#include "fileutils.cuh"
#include "dcat.cuh"
#include "dchgrp.cuh"
#include "dchgrp_getgrnam.cuh"
#include "dchmod.cuh"
#include "dchown.cuh"
#include "dchown_getpwnam.cuh"
#include "dcmp.cuh"
#include "dcp.cuh"
#include "dcp_isadir.cuh"
#include "dgrep.cuh"
#include "dls.cuh"
#include "dmkdir.cuh"
#include "dmore.cuh"
#include "dmv.cuh"
#include "drm.cuh"
#include "drmdir.cuh"
#include "dpwd.cuh"
#include "dcd.cuh"

extern "C" bool sentinelFileUtilsExecutor(void *tag, sentinelMessage *data, int length, char *(**hostPrepare)(void*, char*, char*, intptr_t));
static sentinelExecutor _fileUtilsExecutor = { nullptr, "fileutils", sentinelFileUtilsExecutor, nullptr };
void sentinelRegisterFileUtils() {
	sentinelRegisterExecutor(&_fileUtilsExecutor, false, false);
}