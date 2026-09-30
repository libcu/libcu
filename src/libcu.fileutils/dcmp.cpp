#define _CRT_SECURE_NO_WARNINGS
#include <stdlib.h>
#include <stdio.h>
#include "sentinel-fileutilsmsg.h"
#include <sentinel-client.cpp>
#include <ext/pipeline.cpp>

__forceinline__ int dcmp_(pipelineRedir *redir, char *str, char *str2) { fileutils_dcmp msg(redir[0], str, str2); pipelineRead(redir[1]); return msg.rc; }

int main(int argc, char	**argv) {
	atexit(sentinelClientShutdown);
	sentinelClientInitialize();
	pipelineRedir redir[2]; sentinelClientRedir(redir);
	int r = dcmp_(redir, argv[1], argv[2]);
	exit(r);
}
