#define _CRT_SECURE_NO_WARNINGS
#include <stdlib.h>
#include <stdio.h>
#include "sentinel-fileutilsmsg.h"
#include <sentinel-client.cpp>
#include <ext/pipeline.cpp>

__forceinline__ int dpwd_(pipelineRedir *redir, char *ptr) { fileutils_dpwd msg(redir[0]); strcpy(ptr, msg.ptr); pipelineRead(redir[1]); return msg.rc; }

int main(int argc, char **argv) {
	atexit(sentinelClientShutdown);
	sentinelClientInitialize();
	pipelineRedir redir[2]; sentinelClientRedir(redir);
	char pwd[FILENAME_MAX];
	if (dpwd_(redir, pwd)) {
		fprintf(stderr, "pwd: cannot get current directory\n");
		exit(1);
	}
	printf("%s\n", pwd);
	exit(0);
}
