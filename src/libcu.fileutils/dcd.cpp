#define _CRT_SECURE_NO_WARNINGS
#include <stdlib.h>
#include <stdio.h>
#include "sentinel-fileutilsmsg.h"
#include <sentinel-client.cpp>
#include <ext/pipeline.cpp>

__forceinline__ int dpwd_(pipelineRedir *redir, char *ptr) { fileutils_dpwd msg(redir[0]); strcpy(ptr, msg.ptr); pipelineRead(redir[1]); return msg.rc; }
__forceinline__ int dcd_(pipelineRedir *redir, char *str) { fileutils_dcd msg(redir[0], str); pipelineRead(redir[1]); return msg.rc; }

int main(int argc, const char **argv) {
	atexit(sentinelClientShutdown);
	sentinelClientInitialize();
	pipelineRedir redir[2]; sentinelClientRedir(redir);
	if (argc <= 1 || argc > 2) {
		char pwd[FILENAME_MAX];
		if (!dpwd_(redir, pwd)) {
			printf("%s\n", pwd);
			exit(1);
		}
	}
	int r = dcd_(redir, (char *)argv[1]);
	if (!r) {
		fprintf(stderr, "%s: %s: %s\n", argv[0], argv[1], strerror(r));
		exit(0);
	}
	exit(0);
}
