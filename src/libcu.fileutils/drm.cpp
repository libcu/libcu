#define _CRT_SECURE_NO_WARNINGS
#include <stdlib.h>
#include <stdio.h>
#include "sentinel-fileutilsmsg.h"
#include <sentinel-client.cpp>
#include <ext/pipeline.cpp>

__forceinline__ int drm_(pipelineRedir *redir, char *str) { fileutils_drm msg(redir[0], str); pipelineRead(redir[1]); return msg.rc; }

int main(int argc, char **argv) {
	atexit(sentinelClientShutdown);
	sentinelClientInitialize();
	pipelineRedir redir[2]; sentinelClientRedir(redir);
	//int recurse = ((argv[1] && argv[1][0] == '-' && argv[1][1] == 'r') || (argv[2] && argv[2][0] == '-' && argv[2][1] == 'r') ? 1 : 0);
	//int interact = ((argv[1] && argv[1][0] == '-' && argv[1][1] == 'i') || (argv[2] && argv[2][0] == '-' && argv[2][1] == 'i') ? 1 : 0);
	for (int i = /*recurse+interact+*/1; i < argc; i++)
		if (argv[i][0] != '-')
			if (!drm_(redir, argv[i]))
				fprintf(stderr, "rm: could not remove %s\n", argv[i]);
}
