#define _CRT_SECURE_NO_WARNINGS
#include <stdlib.h>
#include <stdio.h>
#include "sentinel-fileutilsmsg.h"
#include <sentinel-client.cpp>
#include <ext/pipeline.cpp>

unsigned short _newMode = 0666; // & ~umask(0);
__forceinline__ int dmkdir_(pipelineRedir *redir, char *name, unsigned short mode) { fileutils_dmkdir msg(redir[0], name, mode); pipelineRead(redir[1]); return msg.rc; }

int makeDir(pipelineRedir *redir, char *name, int f) {
	char iname[256];
	strcpy(iname, name);
	char *line;
	if ((line = strchr(iname, '/')) && f) {
		while (line > iname && *line == '/')
			--line;
		line[1] = 0;
		makeDir(redir, iname, 1);
	}
	return dmkdir_(redir, name, _newMode) && !f ? 1 : 0;
}

int main(int argc, char **argv) {
	atexit(sentinelClientShutdown);
	sentinelClientInitialize();
	pipelineRedir redir[2]; sentinelClientRedir(redir);
	int parent = argv[1] && argv[1][0] == '-' && argv[1][1] == 'p' ? 1 : 0;

	int r = 0;
	for (int i = parent + 1; i < argc; i++) {
		if (argv[i][0] != '-') {
			if (argv[i][strlen(argv[i]) - 1] == '/')
				argv[i][strlen(argv[i]) - 1] = '\0';
			if (makeDir(redir, argv[i], parent)) {
				fprintf(stderr, "mkdir: cannot create directory %s\n", argv[i]);
				r = 1;
			}

		}
		else {
			fprintf(stderr, "mkdir: usage error.\n");
			exit(1);
		}
	}
	exit(r);
}
