#define _CRT_SECURE_NO_WARNINGS
#include <stdlib.h>
#include <stdio.h>
#include "sentinel-fileutilsmsg.h"
#include <sentinel-client.cpp>
#include <ext/pipeline.cpp>

unsigned short _newMode = 0666; // & ~umask(0);
__forceinline__ int drmdir_(pipelineRedir *redir, char *str) { fileutils_drmdir msg(redir[0], str); pipelineRead(redir[1]); return msg.rc; }

int removeDir(pipelineRedir *redir, char *name, int f) {
	int r, r2 = 2;
	char *line;
	while (!(r = drmdir_(redir, name)) && (line = strchr(name, '/')) && f) {
		while (line > name && *line == '/')
			--line;
		line[1] = 0;
		r2 = 0;
	}
	return r && r2;
}

int main(int argc, char **argv) {
	atexit(sentinelClientShutdown);
	sentinelClientInitialize();
	pipelineRedir redir[2]; sentinelClientRedir(redir);
	int parent = argv[1] && argv[1][0] == '-' && argv[1][1] == 'p' ? 1 : 0;

	int r = 0;
	for (int i = parent + 1; i < argc; i++) {
		if (argv[i][0] != '-') {
			while (argv[i][strlen(argv[i]) - 1] == '/')
				argv[i][strlen(argv[i]) - 1] = '\0';
			if (removeDir(redir, argv[i], parent)) {
				fprintf(stderr, "rmdir: cannot remove directory %s\n", argv[i]);
				r = 1;
			}
		}
		else {
			fprintf(stderr, "rmdir: usage error.\n");
			exit(1);
		}
	}
	exit(r);
}
