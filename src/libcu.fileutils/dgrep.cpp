#define _CRT_SECURE_NO_WARNINGS
#include <stdlib.h>
#include <stdio.h>
#include "sentinel-fileutilsmsg.h"
#include <sentinel-client.cpp>
#include <ext/pipeline.cpp>

__forceinline__ int dgrep_(pipelineRedir *redir, char *str, char *str2, bool ignoreCase, bool tellName, bool tellLine) { fileutils_dgrep msg(redir[0], str, str2, ignoreCase, tellName, tellLine); pipelineRead(redir[1]); return msg.rc; }

int main(int argc, char **argv) {
	atexit(sentinelClientShutdown);
	sentinelClientInitialize();
	pipelineRedir redir[2]; sentinelClientRedir(redir);
	argc--;
	argv++;
	bool ignoreCase = false;
	bool tellLine = false;
	if (argc > 0 && **argv == '-') {
		argc--;
		char *cp = *argv++;
		while (*++cp) switch (*cp) {
		case 'i': ignoreCase = true; break;
		case 'n': tellLine = true; break;
		default: fprintf(stderr, "Unknown option\n"); exit(1);
		}
	}
	char *word = *argv++;
	argc--;
	bool tellName = argc > 1;
	//
	while (argc-- > 0) {
		char *name = *argv++;
		if (!dgrep_(redir, name, word, ignoreCase, tellName, tellLine))
			continue;
	}
	exit(0);
}

