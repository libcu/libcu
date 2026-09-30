#define _CRT_SECURE_NO_WARNINGS
#include <stdlib.h>
#include <stdio.h>
#include "sentinel-fileutilsmsg.h"
#include <sentinel-client.cpp>
#include <ext/pipeline.cpp>

__forceinline__ bool dmv_isadir_(pipelineRedir *redir, char *str) { fileutils_isadir msg(redir[0], str); pipelineRead(redir[1]); return msg.rc; }
__forceinline__ int dmv_(pipelineRedir *redir, char *str, char *str2) { fileutils_dmv msg(redir[0], str, str2); pipelineRead(redir[1]); return msg.rc; }

// Build a path name from the specified directory name and file name. If the directory name is NULL, then the original filename is returned.
// The built path is in a static area, and is overwritten for each call.
char *buildName(char *dirName, char *fileName) {
	if (!dirName || *dirName == '\0')
		return fileName;
	char *cp = strrchr(fileName, '/');
	if (cp)
		fileName = cp + 1;
	static char buf[FILENAME_MAX];
	strcpy(buf, dirName);
	strcat(buf, "/");
	strcat(buf, fileName);
	return buf;
}

int main(int argc, char **argv) {
	atexit(sentinelClientShutdown);
	sentinelClientInitialize();
	pipelineRedir redir[2]; sentinelClientRedir(redir);
	char *lastArg = argv[argc - 1];
	bool dirflag = dmv_isadir_(redir, lastArg);
	if (argc > 3 && !dirflag) {
		fprintf(stderr, "%s: not a directory\n", lastArg);
		exit(1);
	}
	while (argc-- > 2) {
		char *srcName = *(++argv);
		char *destName = lastArg;
		if (dirflag)
			destName = buildName(destName, srcName);
		dmv_(redir, srcName, destName);
	}
	exit(0);
}
