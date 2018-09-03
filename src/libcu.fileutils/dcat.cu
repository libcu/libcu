#define _CRT_SECURE_NO_WARNINGS
#include <stdlib.h>
#include <stdio.h>
#include "sentinel-fileutilsmsg.h"

#define CAT_BUF_SIZE 4096
void dumpfile(FILE *f)
{
	size_t nred;
	static char readbuf[CAT_BUF_SIZE];
	while ((nred = fread(readbuf, 1, CAT_BUF_SIZE, f)) > 0)
		fwrite(readbuf, nred, 1, stdout);
}

__forceinline__ int dcat_(char *str) { fileutils_dcat msg(str); return msg.RC; }

int main(int argc, char **argv)
{
	atexit(sentinelClientShutdown);
	sentinelClientInitialize();
	if (argc <= 1)
		dumpfile(stdin);
	else for (int i = 1; i < argc; i++) {
		int r = dcat_(argv[i]);
		if (!r)
			fprintf(stderr, "%s: %s: %s\n", argv[0], argv[i], strerror(r));
	}
	exit(0);
}
