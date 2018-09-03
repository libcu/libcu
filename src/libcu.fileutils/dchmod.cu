#define _CRT_SECURE_NO_WARNINGS
#include <stdlib.h>
#include <stdio.h>
#include "sentinel-fileutilsmsg.h"

#define	isoctal(ch)	((ch) >= '0' && (ch) <= '7')

__forceinline int dchmod_(char *str, int mode) { fileutils_dchmod msg(str, mode); return msg.RC; }

int main(int argc, char **argv)
{
	atexit(sentinelClientShutdown);
	sentinelClientInitialize();
	int	mode = 0;
	char *cp = argv[1];
	while (isoctal(*cp))
		mode = mode * 8 + (*cp++ - '0');
	if (*cp) {
		fprintf(stderr, "Mode must be octal\n");
		exit(1);
	}
	//
	argc--;
	argv++;
	while (argc-- > 1) {
		if (dchmod_(argv[1], mode))
			perror(argv[1]);
		argv++;
	}
	exit(0);
}