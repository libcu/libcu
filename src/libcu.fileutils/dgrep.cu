#define _CRT_SECURE_NO_WARNINGS
#include <stdlib.h>
#include <stdio.h>
#include "sentinel-fileutilsmsg.h"

__forceinline int dgrep_(char *str, char *str2, bool ignoreCase, bool tellName, bool tellLine) { fileutils_dgrep msg(str, str2, ignoreCase, tellName, tellLine); return msg.RC; }

int main(int argc, char **argv)
{
	atexit(sentinelClientShutdown);
	sentinelClientInitialize();
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
	bool tellName = (argc > 1);
	//
	while (argc-- > 0) {
		char *name = *argv++;
		if (!dgrep_(name, word, ignoreCase, tellName, tellLine))
			continue;
	}
	exit(0);
}

