#define _CRT_SECURE_NO_WARNINGS
#include <stdlib.h>
#include <stdio.h>
#include "sentinel-fileutilsmsg.h"

// Flags for the LS command.
#define	LSF_LONG	0x01
#define	LSF_DIR		0x02
#define	LSF_INODE	0x04
#define	LSF_MULT	0x08
#define LSF_ALL		0x10		// List files starting with `.'
#define LSF_CLASS	0x20		// Classify files (append symbol)

__forceinline int dls_(char *str, int flags, bool endSlash) { fileutils_dls msg(str, flags, endSlash); return msg.RC; }

int main(int argc, char **argv)
{
	atexit(sentinelClientShutdown);
	sentinelClientInitialize();

	// setup
	dls_(nullptr, 1, false);

	// flags
	int flags = 0;
	if (argc > 1 && argv[1][0] == '-') {
		argc--;
		char *cp = *(++argv) + 1;
		while (*cp) switch (*cp++) {
		case 'g':
		case 'l': flags |= LSF_LONG; break;
		case 'd': flags |= LSF_DIR; break;
		case 'i': flags |= LSF_INODE; break;
		case 'a': flags |= LSF_ALL; break;
		case 'F': flags |= LSF_CLASS; break;
		case 'A': break;
		default: fprintf(stderr, "Unknown option -%c\n", cp[-1]); exit(1);
		}
	}
	static char *defaultArgs[2] = { "-ls", "." };
	if (argc <= 1) { argc = 2; argv = defaultArgs; }
	if (argc > 2)
		flags |= LSF_MULT;

	char *name;
	while (argc-- > 1) {
		if (!(name = (char *)malloc(strlen(*(++argv)) + 2))) {
			fprintf(stderr, "No memory for filenames\n");
			exit(1);
		}
		strcpy(name, *argv);
		bool endSlash = (*name && (name[strlen(name) - 1] == '/'));

		if (dls_(name, flags, endSlash))
			continue;
	}

	/*fflush(stdout);*/
	exit(0);
}
