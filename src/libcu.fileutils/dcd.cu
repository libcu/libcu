#define _CRT_SECURE_NO_WARNINGS
#include <stdlib.h>
#include <stdio.h>
#include "sentinel-fileutilsmsg.h"
#include <Windows.h>

__forceinline int dpwd_(char *ptr) { fileutils_dpwd msg; strcpy(ptr, msg.Ptr); return msg.RC; }
__forceinline int dcd_(char *str) { fileutils_dcd msg(str); return msg.RC; }

int main(int argc, char **argv)
{
	atexit(sentinelClientShutdown);
	sentinelClientInitialize();
	if (argc <= 1 || argc > 2) {
		char pwd[MAX_PATH];
		if (!dpwd_(pwd)) {
			printf("%s\n", pwd);
			exit(1);
		}
	}
	int r = dcd_(argv[1]);
	if (!r) {
		fprintf(stderr, "%s: %s: %s\n", argv[0], argv[1], strerror(r));
		exit(0);
	}
	exit(0);
}
