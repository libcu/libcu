#define _CRT_SECURE_NO_WARNINGS
#include <stdlib.h>
#include <stdio.h>
#include "sentinel-fileutilsmsg.h"
#include <Windows.h>

__forceinline int dpwd_(char *ptr) { fileutils_dpwd msg; strcpy(ptr, msg.Ptr); return msg.RC; }

int main(int argc, char **argv)
{
	atexit(sentinelClientShutdown);
	sentinelClientInitialize();
	char pwd[MAX_PATH];
	if (dpwd_(pwd)) {
		fprintf(stderr, "pwd: cannot get current directory\n");
		exit(1);
	}
	printf("%s\n", pwd);
	exit(0);
}
