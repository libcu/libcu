#define _CRT_SECURE_NO_WARNINGS
#include <stdlib.h>
#include <stdio.h>
#include "sentinel-fileutilsmsg.h"

__forceinline int dcmp_(char *str, char *str2) { fileutils_dcmp msg(str, str2); return msg.RC; }

int main(int argc, char	**argv)
{
	atexit(sentinelClientShutdown);
	sentinelClientInitialize();
	int r = dcmp_(argv[1], argv[2]);
	exit(r);
}
