#define _CRT_SECURE_NO_WARNINGS
#include <stdlib.h>
#include <stdio.h>
#include "sentinel-fileutilsmsg.h"

#define	isdecimal(ch) ((ch) >= '0' && (ch) <= '9')
struct group { short gr_gid; };
__device__ struct group *getgrnam(char *name) { return nullptr; }

__device__ __managed__ struct group *m_getgrnam_rc;
__global__ void g_getgrnam(char *name)
{
	m_getgrnam_rc = getgrnam(name);
}
struct group *getgrnam_(char *str)
{
	size_t strLength = strlen(str) + 1;
	char *d_str;
	cudaMalloc(&d_str, strLength);
	cudaMemcpy(d_str, str, strLength, cudaMemcpyHostToDevice);
	g_getgrnam<<<1,1>>>(d_str);
	cudaFree(d_str);
	return m_getgrnam_rc;
}

__forceinline int dchgrp_(char *str, int gid) { fileutils_dchgrp msg(str, gid); return msg.RC; }

int main(int argc, char **argv)
{
	atexit(sentinelClientShutdown);
	sentinelClientInitialize();
	char *cp = argv[1];
	int gid;
	struct group *grp;
	if (isdecimal(*cp)) {
		gid = 0;
		while (isdecimal(*cp))
			gid = gid * 10 + (*cp++ - '0');
		if (*cp) {
			fprintf(stderr, "Bad gid value\n");
			exit(1);
		}
	}
	else {
		grp = getgrnam_(cp);
		if (!grp) {
			fprintf(stderr, "Unknown group name\n");
			exit(1);
		}
		gid = grp->gr_gid;
	}
	//
	argc--;
	argv++;
	while (argc-- > 1) {
		argv++;
		if (dchgrp_(*argv, gid))
			perror(*argv);
	}
	exit(0);
}
