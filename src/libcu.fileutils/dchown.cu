#define _CRT_SECURE_NO_WARNINGS
#include <stdlib.h>
#include <stdio.h>
#include "sentinel-fileutilsmsg.h"

#define	isdecimal(ch) ((ch) >= '0' && (ch) <= '9')
struct passwd { short pw_uid; };
__device__ struct passwd *getpwnam(char *name) { return nullptr; }

__device__ __managed__ struct passwd *m_getpwnam_rc;
__global__ void g_getpwnam(char *name)
{
	m_getpwnam_rc = getpwnam(name);
}
struct passwd *getpwnam_(char *str)
{
	int strLength = strlen(str) + 1;
	char *d_str;
	cudaMalloc(&d_str, strLength);
	cudaMemcpy(d_str, str, strLength, cudaMemcpyHostToDevice);
	g_getpwnam<<<1,1>>>(d_str);
	cudaFree(d_str);
	return m_getpwnam_rc;
}

__forceinline int dchown_(char *str, int uid) { fileutils_dchown msg(str, uid); return msg.RC; }

int main(int argc, char	**argv)
{
	atexit(sentinelClientShutdown);
	sentinelClientInitialize();
	char *cp = argv[1];
	int uid;
	if (isdecimal(*cp)) {
		uid = 0;
		while (isdecimal(*cp))
			uid = uid * 10 + (*cp++ - '0');
		if (*cp) {
			fprintf(stderr, "Bad uid value\n");
			exit(1);
		}
	}
	else {
		struct passwd *pwd = getpwnam_(cp);
		if (!pwd) {
			fprintf(stderr, "Unknown user name\n");
			exit(1);
		}
		uid = pwd->pw_uid;
	}
	//
	argc--;
	argv++;
	while (argc-- > 1) {
		argv++;
		if (dchown_(*argv, uid))
			perror(*argv);
	}
	exit(0);
}
