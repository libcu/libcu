#define _CRT_SECURE_NO_WARNINGS
#include <stdlib.h>
#include <stdio.h>
#include "sentinel-fileutilsmsg.h"
#include <sys/statcu.h>
#include <unistdcu.h>
#define	PATHLEN 256	

// Return TRUE if a filename is a directory. Nonexistant files return FALSE.
__device__ __managed__ bool m_isadir_rc;
__global__ void g_isadir(char *name)
{
	struct stat statbuf;
	if (stat(name, &statbuf) < 0) {
		m_isadir_rc = false;
		return;
	}
	m_isadir_rc = S_ISDIR(statbuf.st_mode);
	return;
}

bool isadir_(char *str)
{
	size_t strLength = strlen(str) + 1;
	char *d_str;
	cudaMalloc(&d_str, strLength);
	cudaMemcpy(d_str, str, strLength, cudaMemcpyHostToDevice);
	g_isadir<<<1,1>>>(d_str);
	cudaFree(d_str);
	return m_isadir_rc;
}

__forceinline int dcp_(char *str, char *str2, bool setModes) { fileutils_dcp msg(str, str2, setModes); return msg.RC; }

// Build a path name from the specified directory name and file name. If the directory name is NULL, then the original filename is returned.
// The built path is in a static area, and is overwritten for each call.
char *buildName(char *dirName, char *fileName)
{
	if (!dirName || *dirName == '\0')
		return fileName;
	char *cp = strrchr(fileName, '/');
	if (cp)
		fileName = cp + 1;
	static char buf[PATHLEN];
	strcpy(buf, dirName);
	strcat(buf, "/");
	strcat(buf, fileName);
	return buf;
}

int main(int argc, char	**argv)
{
	atexit(sentinelClientShutdown);
	sentinelClientInitialize();
	char *lastArg = argv[argc - 1];
	bool dirflag = isadir_(lastArg);
	if (argc > 3 && !dirflag) {
		fprintf(stderr, "%s: not a directory\n", lastArg);
		exit(1);
	}
	while (argc-- > 2) {
		char *srcName = argv[1];
		char *destName = lastArg;
		if (dirflag)
			destName = buildName(destName, srcName);
		dcp_(*++argv, destName, false);
	}
	exit(0);
}
