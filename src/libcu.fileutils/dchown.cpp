#define _CRT_SECURE_NO_WARNINGS
#include <stdlib.h>
#include <stdio.h>
#include "sentinel-fileutilsmsg.h"
#include <sentinel-client.cpp>
#include <ext/pipeline.cpp>
#include <pwdcu.h>

#define	isdecimal(ch) ((ch) >= '0' && (ch) <= '9')
__forceinline__ struct passwd *dchgrp_getpwnam_(pipelineRedir *redir, char *str) { fileutils_getpwnam msg(redir[0], str); pipelineRead(redir[1]); return msg.rc; }
__forceinline__ int dchown_(pipelineRedir *redir, char *str, int uid) { fileutils_dchown msg(redir[0], str, uid); pipelineRead(redir[1]); return msg.rc; }

int main(int argc, char	**argv) {
	atexit(sentinelClientShutdown);
	sentinelClientInitialize();
	pipelineRedir redir[2]; sentinelClientRedir(redir);
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
		struct passwd *pwd = dchgrp_getpwnam_(redir, cp);
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
		if (dchown_(redir, *argv, uid))
			perror(*argv);
	}
	exit(0);
}
