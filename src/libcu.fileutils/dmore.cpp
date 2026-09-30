#define _CRT_SECURE_NO_WARNINGS
#include <stdlib.h>
#include <stdio.h>
#include "sentinel-fileutilsmsg.h"
#include <sentinel-client.cpp>
#include <ext/pipeline.cpp>
#include <unistdcu.h>
#ifndef _MSC_VER
#define _read read
#endif

__forceinline__ int dmore_(pipelineRedir *redir, char *str, int fd) { fileutils_dmore msg(redir[0], str, fd); pipelineRead(redir[1]); return msg.rc; }

int main(int argc, char **argv) {
	atexit(sentinelClientShutdown);
	sentinelClientInitialize();
	pipelineRedir redir[2]; sentinelClientRedir(redir);
	while (argc-- > 1) {
		char *name = *(++argv);
		int fd = -1;
		while (true) {
			fd = dmore_(redir, name, fd);
			if (fd == -1)
				break;
			static char buf[80];
			if (_read(0, buf, sizeof(buf)) < 0) {
				if (fd > -1)
					fd = dmore_(redir, nullptr, fd); // close(fd);
				exit(0);
			}
			unsigned char ch = buf[0];
			if (ch == ':') ch = buf[1];
			switch (ch) {
			case 'N':
			case 'n':
				fd = dmore_(redir, nullptr, fd); // close(fd);
				break;
			case 'Q':
			case 'q':
				fd = dmore_(redir, nullptr, fd); // close(fd);
				exit(0);
			}
		}
	}
	exit(0);
}
