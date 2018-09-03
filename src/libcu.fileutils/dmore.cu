#define _CRT_SECURE_NO_WARNINGS
#include <stdlib.h>
#include <stdio.h>
#include "sentinel-fileutilsmsg.h"
#include <unistdcu.h>

__forceinline int dmore_(char *str, int fd) { fileutils_dmore msg(str, fd); return msg.RC; }

int main(int argc, char **argv)
{
	while (argc-- > 1) {
		char *name = *(++argv);
		int fd = -1;
		while (true) {
			fd = dmore_(name, fd);
			if (fd == -1)
				break;
			static char buf[80];
			if (read(0, buf, sizeof(buf)) < 0) {
				if (fd > -1)
					fd = dmore_(nullptr, fd); // close(fd);
				exit(0);
			}
			unsigned char ch = buf[0];
			if (ch == ':') ch = buf[1];
			switch (ch) {
			case 'N':
			case 'n':
				fd = dmore_(nullptr, fd); // close(fd);
				break;
			case 'Q':
			case 'q':
				fd = dmore_(nullptr, fd); // close(fd);
				exit(0);
			}
		}
	}
	exit(0);
}












