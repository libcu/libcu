#include <sys/statcu.h>
#include <stdiocu.h>
#include <unistdcu.h>
#include <fcntlcu.h>
#include <timecu.h>
#include "fileutils.h"
#ifndef BUF_SIZE
#define	BUF_SIZE 1024
#endif

// Copy one file to another, while possibly preserving its modes, times, and modes.  Returns TRUE if successful, or FALSE on a failure with an
// error message output.  (Failure is not indicted if the attributes cannot be set.)
__device__ bool copyFile(char *srcName, char *destName, bool setModes) {
	struct stat statbuf1;
	if (stat(srcName, &statbuf1) < 0) {
		perror(srcName);
		return false;
	}
	struct stat statbuf2;
	if (stat(destName, &statbuf2) < 0) {
		statbuf2.st_ino = 0;
		statbuf2.st_dev = 0;
	}
	if (statbuf1.st_dev == statbuf2.st_dev && statbuf1.st_ino == statbuf2.st_ino) {
		printf("Copying file \"%s\" to itself\n", srcName);
		return false;
	}
	//
	int rfd = open(srcName, 0);
	if (rfd < 0) {
		perror(srcName);
		return false;
	}
	int wfd = creat(destName, statbuf1.st_mode);
	if (wfd < 0) {
		perror(destName);
		close(rfd);
		return false;
	}
	//
	char *buf = (char *)malloc(BUF_SIZE);
	int rcc;
	while ((rcc = read(rfd, buf, BUF_SIZE)) > 0) {
		char *bp = buf;
		while (rcc > 0) {
			int wcc = write(wfd, bp, rcc);
			if (wcc < 0) {
				perror(destName);
				goto error_exit;
			}
			bp += wcc;
			rcc -= wcc;
		}
	}
	if (rcc < 0) {
		perror(srcName);
		goto error_exit;
	}
	free(buf);
	close(rfd);
	if (close(wfd) < 0) {
		perror(destName);
		return false;
	}
	if (setModes) {
		chmod(destName, statbuf1.st_mode);
		chown(destName, statbuf1.st_uid, statbuf1.st_gid);
		//struct utimbuf times;
		//times.actime = statbuf1.st_atime;
		//times.modtime = statbuf1.st_mtime;
		//utime(destName, &times);
	}
	return true;

error_exit:
	free(buf);
	close(rfd);
	close(wfd);
	return false;
}