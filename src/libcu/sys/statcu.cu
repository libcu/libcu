#include <sys/statcu.h>
#include <sentinel-fcntlmsg.h>
#include "../fsystem.h"
#include <stdio.h> // panic's printf

/* Get file attributes for FILE and put them in BUF.  */
__device__ int stat_(const char *__restrict file, struct stat *__restrict buf, bool lstat_) {
	if (ISHOSTPATH(file)) { fcntl_stat msg(file, buf, nullptr, false, lstat_); return msg.rc; }
#ifdef LIBCU_LEAN_FSYSTEM
	return panic_no_fsystem();
#else
	return fsystemStat(file, buf, nullptr, lstat_);
#endif
}

/* Get file attributes for the file, device, pipe, or socket that file descriptor FD is open on and put them in BUF.  */
__device__ int fstat_(int fd, struct stat *buf) {
	if (ISHOSTHANDLE(fd)) { fcntl_fstat msg(fd, buf, nullptr, false); return msg.rc; }
#ifdef LIBCU_LEAN_FSYSTEM
	return panic_no_fsystem();
#else
	return fsystemFStat(fd, buf, nullptr);
#endif
}

#ifdef __USE_LARGEFILE64
/* Get file attributes for FILE and put them in BUF.  */
__device__ int stat64_(const char *__restrict file, struct stat64 *__restrict buf, bool lstat_) {
	if (ISHOSTPATH(file)) { fcntl_stat msg(file, nullptr, buf, lstat_, true); return msg.rc; }
#ifdef LIBCU_LEAN_FSYSTEM
	return panic_no_fsystem();
#else
	return fsystemStat(file, nullptr, buf, lstat_);
#endif
}

/* Get file attributes for the file, device, pipe, or socket that file descriptor FD is open on and put them in BUF.  */
__device__ int fstat64_(int fd, struct stat64 *buf) {
	if (ISHOSTHANDLE(fd)) { fcntl_fstat msg(fd, nullptr, buf, true); return msg.rc; }
#ifdef LIBCU_LEAN_FSYSTEM
	return panic_no_fsystem();
#else
	return fsystemFStat(fd, nullptr, buf);
#endif
}
#endif

/* Set file access permissions for FILE to MODE. If FILE is a symbolic link, this affects its target instead.  */
__device__ int chmod_(const char *file, mode_t mode) {
	if (ISHOSTPATH(file)) { fcntl_chmod msg(file, mode); return msg.rc; }
#ifdef LIBCU_LEAN_FSYSTEM
	return panic_no_fsystem();
#else
	return fsystemChmod(file, mode);
#endif
}

/* Set the file creation mask of the current process to MASK, and return the old creation mask.  */
__device__ mode_t umask_(mode_t mask) {
	// unable to host umask
#ifdef LIBCU_LEAN_FSYSTEM
	return panic_no_fsystem();
#else
	return fsystemUmask(mask);
#endif
}

/* Create a new directory named PATH, with permission bits MODE.  */
__device__ int mkdir_(const char *path, mode_t mode) {
	if (ISHOSTPATH(path)) { fcntl_mkdir msg(path, mode); return msg.rc; }
#ifdef LIBCU_LEAN_FSYSTEM
	return panic_no_fsystem();
#else
	int r; fsystemMkdir(path, mode, &r); return r;
#endif
}

/* Create a new FIFO named PATH, with permission bits MODE.  */
__device__ int mkfifo_(const char *path, mode_t mode) {
	if (ISHOSTPATH(path)) { fcntl_mkfifo msg(path, mode); return msg.rc; }
#ifdef LIBCU_LEAN_FSYSTEM
	return panic_no_fsystem();
#else
	int r; fsystemMkfifo(path, mode, &r); return r;
#endif
}
