#include <unistdcu.h>
#include <sentinel-unistdmsg.h>
#include "fsystem.h"
#include <stdio.h> // panic's printf

__BEGIN_DECLS;

/* Test for access to NAME using the real UID and real GID.  */
__device__ int access_(const char *name, int mode) {
	if (ISHOSTPATH(name)) { unistd_access msg(name, mode); return msg.rc; }
#ifdef LIBCU_LEAN_FSYSTEM
	return panic_no_fsystem();
#else
	int r; fsystemAccess(name, mode, &r); return r;
#endif
}

/* Move FD's file position to OFFSET bytes from the beginning of the file (if WHENCE is SEEK_SET),
the current position (if WHENCE is SEEK_CUR), or the end of the file (if WHENCE is SEEK_END).
Return the new file position.  */
__device__ off_t lseek_(int fd, off_t offset, int whence) {
	if (ISHOSTHANDLE(fd)) { unistd_lseek msg(fd, offset, whence, false); return msg.rc; }
#ifdef LIBCU_LEAN_FSYSTEM
	return panic_no_fsystem();
#else
	register file_t *s = GETFILE(fd);
	dirEnt_t *f;
	if (!s || !(f = (dirEnt_t *)s->base))
		panic("lseek: !stream");
	switch (whence) {
	case SEEK_SET: return (s->off = offset);
	case SEEK_CUR: return (s->off += offset);
	case SEEK_END: int64_t size; memfileFileSize(f->u.file, &size); return (s->off = size - offset);
	default: return -1;
	}
#endif
}

#ifdef __USE_LARGEFILE64
__device__ off64_t lseek64_(int fd, off64_t offset, int whence) {
	if (ISHOSTHANDLE(fd)) { unistd_lseek msg(fd, offset, whence, true); return msg.rc; }
#ifdef LIBCU_LEAN_FSYSTEM
	return panic_no_fsystem();
#else
	register file_t *s = GETFILE(fd);
	dirEnt_t *f;
	if (!s || !(f = (dirEnt_t *)s->base))
		panic("lseek: !stream");
	switch (whence) {
	case SEEK_SET: return (s->off = offset);
	case SEEK_CUR: return (s->off += offset);
	case SEEK_END: int64_t size; memfileFileSize(f->u.file, &size); return (s->off = size - offset);
	default: return -1;
#endif
}
#endif

/* Close the file descriptor FD.  */
//__device__ int close_(int fd); // defined in fcntlcu.cu

/* Read NBYTES into BUF from FD.  Return the number read, -1 for errors or 0 for EOF.  */
__device__ size_t read_(int fd, void *buf, size_t nbytes, bool wait) {
	if (ISHOSTHANDLE(fd)) { unistd_read msg(wait, fd, buf, nbytes); return msg.rc; }
#ifdef LIBCU_LEAN_FSYSTEM
	return panic_no_fsystem();
#else
	register file_t *s = GETFILE(fd);
	dirEnt_t *f;
	if (!s || !(f = (dirEnt_t *)s->base))
		panic("read: !stream");
	if (f->dir.d_type != DIRTYPE_FILE)
		panic("read: stream !file");
	memfileRead(f->u.file, buf, nbytes, s->off); s->off += nbytes;
	return nbytes;
#endif
}

/* Write N bytes of BUF to FD.  Return the number written, or -1.  */
__device__ size_t write_(int fd, const void *buf, size_t nbytes, bool wait) {
	if (ISHOSTHANDLE(fd)) { unistd_write msg(wait, fd, buf, nbytes); return msg.rc; }
#ifdef LIBCU_LEAN_FSYSTEM
	return panic_no_fsystem();
#else
	register file_t *s = GETFILE(fd);
	dirEnt_t *f;
	if (!s || !(f = (dirEnt_t *)s->base))
		panic("write: !stream");
	if (f->dir.d_type != DIRTYPE_FILE)
		panic("write: stream !file");
	memfileWrite(f->u.file, buf, nbytes, s->off); s->off += nbytes;
	return nbytes;
#endif
}

/* Make the process sleep for SECONDS seconds, or until a signal arrives and is not ignored.  The function returns the number of seconds less
than SECONDS which it actually slept (thus zero if it slept the full time). If a signal handler does a `longjmp' or modifies the handling of the
SIGALRM signal while inside `sleep' call, the handling of the SIGALRM signal afterwards is undefined.  There is no return value to indicate
error, but if `sleep' returns SECONDS, it probably didn't work.  */
__device__ void usleep_(unsigned long milliseconds) {
	clock_t start = clock();
	clock_t end = milliseconds * 10;
	for (;;) {
		clock_t now = clock();
		clock_t cycles = now > start ? now - start : now + (0xffffffff - start);
		if (cycles >= end) break;
	}
}

/* Change the owner and group of FILE.  */
__device__ int chown_(const char *file, uid_t owner, gid_t group) {
	if (ISHOSTPATH(file)) { __cwd[0] = 0; unistd_chown msg(file, owner, group); return msg.rc; }
	return 0;
}

/* Change the process's working directory to PATH.  */
__device__ int chdir_(const char *path) {
	if (ISHOSTPATH(path)) { __cwd[0] = 0; unistd_chdir msg(path); return msg.rc; }
#ifdef LIBCU_LEAN_FSYSTEM
	return panic_no_fsystem();
#else
	return fsystemChdir(path);
#endif
}

/* Get the pathname of the current working directory, and put it in SIZE bytes of BUF.  Returns NULL if the
directory couldn't be determined or SIZE was too small. If successful, returns BUF.  In GNU, if BUF is NULL,
an array is allocated with `malloc'; the array is SIZE bytes long, unless SIZE == 0, in which case it is as
big as necessary.  */
__device__ char *getcwd_(char *buf, size_t size) {
	if (!__cwd[0]) { unistd_getcwd msg(buf, size); return msg.rc; }
	int pathLength = strlen(__cwd);
	return size > pathLength ? strncpy(buf, __cwd, size) : nullptr;
}

/* dup1:true - Duplicate FD, returning a new file descriptor on the same file.  */
/* dup1:false - Duplicate FD to FD2, closing FD2 and making it open on the same file.  */
__device__ int dup_(int fd) {
	if (ISHOSTHANDLE(fd)) { unistd_dup msg(fd, -1, true); return msg.rc; }
#ifdef LIBCU_LEAN_FSYSTEM
	return panic_no_fsystem();
#else
	panic("Not Implemented");
	return 0;
#endif
}

/* Duplicate FD to FD2, closing FD2 and making it open on the same file.  */
__device__ int dup2_(int fd, int fd2) {
	if (ISHOSTHANDLE(fd)) { unistd_dup msg(fd, fd2, false); return msg.rc; }
#ifdef LIBCU_LEAN_FSYSTEM
	return panic_no_fsystem();
#else
	panic("Not Implemented");
	return 0;
#endif
}

/* NULL-terminated array of "NAME=VALUE" environment variables.  */
extern __device__ char **__environ_ = nullptr;

/* Remove the link NAME.  */
__device__ int unlink_(const char *filename) {
	if (ISHOSTPATH(filename)) { unistd_unlink msg(filename); return msg.rc; }
#ifdef LIBCU_LEAN_FSYSTEM
	return panic_no_fsystem();
#else
	return fsystemUnlink(filename, false);
#endif
}

/* Remove the directory PATH.  */
__device__ int rmdir_(const char *path) {
	if (ISHOSTPATH(path)) { unistd_rmdir msg(path); return msg.rc; }
#ifdef LIBCU_LEAN_FSYSTEM
	return panic_no_fsystem();
#else
	return fsystemUnlink(path, true);
#endif
}

__END_DECLS;