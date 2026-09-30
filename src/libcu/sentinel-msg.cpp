#define _CRT_SECURE_NO_WARNINGS
//#include <host_defines.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/statcu.h>
#include <math.h>
#include <fcntl.h>
#include <sentinel.h>
#if __OS_WIN
#include <io.h>
#elif __OS_UNIX
#include <unistd.h>
#endif
#include <sentinel-hostmsg.h>
#include <sentinel-direntmsg.h>
#include <sentinel-fcntlmsg.h>
#include <sentinel-unistdmsg.h>
#include <sentinel-stdiomsg.h>
#include <sentinel-stdlibmsg.h>
#include <sentinel-timemsg.h>

//#define panic(fmt, ...) { printf(fmt, __VA_ARGS__); exit(1); }

#if __OS_WIN
#define fcntl(fd, cmd, ...) 0
#define mkfifo(path, mode) 0
int setenv(const char *name, const char *value, int overwrite) {
	int errcode = 0;
	if (!overwrite) {
		size_t envsize = 0;
		errcode = getenv_s(&envsize, NULL, 0, name);
		if (errcode || envsize) return errcode;
	}
	return _putenv_s(name, value);
}
#define fileno _fileno
int unsetenv(const char *name) {
	size_t nameLength = strlen(name);
	char *remove = (char *)alloca(nameLength + 2);
	memcpy(remove, name, nameLength);
	remove[nameLength] = '=';
	remove[nameLength + 1] = 0;
	return _putenv(remove);
}
#define mktemp _mktemp
#define mkstemp(p) _open(_mktemp(p), O_CREAT | O_EXCL | O_RDWR)
#define access(a, b) (b)!=1?_access(a,b):0
#define lseek _lseek
#define close _close
#define read _read
#define write _write
#define chown(a, b, c) 0
#define chdir _chdir
#define getcwd _getcwd	
#define dup _dup
#define dup2 _dup2
#define unlink _unlink
#define rmdir _rmdir
#define open _open
#define stat64 _stat64
#define fstat64 _fstat64
#define chmod _chmod

#endif

bool sentinelDefaultHostExecutor(void *tag, sentinelMessage *data, int length, char *(**hostPrepare)(void*, char*, char*, intptr_t)) {
#if HAS_HOSTSENTINEL
	if (data->op > TIME_STRFTIME) return false;
	switch (data->op) {
#if __OS_WIN
	case HOST_GETPROCESSID: { host_getprocessid *msg = (host_getprocessid *)data; msg->rc = GetCurrentProcessId(); return true; }
#elif __OS_UNIX
	case HOST_GETPROCESSID: { host_getprocessid *msg = (host_getprocessid *)data; msg->rc = getpid(); return true; }
#endif
	}
#endif
	return false;
}

bool sentinelDefaultDeviceExecutor(void *tag, sentinelMessage *data, int length, char *(**hostPrepare)(void*, char*, char*, intptr_t)) {
	if (data->op > TIME_STRFTIME) return false;
	switch (data->op) {
	case STDIO_REMOVE: { stdio_remove *msg = (stdio_remove *)data; msg->rc = remove(msg->str); return true; }
	case STDIO_RENAME: { stdio_rename *msg = (stdio_rename *)data; msg->rc = rename(msg->str, msg->str2); return true; }
	case STDIO_FCLOSE: { stdio_fclose *msg = (stdio_fclose *)data; msg->rc = fclose(msg->file); return true; }
	case STDIO_FFLUSH: { stdio_fflush *msg = (stdio_fflush *)data; msg->rc = fflush(msg->file); return true; }
	case STDIO_FREOPEN: { stdio_freopen *msg = (stdio_freopen *)data; FILE *f = (!msg->stream ? fopen(msg->str, msg->str2) : freopen(msg->str, msg->str2, msg->stream)); msg->rc = f; return true; }
	case STDIO_SETVBUF: { stdio_setvbuf *msg = (stdio_setvbuf *)data; if (msg->mode != -1) msg->rc = setvbuf(msg->file, msg->buf, msg->mode, msg->size); else setbuf(msg->file, msg->buf); return true; }
	case STDIO_FGETC: { stdio_fgetc *msg = (stdio_fgetc *)data; msg->rc = fgetc(msg->file); return true; }
	case STDIO_FGETS: { stdio_fgets *msg = (stdio_fgets *)data; msg->rc = fgets(msg->str, msg->num, msg->file); return true; }
	case STDIO_FPUTC: { stdio_fputc *msg = (stdio_fputc *)data; msg->rc = fputc(msg->ch, msg->file); return true; }
	case STDIO_FPUTS: { stdio_fputs *msg = (stdio_fputs *)data; msg->rc = fputs(msg->str, msg->file); return true; }
	case STDIO_UNGETC: { stdio_ungetc *msg = (stdio_ungetc *)data; msg->rc = ungetc(msg->ch, msg->file); return true; }
	case STDIO_FREAD: { stdio_fread *msg = (stdio_fread *)data; msg->rc = fread(msg->ptr, msg->size, msg->num, msg->file); return true; }
	case STDIO_FWRITE: { stdio_fwrite *msg = (stdio_fwrite *)data; msg->rc = fwrite(msg->ptr, msg->size, msg->num, msg->file); return true; }
	case STDIO_FSEEK: { stdio_fseek *msg = (stdio_fseek *)data; msg->rc = fseek(msg->file, msg->offset, msg->origin); return true; }
	case STDIO_FTELL: { stdio_ftell *msg = (stdio_ftell *)data; msg->rc = ftell(msg->file); return true; }
	case STDIO_REWIND: { stdio_rewind *msg = (stdio_rewind *)data; rewind(msg->file); return true; }
#if defined(__USE_LARGEFILE)
	case STDIO_FSEEKO: { stdio_fseeko *msg = (stdio_fseeko *)data;
		if (!msg->bit64) msg->rc = fseeko(msg->file, msg->offset, msg->origin);
#ifdef __USE_LARGEFILE64
		else msg->rc = fseeko64(msg->file, msg->offset64, msg->origin);
#endif
		return true; }
	case STDIO_FTELLO: { stdio_ftello *msg = (stdio_ftello *)data;
		if (!msg->bit64) msg->rc = ftello(msg->file);
#ifdef __USE_LARGEFILE64
		else msg->rc64 = ftello64(msg->file);
#endif
		return true; }
#endif
	case STDIO_FGETPOS: { stdio_fgetpos *msg = (stdio_fgetpos *)data;
		if (!msg->bit64) msg->rc = fgetpos(msg->file, &msg->pos);
#ifdef __USE_LARGEFILE64
		else msg->rc = fgetpos64(msg->file, &msg->pos64);
#endif
		return true; }
	case STDIO_FSETPOS: { stdio_fsetpos *msg = (stdio_fsetpos *)data;
		if (!msg->bit64) msg->rc = fsetpos(msg->file, &msg->pos);
#ifdef __USE_LARGEFILE64
		else msg->rc = fsetpos64(msg->file, &msg->pos64);
#endif
		return true; }
	case STDIO_CLEARERR: { stdio_clearerr *msg = (stdio_clearerr *)data; clearerr(msg->file); return true; }
	case STDIO_FEOF: { stdio_feof *msg = (stdio_feof *)data; msg->rc = feof(msg->file); return true; }
	case STDIO_FERROR: { stdio_ferror *msg = (stdio_ferror *)data; msg->rc = ferror(msg->file); return true; }
	case STDIO_FILENO: { stdio_fileno *msg = (stdio_fileno *)data; msg->rc = fileno(msg->file); return true; }
	case STDLIB_SYSTEM: { stdlib_system *msg = (stdlib_system *)data; msg->rc = system(msg->str); return true; }
#if __OS_WIN
	case STDLIB_EXIT: { stdlib_exit *msg = (stdlib_exit *)data; if (msg->std) exit(msg->status); else _exit(msg->status); return true; }
#elif __OS_UNIX
	case STDLIB_EXIT: { stdlib_exit *msg = (stdlib_exit *)data; exit(msg->status); return true; }
#endif
	case STDLIB_GETENV: { stdlib_getenv *msg = (stdlib_getenv *)data; msg->rc = getenv(msg->str); *hostPrepare = SENTINELPREPARE(stdlib_getenv::hostPrepare); return true; }
	case STDLIB_SETENV: { stdlib_setenv *msg = (stdlib_setenv *)data; msg->rc = setenv(msg->str, msg->str2, msg->replace); return true; }
	case STDLIB_UNSETENV: { stdlib_unsetenv *msg = (stdlib_unsetenv *)data; msg->rc = unsetenv(msg->str); return true; }
	case STDLIB_MKTEMP: { stdlib_mktemp *msg = (stdlib_mktemp *)data; msg->rc = mktemp(msg->str); return true; }
	case STDLIB_MKSTEMP: { stdlib_mkstemp *msg = (stdlib_mkstemp *)data; msg->rc = mkstemp(msg->str); return true; }
	case UNISTD_ACCESS: { unistd_access *msg = (unistd_access *)data; msg->rc = access(msg->str, msg->type); return true; }
	case UNISTD_LSEEK: { unistd_lseek *msg = (unistd_lseek *)data;
		if (!msg->bit64) msg->rc = lseek(msg->handle, (long)msg->offset, msg->whence);
#ifdef __USE_LARGEFILE64
		else msg->rc = lseek64(msg->handle, msg->offset, msg->whence);
#endif
		return true; }
	case UNISTD_CLOSE: { unistd_close *msg = (unistd_close *)data; msg->rc = close(msg->handle); return true; }
	case UNISTD_READ: { unistd_read *msg = (unistd_read *)data; msg->rc = read(msg->handle, msg->ptr, (int)msg->size); return true; }
	case UNISTD_WRITE: { unistd_write *msg = (unistd_write *)data; msg->rc = write(msg->handle, msg->ptr, (int)msg->size); return true; }
	case UNISTD_CHOWN: { unistd_chown *msg = (unistd_chown *)data; msg->rc = chown(msg->str, msg->owner, msg->group); return true; }
	case UNISTD_CHDIR: { unistd_chdir *msg = (unistd_chdir *)data; msg->rc = chdir(msg->str); return true; }
	case UNISTD_GETCWD: { unistd_getcwd *msg = (unistd_getcwd *)data; msg->rc = getcwd(msg->ptr, (int)msg->size); return true; }
	case UNISTD_DUP: { unistd_dup *msg = (unistd_dup *)data; msg->rc = (msg->dup1 ? dup(msg->handle) : dup2(msg->handle, msg->handle2)); return true; }
	case UNISTD_UNLINK: { unistd_unlink *msg = (unistd_unlink *)data; msg->rc = unlink(msg->str); return true; }
	case UNISTD_RMDIR: { unistd_rmdir *msg = (unistd_rmdir *)data; msg->rc = rmdir(msg->str); return true; }
	case FCNTL_FCNTL: { fcntl_fcntl *msg = (fcntl_fcntl *)data;
		if (!msg->bit64) msg->rc = fcntl(msg->handle, msg->cmd, msg->p0);
#ifdef __USE_LARGEFILE64
		else panic("Not Implemented");
		//else msg->rc = fcntl64(msg->handle, msg->cmd, msg->p0);
#endif
		return true; }
	case FCNTL_OPEN: { fcntl_open *msg = (fcntl_open *)data;
		if (!msg->bit64) msg->rc = open(msg->str, msg->oflag, msg->p0);
#ifdef __USE_LARGEFILE64
		else msg->rc = open64(msg->str, msg->oflag, msg->p0);
#endif
		return true; }
	case FCNTL_STAT: { fcntl_stat *msg = (fcntl_stat *)data;
		if (!msg->bit64) msg->rc = !msg->lstat_ ? stat(msg->str, (struct stat *)msg->ptr) : lstat(msg->str, (struct stat *)msg->ptr);
#ifdef __USE_LARGEFILE64
		else msg->rc = !msg->lstat_ ? stat64(msg->str, (struct _stat64 *)msg->ptr) : lstat64(msg->str, (struct _stat64 *)msg->ptr);
#endif
		return true; }
	case FCNTL_FSTAT: { fcntl_fstat *msg = (fcntl_fstat *)data;
		if (!msg->bit64) msg->rc = fstat(msg->handle, (struct stat *)msg->ptr);
#ifdef __USE_LARGEFILE64
		else msg->rc = fstat64(msg->handle, (struct _stat64 *)msg->ptr);
#endif
		return true; }
	case FCNTL_CHMOD: { fcntl_chmod *msg = (fcntl_chmod *)data; msg->rc = chmod(msg->str, msg->mode); return true; }
	case FCNTL_MKDIR: { fcntl_mkdir *msg = (fcntl_mkdir *)data; msg->rc = mkdir(msg->str, msg->mode); return true; }
	case FCNTL_MKFIFO: { fcntl_mkfifo *msg = (fcntl_mkfifo *)data; msg->rc = mkfifo(msg->str, msg->mode); return true; }
	case DIRENT_OPENDIR: { dirent_opendir *msg = (dirent_opendir *)data; msg->rc = opendir(msg->str); return true; }
	case DIRENT_CLOSEDIR: { dirent_closedir *msg = (dirent_closedir *)data; msg->rc = closedir(msg->ptr); return true; }
	case DIRENT_READDIR: { dirent_readdir *msg = (dirent_readdir *)data;
		if (!msg->bit64) { msg->rc = readdir(msg->ptr); *hostPrepare = SENTINELPREPARE(dirent_readdir::hostPrepare); }
#ifdef __USE_LARGEFILE64
		else { msg->rc64 = readdir64(msg->ptr); *hostPrepare = SENTINELPREPARE(dirent_readdir::hostPrepare64); }
#endif
		return true; }
	case DIRENT_REWINDDIR: { dirent_rewinddir *msg = (dirent_rewinddir *)data; rewinddir(msg->ptr); return true; }
	case TIME_TIME: { time_time *msg = (time_time *)data; msg->rc = time(nullptr); return true; }
	case TIME_MKTIME: { time_mktime *msg = (time_mktime *)data; msg->rc = mktime(msg->tp); return true; }
	case TIME_STRFTIME: { time_strftime *msg = (time_strftime *)data; msg->rc = strftime((char *)msg->ptr, msg->maxsize, msg->str, &msg->tp); return true; }
	}
	return false;
}