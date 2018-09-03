#define _CRT_SECURE_NO_WARNINGS
#include <host_defines.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/statcu.h>
#include <math.h>
#if __OS_WIN
#include <io.h>
#elif __OS_UNIX
#include <unistd.h>
#include <fcntl.h>
#endif
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
#define unsetenv(a) _putenv_s(a, nullptr)
#define access _access
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

bool sentinelDefaultExecutor(void *tag, sentinelMessage *data, int length, char *(**hostPrepare)(void*, char*, char*, intptr_t)) {
	if (data->OP > TIME_STRFTIME) return false;
	switch (data->OP) {
	case STDIO_REMOVE: { stdio_remove *msg = (stdio_remove *)data; msg->RC = remove(msg->Str); return true; }
	case STDIO_RENAME: { stdio_rename *msg = (stdio_rename *)data; msg->RC = rename(msg->Str, msg->Str2); return true; }
	case STDIO_FCLOSE: { stdio_fclose *msg = (stdio_fclose *)data; msg->RC = fclose(msg->File); return true; }
	case STDIO_FFLUSH: { stdio_fflush *msg = (stdio_fflush *)data; msg->RC = fflush(msg->File); return true; }
	case STDIO_FREOPEN: { stdio_freopen *msg = (stdio_freopen *)data; FILE *f = (!msg->Stream ? fopen(msg->Str, msg->Str2) : freopen(msg->Str, msg->Str2, msg->Stream)); msg->RC = f; return true; }
	case STDIO_SETVBUF: { stdio_setvbuf *msg = (stdio_setvbuf *)data; if (msg->Mode != -1) msg->RC = setvbuf(msg->File, msg->Buffer, msg->Mode, msg->Size); else setbuf(msg->File, msg->Buffer); return true; }
	case STDIO_FGETC: { stdio_fgetc *msg = (stdio_fgetc *)data; msg->RC = fgetc(msg->File); return true; }
	case STDIO_FGETS: { stdio_fgets *msg = (stdio_fgets *)data; msg->RC = fgets(msg->Str, msg->Num, msg->File); return true; }
	case STDIO_FPUTC: { stdio_fputc *msg = (stdio_fputc *)data; msg->RC = fputc(msg->Ch, msg->File); return true; }
	case STDIO_FPUTS: { stdio_fputs *msg = (stdio_fputs *)data; msg->RC = fputs(msg->Str, msg->File); return true; }
	case STDIO_UNGETC: { stdio_ungetc *msg = (stdio_ungetc *)data; msg->RC = ungetc(msg->Ch, msg->File); return true; }
	case STDIO_FREAD: { stdio_fread *msg = (stdio_fread *)data; msg->RC = fread(msg->Ptr, msg->Size, msg->Num, msg->File); return true; }
	case STDIO_FWRITE: { stdio_fwrite *msg = (stdio_fwrite *)data; msg->RC = fwrite(msg->Ptr, msg->Size, msg->Num, msg->File); return true; }
	case STDIO_FSEEK: { stdio_fseek *msg = (stdio_fseek *)data; msg->RC = fseek(msg->File, msg->Offset, msg->Origin); return true; }
	case STDIO_FTELL: { stdio_ftell *msg = (stdio_ftell *)data; msg->RC = ftell(msg->File); return true; }
	case STDIO_REWIND: { stdio_rewind *msg = (stdio_rewind *)data; rewind(msg->File); return true; }
#if defined(__USE_LARGEFILE)
	case STDIO_FSEEKO: { stdio_fseeko *msg = (stdio_fseeko *)data;
		if (!msg->Bit64) msg->RC = fseeko(msg->File, msg->Offset, msg->Origin);
#ifdef __USE_LARGEFILE64
		else msg->RC = fseeko64(msg->File, msg->Offset64, msg->Origin);
#endif
		return true; }
	case STDIO_FTELLO: { stdio_ftello *msg = (stdio_ftello *)data;
		if (!msg->Bit64) msg->RC = ftello(msg->File);
#ifdef __USE_LARGEFILE64
		else msg->RC64 = ftello64(msg->File);
#endif
		return true; }
#endif
	case STDIO_FGETPOS: { stdio_fgetpos *msg = (stdio_fgetpos *)data; msg->RC = fgetpos(msg->File, msg->Pos); return true; }
	case STDIO_FSETPOS: { stdio_fsetpos *msg = (stdio_fsetpos *)data; msg->RC = fsetpos(msg->File, msg->Pos); return true; }
	case STDIO_CLEARERR: { stdio_clearerr *msg = (stdio_clearerr *)data; clearerr(msg->File); return true; }
	case STDIO_FEOF: { stdio_feof *msg = (stdio_feof *)data; msg->RC = feof(msg->File); return true; }
	case STDIO_FERROR: { stdio_ferror *msg = (stdio_ferror *)data; msg->RC = ferror(msg->File); return true; }
	case STDIO_FILENO: { stdio_fileno *msg = (stdio_fileno *)data; msg->RC = fileno(msg->File); return true; }
	case STDLIB_SYSTEM: { stdlib_system *msg = (stdlib_system *)data; msg->RC = system(msg->Str); return true; }
#if __OS_WIN
	case STDLIB_EXIT: { stdlib_exit *msg = (stdlib_exit *)data; if (msg->Std) exit(msg->Status); else _exit(msg->Status); return true; }
#elif __OS_UNIX
	case STDLIB_EXIT: { stdlib_exit *msg = (stdlib_exit *)data; exit(msg->Status); return true; }
#endif
	case STDLIB_GETENV: { stdlib_getenv *msg = (stdlib_getenv *)data; msg->RC = getenv(msg->Str); return true; }
	case STDLIB_SETENV: { stdlib_setenv *msg = (stdlib_setenv *)data; msg->RC = setenv(msg->Str, msg->Str2, msg->Replace); return true; }
	case STDLIB_UNSETENV: { stdlib_unsetenv *msg = (stdlib_unsetenv *)data; msg->RC = unsetenv(msg->Str); return true; }
	case UNISTD_ACCESS: { unistd_access *msg = (unistd_access *)data; msg->RC = access(msg->Name, msg->Type); return true; }
	case UNISTD_LSEEK: { unistd_lseek *msg = (unistd_lseek *)data;
		if (!msg->Bit64) msg->RC = lseek(msg->Handle, (long)msg->Offset, msg->Whence);
#ifdef __USE_LARGEFILE64
		else msg->RC = lseek64(msg->Handle, msg->Offset, msg->Whence);
#endif
		return true; }
	case UNISTD_CLOSE: { unistd_close *msg = (unistd_close *)data; msg->RC = close(msg->Handle); return true; }
	case UNISTD_READ: { unistd_read *msg = (unistd_read *)data; msg->RC = read(msg->Handle, msg->Ptr, (int)msg->Size); return true; }
	case UNISTD_WRITE: { unistd_write *msg = (unistd_write *)data; msg->RC = write(msg->Handle, msg->Ptr, (int)msg->Size); return true; }
	case UNISTD_CHOWN: { unistd_chown *msg = (unistd_chown *)data; msg->RC = chown(msg->Str, msg->Owner, msg->Group); return true; }
	case UNISTD_CHDIR: { unistd_chdir *msg = (unistd_chdir *)data; msg->RC = chdir(msg->Str); return true; }
	case UNISTD_GETCWD: { unistd_getcwd *msg = (unistd_getcwd *)data; msg->RC = getcwd(msg->Ptr, (int)msg->Size); return true; }
	case UNISTD_DUP: { unistd_dup *msg = (unistd_dup *)data; msg->RC = (msg->Dup1 ? dup(msg->Handle) : dup2(msg->Handle, msg->Handle2)); return true; }
	case UNISTD_UNLINK: { unistd_unlink *msg = (unistd_unlink *)data; msg->RC = unlink(msg->Str); return true; }
	case UNISTD_RMDIR: { unistd_rmdir *msg = (unistd_rmdir *)data; msg->RC = rmdir(msg->Str); return true; }
	case FCNTL_FCNTL: { fcntl_fcntl *msg = (fcntl_fcntl *)data;
		if (!msg->Bit64) msg->RC = fcntl(msg->Handle, msg->Cmd, msg->P0);
#ifdef __USE_LARGEFILE64
		else panic("Not Implemented");
		//else msg->RC = fcntl64(msg->Handle, msg->Cmd, msg->P0);
#endif
		return true; }
	case FCNTL_OPEN: { fcntl_open *msg = (fcntl_open *)data;
		if (!msg->Bit64) msg->RC = open(msg->Str, msg->OFlag, msg->P0);
#ifdef __USE_LARGEFILE64
		else msg->RC = open64(msg->Str, msg->OFlag, msg->P0);
#endif
		return true; }
	case FCNTL_STAT: { fcntl_stat *msg = (fcntl_stat *)data;
		if (!msg->Bit64) msg->RC = !msg->LStat ? stat(msg->Str, msg->Ptr) : lstat(msg->Str, msg->Ptr);
#ifdef __USE_LARGEFILE64
		else msg->RC = !msg->LStat ? stat64(msg->Str, msg->Ptr64) : lstat64(msg->Str, msg->Ptr64);
#endif
		return true; }
	case FCNTL_FSTAT: { fcntl_fstat *msg = (fcntl_fstat *)data;
		if (!msg->Bit64) msg->RC = fstat(msg->Handle, msg->Ptr);
#ifdef __USE_LARGEFILE64
		else msg->RC = fstat64(msg->Handle, msg->Ptr64);
#endif
		return true; }
	case FCNTL_CHMOD: { fcntl_chmod *msg = (fcntl_chmod *)data; msg->RC = chmod(msg->Str, msg->Mode); return true; }
	case FCNTL_MKDIR: { fcntl_mkdir *msg = (fcntl_mkdir *)data; msg->RC = mkdir(msg->Str, msg->Mode); return true; }
	case FCNTL_MKFIFO: { fcntl_mkfifo *msg = (fcntl_mkfifo *)data; msg->RC = mkfifo(msg->Str, msg->Mode); return true; }
	case DIRENT_OPENDIR: { dirent_opendir *msg = (dirent_opendir *)data; msg->RC = opendir(msg->Str); return true; }
	case DIRENT_CLOSEDIR: { dirent_closedir *msg = (dirent_closedir *)data; msg->RC = closedir(msg->Ptr); return true; }
	case DIRENT_READDIR: { dirent_readdir *msg = (dirent_readdir *)data;
		if (!msg->Bit64) { msg->RC = readdir(msg->Ptr); *hostPrepare = SENTINELPREPARE(dirent_readdir::HostPrepare); }
#ifdef __USE_LARGEFILE64
		else { msg->RC64 = readdir64(msg->Ptr); *hostPrepare = SENTINELPREPARE(dirent_readdir::HostPrepare64); }
#endif
		return true; }
	case DIRENT_REWINDDIR: { dirent_rewinddir *msg = (dirent_rewinddir *)data; rewinddir(msg->Ptr); return true; }
	case TIME_TIME: { time_time *msg = (time_time *)data; msg->RC = time(nullptr); return true; }
	case TIME_MKTIME: { time_mktime *msg = (time_mktime *)data; msg->RC = mktime(msg->Tp); return true; }
	case TIME_STRFTIME: { time_strftime *msg = (time_strftime *)data; msg->RC = strftime((char *)msg->Ptr, msg->Maxsize, msg->Fmt, &msg->Tp); return true; }
	}
	return false;
}