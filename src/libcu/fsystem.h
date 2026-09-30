#ifndef _FSYSTEM_H
#define _FSYSTEM_H
#include <crtdefscu.h>
#include <sys/statcu.h>
#include <fcntl.h>
#include <ext/memfile.h>
#if __OS_WIN
#include <_dirent.h>
#elif __OS_UNIX
#include <dirent.h>
#define _stat64 stat64
#endif

__BEGIN_DECLS;

enum {
	DIRTYPE_DIR = S_IFDIR,
	DIRTYPE_FILE = S_IFREG,
	DIRTYPE_FIFO = S_IFIFO,
};

struct dirEnt_t {
	dirent dir;		// Entry information
	struct stat stat;
	dirEnt_t *next;	// Next entity in the directory.
	char *path;		// Path/Key
	union {
		dirEnt_t *list;	// List of entities in the directory
		vsysfile *file; // Memory file associated with this element
	} u;
};

struct file_t {
	char *base;
	int flag;
	off_t off;
};

__device__ int expandPath(const char *path, char *newPath);
__device__ mode_t fsystemUmask(mode_t mask);
__device__ int fsystemChdir(const char *path);
__device__ dirEnt_t *fsystemOpendir(const char *path);
__device__ int fsystemRename(const char *old, const char *new_);
__device__ int fsystemUnlink(const char *path, bool enotdir);
__device__ int fsystemStat(const char *path, struct stat *buf, struct _stat64 *buf64, bool lstat_);
__device__ int fsystemFStat(int fd, struct stat *buf, struct _stat64 *buf64);
__device__ int fsystemChmod(const char *path, mode_t mode);
__device__ dirEnt_t *fsystemMkdir(const char *__restrict path, int mode, int *r);
__device__ dirEnt_t *fsystemMkfifo(const char *__restrict path, int mode, int *r);
__device__ dirEnt_t *fsystemAccess(const char *__restrict path, int mode, int *r);
__device__ dirEnt_t *fsystemOpen(const char *__restrict path, int mode, int *fd);
__device__ void fsystemClose(int fd);
__device__ void fsystemReset();
__device__ void fsystemSetFlag(int fd, int flag);
#define fsystemIsDir(ent) (ent->dir.d_type == DIRTYPE_DIR)

/* File support  */
extern __device__ dirEnt_t __iob_root;
extern __constant__ file_t __iob_files[LIBCU_MAXFILESTREAM];
#define GETFD(fd) (INT_MAX-(fd))
#define GETFILE(fd) (&__iob_files[GETFD(fd)])

__END_DECLS;
#endif  /* _FSYSTEM_H */