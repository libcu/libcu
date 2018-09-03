#ifndef _FSYSTEM_H
#define _FSYSTEM_H
#include <crtdefscu.h>
#include <fcntl.h>
#include <ext/memfile.h>
#if __OS_WIN
#include <_dirent.h>
#elif __OS_UNIX
#include <dirent.h>
#endif

__BEGIN_DECLS;

struct dirEnt_t {
	dirent dir;		// Entry information
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
};

__device__ int expandPath(const char *path, char *newPath);
__device__ int fsystemChdir(const char *path);
__device__ dirEnt_t *fsystemOpendir(const char *path);
__device__ int fsystemRename(const char *old, const char *new_);
__device__ int fsystemUnlink(const char *path, bool enotdir);
__device__ dirEnt_t *fsystemMkdir(const char *__restrict path, int mode, int *r);
__device__ dirEnt_t *fsystemAccess(const char *__restrict path, int mode, int *r);
__device__ dirEnt_t *fsystemOpen(const char *__restrict path, int mode, int *fd);
__device__ void fsystemClose(int fd);
__device__ void fsystemReset();
__device__ void fsystemSetFlag(int fd, int flag);

/* File support  */
extern __device__ dirEnt_t __iob_root;
extern __constant__ file_t __iob_files[LIBCU_MAXFILESTREAM];
#define GETFD(fd) (INT_MAX-(fd))
#define GETFILE(fd) (&__iob_files[GETFD(fd)])

__END_DECLS;
#endif  /* _FSYSTEM_H */