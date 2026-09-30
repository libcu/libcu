#ifdef LIBCU_LEAN_FSYSTEM
__device__ char __cwd[MAX_PATH] = "";
#else
#include "fsystem.h"
#include <stdlibcu.h>
#include <stdiocu.h>
#include <stringcu.h>
#include <ext/hash.h>
#include <errnocu.h>
#include <assert.h>

__BEGIN_DECLS;

// FILES
#pragma region FILES

typedef struct __align__(8) {
	file_t *file;			// reference
	unsigned short id;		// ID of author
	unsigned short threadid;// thread ID of author
} fileRef;

__device__ fileRef __iob_fileRefs[LIBCU_MAXFILESTREAM]; // Start of circular buffer (set up by host)
volatile __device__ fileRef *__iob_freeFilePtr = __iob_fileRefs; // Current atomically-incremented non-wrapped offset
volatile __device__ fileRef *__iob_retnFilePtr = __iob_fileRefs; // Current atomically-incremented non-wrapped offset
__constant__ file_t __iob_files[LIBCU_MAXFILESTREAM];

static __forceinline__ __device__ void writeFileRef(fileRef *ref, file_t *f) {
	ref->file = f;
	ref->id = gridDim.x*blockIdx.y + blockIdx.x;
	ref->threadid = blockDim.x*blockDim.y*threadIdx.z + blockDim.x*threadIdx.y + threadIdx.x;
}

static __device__ int fileGet(file_t **file) {
	// advance circular buffer
	size_t offset = atomicAdd((_uintptr_t *)&__iob_freeFilePtr, sizeof(fileRef)) - (size_t)&__iob_fileRefs;
	offset %= (sizeof(fileRef)*LIBCU_MAXFILESTREAM);
	int offsetId = offset / sizeof(fileRef);
	fileRef *ref = (fileRef *)((char *)&__iob_fileRefs + offset);
	file_t *f = ref->file;
	if (!f) {
		f = &__iob_files[offsetId];
		writeFileRef(ref, f);
	}
	*file = f;
	return GETFD(offsetId);
}

static __device__ void fileFree(int fd) {
	//if (!f) return;
	file_t *f = GETFILE(fd);
	// advance circular buffer
	size_t offset = atomicAdd((_uintptr_t *)&__iob_retnFilePtr, sizeof(fileRef)) - (size_t)&__iob_fileRefs;
	offset %= (sizeof(fileRef)*LIBCU_MAXFILESTREAM);
	fileRef *ref = (fileRef *)((char *)&__iob_fileRefs + offset);
	writeFileRef(ref, f);
}

#pragma endregion

__device__ char __cwd[MAX_PATH] = ""; // ":\\";
__device__ dirEnt_t __iob_root = {
#ifdef __APPLE__
{ 0, 0, 0, 0, 1, ":\\" }, { 0, 0, 0x4000 }, nullptr, nullptr
#else
{ 0, 0, 0, 1, ":\\" }, { 0, 0, 0x4000 }, nullptr, nullptr
#endif
};
static __device__ hash_t __iob_dir = HASHINIT;
static __device__ mode_t __umask = 0;

__device__ int expandPath(const char *path, char *newPath) {
	register unsigned char *d = (unsigned char *)newPath;
	register unsigned char *s;
	if (!path)
		panic("expandPath: null path");
	// add cwd
	if (path[0] != ':') {
		s = (unsigned char *)__cwd;
		if (path[0] != '\\' && path[0] != '/') { while (*s) *d++ = *s++; *d++ = '\\'; } // relative
		else *d++ = *s++; // absolute
	}
	// add path if not .
	if (path[0] != '.' && path[1] != 0) {
		s = (unsigned char *)path;
		int i = 0; int c;
		while (*s) {
			c = *s;
			if (c == '/') c = '\\'; // switch from unix path
			if (c == '\\') {
				// directory reached
				if (i == 2 && s[-1] == '.') d -= 2; // self directory
				else if (i == 3 && s[-1] == '.' && s[-2] == '.') { d -= 4; while (*d >= *newPath && *d != '\\') *d--; } // parent directory
				i = 0;
			}
			// advance
			*d++ = c; s++; i++;
		}
		// remove trailing '\.' && '\'
		d[c == '.' && i == 2 ? -2 : i == 1 ? -1 : 0] = 0;
	}
	else d[-1] = 0; // terminate if .
	return d - (unsigned char *)newPath;
}

static __device__ dirEnt_t *expandAndFindEnt(const char *path, char *newPath, int *pathLength = 0) {
	int len = expandPath(path, newPath);
	if (pathLength) *pathLength = len;
	dirEnt_t *ent = (newPath[0] == ':' && newPath[1] == '\\' && !newPath[2])
		? &__iob_root
		: (dirEnt_t *)hashFind(&__iob_dir, newPath);
	return ent;
}

static __device__ dirEnt_t *findDirInPath(const char *path, const char **file) {
	char *file2 = strrchr((char *)path, '\\');
	if (!file2) {
		_set_errno(EINVAL);
		return nullptr;
	}
	*file2 = 0;
	dirEnt_t *ent = (path[0] == ':' && !path[1])
		? &__iob_root
		: (dirEnt_t *)hashFind(&__iob_dir, path);
	*file2 = '\\';
	*file = file2 + 1;
	return ent;
}

static __device__ dirEnt_t *createEnt(dirEnt_t *parentEnt, const char *path, const char *name, int type, int extraSize) {
	dirEnt_t *ent = (dirEnt_t *)malloc(ROUND64_(sizeof(dirEnt_t)) + extraSize);
	char *newPath = (char *)malloc(strlen(path));
	strcpy(newPath, path);
	if (hashInsert(&__iob_dir, newPath, ent))
		panic("removed entity");
	ent->path = newPath;
	ent->dir.d_type = type;
	strcpy(ent->dir.d_name, name);
	// stat
	struct stat *stat = &ent->stat;
	memset(stat, 0, sizeof(struct stat));
	stat->st_mode = type;
#ifndef LIBCU_LEAN_AND_MEAN
	time(&stat->st_ctime);
	memcpy(&stat->st_atime, &stat->st_ctime, sizeof(time_t));
	memcpy(&stat->st_mtime, &stat->st_ctime, sizeof(time_t));
#endif
	// add to directory
	ent->next = parentEnt->u.list; parentEnt->u.list = ent;
	return ent;
}

static __device__ void freeEnt(dirEnt_t *ent) {
	if (ent->dir.d_type == DIRTYPE_DIR) {
		dirEnt_t *p = ent->u.list;
		while (p) {
			dirEnt_t *next = p->next;
			freeEnt(p);
			p = next;
		}
	}
	else if (ent->dir.d_type == DIRTYPE_FILE)
		memfileClose(ent->u.file);
	if (ent != &__iob_root) {
		hashInsert(&__iob_dir, ent->path, nullptr);
		free(ent->path);
		free(ent);
	}
	else __iob_root.u.list = nullptr;
}

__device__ mode_t fsystemUmask(mode_t mask) {
	mode_t r = __umask;
	__umask = mask;
	return r;
}

__device__ int fsystemChdir(const char *path) {
	char newPath[MAX_PATH]; expandPath(path, newPath);
	strncpy(__cwd, newPath, MAX_PATH);
	return 0;
}

__device__ dirEnt_t *fsystemOpendir(const char *path) {
	char newPath[MAX_PATH];
	dirEnt_t *ent = expandAndFindEnt(path, newPath);
	if (!ent || ent->dir.d_type != DIRTYPE_DIR) {
		_set_errno(!ent ? ENOENT : ENOTDIR);
		return nullptr;
	}
	return ent;
}

__device__ int fsystemRename(const char *old, const char *new_) {
	char oldPath[MAX_PATH], newPath[MAX_PATH]; int oldPathLength;
	dirEnt_t *ent = expandAndFindEnt(old, oldPath, &oldPathLength);
	if (!ent) {
		_set_errno(ENOENT);
		return -1;
	}
	register char *oldPathEnd = oldPath + oldPathLength - 1; while (*oldPathEnd && *oldPathEnd != '\\') oldPathEnd--;
	strcpy(oldPathEnd + 1, new_);
	//
	int newPathLength;
	dirEnt_t *ent2 = expandAndFindEnt(oldPath, newPath, &newPathLength);
	if (ent2) {
		_set_errno(EEXIST);
		return -1;
	}
	const char *name;
	dirEnt_t *parentEnt = findDirInPath(newPath, &name);
	if (!parentEnt) {
		_set_errno(ENOENT);
		return -1;
	}
	//
	char *newPath2 = (char *)malloc(newPathLength + 1);
	strcpy(newPath2, newPath);
	char *lastPath = ent->path;
	ent->path = newPath2;
	if (hashInsert(&__iob_dir, newPath2, ent))
		panic("removed entity");
	hashInsert(&__iob_dir, lastPath, nullptr);
	free(lastPath);
	return 0;
}

__device__ int fsystemUnlink(const char *path, bool enotdir) {
	char newPath[MAX_PATH];
	dirEnt_t *ent = expandAndFindEnt(path, newPath);
	if (!ent) {
		_set_errno(ENOENT);
		return -1;
	}
	const char *name;
	dirEnt_t *parentEnt = findDirInPath(newPath, &name);
	if (!parentEnt) {
		_set_errno(ENOENT);
		return -1;
	}

	// error if not directory
	if (enotdir && ent->dir.d_type != DIRTYPE_DIR) {
		_set_errno(ENOTDIR);
		return -1;
	}

	// directory not empty
	if (ent->dir.d_type == DIRTYPE_DIR && ent->u.list) {
		_set_errno(ENOENT);
		return -1;
	}

	// remove from directory
	dirEnt_t *list = parentEnt->u.list;
	if (list == ent)
		parentEnt->u.list = ent->next;
	else if (list) {
		dirEnt_t *p = list;
		while (p->next && p->next != ent)
			p = p->next;
		if (p->next == ent)
			p->next = ent->next;
	}

	// free entity
	freeEnt(ent);
	return 0;
}

static __device__ int stat__(dirEnt_t *ent, struct stat *buf) {
	memcpy(buf, &ent->stat, sizeof(*buf));
	return 0;
}

static __device__ int stat64__(dirEnt_t *ent, struct _stat64 *buf) {
	struct stat *estat = &ent->stat;
	buf->st_mode = estat->st_mode;
	buf->st_uid = estat->st_uid;
	buf->st_gid = estat->st_gid;
	buf->st_size = estat->st_size;
	buf->st_atime = estat->st_atime;
	buf->st_mtime = estat->st_mtime;
	buf->st_ctime = estat->st_ctime;
	return 0;
}

__device__ int fsystemStat(const char *path, struct stat *buf, struct _stat64 *buf64, bool lstat_) {
	char newPath[MAX_PATH];
	dirEnt_t *ent = expandAndFindEnt(path, newPath);
	if (!ent) {
		_set_errno(ENOENT);
		return -1;
	}
	return buf ? stat__(ent, buf) : stat64__(ent, buf64);
}

__device__ int fsystemFStat(int fd, struct stat *buf, struct _stat64 *buf64) {
	file_t *f = GETFILE(fd);
	if (!f) {
		_set_errno(ENOENT);
		return -1;
	}
	dirEnt_t *ent = (dirEnt_t *)f->base;
	return buf ? stat__(ent, buf) : stat64__(ent, buf64);
}

__device__ int fsystemChmod(const char *path, mode_t mode) {
	char newPath[MAX_PATH];
	dirEnt_t *ent = expandAndFindEnt(path, newPath);
	if (!ent) {
		_set_errno(ENOENT);
		return -1;
	}
	ent->stat.st_mode = mode;
	return 0;
}

__device__ dirEnt_t *fsystemMkdir(const char *__restrict path, int mode, int *r) {
	char newPath[MAX_PATH];
	dirEnt_t *ent = expandAndFindEnt(path, newPath);
	if (ent) {
		*r = 1;
		return ent;
	}
	const char *name;
	dirEnt_t *parentEnt = findDirInPath(newPath, &name);
	if (!parentEnt) {
		_set_errno(ENOENT);
		*r = -1;
		return nullptr;
	}
	// create directory
	ent = createEnt(parentEnt, newPath, name, DIRTYPE_DIR, 0);
	*r = 0;
	return ent;
}

__device__ dirEnt_t *fsystemMkfifo(const char *__restrict path, int mode, int *r) {
	char newPath[MAX_PATH];
	dirEnt_t *ent = expandAndFindEnt(path, newPath);
	if (ent) {
		*r = 1;
		return ent;
	}
	const char *name;
	dirEnt_t *parentEnt = findDirInPath(newPath, &name);
	if (!parentEnt) {
		_set_errno(ENOENT);
		*r = -1;
		return nullptr;
	}
	// create directory
	ent = createEnt(parentEnt, newPath, name, DIRTYPE_FIFO, 0);
	*r = 0;
	return ent;
}


__device__ dirEnt_t *fsystemAccess(const char *__restrict path, int mode, int *r) {
	char newPath[MAX_PATH];
	dirEnt_t *ent = expandAndFindEnt(path, newPath);
	if (!ent) {
		_set_errno(ENOENT);
		*r = -1;
		return nullptr;
	}
	//if ((mode & 2) && false) {
	//	_set_errno(EACCES);
	//	*r = -1;
	//	return ent;
	//}
	*r = 0;
	return ent;
}

__device__ dirEnt_t *fsystemOpen(const char *__restrict path, int mode, int *fd) {
	char newPath[MAX_PATH];
	dirEnt_t *ent = expandAndFindEnt(path, newPath);
	if (ent) {
		if (mode & O_TRUNC)
			memfileTruncate(ent->u.file, 0);
		file_t *f; *fd = fileGet(&f);
		f->base = (char *)ent;
		return ent;
	}
	if ((mode & 0xF) == O_RDONLY) {
		_set_errno(EINVAL); // So illegal mode.
		*fd = -1;
		return nullptr;
	}
	const char *name;
	dirEnt_t *parentEnt = findDirInPath(newPath, &name);
	if (!parentEnt) {
		_set_errno(ENOENT);
		*fd = -1;
		return nullptr;
	}
	// create file
	ent = createEnt(parentEnt, newPath, name, DIRTYPE_FILE, memfileSize(nullptr));
	ent->u.file = (vsysfile *)((char *)ent + ROUND64_(sizeof(dirEnt_t)));
	memfileMemOpen(ent->u.file);
	// set to file
	file_t *f; *fd = fileGet(&f);
	f->base = (char *)ent;
	return ent;
}

__device__ void fsystemClose(int fd) {
	file_t *f = GETFILE(fd);
	if (f->flag & DELETE) {
		dirEnt_t *ent = (dirEnt_t *)f->base;
		fsystemUnlink(ent->path, false);
	}
	fileFree(fd);
}

__device__ void fsystemReset() {
	freeEnt(&__iob_root);
	strcpy(__cwd, ":\\");
}

__device__ void fsystemSetFlag(int fd, int flag) {
	file_t *f = GETFILE(fd);
	f->flag |= flag;
}

__END_DECLS;
#endif