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

__device__ char __cwd[MAX_PATH] = ":\\";
__device__ dirEnt_t __iob_root = { { 0, 0, 0, 1, ":\\" }, nullptr, nullptr };
__device__ hash_t __iob_dir = HASHINIT;

__device__ int expandPath(const char *path, char *newPath) {
	register unsigned char *d = (unsigned char *)newPath;
	register unsigned char *s;
	// add cwd
	if (path[0] != ':') {
		s = (unsigned char *)__cwd;
		if (path[0] != '\\' && path[0] != '/') { while (*s) *d++ = *s++; *d++ = '\\'; } // relative
		else *d++ = *s++; // absolute
	}
	// add path
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
	return d - (unsigned char *)newPath;
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
	// add to directory
	ent->next = parentEnt->u.list; parentEnt->u.list = ent;
	return ent;
}

static __device__ void freeEnt(dirEnt_t *ent) {
	if (ent->dir.d_type == 1) {
		dirEnt_t *p = ent->u.list;
		while (p) {
			dirEnt_t *next = p->next;
			freeEnt(p);
			p = next;
		}
	}
	else if (ent->dir.d_type == 2)
		memfileClose(ent->u.file);
	if (ent != &__iob_root) {
		hashInsert(&__iob_dir, ent->path, nullptr);
		free(ent->path);
		free(ent);
	}
	else __iob_root.u.list = nullptr;
}

static __device__ dirEnt_t *findDir(const char *path) {
	dirEnt_t *ent = !strcmp(path, ":")
		? &__iob_root
		: (dirEnt_t *)hashFind(&__iob_dir, path);
	return ent;
}

static __device__ dirEnt_t *findDirInPath(const char *path, const char **file) {
	char *file2 = strrchr((char *)path, '\\');
	if (!file2) {
		_set_errno(EINVAL);
		return nullptr;
	}
	*file2 = 0;
	dirEnt_t *ent = !strcmp(path, ":")
		? &__iob_root
		: (dirEnt_t *)hashFind(&__iob_dir, path);
	*file2 = '\\';
	*file = file2 + 1;
	return ent;
}

__device__ int fsystemChdir(const char *path) {
	char newPath[MAX_PATH]; expandPath(path, newPath);
	strncpy(__cwd, newPath, MAX_PATH);
	return 0;
}

__device__ dirEnt_t *fsystemOpendir(const char *path) {
	char newPath[MAX_PATH]; expandPath(path, newPath);
	dirEnt_t *ent = findDir(newPath);
	if (!ent || ent->dir.d_type != 1) {
		_set_errno(!ent ? ENOENT : ENOTDIR);
		return nullptr;
	}
	return ent;
}

__device__ int fsystemRename(const char *old, const char *new_) {
	char oldPath[MAX_PATH], newPath[MAX_PATH]; int oldPathLength = expandPath(old, oldPath);
	dirEnt_t *ent = (dirEnt_t *)hashFind(&__iob_dir, oldPath);
	if (!ent) {
		_set_errno(ENOENT);
		return -1;
	}
	register char *oldPathEnd = oldPath + oldPathLength - 1; while (*oldPathEnd && *oldPathEnd != '\\') oldPathEnd--;
	strcpy(oldPathEnd + 1, new_);
	int newPathLength = expandPath(oldPath, newPath);
	//
	dirEnt_t *ent2 = (dirEnt_t *)hashFind(&__iob_dir, newPath);
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
	char newPath[MAX_PATH]; expandPath(path, newPath);
	dirEnt_t *ent = (dirEnt_t *)hashFind(&__iob_dir, newPath);
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
	if (enotdir && ent->dir.d_type != 1) {
		_set_errno(ENOTDIR);
		return -1;
	}

	// directory not empty
	if (ent->dir.d_type == 1 && ent->u.list) {
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

__device__ dirEnt_t *fsystemMkdir(const char *__restrict path, int mode, int *r) {
	char newPath[MAX_PATH]; expandPath(path, newPath);
	dirEnt_t *dirEnt = (dirEnt_t *)hashFind(&__iob_dir, newPath);
	if (dirEnt) {
		*r = 1;
		return dirEnt;
	}
	const char *name;
	dirEnt_t *parentEnt = findDirInPath(newPath, &name);
	if (!parentEnt) {
		_set_errno(ENOENT);
		*r = -1;
		return nullptr;
	}
	// create directory
	dirEnt = createEnt(parentEnt, newPath, name, 1, 0);
	*r = 0;
	return dirEnt;
}

__device__ dirEnt_t *fsystemAccess(const char *__restrict path, int mode, int *r) {
	char newPath[MAX_PATH]; expandPath(path, newPath);
	dirEnt_t *dirEnt = (dirEnt_t *)hashFind(&__iob_dir, newPath);
	if (!dirEnt) {
		_set_errno(ENOENT);
		*r = -1;
		return nullptr;
	}
	//if ((mode & 2) && false) {
	//	_set_errno(EACCES);
	//	*r = -1;
	//	return dirEnt;
	//}
	*r = 0;
	return dirEnt;
}

__device__ dirEnt_t *fsystemOpen(const char *__restrict path, int mode, int *fd) {
	char newPath[MAX_PATH]; expandPath(path, newPath);
	dirEnt_t *fileEnt = (dirEnt_t *)hashFind(&__iob_dir, newPath);
	if (fileEnt) {
		if (mode & O_TRUNC)
			memfileTruncate(fileEnt->u.file, 0);
		file_t *f; *fd = fileGet(&f);
		f->base = (char *)fileEnt;
		return fileEnt;
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
	fileEnt = createEnt(parentEnt, newPath, name, 2, memfileSize(nullptr));
	fileEnt->u.file = (vsysfile *)((char *)fileEnt + ROUND64_(sizeof(dirEnt_t)));
	memfileMemOpen(fileEnt->u.file);
	// set to file
	file_t *f; *fd = fileGet(&f);
	f->base = (char *)fileEnt;
	return fileEnt;
}

__device__ void fsystemClose(int fd) {
	file_t *f = GETFILE(fd);
	if (f->flag & DELETE) {
		dirEnt_t *fileEnt = (dirEnt_t *)f->base;
		fsystemUnlink(fileEnt->path, false);
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