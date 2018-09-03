#include <direntcu.h>
#include <sentinel-direntmsg.h>
#include "fsystem.h"

struct cuDIR {
#if __OS_WIN
	struct DIR dir;
#elif __OS_UNIX
	//	struct __dirstream dir;
#endif
	void *listIdx; void *list;
	int fakeIdx; int fake;
};

__constant__ const struct dirent _dirpFakes[2] = { { 0, 0, 2, 0, ".." }, { 0, 0, 1, 0, "." } };

/* Open a directory stream on NAME. Return a DIR stream on the directory, or NULL if it could not be opened. */
__device__ DIR *opendir_(const char *name) {
	if (ISHOSTPATH(name)) { dirent_opendir msg(name); return newhostptr<DIR>(msg.RC); }
	dirEnt_t *ent = fsystemOpendir(name);
	if (!ent) return nullptr;
	cuDIR *dirp = (cuDIR *)malloc(sizeof(cuDIR));
	memcpy(dirp, ent, sizeof(dirEnt_t));
	// set first file
	dirp->list = dirp->listIdx = ent->u.list;
	dirp->fake = dirp->fakeIdx = ent != &__iob_root ? 2 : 0;
	return (DIR *)dirp;
}

/* Close the directory stream DIRP. Return 0 if successful, -1 if not.  */
__device__ int closedir_(DIR *dirp) {
	if (ISHOSTPTR(dirp)) { dirent_closedir msg(hostptr<DIR>(dirp)); freehostptr<DIR>(dirp); return msg.RC; }
	if (!dirp) return -1;
	free(dirp);
	return 0;
}

/* Read a directory entry from DIRP.  Return a pointer to a `struct dirent' describing the entry, or NULL for EOF or error.  The
storage returned may be overwritten by a later readdir call on the same DIR stream.

If the Large File Support API is selected we have to use the appropriate interface.  */
__device__ struct dirent *readdir_(DIR *dirp) {
	if (ISHOSTPTR(dirp)) { dirent_readdir msg(hostptr<DIR>(dirp), false); return msg.RC; }
	cuDIR *p = (cuDIR *)dirp;
	if (!p || !p->listIdx) return nullptr;
	if (p->fakeIdx)
		return (struct dirent *)&_dirpFakes[--p->fakeIdx];
	memcpy(p, p->listIdx, sizeof(dirent));
	p->listIdx = ((dirEnt_t *)p->listIdx)->next;
	return (struct dirent *)p;
}

#ifdef __USE_LARGEFILE64
__device__ struct dirent64 *readdir64_(DIR *dirp) {
	if (ISHOSTPTR(dirp)) { dirent_readdir msg(hostptr<DIR>(dirp), true); return msg.RC64; }
	cuDIR *p = (cuDIR *)dirp;
	if (!p || !p->listIdx) return nullptr;
	if (p->fakeIdx)
		return (struct dirent64 *)&_dirpFakes[--p->fakeIdx];
	memcpy(p, p->listIdx, sizeof(dirent));
	p->listIdx = ((dirEnt_t *)p->listIdx)->next;
	return (struct dirent64 *)p;
}
#endif

/* Rewind DIRP to the beginning of the directory.  */
__device__ void rewinddir_(DIR *dirp) {
	if (ISHOSTPTR(dirp)) { dirent_rewinddir msg(hostptr<DIR>(dirp)); return; }
	cuDIR *p = (cuDIR *)dirp;
	memcpy(p, p->list, sizeof(dirent));
	p->listIdx = (dirEnt_t *)p->list;
	p->fakeIdx = p->fake;
}

#if 0
#ifdef _MSC_VER

DIR *opendir(const char *name) {
	DIR *dir = nullptr;
	if (name && name[0]) {
		size_t base_length = strlen(name);
		const char *all = strchr("/\\", name[base_length - 1]) ? "*" : "/*"; // search pattern must end with suitable wildcard
		if ((dir = (DIR *)malloc(sizeof(*dir))) != 0 && (dir->name = (char *)malloc((int)(base_length + strlen(all) + 1))) != 0) {
			strcat(strcpy(dir->name, name), all);
			if ((dir->handle = (long)_findfirst(dir->name, &dir->info)) != -1)
				dir->result.d_name = 0;
			else { // rollback
				free(dir->name);
				free(dir);
				dir = nullptr;
			}
		}
		else { // rollback
			free(dir);
			dir = 0;
			errno = ENOMEM;
		}
	}
	else
		errno = EINVAL;
	return dir;
}

int closedir(DIR *dir) {
	int result = -1;
	if (dir) {
		if (dir->handle != -1)
			result = _findclose(dir->handle);
		free(dir->name);
		free(dir);
	}
	if (result == -1) // map all errors to EBADF
		errno = EBADF;
	return result;
}

struct dirent *readdir(DIR * dir) {
	struct dirent *result = nullptr;
	if (dir && dir->handle != -1) {
		if (!dir->result.d_name || _findnext(dir->handle, &dir->info) != -1) {
			result = &dir->result;
			result->d_name = dir->info.name;
		}
	}
	else
		errno = EBADF;
	return result;
}

#endif
#endif