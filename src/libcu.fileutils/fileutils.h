#ifndef _FILEUTILS_H
#define _FILEUTILS_H

#define	PATHLEN 256

#ifdef S_ISLNK
#define	LSTAT lstat
#else
#define	LSTAT stat
#endif

extern __device__ bool copyFile(char *srcName, char *destName, bool setModes);

#endif  /* _FILEUTILS_H */