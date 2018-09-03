/*
dirent.h - Directory Entities
The MIT License

Copyright (c) 2016 Sky Morey

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

//#pragma once
#ifndef _DIRENTCU_H
#define	_DIRENTCU_H
#include <crtdefscu.h>

#if __OS_WIN
#include <_dirent.h>
#elif __OS_UNIX
#include <dirent.h>
#endif
#if defined(__CUDA_ARCH__)
__BEGIN_DECLS;

/* Open a directory stream on NAME. Return a DIR stream on the directory, or NULL if it could not be opened. */
extern __device__ DIR *opendir_(const char *name);
#define opendir opendir_

/* Close the directory stream DIRP. Return 0 if successful, -1 if not.  */
extern __device__ int closedir_(DIR *dirp);
#define closedir closedir_

/* Read a directory entry from DIRP.  Return a pointer to a `struct dirent' describing the entry, or NULL for EOF or error.  The
storage returned may be overwritten by a later readdir call on the same DIR stream.

If the Large File Support API is selected we have to use the appropriate interface.  */
#ifndef __USE_FILE_OFFSET64
extern __device__ struct dirent *readdir_(DIR *dirp);
#define readdir readdir_
#else
#define readdir readdir64_
#endif
#ifdef __USE_LARGEFILE64
extern __device__ struct dirent64 *readdir64_(DIR *dirp);
#define readdir64 readdir64_
#endif

/* Rewind DIRP to the beginning of the directory.  */
extern __device__ void rewinddir_(DIR *dirp);
#define rewinddir rewinddir_

__END_DECLS;
#endif  /* __CUDA_ARCH__ */

#endif  /* _DIRENTCU_H */