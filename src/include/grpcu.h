/*
grp.h - Group structure
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
#ifndef _GRPCU_H
#define _GRPCU_H
#include <crtdefscu.h>

#if __OS_WIN
#define gid_t short
struct group {
	char *gr_name;		// the name of the group
	gid_t gr_gid;		// numerical group ID
	char  **gr_mem;		// pointer to a null-terminated array of character pointers to member names
};
#elif __OS_UNIX
#include <grp.h>
#endif

#if defined(__CUDA_ARCH__)
__BEGIN_DECLS;

/* get group database entry for a group ID */
extern __device__ struct group *getgrgid_(gid_t gid);
#define getgrgid getgrgid_

/* search group database for a name */
extern __device__ struct group *getgrnam_(const char *name);
#define getgrnam getgrnam_

/* get the group database entry */
extern __device__ struct group *getgrent_();
#define getgrent getgrent_

/* close the group database */
/* setgrent - reset group database to first entry */
extern __device__ void endgrent_();
#define endgrent endgrent_
#define setgrent endgrent_

__END_DECLS;
#else
#define getgrgid(gid) nullptr
#define getgrnam(name) nullptr
#define getgrent() nullptr
#define endgrent()
#define setgrent()
#endif  /* __CUDA_ARCH__ */

#endif  /* _GRPCU_H */
