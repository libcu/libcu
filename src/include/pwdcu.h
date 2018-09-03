/*
pwd.h - Password structure
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
#ifndef _PWDCU_H
#define _PWDCU_H
#include <crtdefscu.h>

#if __OS_WIN
#include <sys/types.h>
#include <grpcu.h>
#define uid_t short
struct passwd {
	char *pw_name;		// user's login name
	uid_t pw_uid;		// numerical user ID
	gid_t pw_gid;		// numerical group ID
	//char *pw_dir;		// initial working directory
	//char *pw_shell;	// program to use as shell
};
#elif __OS_UNIX
#include <pwd.h>
#endif

#if defined(__CUDA_ARCH__)
__BEGIN_DECLS;

/* search user database for a user ID */
extern __device__ struct passwd *getpwuid_(uid_t uid);
#define getpwuid getpwuid_

/* search user database for a name */
extern __device__ struct passwd *getpwnam_(const char *name);
#define getpwnam getpwnam_

/* get user database entry */
extern __device__ struct passwd *getpwent_();
#define getpwent getpwent_

/* close the user database */
/* setpwent - reset user database to first entry */
extern __device__ void endpwent_();
#define endpwent endpwent_
#define setpwent endpwent_

__END_DECLS;
#else
#define getpwuid(uid) nullptr
#define getpwnam(name) nullptr
#define getpwent() nullptr
#define endpwent()
#define setpwent()
#endif  /* __CUDA_ARCH__ */

#endif  /* _PWDCU_H */
