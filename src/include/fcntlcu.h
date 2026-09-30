/*
fcntl.h - File control
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

#ifndef _FCNTLCU_H
#define _FCNTLCU_H
#include <crtdefscu.h>
#include <sys/statcu.h>

#include <fcntl.h>
#if __OS_WIN
#include <io.h>
#endif
#ifdef __CUDA_ARCH__
#include <stdarg.h>
__BEGIN_DECLS;

/* Do the file control operation described by CMD on FD. The remaining arguments are interpreted depending on CMD. */
#ifndef __USE_FILE_OFFSET64
extern __device__ int fcntl_(int fd, int cmd, ...);
#define fcntl fcntl_
extern __device__ int vfcntl_(int fd, int cmd, va_list va);
#define vfcntlv vfcntl_
#else
#define fcntl fcntl64_
#define vfcntlv vfcntl64_
#endif
#ifdef __USE_LARGEFILE64
extern __device__ int fcntl64_(int fd, int cmd, ...);
#define fcntl64 fcntl64_
extern __device__ int vfcntl64_(int fd, int cmd, va_list va);
#define vfcntl64v vfcntl64_
#endif

/* Open FILE and return a new file descriptor for it, or -1 on error. OFLAG determines the type of access used.  If O_CREAT is on OFLAG,
   the third argument is taken as a `mode_t', the mode of the created file. */
#ifndef __USE_FILE_OFFSET64
extern __device__ int open_(const char *file, int oflag, ...);
#define open open_
extern __device__ int vopen_(const char *file, int oflag, va_list va);
#define vopen vopen_
#else
#define open open64_
#define vopen vopen64_
#endif
#ifdef __USE_LARGEFILE64
extern __device__ int open64_(const char *file, int oflag, ...);
#define open64 open64_
extern __device__ int open64v_(const char *file, int oflag, va_list va);
#define vopen64 vopen64_
#endif

/* Create and open FILE, with mode MODE.  This takes an `int' MODE argument because that is what `mode_t' will be widened to. */
#define creat(file, mode) open_(file, O_WRONLY|O_CREAT|O_TRUNC, mode)
#ifdef __USE_LARGEFILE64
#define creat64(file, mode) open64_(file, O_WRONLY|O_CREAT|O_TRUNC, mode)
#endif

__END_DECLS;

#endif  /* __CUDA_ARCH__ */

#endif  /* _FCNTLCU_H */
