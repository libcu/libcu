/*
stdio.h - definitions/declarations for standard I/O routines
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

#ifndef _STDIOCU_H
#define _STDIOCU_H
#include <crtdefscu.h>

#include <stdio.h>
typedef struct {
	char *_base;
	int _flag;
	int _file;
	int off;
	int state;
} cuFILE;
#ifdef __APPLE__
#define __off_t off_t
// #define __off64_t off64_t
#endif
#ifdef __CUDA_ARCH__
#include <stdarg.h>

__BEGIN_DECLS;

/* IsHost support  */
#define ISHOSTFILE(stream) ((cuFILE*)(stream) < __iob_streams || (cuFILE*)(stream) > __iob_streams + LIBCU_MAXFILESTREAM+3)
extern __constant__ cuFILE __iob_streams[LIBCU_MAXFILESTREAM + 3];
#undef stdin
#undef stdout
#undef stderr
#define stdin  ((FILE*)&__iob_streams[0]) /* Standard input stream.  */
#define stdout ((FILE*)&__iob_streams[1]) /* Standard output stream.  */
#define stderr ((FILE*)&__iob_streams[2]) /* Standard error stream.  */

__BEGIN_NAMESPACE_STD;
/* Remove file FILENAME.  */
extern __device__ int remove_(const char *filename);
#define remove remove_
/* Rename file OLD to NEW.  */
extern  __device__ int rename_(const char *old, const char *new_);
#define rename rename_
__END_NAMESPACE_STD;

__BEGIN_NAMESPACE_STD;
// mktemp
/* Create a temporary file and open it read/write. */
#ifndef __USE_FILE_OFFSET64
extern __device__ FILE *tmpfile_(void);
#define tmpfile tmpfile_
#else
#define tmpfile tmpfile64
#endif
__END_NAMESPACE_STD;

__BEGIN_NAMESPACE_STD;
/* Close STREAM. */
extern __device__ int fclose_(FILE *stream, bool wait = true);
#define fclose fclose_
/* Flush STREAM, or all streams if STREAM is NULL. */
extern __device__ int fflush_(FILE *stream, bool wait = true);
#define fflush fflush_
__END_NAMESPACE_STD;

__BEGIN_NAMESPACE_STD;
#ifndef __USE_FILE_OFFSET64
/* Open a file, replacing an existing stream with it. */
extern __device__ FILE *freopen_(const char *__restrict filename, const char *__restrict modes, FILE *__restrict stream);
#define freopen freopen_
/* Open a file and create a new stream for it. */
extern __device__ FILE *fopen_(const char *__restrict filename, const char *__restrict modes);
#define fopen fopen_
#else
#define fopen fopen64
#define freopen freopen64
#endif
__END_NAMESPACE_STD;
#ifdef __USE_LARGEFILE64
/* Open a file, replacing an existing stream with it. */
extern __device__ FILE *freopen64_(const char *__restrict filename, const char *__restrict modes, FILE *__restrict stream);
#define freopen64 freopen64_
/* Open a file and create a new stream for it. */
extern __device__ FILE *fopen64_(const char *__restrict filename, const char *__restrict modes);
#define fopen64 fopen64_
#endif

__BEGIN_NAMESPACE_STD;
/* Make STREAM use buffering mode MODE. If BUF is not NULL, use N bytes of it for buffering; else allocate an internal buffer N bytes long.  */
extern __device__ int setvbuf_(FILE *__restrict stream, char *__restrict buf, int modes, size_t n);
#define setvbuf setvbuf_
/* If BUF is NULL, make STREAM unbuffered. Else make it use buffer BUF, of size BUFSIZ.  */
extern __device__ void setbuf_(FILE *__restrict stream, char *__restrict buf);
#define setbuf setbuf_
__END_NAMESPACE_STD;

__BEGIN_NAMESPACE_C99;
/* Maximum chars of output to write in MAXLEN.  */
extern __device__ int snprintf_(char *__restrict s, size_t maxlen, const char *__restrict format, ...);
#define snprintf snprintf_
extern __device__ int vsnprintf_(char *__restrict s, size_t maxlen, const char *__restrict format, va_list va);
#define vsnprintf vsnprintf_
__END_NAMESPACE_C99;

__BEGIN_NAMESPACE_STD;
/* Write formatted output to STREAM. */
extern __device__ int fprintf_(FILE *__restrict stream, const char *__restrict format, ...);
#define fprintf fprintf_
/* Write formatted output to stdout. */
//builtin: extern __device__ int printf_(const char *__restrict format, ...);
/* Write formatted output to S.  */
#define sprintf(s, format, ...) snprintf_(s, 0xffffffff, format, __VA_ARGS__)
//extern __device__ int sprintf_(char *__restrict s, const char *__restrict format, ...);
//#define sprintf sprintf_

/* Write formatted output to S from argument list ARG. */
extern __device__ int vfprintf_(FILE *__restrict s, const char *__restrict format, va_list va, bool wait = true);
#define vfprintf vfprintf_
/* Write formatted output to stdout from argument list ARG. */
//builtin: __forceinline__ __device__ int vprintf_(const char *__restrict format, va_list va) { return vfprintf(stdout, format, va, true); };
/* Write formatted output to S from argument list ARG.  */
__forceinline__ __device__ int vsprintf_(char *__restrict s, const char *__restrict format, va_list va) { return vsnprintf(s, 0xffffffff, format, va); }
#define vsprintf vsprintf_

__END_NAMESPACE_STD;

__BEGIN_NAMESPACE_STD;
/* Read formatted input from STREAM.  */
extern __device__ int fscanf_(FILE *__restrict stream, const char *__restrict format, ...);
#define fscanf fscanf_
/* Read formatted input from stdin.  */
extern __device__ int scanf_(const char *__restrict format, ...);
#define scanf scanf_
/* Read formatted input from S.  */
extern __device__ int sscanf_(const char *__restrict s, const char *__restrict format, ...);
#define sscanf sscanf_
__END_NAMESPACE_STD;

__BEGIN_NAMESPACE_C99;
/* Read formatted input from S into argument list ARG.  */
extern __device__ int vfscanf_(FILE *__restrict s, const char *__restrict format, va_list va, bool wait = true);
#define vfscanf vfscanf_
/* Read formatted input from stdin into argument list ARG. */
extern __device__ int vscanf_(const char *__restrict format, va_list va);
#define vscanf vscanf_
/* Read formatted input from S into argument list ARG.  */
extern __device__ int vsscanf_(const char *__restrict s, const char *__restrict format, va_list va);
#define vsscanf vsscanf_
__END_NAMESPACE_C99;

__BEGIN_NAMESPACE_STD;
/* Read a character from STREAM.  */
extern __device__ int fgetc_(FILE *stream);
#define fgetc fgetc_
#undef getc
#define getc(stream) fgetc(stream)
/* Read a character from stdin.  */
__forceinline__ __device__ int getchar_(void) { return fgetc(stdin); }
#define getchar getchar_
__END_NAMESPACE_STD;

/* The C standard explicitly says this is a macro, so we always do the optimization for it.  */
//sky: #define getc(fp) __GETC(fp)

__BEGIN_NAMESPACE_STD;
/* Write a character to STREAM.  */
extern __device__ int fputc_(int c, FILE *stream, bool wait = true);
#define fputc fputc_
#undef putc
#define putc(c, stream) fputc(c, stream)
/* Write a character to stdout.  */
__forceinline__ __device__ int putchar_(int c) { return fputc(c, stdout); }
#define putchar putchar_
__END_NAMESPACE_STD;

/* The C standard explicitly says this can be a macro, so we always do the optimization for it.  */
//sky: #define putc(ch, fp) __PUTC(ch, fp)

__BEGIN_NAMESPACE_STD;
/* Get a newline-terminated string of finite length from STREAM.  */
extern __device__ char *fgets_(char *__restrict s, int n, FILE *__restrict stream);
#define fgets fgets_
__END_NAMESPACE_STD;

__BEGIN_NAMESPACE_STD;
/* Write a string to STREAM.  */
extern __device__ int fputs_(const char *__restrict s, FILE *__restrict stream, bool wait = true);
#define fputs fputs_

/* Write a string, followed by a newline, to stdout.  */
//extern __device__ int puts(const char *s);
__forceinline__ __device__ int puts_(const char *s) { fputs(s, stdout); return fputs("\n", stdout); }
#define puts puts_

/* Push a character back onto the input buffer of STREAM.  */
extern __device__ int ungetc_(int c, FILE *stream, bool wait = true);
#define ungetc ungetc_

/* Read chunks of generic data from STREAM.  */
extern __device__ size_t fread_(void *__restrict ptr, size_t size, size_t n, FILE *__restrict stream, bool wait = true);
#define fread fread_
/* Write chunks of generic data to STREAM.  */
extern __device__ size_t fwrite_(const void *__restrict ptr, size_t size, size_t n, FILE *__restrict stream, bool wait = true);
#define fwrite fwrite_
__END_NAMESPACE_STD;

__BEGIN_NAMESPACE_STD;
/* Seek to a certain position on STREAM.  */
extern __device__ int fseek_(FILE *stream, long int off, int whence);
#define fseek fseek_
/* Return the current position of STREAM.  */
extern __device__ long int ftell_(FILE *stream);
#define ftell ftell_
/* Rewind to the beginning of STREAM.  */
extern __device__ void rewind_(FILE *stream);
#define rewind rewind_
__END_NAMESPACE_STD;

/* The Single Unix Specification, Version 2, specifies an alternative,
more adequate interface for the two functions above which deal with
file offset.  `long int' is not the right type.  These definitions
are originally defined in the Large File Support API.  */
#if defined(__USE_LARGEFILE)
#ifndef __USE_FILE_OFFSET64
/* Seek to a certain position on STREAM.   */
extern __device__ int fseeko_(FILE *stream, __off_t off, int whence);
#define fseeko fseeko_
/* Return the current position of STREAM.  */
extern __device__ __off_t ftello_(FILE *stream);
#define ftello ftello_
#else
#define fseeko fseeko64
#define ftello ftello64
#endif
#endif

__BEGIN_NAMESPACE_STD;
#ifndef __USE_FILE_OFFSET64
/* Get STREAM's position.  */
extern __device__ int fgetpos_(FILE *__restrict stream, fpos_t *__restrict pos);
#define fgetpos fgetpos_
/* Set STREAM's position.  */
extern __device__ int fsetpos_(FILE *stream, const fpos_t *pos);
#define fsetpos fsetpos_
#else
#define fgetpos fgetpos64
#define fsetpos fsetpos64
#endif
__END_NAMESPACE_STD;

#ifdef __USE_LARGEFILE64
extern __device__ int fseeko64_(FILE *stream, __off64_t off, int whence);
#define fseeko64 fseeko64_
extern __device__ __off64_t ftello64_(FILE *stream);
#define ftello64 ftello64_
extern __device__ int fgetpos64_(FILE *__restrict stream, fpos64_t *__restrict pos);
#define fgetpos64 fgetpos64_
extern __device__ int fsetpos64_(FILE *stream, const fpos64_t *pos);
#define fsetpos64 fsetpos64_
#endif

__BEGIN_NAMESPACE_STD;
/* Clear the error and EOF indicators for STREAM.  */
extern __device__ void clearerr_(FILE *stream);
#define clearerr clearerr_
/* Return the EOF indicator for STREAM.  */
extern __device__ int feof_(FILE *stream);
#define feof feof_
/* Return the error indicator for STREAM.  */
extern __device__ int ferror_(FILE *stream);
#define ferror ferror_
__END_NAMESPACE_STD;

__BEGIN_NAMESPACE_STD;
/* Print a message describing the meaning of the value of errno. */
extern __device__ void fperror_(FILE *stream, const char *s); //: ext
#define fperror fperror_
/* Print a message describing the meaning of the value of errno. */
extern __device__ void perror_(const char *s);
#define perror perror_
__END_NAMESPACE_STD;

/* Return the system file descriptor for STREAM.  */
extern __device__ int fileno_(FILE *stream);
#define fileno fileno_

/* If we are compiling with optimizing read this file.  It contains
several optimizing inline functions and macros.  */
#ifdef __LIBCUx__
#define fgetc(fp)                   __FGETC(fp)
#define fputc(ch, fp)				__FPUTC(ch, fp)
#define getchar()                   __GETC(__stdin)
#define putchar(ch)                 __PUTC((ch), __stdout)
/* Clear the error and EOF indicators for STREAM.  */
#define clearerr(fp)                __CLEARERR(fp)
#define feof(fp)                    __FEOF(fp)
#define ferror(fp)                  __FERROR(fp)
#endif

__END_DECLS;

#else
#define fperror(s, e) 0
#define ISHOSTFILE(stream) false
#if __OS_WIN
#define snprintf _snprintf
#define fprintf_ fprintf
#elif __OS_UNIX
#define fprintf_ fprintf
#endif
#endif  /* __CUDA_ARCH__ */

#endif  /* _STDIOCU_H */