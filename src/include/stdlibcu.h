/*
stdlibcu.h - declarations/definitions for commonly used library functions
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

#ifndef _STDLIBCU_H
#define _STDLIBCU_H
#include <crtdefscu.h>

//* Shorthand for type of comparison functions.  */
#ifndef __COMPAR_FN_T
#define __COMPAR_FN_T
typedef int(*__compar_fn_t)(const void *, const void *);
#endif

#include <stdlib.h>
#if __OS_UNIX
#include <unistd.h>
#endif
#ifdef __CUDA_ARCH__
__BEGIN_DECLS;

extern __device__ unsigned long __strtol(register const char *__restrict str, char **__restrict endptr, int base, int sflag);
#if defined(ULLONG_MAX)
extern __device__ unsigned long long __strtoll(register const char *__restrict str, char **__restrict endptr, int base, int sflag);
#endif

//__BEGIN_NAMESPACE_STD;
///* Returned by `div'.  */
//typedef struct {
//	int quot;			/* Quotient.  */
//	int rem;			/* Remainder.  */
//} div_t;
///* Returned by `ldiv'.  */
//typedef struct {
//	long int quot;		/* Quotient.  */
//	long int rem;		/* Remainder.  */
//} ldiv_t;
//__END_NAMESPACE_STD;
//
//#if defined(ULLONG_MAX)
//__BEGIN_NAMESPACE_C99;
///* Returned by `lldiv'.  */
//typedef struct {
//	long long int quot;		/* Quotient.  */
//	long long int rem;		/* Remainder.  */
//} lldiv_t;
//__END_NAMESPACE_C99;
//#endif
//
///* The largest number rand will return (same as INT_MAX).  */
//#define	RAND_MAX	2147483647
//
///* We define these the same for all machines. Changes from this to the outside world should be done in `_exit'.  */
//#define	EXIT_FAILURE	1	/* Failing exit status.  */
//#define	EXIT_SUCCESS	0	/* Successful exit status.  */

__BEGIN_NAMESPACE_STD;
/* prototype */
extern __device__ double strtod_(const char *__restrict nptr, char **__restrict endptr);

/* Convert a string to a floating-point number.  */
__forceinline__ __device__ double atof_(const char *nptr) { return strtod_(nptr, NULL); }
#define atof atof_
/* Convert a string to an integer.  */
__forceinline__ __device__ int atoi_(const char *nptr) { return (int)__strtol(nptr, (char **)NULL, 10, 1); }
#define atoi atoi_
/* Convert a string to a long integer.  */
__forceinline__ __device__ long int atol_(const char *nptr) { return __strtol(nptr, (char **)NULL, 10, 1); }
#define atol atol_
__END_NAMESPACE_STD;

#if defined(ULLONG_MAX)
__BEGIN_NAMESPACE_C99;
/* Convert a string to a long long integer.  */
__forceinline__ __device__ long long int atoll_(const char *nptr) { return __strtoll(nptr, (char **)NULL, 10, 1); }
#define atoll atoll_
__END_NAMESPACE_C99;
#endif

__BEGIN_NAMESPACE_STD;
/* Convert a string to a floating-point number.  */
extern __device__ double strtod_(const char *__restrict nptr, char **__restrict endptr);
#define strtod strtod_
__END_NAMESPACE_STD;

__BEGIN_NAMESPACE_C99;
/* Likewise for `float' and `long double' sizes of floating-point numbers.  */
extern __device__ float strtof_(const char *__restrict nptr, char **__restrict endptr);
#define strtof strtof_
//extern __device__ long double strtold_(const char *__restrict nptr, char **__restrict endptr);
#define strtold strtof_
__END_NAMESPACE_C99;

__BEGIN_NAMESPACE_STD;
/* Convert a string to a long integer.  */
__forceinline__ __device__ long int strtol_(const char *__restrict nptr, char **__restrict endptr, int base) { return __strtol(nptr, endptr, base, 1); }
#define strtol strtol_
/* Convert a string to an unsigned long integer.  */
__forceinline__ __device__ unsigned long int strtoul_(const char *__restrict nptr, char **__restrict endptr, int base) { return __strtol(nptr, endptr, base, 0); }
#define strtoul strtoul_
__END_NAMESPACE_STD;

#if defined(ULLONG_MAX)
__BEGIN_NAMESPACE_C99;
/* Convert a string to a quadword integer.  */
__forceinline__ __device__ long long int strtoll_(const char *__restrict nptr, char **__restrict endptr, int base) { return __strtoll(nptr, endptr, base, 1); }
#define strtoll strtoll_
/* Convert a string to an unsigned quadword integer.  */
__forceinline__ __device__ unsigned long long int strtoull_(const char *__restrict nptr, char **__restrict endptr, int base) { return __strtoll(nptr, endptr, base, 0); }
#define strtoull strtoull_
__END_NAMESPACE_C99;
#endif

__BEGIN_NAMESPACE_STD;
/* Return a random integer between 0 and RAND_MAX inclusive.  */
extern __device__ int rand_(void);
#define rand rand_
/* Seed the random number generator with the given number.  */
extern __device__ void srand_(unsigned int seed);
#define srand srand_
__END_NAMESPACE_STD;

__BEGIN_NAMESPACE_STD;
/* Allocate SIZE bytes of memory.  */
extern __device__ void *malloc_(size_t size);
#ifdef GANGING
__forceinline__ __device__ void *mallocG(size_t size) { __shared__ void *p; if (!threadIdx.x) p = malloc_(size); __syncthreads(); return p; }
#define malloc mallocG
#else
#define malloc malloc_
#endif
/* Allocate NMEMB elements of SIZE bytes each, all initialized to 0.  */
//extern __device__ void *calloc_(size_t nmemb, size_t size);

__forceinline__ __device__ void *calloc_(size_t nmemb, size_t size) { void *p = malloc_(nmemb * size); if (p) memset(p, 0, size); return p; }
#ifdef GANGING
__forceinline__ __device__ void *callocG(size_t nmemb, size_t size) { __shared__ void *p; if (!threadIdx.x) p = calloc_(nmemb, size); __syncthreads(); return p; }
#define calloc callocG
#else
#define calloc calloc_
#endif
__END_NAMESPACE_STD;

__BEGIN_NAMESPACE_EXT;
/* SIZE bytes of memory.  */
extern __device__ size_t _msize_(void *ptr);
#define _msize _msize_
__END_NAMESPACE_STD;

__BEGIN_NAMESPACE_STD;
/* Re-allocate the previously allocated block in PTR, making the new block SIZE bytes long.  */
extern __device__ void *realloc_(void *ptr, size_t size);
#ifdef GANGING
__forceinline__ __device__ void *reallocG(void *ptr, size_t size) { __shared__ void *v; if (!threadIdx.x) v = realloc_(ptr, size); __syncthreads(); return v; }
#define realloc reallocG
#else
#define realloc realloc_
#endif
/* Free a block allocated by `malloc', `realloc' or `calloc'.  */
extern __device__ void free_(void *ptr);
#ifdef GANGING
__forceinline__ __device__ void freeG(void *p) { if (!threadIdx.x) free_(p); __syncthreads(); }
#define free freeG
#else
#define free free_
#endif
__END_NAMESPACE_STD;

__BEGIN_NAMESPACE_STD;
/* Abort execution and generate a core-dump.  */
__forceinline__ __device__ void abort_(void) { asm("trap;"); }
#define abort abort_
/* Register a function to be called when `exit' is called.  */
__forceinline__ __device__ int atexit_(void(*func)(void)) { panic("Not Supported"); return -1; }
#define atexit atexit_
//extern __device__ int atexit_(void(*func)(void)); #define atexit atexit_
__END_NAMESPACE_STD;

__BEGIN_NAMESPACE_STD;
/* Call all functions registered with `atexit' and `on_exit', in the reverse of the order in which they were registered, perform stdio cleanup, and terminate program execution with STATUS.  */
extern __device__ void exit_(int status);
#define exit exit_
__END_NAMESPACE_STD;

__BEGIN_NAMESPACE_C99;
/* Terminate the program with STATUS without calling any of the functions registered with `atexit' or `on_exit'.  */
extern __device__ void _Exit_(int status);
#define _Exit _Exit_
__END_NAMESPACE_C99;

__BEGIN_NAMESPACE_STD;
/* Return the value of envariable NAME, or NULL if it doesn't exist.  */
extern __device__ char *getenv_(const char *name);
#define getenv getenv_
__END_NAMESPACE_STD;

/* Set NAME to VALUE in the environment. If REPLACE is nonzero, overwrite an existing value.  */
extern __device__ int setenv_(const char *name, const char *value, int replace);
#define setenv setenv_
/* Remove the variable NAME from the environment.  */
extern __device__ int unsetenv_(const char *name);
#define unsetenv unsetenv_

/* Generate a unique temporary file name from TEMPLATE.
The last six characters of TEMPLATE must be "XXXXXX"; they are replaced with a string that makes the file name unique.
Returns TEMPLATE, or a null pointer if it cannot get a unique file name.  */
extern __device__ char *mktemp_(char *template_);
#define mktemp mktemp_
/* Generate a unique temporary file name from TEMPLATE.
The last six characters of TEMPLATE must be "XXXXXX"; they are replaced with a string that makes the filename unique.
Returns a file descriptor open on the file for reading and writing, or -1 if it cannot create a uniquely-named file. */
#ifndef __USE_FILE_OFFSET64
extern __device__ int mkstemp_(char *template_);
#define mkstemp mkstemp_
#else
#define mkstemp mkstemp64
#endif
#ifdef __USE_LARGEFILE64
extern __device__ int mkstemp64_(char *template_);
#define mkstemp64 mkstemp64_
#endif

__BEGIN_NAMESPACE_STD;
/* Execute the given line as a shell command.  */
extern __device__ int system_(const char *command);
#define system system_
__END_NAMESPACE_STD;

__BEGIN_NAMESPACE_STD;
/* Do a binary search for KEY in BASE, which consists of NMEMB elements of SIZE bytes each, using COMPAR to perform the comparisons.  */
extern __device__ void *bsearch_(const void *key, const void *base, size_t nmemb, size_t size, __compar_fn_t compar);
#define bsearch bsearch_
/* Sort NMEMB elements of BASE, of SIZE bytes each, using COMPAR to perform the comparisons.  */
extern __device__ void qsort_(void *base, size_t nmemb, size_t size, __compar_fn_t compar);
#define qsort qsort_

/* Return the absolute value of X.  */
__forceinline__ __device__ int abs_(int x) { return x >= 0 ? x : -x; }
#define abs abs_
__forceinline__ __device__ long int labs_(long int x) { return x >= 0 ? x : -x; }
#define labs labs_
__END_NAMESPACE_STD;
#if defined(ULLONG_MAX)
__BEGIN_NAMESPACE_C99;
__forceinline__ __device__ long long int llabs_(long long int x) { return x >= 0 ? x : -x; }
#define llabs llabs_
__END_NAMESPACE_C99;
#endif

__BEGIN_NAMESPACE_STD;
/* Return the `div_t', `ldiv_t' or `lldiv_t' representation of the value of NUMER over DENOM. */
extern __device__ div_t div_(int numer, int denom);
#define div div_
extern __device__ ldiv_t ldiv_(long int numer, long int denom);
#define ldiv ldiv_
__END_NAMESPACE_STD;
#if defined(ULLONG_MAX)
__BEGIN_NAMESPACE_C99;
extern __device__ lldiv_t lldiv_(long long int numer, long long int denom);
#define lldiv lldiv_
__END_NAMESPACE_C99;
#endif

__BEGIN_NAMESPACE_STD;
/* Return the length of the multibyte character in S, which is no longer than N.  */
extern __device__ int mblen_(const char *s, size_t n);
#define mblen mblen_
/* Return the length of the given multibyte character, putting its `wchar_t' representation in *PWC.  */
extern __device__ int mbtowc_(wchar_t *__restrict pwc, const char *__restrict s, size_t n);
#define mbtowc mbtowc_
/* Put the multibyte character represented by WCHAR in S, returning its length.  */
extern __device__ int wctomb_(char *s, wchar_t wchar);
#define wctomb wctomb_

/* Convert a multibyte string to a wide char string.  */
extern __device__ size_t mbstowcs_(wchar_t *__restrict pwcs, const char *__restrict s, size_t n);
#define mbstowcs mbstowcs_
/* Convert a wide char string to multibyte string.  */
extern __device__ size_t wcstombs_(char *__restrict s, const wchar_t *__restrict pwcs, size_t n);
#define wcstombs wcstombs_
__END_NAMESPACE_STD;

__END_NAMESPACE_EXT;
#if defined(__GNUC__)
extern __device__ uint16_t __builtin_bswap16_(uint16_t x);
#define __builtin_bswap16 __builtin_bswap16_
extern __device__ uint32_t __builtin_bswap32_(uint32_t x);
#define __builtin_bswap32 __builtin_bswap32_
extern __device__ uint64_t __builtin_bswap64_(uint64_t x);
#define __builtin_bswap64 __builtin_bswap64_
#elif defined(_MSC_VER)
extern __device__ unsigned short _byteswap_ushort_(unsigned short x);
#define _byteswap_ushort _byteswap_ushort_
extern __device__ unsigned long _byteswap_ulong_(unsigned long x);
#define _byteswap_ulong _byteswap_ulong_
extern __device__ unsigned __int64 _byteswap_uint64_(unsigned __int64 x);
#define _byteswap_uint64 _byteswap_uint64_
#endif
__END_NAMESPACE_EXT;

__END_DECLS;
#else
#define atoll(s) 0
#define strtof(s,e) 0.0
#define strtold(s,e) 0.0
#define strtoll(s,e,b) 0
#define strtoull(s,e,b) 0
#define setenv(n,v,r) 0
#define unsetenv(n) 0
//#define mktemp(t) 0
#define mkstemp(t) 0
#ifdef _MSC_VER
#include <malloc.h>
#else
#ifdef __APPLE__
#include <malloc/malloc.h>
#define _msize(p) malloc_size(p)
#else
#include <malloc.h>
#define _msize(p) malloc_usable_size(p)
#endif
#endif
#endif  /* _STDLIBCU_H */
__BEGIN_DECLS;

#if defined(ULLONG_MAX)
#if __OS_WIN
/* Returned by `strtoq'.  */
typedef long long int quad_t;
/* Returned by `strtouq'.  */
typedef unsigned long long int u_quad_t;
//#else
//#include <sys/types.h>
#endif
/* Convert a string to a quadword integer.  */
__forceinline__ __device__ quad_t strtoq_(const char *__restrict nptr, char **__restrict endptr, int base) { return (quad_t)strtol(nptr, endptr, base); }
#define strtoq strtoq_
/* Convert a string to an unsigned quadword integer.  */
__forceinline__ __device__ u_quad_t strtouq_(const char *__restrict nptr, char **__restrict endptr, int base) { return (u_quad_t)strtoul(nptr, endptr, base); }
#define strtouq strtouq_
#endif

__END_DECLS;
#endif  /* _STDLIBCU_H */