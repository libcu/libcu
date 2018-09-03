/*
crtdefscu.h - xxx
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

#ifndef _CRTDEFSCU_H
#define _CRTDEFSCU_H

/* Figure out if we are dealing with Unix, Windows, or some other operating system. */
#if defined(__OS_OTHER)
# if __OS_OTHER == 1
#  undef __OS_UNIX
#  define __OS_UNIX 0
#  undef __OS_WIN
#  define __OS_WIN 0
# else
#  undef __OS_OTHER
# endif
#endif
#if !defined(__OS_UNIX) && !defined(__OS_OTHER)
# define __OS_OTHER 0
# ifndef __OS_WIN
#  if defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__) || defined(__MINGW32__) || defined(__BORLANDC__)
#   define __OS_WIN 1
#   define __OS_UNIX 0
#  else
#   define __OS_WIN 0
#   define __OS_UNIX 1
#  endif
# else
#  define __OS_UNIX 0
# endif
#else
# ifndef __OS_WIN
#  define __OS_WIN 0
# endif
#endif

/*
** Macros to determine whether the machine is big or little endian, and whether or not that determination is run-time or compile-time.
**
** For best performance, an attempt is made to guess at the byte-order using C-preprocessor macros.  If that is unsuccessful, or if
** -DLIBCU_BYTEORDER=0 is set, then byte-order is determined at run-time.
*/
#ifndef LIBCU_BYTEORDER
#if defined(i386) || defined(__i386__) || defined(_M_IX86) || defined(__x86_64) || defined(__x86_64__) || defined(_M_X64) || \
	defined(_M_AMD64) || defined(_M_ARM) || defined(__x86) || defined(__arm__)
#define LIBCU_BYTEORDER 1234
#elif defined(sparc) || defined(__ppc__)
#define LIBCU_BYTEORDER 4321
#else
#define LIBCU_BYTEORDER 0
#endif
#endif
#if LIBCU_BYTEORDER == 4321
#define LIBCU_BIGENDIAN 1
#define LIBCU_LITTLEENDIAN 0
#define LIBCU_UTF16NATIVE TEXTENCODE_UTF16BE
#elif LIBCU_BYTEORDER == 1234
#define LIBCU_BIGENDIAN 0
#define LIBCU_LITTLEENDIAN 1
#define LIBCU_UTF16NATIVE TEXTENCODE_UTF16LE
#else
extern __host_constant__ const int __libcuone;
#define LIBCU_BIGENDIAN (*(char *)(&__libcuone)==0)
#define LIBCU_LITTLEENDIAN (*(char *)(&__libcuone)==1)
#define LIBCU_UTF16NATIVE (SQLITE_BIGENDIAN?TEXTENCODE_UTF16BE:TEXTENCODE_UTF16LE)
#endif

#if __OS_WIN
#include <crtdefs.h>
//#include <corecrt_io.h>
#define _uintptr_t uintptr_t
//#define __USE_LARGEFILE64 1
#elif __OS_UNIX
#define MAX_PATH 260
#define DELETE 0x00010000L
#if defined(__LP64__) || defined(_LP64)
# define _WIN64 1
typedef unsigned int long long _uintptr_t;
# else
typedef unsigned int _uintptr_t;
# endif
#endif

#include <cuda_runtime.h>
#include <stdint.h>
#define uint unsigned int
#define __LIBCU__

#define HAS_STDIO_BUFSIZ_NONE__
//#define _LARGEFILE64_SOURCE

/* These are defined by the user (or the compiler) to specify the desired environment:
_LARGEFILE_SOURCE	Some more functions for correct standard I/O.
_LARGEFILE64_SOURCE	Additional functionality from LFS for large files.
_FILE_OFFSET_BITS=N	Select default filesystem interface.

All macros listed above as possibly being defined by this file are explicitly undefined if they are not explicitly defined. */

#ifdef _LARGEFILE_SOURCE
#define __USE_LARGEFILE		1
#endif

#ifdef _LARGEFILE64_SOURCE
#define __USE_LARGEFILE64	1
#endif

#if defined(_FILE_OFFSET_BITS) && _FILE_OFFSET_BITS == 64
#define __USE_FILE_OFFSET64	1
#endif

#ifndef LIBCU_MAXENVIRON
#define LIBCU_MAXENVIRON 5
#endif

#ifndef LIBCU_MAXFILESTREAM
#define LIBCU_MAXFILESTREAM 10
#endif

#ifndef LIBCU_MAXHOSTPTR
#define LIBCU_MAXHOSTPTR 10
#endif

#if defined(__CUDA_ARCH__)
#if __OS_WIN
#define panic(fmt, ...) { printf(fmt"\n", __VA_ARGS__); asm("trap;"); }
#elif __OS_UNIX
#define panic(fmt, ...) { printf(fmt"\n"); asm("trap;"); }
#endif
#else
//__forceinline__ void Coverage(int line) { }
#if __OS_WIN
#define panic(fmt, ...) { printf(fmt"\n", __VA_ARGS__); exit(1); }
#elif __OS_UNIX
#define panic(fmt, ...) { printf(fmt"\n"); exit(1); }
#endif
#endif  /* __CUDA_ARCH__ */

/* GCC does not define the offsetof() macro so we'll have to do it ourselves. */
#ifndef offsetof
#define offsetof(STRUCTURE,FIELD) ((int)((char*)&((STRUCTURE*)0)->FIELD))
#endif

//////////////////////
// UTILITY
#pragma region UTILITY

/* For these things, GCC behaves the ANSI way normally, and the non-ANSI way under -traditional.  */
#define __CONCAT(x,y) x ## y
#define __STRING(x) #x

/* PTX conditionals */
#ifdef _WIN64
#define _UX ".u64"
#define _BX ".b64"
#define __R "l"
#define __I "r"
#else
#define _UX ".u32"
#define _BX ".b32"
#define __R "r"
#define __I "r"
#endif

/* This is not a typedef so `const __ptr_t' does the right thing.  */
#define __ptr_t void *
//#define __long_double_t long double
/* CUDA long_double is double */
#define long_double double

#define MEMORY_ALIGNMENT 4096
/* Memory allocation - rounds to the type in T */
#define ROUNDT_(x, T)		(((x)+sizeof(T)-1)&~(sizeof(T)-1))
/* Memory allocation - rounds up to 8 */
#define ROUND8_(x)			(((x)+7)&~7)
/* Memory allocation - rounds up to 64 */
#define ROUND64_(x)			(((x)+63)&~63)
/* Memory allocation - rounds up to "size" */
#define ROUNDN_(x, size)	(((size_t)(x)+(size-1))&~(size-1))
/* Memory allocation - rounds down to 8 */
#define ROUNDDOWN8_(x)		((x)&~7)
/* Memory allocation - rounds down to "size" */
#define ROUNDDOWNN_(x, size) (((size_t)(x))&~(size-1))
/* Test to see if you are on aligned boundary, affected by BYTEALIGNED4 */
#ifdef BYTEALIGNED4
#define HASALIGNMENT8_(x) ((((char *)(x) - (char *)0)&3) == 0)
#else
#define HASALIGNMENT8_(x) ((((char *)(x) - (char *)0)&7) == 0)
#endif
/* Returns the length of an array at compile time (via math) */
#define ARRAYSIZE_(symbol) (sizeof(symbol) / sizeof(symbol[0]))
/* Removes compiler warning for unused parameter(s) */
#define UNUSED_SYMBOL(x) (void)(x)
#define UNUSED_SYMBOL2(x,y) (void)(x),(void)(y)

/* Macros to compute minimum and maximum of two numbers. */
#ifndef MIN_
#define MIN_(A,B) ((A)<(B)?(A):(B))
#endif
#ifndef MAX_
#define MAX_(A,B) ((A)>(B)?(A):(B))
#endif
/* Swap two objects of type TYPE. */
#define SWAP_(TYPE,A,B) { TYPE t=A; A=B; B=t; }

#pragma endregion

//////////////////////
// PTRSIZE
#pragma region PTRSIZE

/* Set the SQLITE_PTRSIZE macro to the number of bytes in a pointer */
#ifndef PTRSIZE_
#if defined(__SIZEOF_POINTER__)
#define PTRSIZE_ __SIZEOF_POINTER__
#elif defined(i386) || defined(__i386__) || defined(_M_IX86) || defined(_M_ARM) || defined(__arm__) || defined(__x86)
#define PTRSIZE_ 4
#else
#define PTRSIZE_ 8
#endif
#endif

/* The uptr type is an unsigned integer large enough to hold a pointer */
#if defined(HAVE_STDINT_H)
//typedef uintptr_t uintptr_t;
#elif PTRSIZE_ == 4
typedef uint32_t uintptr_t;
#else
typedef uint64_t uintptr_t;
#endif

/*
** The WITHIN_(P,S,E) macro checks to see if pointer P points to something between S (inclusive) and E (exclusive).
**
** In other words, S is a buffer and E is a pointer to the first byte after the end of buffer S.  This macro returns true if P points to something
** contained within the buffer S.
*/
#define WITHIN_(P,S,E) (((uintptr_t)(P)>=(uintptr_t)(S))&&((uintptr_t)(P)<(uintptr_t)(E)))

#pragma endregion	

//////////////////////
// NAMESPACE
#pragma region NAMESPACE

/* C++ needs to know that types and declarations are C, not C++.  */
#ifdef	__cplusplus
#define __BEGIN_DECLS extern "C" {
#define __END_DECLS }
#else
#define __BEGIN_DECLS
#define __END_DECLS
#endif

/* The standard library needs the functions from the ISO C90 standard
in the std namespace.  At the same time we want to be safe for
future changes and we include the ISO C99 code in the non-standard
namespace __c99.  The C++ wrapper header take case of adding the
definitions to the global namespace.  */
#if defined(__cplusplus) && defined(_GLIBCPP_USE_NAMESPACES)
#define __BEGIN_NAMESPACE_STD namespace std {
#define __END_NAMESPACE_STD }
#define __USING_NAMESPACE_STD(name) using std::name;
#define __BEGIN_NAMESPACE_C99 namespace __c99 {
#define __END_NAMESPACE_C99 }
#define __USING_NAMESPACE_C99(name) using ext::name;
#define __BEGIN_NAMESPACE_EXT namespace ext {
#define __END_NAMESPACE_EXT }
#define __USING_NAMESPACE_EXT(name) using ext::name;
#else
/* For compatibility we do not add the declarations into any
namespace.  They will end up in the global namespace which is what
old code expects.  */
#define __BEGIN_NAMESPACE_STD
#define __END_NAMESPACE_STD
#define __USING_NAMESPACE_STD(name)
#define __BEGIN_NAMESPACE_C99
#define __END_NAMESPACE_C99
#define __USING_NAMESPACE_C99(name)
#define __BEGIN_NAMESPACE_EXT
#define __END_NAMESPACE_EXT
#define __USING_NAMESPACE_EXT(name)
#endif

#pragma endregion

//////////////////////
// DEVICE/HOST
#pragma region DEVICE/HOST
__BEGIN_DECLS;

#ifndef __CUDA_ARCH__
#define __host_device__ __host__
#define __hostb_device__
#define __host_constant__
#else
#define __host_device__ __device__
#define __hostb_device__ __device__
#define __host_constant__ __constant__
#endif

#ifndef	__cplusplus
#define bool int
#define false 0
#define true 1
#endif

typedef struct hostptr_t {
	void *host;
} hostptr_t;

/* IsHost support  */
extern __device__ char __cwd[];
#define ISHOSTENV(name) (name[0] != ':')
#define ISHOSTPATH(path) ((path)[1] == ':' || ((path)[0] != ':' && __cwd[0] == 0))
#define ISHOSTHANDLE(handle) (handle < INT_MAX-LIBCU_MAXFILESTREAM)
#define ISHOSTPTR(ptr) ((hostptr_t *)(ptr) >= __iob_hostptrs && (hostptr_t *)(ptr) <= __iob_hostptrs+LIBCU_MAXHOSTPTR)

/* Host pointer support  */
extern __constant__ hostptr_t __iob_hostptrs[LIBCU_MAXHOSTPTR];
extern __device__ hostptr_t *__hostptrGet(void *host);
extern __device__ void __hostptrFree(hostptr_t *p);

/* Reset library */
extern __device__ void libcuReset();

__END_DECLS;
#ifdef	__cplusplus
template <typename T> __forceinline__ __device__ T *newhostptr(T *p) { return (T *)(p ? __hostptrGet(p) : nullptr); }
template <typename T> __forceinline__ __device__ void freehostptr(T *p) { if (p) __hostptrFree((hostptr_t *)p); }
template <typename T> __forceinline__ __device__ T *hostptr(T *p) { return (T *)(p ? ((hostptr_t *)p)->host : nullptr); }
#endif
#pragma endregion

//////////////////////
// ASSERT
#pragma region ASSERT
__BEGIN_DECLS;

/*
** NDEBUG and _DEBUG are opposites.  It should always be true that defined(NDEBUG) == !defined(_DEBUG).  If this is not currently true,
** make it true by defining or undefining NDEBUG.
**
** Setting NDEBUG makes the code smaller and faster by disabling the assert() statements in the code.  So we want the default action
** to be for NDEBUG to be set and NDEBUG to be undefined only if _DEBUG is set.  Thus NDEBUG becomes an opt-in rather than an opt-out feature.
*/
#if !defined(NDEBUG) && !defined(_DEBUG)
#define NDEBUG 1
#endif
#if defined(NDEBUG) && defined(_DEBUG)
#undef NDEBUG
#endif

/* The testcase() macro is used to aid in coverage testing.  When doing coverage testing, the condition inside the argument to
** testcase() must be evaluated both true and false in order to get full branch coverage.  The testcase() macro is inserted
** to help ensure adequate test coverage in places where simple condition/decision coverage is inadequate.  For example, testcase()
** can be used to make sure boundary values are tested.  For bitmask tests, testcase() can be used to make sure each bit
** is significant and used at least once.  On switch statements where multiple cases go to the same block of code, testcase()
** can insure that all cases are evaluated.
*/
#ifdef _COVERAGE_TEST
#if defined(__CUDA_ARCH__)
__device__ void __coverage(int line);
#else
void __coverage(int line);
#endif
#define TESTCASE_(X)  if (X) { __coverage(__LINE__); }
#else
#define TESTCASE_(X)
#endif

/* The TESTONLY_ macro is used to enclose variable declarations or other bits of code that are needed to support the arguments
** within testcase() and assert() macros.
*/
#if !defined(NDEBUG) || defined(_COVERAGE_TEST)
#define TESTONLY_(X)  X
#else
#define TESTONLY_(X)
#endif

/*
** Sometimes we need a small amount of code such as a variable initialization to setup for a later assert() statement.  We do not want this code to
** appear when assert() is disabled.  The following macro is therefore used to contain that setup code.  The "VVA" acronym stands for
** "Verification, Validation, and Accreditation".  In other words, the code within VVA_ONLY() will only run during verification processes.
*/
#ifndef NDEBUG
#define DEBUGONLY_(X)  X
#else
#define DEBUGONLY_(X)
#endif

/* The ALWAYS and NEVER macros surround boolean expressions which are intended to always be true or false, respectively.  Such
** expressions could be omitted from the code completely.  But they are included in a few cases in order to enhance the resilience
** of SQLite to unexpected behavior - to make the code "self-healing" or "ductile" rather than being "brittle" and crashing at the first
** hint of unplanned behavior.
**
** In other words, ALWAYS and NEVER are added for defensive code.
**
** When doing coverage testing ALWAYS and NEVER are hard-coded to be true and false so that the unreachable code they specify will
** not be counted as untested code.
*/
#if defined(_COVERAGE_TEST) || defined(_MUTATION_TEST)
# define ALWAYS_(X)      (1)
# define NEVER_(X)       (0)
#elif !defined(NDEBUG)
# define ALWAYS_(X)      ((X)?1:(assert(0),0))
# define NEVER_(X)       ((X)?(assert(0),1):0)
#else
# define ALWAYS_(X)      (X)
# define NEVER_(X)       (X)
#endif

__END_DECLS;
#pragma endregion

//////////////////////
// WSD
#pragma region WSD
__BEGIN_DECLS;

// When NO_WSD is defined, it means that the target platform does not support Writable Static Data (WSD) such as global and static variables.
// All variables must either be on the stack or dynamically allocated from the heap.  When WSD is unsupported, the variable declarations scattered
// throughout the code must become constants instead.  The WSD_ macro is used for this purpose.  And instead of referencing the variable
// directly, we use its constant as a key to lookup the run-time allocated buffer that holds real variable.  The constant is also the initializer
// for the run-time allocated buffer.
//
// In the usual case where WSD is supported, the WSD_ and GLOBAL_ macros become no-ops and have zero performance impact.
#ifdef NO_WSD
int __wsdinit(int n, int j);
void *__wsdfind(void *k, int l);
#define WSD_ const
#define GLOBAL_(t, v) (*(t*)__wsdfind((void *)&(v), sizeof(v)))
#else
#define WSD_
#define GLOBAL_(t, v) v
#endif

__END_DECLS;
#pragma endregion

//////////////////////
// EXT METHODS
#pragma region EXT-METHODS
__BEGIN_DECLS;

// CAPI3REF: OS Interface Open File Handle
typedef struct vsysfile vsysfile;
struct vsysfile {
	const struct vsysfile_methods *methods;  // Methods for an open file
};

// CAPI3REF: OS Interface File Virtual Methods Object
typedef struct vsysfile_methods vsysfile_methods;
struct vsysfile_methods {
	int version;
	int(*close)(vsysfile *);
	int(*read)(vsysfile *, void *, int amount, int64_t offset);
	int(*write)(vsysfile *, const void *, int amount, int64_t offset);
	int(*truncate)(vsysfile *, int64_t size);
	int(*sync)(vsysfile *, int flags);
	int(*fileSize)(vsysfile *, int64_t *size);
	int(*lock)(vsysfile *, int);
	int(*unlock)(vsysfile *, int);
	int(*checkReservedLock)(vsysfile *, int *resOut);
	int(*fileControl)(vsysfile *, int op, void *args);
	int(*sectorSize)(vsysfile *);
	int(*deviceCharacteristics)(vsysfile *);
	int(*shmMap)(vsysfile *, int page, int pageSize, int, void volatile **);
	int(*shmLock)(vsysfile *, int offset, int n, int flags);
	void(*shmBarrier)(vsysfile *);
	int(*shmUnmap)(vsysfile *, int deleteFlag);
	int(*fetch)(vsysfile *, int64_t offset, int amount, void **p);
	int(*unfetch)(vsysfile *, int64_t offset, void *p);
};

// CAPI3REF: OS Interface Object
typedef struct vsystem vsystem;

typedef struct strbld_t strbld_t;
typedef struct ext_methods ext_methods;
struct ext_methods {
	void *(*tagallocRaw)(void *tag, uint64_t size);
	void *(*tagrealloc)(void *tag, void *old, uint64_t newSize);
	int *(*tagallocSize)(void *tag, void *p);
	void(*tagfree)(void *tag, void *p);
	void(*appendFormat[2])(strbld_t *b, void *va);
	int64_t(*getIntegerArg)(void *args);
	double(*getDoubleArg)(void *args);
	char *(*getStringArg)(void *args);
	//
	void(*vsys_close)(vsysfile *);
	int(*vsys_write)(vsysfile *, const void *, int amount, int64_t offset);
	int(*vsys_open)(vsystem *, const char *, vsysfile *, int, int *);
};
extern __hostb_device__ ext_methods __extsystem;

__END_DECLS;
#pragma endregion	

#endif  /* _CRTDEFSCU_H */