/*
string.h - declarations for string manipulation functions
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

#ifndef _STRINGCU_H
#define _STRINGCU_H
#include <crtdefscu.h>

#include <string.h>
#include <stdarg.h>
#ifdef __CUDA_ARCH__
__BEGIN_DECLS;

__BEGIN_NAMESPACE_STD;
/* Copy N bytes of SRC to DEST.  */
//builtin: extern void *__cdecl memcpy(void *, const void *, size_t);
extern __device__ void *memcpy_(void *__restrict dest, const void *__restrict src, size_t n);
#define memcpy memcpy_

/* Copy N bytes of SRC to DEST, guaranteeing correct behavior for overlapping strings.  */
//extern __device__ void *memmove_(void *__restrict dest, const void *__restrict src, size_t n);
#define memmove memcpy_
__END_NAMESPACE_STD;

__BEGIN_NAMESPACE_STD;
/* Set N bytes of S to C.  */
//builtin: extern void *__cdecl memset(void *, int, size_t);
extern __device__ void *memset_(void *s, int c, size_t n);
#define memset memset_
/* Compare N bytes of S1 and S2.  */
extern __device__ int memcmp_(const void *s1, const void *s2, size_t n);
#define memcmp memcmp_
/* Search N bytes of S for C.  */
extern __device__ void *memchr_(const void *s, int c, size_t n);
#define memchr memchr_
__END_NAMESPACE_STD;

__BEGIN_NAMESPACE_STD;
/* Copy SRC to DEST.  */
extern __device__ char *strcpy_(char *__restrict dest, const char *__restrict src);
#define strcpy strcpy_
/* Copy no more than N characters of SRC to DEST.  */
extern __device__ char *strncpy_(char *__restrict dest, const char *__restrict src, size_t n);
#define strncpy strncpy_

/* Append SRC onto DEST.  */
extern __device__ char *strcat_(char *__restrict dest, const char *__restrict src);
#define strcat strcat_
/* Append no more than N characters from SRC onto DEST.  */
extern __device__ char *strncat_(char *__restrict dest, const char *__restrict src, size_t n);
#define strncat strncat_

/* Compare S1 and S2.  */
extern __device__ int strcmp_(const char *s1, const char *s2);
#define strcmp strcmp_
/* Compare S1 and S2. Case insensitive.  */
extern __device__ int stricmp_(const char *s1, const char *s2);
#define stricmp stricmp_
/* Compare N characters of S1 and S2.  */
extern __device__ int strncmp_(const char *s1, const char *s2, size_t n);
#define strncmp strncmp_
/* Compare N characters of S1 and S2. Case insensitive.  */
extern __device__ int strnicmp_(const char *s1, const char *s2, size_t n);
#define strnicmp strnicmp_

/* Compare the collated forms of S1 and S2.  */
extern __device__ int strcoll_(const char *s1, const char *s2);
#define strcoll strcoll_
/* Put a transformation of SRC into no more than N bytes of DEST.  */
extern __device__ size_t strxfrm_(char *__restrict dest, const char *__restrict src, size_t n);
#define strxfrm strxfrm_
__END_NAMESPACE_STD;

/* Duplicate S, returning an identical malloc'd string.  */
extern __device__ char *strdup_(const char *s);
#define strdup strdup_
/* Return a malloc'd copy of at most N bytes of STRING.  The resultant string is terminated even if no null terminator appears before STRING[N].  */
extern __device__ char *strndup_(const char *s, size_t n);
#define strndup strndup_

__BEGIN_NAMESPACE_STD;
/* Find the first occurrence of C in S.  */
extern __device__ char *strchr_(const char *s, int c);
#define strchr strchr_
/* Find the last occurrence of C in S.  */
extern __device__ char *strrchr_(const char *s, int c);
#define strrchr strrchr_
__END_NAMESPACE_STD;

__BEGIN_NAMESPACE_STD;
/* Return the length of the initial segment of S which consists entirely of characters not in REJECT.  */
extern __device__ size_t strcspn_(const char *s, const char *reject);
#define strcspn strcspn_
/* Return the length of the initial segment of S which consists entirely of characters in ACCEPT.  */
extern __device__ size_t strspn_(const char *s, const char *accept);
#define strspn strspn_
/* Find the first occurrence in S of any character in ACCEPT.  */
extern __device__ char *strpbrk_(const char *s, const char *accept);
#define strpbrk strpbrk_
/* Find the first occurrence of NEEDLE in HAYSTACK.  */
extern __device__ char *strstr_(const char *haystack, const char *needle);
#define strstr strstr_

/* Divide S into tokens separated by characters in DELIM.  */
extern __device__ char *strtok_(char *__restrict s, const char *__restrict delim);
#define strtok strtok_
__END_NAMESPACE_STD;

extern __device__ void *mempcpy_(void *__restrict dest, const void *__restrict src, size_t n);
#define mempcpy mempcpy_

__BEGIN_NAMESPACE_STD;
/* Return the length of S.  */
extern __device__ size_t strlen_(const char *s);
#define strlen strlen_
__END_NAMESPACE_STD;

/* Return the length of S.  */
extern __device__ size_t strlen16_(const void *s);
#define strlen16 strlen16_

/* Find the length of STRING, but scan at most MAXLEN characters. If no '\0' terminator is found in that many characters, return MAXLEN.  */
extern __device__ size_t strnlen_(const char *s, size_t maxlen);
#define strnlen strnlen_

__BEGIN_NAMESPACE_STD;
/* Return a string describing the meaning of the `errno' code in ERRNUM.  */
extern __device__ char *strerror_(int errnum);
#define strerror strerror_
__END_NAMESPACE_STD;

__END_DECLS;
#else
#define strndup(s, n) (nullptr)
#define strlen16(s) ((size_t)0)
#define stricmp(s1, s2) (s1 == s2 ? 0 : -1)
#define strnicmp(s1, s2, n) (s1 == s2 ? 0 : -1)
#endif  /* __CUDA_ARCH__ */
__BEGIN_DECLS;

/* On machines with a small stack size, you can redefine the PRINT_BUF_SIZE to be something smaller, if desired. */
#ifndef PRINT_BUF_SIZE
#define PRINT_BUF_SIZE 70
#endif

typedef struct strbld_t {
	void *tag;			// Optional database for lookaside.  Can be NULL //: db
	char *base;			// A base allocation.  Not from malloc. //: zBase
	char *text;			// The string collected so far //: zText
	size_t index;		// Length of the string so far //: nChar
	size_t size;		// Amount of space allocated in zText //: nAlloc
	size_t maxSize;		// Maximum allowed string length //: mxAlloc
	unsigned char error; // Becomes true if any memory allocation fails //: accError
	unsigned char flags; // SQLITE_PRINTF flags below //: printfFlags
} strbld_t;

#define STRACCUM_NOMEM   1
#define STRACCUM_TOOBIG  2
#define PRINTF_INTERNAL 0x01  // Internal-use-only converters allowed
#define PRINTF_SQLFUNC  0x02  // SQL function arguments to VXPrintf
#define PRINTF_MALLOCED 0x04  // True if xText is allocated space
#define PRINTF_ISMALLOCED(X)  (((X)->flags & PRINTF_MALLOCED)!=0)

extern __host_device__ void strbldInit(strbld_t *b, void *tag = nullptr, char *base = nullptr, int capacity = -1, int maxSize = -1);
extern __host_device__ void strbldAppendFormatv(strbld_t *b, const char *fmt, va_list va);
extern __host_device__ void strbldAppendChar(strbld_t *b, int n, int c);
extern __host_device__ void strbldAppend(strbld_t *b, const char *str, int length);
extern __host_device__ void strbldAppendAll(strbld_t *b, const char *str);
extern __forceinline__ __host_device__ void strbldAppendElement(strbld_t *b, const char *str) { strbldAppend(b, ", ", 2); strbldAppend(b, str, (int)strlen(str)); }
extern __host_device__ char *strbldToString(strbld_t *b);
extern __host_device__ void strbldReset(strbld_t *b);

__END_DECLS;
#endif  /* _STRINGCU_H */