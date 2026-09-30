/*
ctype.h - Character handling
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

#ifndef _CTYPECU_H
#define _CTYPECU_H
#include <crtdefscu.h>

#include <ctype.h>
#if __OS_UNIX
/* set bit masks for the possible character types */
#define _DIGIT          0x04     /* digit[0-9] */
#define _HEX            0x08    /* hexadecimal digit */
#endif

#ifdef __CUDA_ARCH__
__BEGIN_DECLS;

extern __constant__ const unsigned char __curtUpperToLower[256];
extern __constant__ const unsigned char __curtCtypeMap[256];

extern __forceinline__ __device__ int isctype_(int c, int type) { return (__curtCtypeMap[(unsigned char)c] & type) != 0; }
#define isctype isctype_

extern __forceinline__ __device__ int isalnum_(int c) { return (__curtCtypeMap[(unsigned char)c] & 0x06) != 0; }
#define isalnum isalnum_
extern __forceinline__ __device__ int isalpha_(int c) { return (__curtCtypeMap[(unsigned char)c] & 0x02) != 0; }
#define isalpha isalpha_
extern __forceinline__ __device__ int iscntrl_(int c) { return (unsigned char)c <= 0x1f || (unsigned char)c == 0x7f; }
#define iscntrl iscntrl_
extern __forceinline__ __device__ int isdigit_(int c) { return (__curtCtypeMap[(unsigned char)c] & 0x04) != 0; }
#define isdigit isdigit_
extern __forceinline__ __device__ int islower_(int c) { return __curtUpperToLower[(unsigned char)c] == c; }
#define islower islower_
extern __forceinline__ __device__ int isgraph_(int c) { return 0; }
#define isgraph isgraph_
extern __forceinline__ __device__ int isprint_(int c) { return (unsigned char)c > 0x1f && (unsigned char)c != 0x7f; }
#define isprint isprint_
extern __forceinline__ __device__ int ispunct_(int c) { return 0; }
#define ispunct ispunct_
extern __forceinline__ __device__ int isspace_(int c) { return (__curtCtypeMap[(unsigned char)c] & 0x01) != 0; }
#define isspace isspace_
extern __forceinline__ __device__ int isupper_(int c) { return (c&~(__curtCtypeMap[(unsigned char)c] & 0x20)) == c; }
#define isupper isupper_
extern __forceinline__ __device__ int isxdigit_(int c) { return (__curtCtypeMap[(unsigned char)c] & 0x08) != 0; }
#define isxdigit isxdigit_

/* Return the lowercase version of C.  */
extern __forceinline__ __device__ int tolower_(int c) { return __curtUpperToLower[(unsigned char)c]; }
#define tolower tolower_

/* Return the uppercase version of C.  */
extern __forceinline__ __device__ int toupper_(int c) { return c & ~(__curtCtypeMap[(unsigned char)c] & 0x20); }
#define toupper toupper_

#ifdef __APPLE__
#define __tolower tolower_
#define __toupper toupper_
#elif __OS_UNIX && !defined(_tolower)
#define _tolower(c) (char)((c)-'A'+'a')
#define _toupper(c) (char)((c)-'a'+'A')
#endif

/*C99*/
extern __forceinline__ __device__ int isblank_(int c) { return c == '\t' || c == ' '; }
#define isblank isblank_

extern __forceinline__ __device__ int isidchar_(int c) { return (__curtCtypeMap[(unsigned char)c] & 0x46) != 0; }
#define isidchar isidchar_

/*EXT*/
extern __forceinline__ __device__ int isquote_(int c) { return (__curtCtypeMap[(unsigned char)c] & 0x80) != 0; }
#define isquote isquote_

__END_DECLS;
#else
#define __curtUpperToLower ((unsigned char *)nullptr)
#define isctype(c, type) 0
#define isidchar(c) 0
#define isblank(c) ((c) == '\t' || (c) == ' ')
#define isquote(c) ((c) == '"'|| (c) == '\'' || (c) == '[' || (c) == '`')

#endif  /* __CUDA_ARCH__ */

#endif  /* _CTYPECU_H */
