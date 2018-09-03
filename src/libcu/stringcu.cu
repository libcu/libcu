#include <stdiocu.h>
#include <stringcu.h>
#include <stdlibcu.h>
#include <ctypecu.h>
#include <limits.h>
#include <assert.h>
#include <stdarg.h>

#define xOMIT_PTX
__BEGIN_DECLS;

/* Copy N bytes of SRC to DEST.  */
#ifdef _WIN64
typedef	long long int word; /* "word" used for optimal copy speed */
#else
typedef	long int word; /* "word" used for optimal copy speed */
#endif
#define	wsize sizeof(word)
#define	wmask (wsize-1)
#define	_wsize "4"
#define	_wsizex8 "32"
#define	_wmask "3"

__device__ void *memcpy_(void *__restrict dest, const void *__restrict src, size_t n) {
#ifndef xOMIT_PTX
	void *r;
	asm(
		".reg .pred p1;\n\t"
		".reg " _UX " z0;\n\t"
		".reg " _UX " t;\n\t"
		".reg .b8 c;\n\t"
		"mov"_BX" 		%0, %1;\n\t"
		// if (!n || dest == src) goto _ret;
		"setp.eq" _BX "	p1, %3, 0;\n\t"
		"setp.eq.or" _BX " p1, %1, %2, p1;\n\t"
		"@p1 bra		_Ret;\n\t"
		// if (a < b)
		"setp.lt" _UX "	p1, %1, %2;\n\t" // Check for destructive overlap
		"@!p1 bra		_WhileDesc;\n\t"
		// Do an ascending copy

		// align to word
		// if (((uintptr_t)a | (uintptr_t)b) & wmask)
		"or" _BX "  	t, %1, %2;\n\t"
		"and" _BX "  	t, t, " _wmask ";\n\t"
		"setp.ne" _UX "	p1, t, 0;\n\t"
		"@!p1 bra		_ascWholepage;\n\t"

		// t = ((uintptr_t)a ^ (uintptr_t)b) & wmask || n < wsize ? n : wsize-((uintptr_t)b & wmask);
		"xor" _BX "		t, %1, %2;\n\t"
		"and" _BX "		t, t, " _wmask ";\n\t"
		"setp.ne" _UX "	p1, t, 0;\n\t"
		"setp.lt.or" _UX " p1, %3, " _wsize ", p1;\n\t"
		"and" _BX "		t, %2, " _wmask ";\n\t"
		"@p1 mov" _UX " t, %2;\n\t"
		"@!p1 sub" _UX "t, " _wsize ", t;\n\t"
		// n -= t;
		"sub" _UX "		%3, %3, t;\n\t"
		// do { *a++ = *b++; } while (--t);
		"_asc0:\n\t"
		"ld.u8 			c, [%2];\n\t"
		"add" _UX " 	%2, %2, 1;\n\t"
		"st.u8 			[%1], c;\n\t"
		"add" _UX " 	%1, %1, 1;\n\t"
		"add" _UX " 	t, t, -1;\n\t"
		"setp.ne" _UX "	p1, t, 0;\n\t"
		"@p1 bra		_asc0;\n\t"

		// asc - set whole pages
		"_ascWholepage:\n\t"
		// t = n / (wsize * 8);
		"div" _UX "		t, %3, " _wsizex8 ";\n\t"
		// if (t)
		"setp.ne" _UX "	p1, t, 0;\n\t"
		"@p1 bra		_ascWholeword;\n\t"
		// n %= wsize * 8;
		"rem" _UX "		%3, %3, " _wsizex8 ";\n\t"
		// do {...} while (--t);
		"_asc1:\n\t"
		"ld" _UX " 		z0, [%2+0];		st" _UX " [%1+0], z0;\n\t"
		"ld" _UX " 		z0, [%2+4];		st" _UX " [%1+4], z0;\n\t"
		"ld" _UX " 		z0, [%2+8];		st" _UX " [%1+8], z0;\n\t"
		"ld" _UX " 		z0, [%2+12];	st" _UX " [%1+12], z0;\n\t"
		"ld" _UX " 		z0, [%2+16];	st" _UX " [%1+16], z0;\n\t"
		"ld" _UX " 		z0, [%2+20];	st" _UX " [%1+20], z0;\n\t"
		"ld" _UX " 		z0, [%2+24];	st" _UX " [%1+24], z0;\n\t"
		"ld" _UX " 		z0, [%2+28];	st" _UX " [%1+28], z0;\n\t"
		"add" _UX "		%1, %1, " _wsizex8 ";\n\t"
		"add" _UX "		%2, %2, " _wsizex8 ";\n\t"
		"add" _UX "		t, t, -1;\n\t"
		"setp.ne" _UX "	p1, t, 0;\n\t"
		"@p1 bra 		_asc1;\n\t"

		// copy whole words
		"_ascWholeword:\n\t"
		// t = n / wsize;
		"div" _UX " 	t, %3, " _wsize ";\n\t"
		// if (t)
		"setp.ne" _UX "	p1, t, 0;\n\t"
		"@p1 bra		_ascTrailing;\n\t"
		// do { *(word *)a = *(word *)b; a += wsize; b += wsize; } while (--t);
		"_asc2:\n\t"
		"ld.u32 		z0, [%2];\n\t"
		"add" _UX " 	%2, %2, " _wsize ";\n\t"
		"st.u32 		[%1], z0;\n\t"
		"add" _UX " 	%1, %1, " _wsize ";\n\t"
		"add" _UX " 	t, t, -1;\n\t"
		"setp.ne" _UX "	p1, t, 0;\n\t"
		"@p1 bra 		_asc2;\n\t"

		// copy trailing bytes
		"_ascTrailing:\n\t"
		// t = n & wmask;
		"and" _BX "		t, %3, " _wmask ";\n\t"
		// if (t)
		"setp.ne" _UX "	p1, t, 0;\n\t"
		"@p1 bra		_Ret;\n\t"
		// do { *a++ = *b++; } while (--t);
		"_asc3:\n\t"
		"ld.u8			c, [%2];\n\t"
		"add" _UX "		%2, %2, 1;\n\t"
		"st.u8			[%1], c;\n\t"
		"add" _UX "		%1, %1, 1;\n\t"
		"add" _UX "		t, t, -1;\n\t"
		"setp.ne" _UX "	p1, t, 0;\n\t"
		"@p1 bra		_asc3;\n\t"
		"bra.uni		_Ret;\n\t"


		// Destructive overlap
		"_WhileDesc:\n\t"
		"add" _UX "		%1, %1, %3;\n\t"
		"add" _UX "		%2, %2, %3;\n\t"

		// align to word
		// if (((uintptr_t)a | (uintptr_t)b) & wmask)
		"or" _BX " 		t, %1, %2;\n\t"
		"and" _BX "		t, t, " _wmask ";\n\t"
		"setp.ne" _UX "	p1, t, 0;\n\t"
		"@!p1 bra		_descWholepage;\n\t"

		// t = ((uintptr_t)a ^ (uintptr_t)b) & wmask || n <= wsize ? n : ((uintptr_t)b & wmask);
		"xor" _BX "		t, %1, %2;\n\t"
		"and" _BX "		t, t, "_wmask";\n\t"
		"setp.ne" _UX "	p1, t, 0;\n\t"
		"setp.le.or" _UX " p1, %3, " _wsize ", p1;\n\t"
		"@!p1 and" _BX " t, %2, " _wmask ";\n\t"
		"@p1 mov" _UX "	t, %2;\n\t"
		// n -= t;
		"sub" _UX " 	%3, %3, t;\n\t"
		// do { *--a = *--b; } while (--t);
		"_desc0:\n\t"
		"add" _UX " 	%2, %2, -1;\n\t"
		"ld.u8 			c, [%2];\n\t"
		"add" _UX "		%1, %1, -1;\n\t"
		"st.u8 			[%1], c;\n\t"
		"add" _UX " 	t, t, -1;\n\t"
		"setp.ne" _UX "	p1, t, 0;\n\t"
		"@p1 bra		_desc0;\n\t"

		// asc - set whole pages
		"_descWholepage:\n\t"
		// t = n / (wsize * 8);
		"div" _UX " 	t, %3, " _wsizex8 ";\n\t"
		// if (t)
		"setp.ne" _UX "	p1, t, 0;\n\t"
		"@p1 bra		_descWholeword;\n\t"
		// n %= wsize * 8;
		"rem" _UX " 		%3, %3, " _wsizex8 ";\n\t"
		// do {...} while (--t);
		"_desc1:\n\t"
		"add" _UX "		%1, %1, -" _wsizex8 ";\n\t"
		"add" _UX "		%2, %2, -" _wsizex8 ";\n\t"
		"ld" _UX " 		z0, [%2+28];	st" _UX "	[%1+28], z0;\n\t"
		"ld" _UX " 		z0, [%2+24];	st" _UX "	[%1+24], z0;\n\t"
		"ld" _UX " 		z0, [%2+20];	st" _UX "	[%1+20], z0;\n\t"
		"ld" _UX " 		z0, [%2+16];	st" _UX "	[%1+16], z0;\n\t"
		"ld" _UX " 		z0, [%2+12];	st" _UX "	[%1+12], z0;\n\t"
		"ld" _UX " 		z0, [%2+8];		st" _UX "	[%1+8], z0;\n\t"
		"ld" _UX " 		z0, [%2+4];		st" _UX "	[%1+4], z0;\n\t"
		"ld" _UX " 		z0, [%2+0];		st" _UX "	[%1+0], z0;\n\t"
		"add" _UX "		t, t, -1;\n\t"
		"setp.ne" _UX "	p1, t, 0;\n\t"
		"@p1 bra 		_desc1;\n\t"

		// copy whole words
		"_descWholeword:\n\t"
		// t = n / wsize;
		"div" _UX " 		t, %3, " _wsize ";\n\t"
		// if (t)
		"setp.ne" _UX "	p1, t, 0;\n\t"
		"@p1 bra		_descTrailing;\n\t"
		// do { a -= wsize; b -= wsize; *(word *)a = *(word *)b; } while (--t);
		"_desc2:\n\t"
		"add" _UX "		%2, %2, -" _wsize ";\n\t"
		"ld.u32 		z0, [%2];\n\t"
		"add" _UX " 	%1, %1, -" _wsize ";\n\t"
		"st.u32 		[%1], z0;\n\t"
		"add" _UX "		t, t, -1;\n\t"
		"setp.ne" _UX "	p1, t, 0;\n\t"
		"@p1 bra 		_desc2;\n\t"

		// copy trailing bytes
		"_descTrailing:\n\t"
		// t = n & wmask;
		"and" _BX "		t, %3, " _wmask ";\n\t"
		// if (t)
		"setp.ne" _UX "	p1, t, 0;\n\t"
		"@p1 bra		_Ret;\n\t"
		// do { *--a = *--b; } while (--t);
		"_desc3:\n\t"
		"add" _UX "		%2, %2, -1;\n\t"
		"ld.u8			c, [%2];\n\t"
		"add" _UX "		%1, %1, -1;\n\t"
		"st.u8			[%1], c;\n\t"
		"add" _UX " 	t, t, -1;\n\t"
		"setp.ne" _UX "	p1, t, 0;\n\t"
		"@p1 bra		_desc3;\n\t"

		"_Ret:\n\t"
		: "="__R(r) : __R(dest), __R(src), __R(n));
	return r;
#else
	if (!n || dest == src) goto _ret;
	register unsigned char *a = (unsigned char *)dest;
	register unsigned char *b = (unsigned char *)src;
	size_t t;
	// Do an ascending copy
	if (a < b) { // Check for destructive overlap
		// align to word
		if (((uintptr_t)a | (uintptr_t)b) & wmask) {
			t = ((uintptr_t)a ^ (uintptr_t)b) & wmask || n < wsize ? n : wsize - ((uintptr_t)b & wmask);
			n -= t;
			do { *a++ = *b++; } while (--t);
		}
		// set whole pages
		t = n / (wsize * 8);
		if (t) {
			n %= wsize * 8;
			do {
				((word *)a)[0] = ((word *)b)[0];
				((word *)a)[1] = ((word *)b)[1];
				((word *)a)[2] = ((word *)b)[2];
				((word *)a)[3] = ((word *)b)[3];
				((word *)a)[4] = ((word *)b)[4];
				((word *)a)[5] = ((word *)b)[5];
				((word *)a)[6] = ((word *)b)[6];
				((word *)a)[7] = ((word *)b)[7];
				a += wsize * 8; b += wsize * 8;
			} while (--t);
		}
		// copy whole words
		t = n / wsize;
		if (t) do { *(word *)a = *(word *)b; a += wsize; b += wsize; } while (--t);
		// copy trailing bytes
		t = n & wmask;
		if (t) do { *a++ = *b++; } while (--t);
	}
	else {
		b += n;
		a += n;
		// align to word
		if (((uintptr_t)a | (uintptr_t)b) & wmask) {
			t = ((uintptr_t)a ^ (uintptr_t)b) & wmask || n <= wsize ? n : ((uintptr_t)b & wmask);
			n -= t;
			do { *--a = *--b; } while (--t);
		}
		// set whole pages
		t = n / (wsize * 8);
		if (t) {
			n %= wsize * 8;
			do {
				a -= wsize * 8; b -= wsize * 8;
				((word *)a)[7] = ((word *)b)[7];
				((word *)a)[6] = ((word *)b)[6];
				((word *)a)[5] = ((word *)b)[5];
				((word *)a)[4] = ((word *)b)[4];
				((word *)a)[3] = ((word *)b)[3];
				((word *)a)[2] = ((word *)b)[2];
				((word *)a)[1] = ((word *)b)[1];
				((word *)a)[0] = ((word *)b)[0];
			} while (--t);
		}
		// copy whole words
		t = n / wsize;
		if (t) do { a -= wsize; b -= wsize; *(word *)a = *(word *)b; } while (--t);
		// copy trailing bytes
		t = n & wmask;
		if (t) do { *--a = *--b; } while (--t);
	}
_ret:
	return dest;
#endif
}

/* Set N bytes of S to C.  */
__device__ void *memset_(void *s, int c, size_t n) {
	//#ifndef OMIT_PTX
	//#else
	if (!n) goto _ret;
	register unsigned char *a = (unsigned char *)s;
	register size_t t;
	// tiny optimize
	if (n < 3 * wsize) {
		while (n) { *a++ = c; --n; }
		goto _ret;
	}
	// set value
#ifdef _WIN64
	long long int cccc;
#else
	long int cccc;
#endif
	if ((cccc = (unsigned char)c)) {
		cccc |= cccc << 8;
		cccc |= cccc << 16;
#ifdef _WIN64
		cccc |= cccc << 32;
#endif
	}
	// align to word
	if ((t = (int)a & wmask)) {
		t = wsize - t;
		n -= t;
		do { *a++ = c; } while (--t);
	}
	// set whole pages
	t = n / (wsize * 8);
	if (t) {
		n %= wsize * 8;
		do {
			((word *)a)[0] = cccc;
			((word *)a)[1] = cccc;
			((word *)a)[2] = cccc;
			((word *)a)[3] = cccc;
			((word *)a)[4] = cccc;
			((word *)a)[5] = cccc;
			((word *)a)[6] = cccc;
			((word *)a)[7] = cccc;
			a += wsize * 8;
		} while (--t);
	}
	// set whole words
	t = n / wsize;
	if (t) do { *(word *)a = cccc; a += wsize; } while (--t);
	// set trailing bytes
	t = n & wmask;
	if (t) do { *a++ = c; } while (--t);
_ret:
	return s;
	//#endif
}

/* Compare N bytes of S1 and S2.  */
__device__ int memcmp_(const void *s1, const void *s2, size_t n) {
#ifndef OMIT_PTX
	int r;
	asm(
		".reg .pred p1;\n\t"
		".reg .s32 c1, c2;\n\t"
		"setp.eq" _BX "	p1, %3, 0;\n\t"
		"@!p1 bra _Start;\n\t"
		"mov.b32		%0, 0;\n\t"
		"bra.uni _End;\n\t"
		"_Start:\n\t"

		//
		"_While0:\n\t"
		"add" _UX " 	%3, %3, -1;\n\t"
		"setp.gt" _UX "	p1, %3, 0;\n\t"
		"@!p1 bra _Ret;\n\t"
		"ld.u8 			c1, [%1];\n\t"
		"ld.u8 			c2, [%2];\n\t"
		"setp.eq.s32	p1, c1, c2;\n\t"
		"@!p1 bra _Ret;\n\t"
		"add" _UX "		%1, %1, 1;\n\t"
		"add" _UX "		%2, %2, 1;\n\t"
		"bra.uni _While0;\n\t"

		"_Ret:\n\t"
		"sub.s32 		%0, c1, c2;\n\t"
		"_End:\n\t"
		: "=" __I(r) : __R(s1), __R(s2), __R(n));
	return r;
#else
	if (!n) return 0;
	register unsigned char *a = (unsigned char *)s1;
	register unsigned char *b = (unsigned char *)s2;
	while (--n && *a == *b) { a++; b++; }
	return *a - *b;
#endif
}

/* Search N bytes of S for C.  */
__device__ void *memchr_(const void *s, int c, size_t n) {
#ifndef OMIT_PTX
	void *r;
	asm(
		".reg .pred p1;\n\t"
		".reg .u32 c1;\n\t"
		"setp.eq" _BX "	p1, %3, 0;\n\t"
		"@!p1 bra _Start;\n\t"
		"mov" _BX "		%0, 0;\n\t"
		"bra.uni _End;\n\t"
		"_Start:\n\t"

		//
		"_While0:\n\t"
		"ld.u8 			c1, [%1];\n\t"
		"setp.eq.u32	p1, c1, %2;\n\t"
		"@p1 bra _Ret;\n\t"
		"add" _UX " 	%1, %1, 1;\n\t"
		"add" _UX " 	%3, %3, -1;\n\t"
		"setp.ne" _UX "	p1, %3, 0;\n\t"
		"@p1 bra _While0;\n\t"
		"mov" _UX " 	%1, 0;\n\t"

		"_Ret:\n\t"
		"mov" _UX "		%0, %1;\n\t"
		"_End:\n\t"
		: "=" __R(r) : __R(s), __I(c), __R(n));
	return r;
#else
	if (!n) goto _ret;
	register const char *p = (const char *)s;
	do {
		if (*p++ == c)
			return (void *)(p - 1);
	} while (--n);
_ret:
	return nullptr;
#endif
}

/* Copy SRC to DEST.  */
__device__ char *strcpy_(char *__restrict dest, const char *__restrict src) {
#ifndef OMIT_PTX
	char *r;
	asm(
		".reg .pred p1;\n\t"
		".reg .u32 c1;\n\t"
		"mov" _UX " 	%0, %1;\n\t"

		//
		"_While0:\n\t"
		"ld.u8 			c1, [%2];\n\t"
		"setp.ne.u32	p1, c1, 0;\n\t"
		"st.u8 			[%1], c1;\n\t"
		"@!p1 bra _Ret;\n\t"
		"add" _UX " 	%1, %1, 1;\n\t"
		"add" _UX " 	%2, %2, 1;\n\t"
		"bra.uni _While0;\n\t"

		"_Ret:\n\t"
		: "=" __R(r) : __R(dest), __R(src));
	return r;
#else
	register unsigned char *d = (unsigned char *)dest;
	register unsigned char *s = (unsigned char *)src;
	while (*s) { *d++ = *s++; } *d = *s;
	return (char *)dest;
#endif
}

/* Copy no more than N characters of SRC to DEST.  */
__device__ char *strncpy_(char *__restrict dest, const char *__restrict src, size_t n) {
#ifndef OMIT_PTX
	char *r;
	asm(
		".reg .pred p1;\n\t"
		".reg .u32 c1;\n\t"
		".reg " _UX "	i1;\n\t"
		"mov" _UX "		%0, %1;\n\t"
		"mov" _UX "		i1, 0;\n\t"

		//
		"_While0:\n\t"
		"setp.lt" _UX "	p1, i1, %3;\n\t"
		"@!p1 bra _While1;\n\t"
		"ld.u8 			c1, [%2];\n\t"
		"setp.ne.u32	p1, c1, 0;\n\t"
		"@!p1 bra _While1;\n\t"
		"st.u8 			[%1], c1;\n\t"
		"add" _UX " 	i1, i1, 1;\n\t"
		"add" _UX " 	%1, %1, 1;\n\t"
		"add" _UX " 	%2, %2, 1;\n\t"
		"bra.uni _While0;\n\t"

		//
		"_While1:\n\t"
		"setp.lt" _UX "	p1, i1, %3;\n\t"
		"@!p1 bra _Ret;\n\t"
		"st.u8 			[%1], 0;\n\t"
		"add" _UX "		i1, i1, 1;\n\t"
		"add" _UX "		%1, %1, 1;\n\t"
		"add" _UX "		%2, %2, 1;\n\t"
		"bra.uni _While1;\n\t"

		"_Ret:\n\t"
		: "=" __R(r) : __R(dest), __R(src), __R(n));
	return r;
#else
	register unsigned char *d = (unsigned char *)dest;
	register unsigned char *s = (unsigned char *)src;
	size_t i = 0;
	for (; i < n && *s; ++i, ++d, ++s) *d = *s;
	for (; i < n; ++i, ++d, ++s) *d = 0;
	return (char *)dest;
#endif
}

/* Append SRC onto DEST.  */
__device__ char *strcat_(char *__restrict dest, const char *__restrict src) {
#ifndef OMIT_PTX
	char *r;
	asm(
		".reg .pred p1;\n\t"
		".reg .u32 c1;\n\t"
		"mov" _UX "		%0, %1;\n\t"

		//
		"_While0:\n\t"
		"ld.u8 			c1, [%1];\n\t"
		"setp.ne.u32	p1, c1, 0;\n\t"
		"@!p1 bra _While1;\n\t"
		"add" _UX " 	%1, %1, 1;\n\t"
		"bra.uni _While0;\n\t"

		//
		"_While1:\n\t"
		"ld.u8 			c1, [%2];\n\t"
		"setp.ne.u32	p1, c1, 0;\n\t"
		"st.u8 			[%1], c1;\n\t"
		"@!p1 bra _Ret;\n\t"
		"add" _UX "		%1, %1, 1;\n\t"
		"add" _UX "		%2, %2, 1;\n\t"
		"bra.uni _While1;\n\t"

		"_Ret:\n\t"
		: "=" __R(r) : __R(dest), __R(src));
	return r;
#else
	register unsigned char *d = (unsigned char *)dest;
	register unsigned char *s = (unsigned char *)src;
	while (*d) d++;
	while (*s) { *d++ = *s++; } *d = *s;
	return (char *)dest;
#endif
}

/* Append no more than N characters from SRC onto DEST.  */
__device__ char *strncat_(char *__restrict dest, const char *__restrict src, size_t n) {
#ifndef OMIT_PTX
	char *r;
	asm(
		".reg .pred p1;\n\t"
		".reg .u32 c1;\n\t"
		"mov" _UX "		%0, %1;\n\t"

		//
		"_While0:\n\t"
		"ld.u8 			c1, [%1];\n\t"
		"setp.ne.u32	p1, c1, 0;\n\t"
		"@!p1 bra _While1;\n\t"
		"add" _UX " 	%1, %1, 1;\n\t"
		"bra.uni _While0;\n\t"

		//
		"_While1:\n\t"
		"ld.u8 			c1, [%2];\n\t"
		"setp.ne.u32	p1, c1, 0;\n\t"
		"setp.ne" _UX ".and p1, %3, 0, p1;\n\t"
		"@!p1 bra _Ret;\n\t"
		"st.u8 			[%1], c1;\n\t"
		"add" _UX "		%3, %3, -1;\n\t"
		"add" _UX "		%1, %1, 1;\n\t"
		"add" _UX "		%2, %2, 1;\n\t"
		"bra.uni _While1;\n\t"

		"_Ret:\n\t"
		"st.u8 			[%1], c1;\n\t"
		: "=" __R(r) : __R(dest), __R(src), __R(n));
	return r;
#else
	register unsigned char *d = (unsigned char *)dest;
	register unsigned char *s = (unsigned char *)src;
	while (*d) d++;
	while (*s && !--n) { *d++ = *s++; } *d = *s;
	return (char *)dest;
#endif
}

/* Compare S1 and S2.  */
__device__ int strcmp_(const char *s1, const char *s2) {
#ifndef OMIT_PTX
	int r;
	asm(
		".reg .pred p1;\n\t"
		".reg .u32 c1, c2;\n\t"

		//
		"_While0:\n\t"
		"ld.u8 			c1, [%1];\n\t"
		"setp.ne.u32	p1, c1, 0;\n\t"
		"ld.u8 			c2, [%2];\n\t"
		"@!p1 bra _Ret;\n\t"
		"setp.eq.u32	p1, c1, c2;\n\t"
		"@!p1 bra _Ret;\n\t"
		"add" _UX "		%1, %1, 1;\n\t"
		"add" _UX "		%2, %2, 1;\n\t"
		"bra.uni _While0;\n\t"

		"_Ret:\n\t"
		"sub.u32 		%0, c1, c2;\n\t"
		: "=" __I(r) : __R(s1), __R(s2));
	return r;
#else
	register unsigned char *a = (unsigned char *)s1;
	register unsigned char *b = (unsigned char *)s2;
	while (*a && *a == *b) { a++; b++; }
	return *a - *b;
#endif
}
/* Compare S1 and S2. Case insensitive.  */
__device__ int stricmp_(const char *s1, const char *s2) {
#ifndef OMIT_PTX
	int r;
	asm(
		".reg .pred p1;\n\t"
		".reg .u32 c1, c2;\n\t"
		".reg " _UX " u2l;\n\t"
#if _WIN64
		".reg " _UX " u2;\n\t"
#endif
		"cvta.const" _UX " u2l, __curtUpperToLower;\n\t"

		//
		"_While0:\n\t"
		"ld.u8 			c1, [%1];\n\t"
		"setp.ne.u32	p1, c1, 0;\n\t"
#if _WIN64
		"cvt.u64.u32	u2, c1;\n\t"
		"add" _UX "		u2, u2, u2l; ld.u8 c1, [u2];\n\t"
#else
		"add" _UX " 	c1, c1, u2l; ld.u8 c1, [c1];\n\t"
#endif
		"ld.u8 			c2, [%2];\n\t"
#if _WIN64
		"cvt.u64.u32	u2, c2;\n\t"
		"add" _UX "		u2, u2, u2l; ld.u8 c2, [u2];\n\t"
#else
		"add" _UX " 	c2, c2, u2l; ld.u8 c2, [c2];\n\t"
#endif
		"@!p1 bra _Ret;\n\t"
		"setp.eq.u32	p1, c1, c2;\n\t"
		"@!p1 bra _Ret;\n\t"
		"add" _UX "		%1, %1, 1;\n\t"
		"add" _UX "		%2, %2, 1;\n\t"
		"bra.uni _While0;\n\t"

		"_Ret:\n\t"
		"sub.u32 		%0, c1, c2;\n\t"
		: "=" __I(r) : __R(s1), __R(s2), __R(__curtUpperToLower));
	return r;
#else
	register unsigned char *a = (unsigned char *)s1;
	register unsigned char *b = (unsigned char *)s2;
	while (*a && __curtUpperToLower[*a] == __curtUpperToLower[*b]) { a++; b++; }
	return __curtUpperToLower[*a] - __curtUpperToLower[*b];
#endif
}

/* Compare N characters of S1 and S2.  */
__device__ int strncmp_(const char *s1, const char *s2, size_t n) {
#ifndef OMIT_PTX
	int r;
	asm(
		".reg .pred p1, p2;\n\t"
		".reg .u32 c1, c2;\n\t"
		//
		"_While0:\n\t"
		"setp.gt" _UX "	p2, %3, 0;\n\t"
		"@!p2 bra _Ret;\n\t"
		"ld.u8 			c1, [%1];\n\t"
		"setp.ne.u32	p1, c1, 0;\n\t"
		"ld.u8 			c2, [%2];\n\t"
		"@!p1 bra _Ret;\n\t"
		"setp.eq.u32	p1, c1, c2;\n\t"
		"@!p1 bra _Ret;\n\t"
		"add" _UX "		%3, %3, -1;\n\t"
		"add" _UX "		%1, %1, 1;\n\t"
		"add" _UX "		%2, %2, 1;\n\t"
		"bra.uni _While0;\n\t"

		"_Ret:\n\t"
		"@!p2 mov.u32 	%0, 0;\n\t"
		"@p2 sub.u32 	%0, c1, c2;\n\t"
		: "=" __I(r) : __R(s1), __R(s2), __R(n));
	return r;
#else
	register unsigned char *a = (unsigned char *)s1;
	register unsigned char *b = (unsigned char *)s2;
	while (--n > 0 && *a && *a == *b) { a++; b++; }
	return !n ? 0 : *a - *b;
#endif
}

/* Compare N characters of S1 and S2. Case insensitive.  */
__device__ int strnicmp_(const char *s1, const char *s2, size_t n) {
#ifndef OMIT_PTX
	int r;
	asm(
		".reg .pred p1, p2;\n\t"
		".reg .u32 c1, c2;\n\t"
		".reg " _UX " u2l;\n\t"
#if _WIN64
		".reg " _UX " u2;\n\t"
#endif
		"cvta.const" _UX " u2l, __curtUpperToLower;\n\t"

		//
		"_While0:\n\t"
		"setp.gt" _UX "	p2, %3, 0;\n\t"
		"@!p2 bra _Ret;\n\t"
		"ld.u8 			c1, [%1];\n\t"
		"setp.ne.u32	p1, c1, 0;\n\t"
#if _WIN64
		"cvt.u64.u32	u2, c1;\n\t"
		"add" _UX "		u2, u2, u2l; ld.u8 c1, [u2];\n\t"
#else
		"add" _UX " 	c1, c1, u2l; ld.u8 c1, [c1];\n\t"
#endif
		"ld.u8 			c2, [%2];\n\t"
#if _WIN64
		"cvt.u64.u32	u2, c2;\n\t"
		"add" _UX "		u2, u2, u2l; ld.u8 c2, [u2];\n\t"
#else
		"add" _UX " 	c2, c2, u2l; ld.u8 c2, [c2];\n\t"
#endif
		"@!p1 bra _Ret;\n\t"
		"setp.eq.u32	p1, c1, c2;\n\t"
		"@!p1 bra _Ret;\n\t"
		"add" _UX "		%3, %3, -1;\n\t"
		"add" _UX "		%1, %1, 1;\n\t"
		"add" _UX "		%2, %2, 1;\n\t"
		"bra.uni _While0;\n\t"

		"_Ret:\n\t"
		"@!p2 mov.u32 	%0, 0;\n\t"
		"@p2 sub.u32 	%0, c1, c2;\n\t"
		: "=" __I(r) : __R(s1), __R(s2), __R(n), __R(__curtUpperToLower));
	return r;
#else
	register unsigned char *a = (unsigned char *)s1;
	register unsigned char *b = (unsigned char *)s2;
	while (n-- > 0 && *a && __curtUpperToLower[*a] == __curtUpperToLower[*b]) { a++; b++; }
	return !n ? 0 : __curtUpperToLower[*a] - __curtUpperToLower[*b];
#endif
}

/* Compare the collated forms of S1 and S2.  */
__device__ int strcoll_(const char *s1, const char *s2) {
	panic("Not Implemented");
	return -1;
}

/* Put a transformation of SRC into no more than N bytes of DEST.  */
__device__ size_t strxfrm_(char *__restrict dest, const char *__restrict src, size_t n) {
	panic("Not Implemented");
	return 0;
}

/* Duplicate S, returning an identical malloc'd string.  */
__device__ char *strdup_(const char *s) {
	const char *old = s;
	size_t len = strlen(old) + 1;
	char *new_ = (char *)malloc(len);
	(char *)memcpy(new_, old, len);
	return new_;
}

/* Return a malloc'd copy of at most N bytes of STRING.  The resultant string is terminated even if no null terminator appears before STRING[N].  */
__device__ char *strndup_(const char *s, size_t n) {
	const char *old = s;
	size_t len = strnlen(old, n);
	char *new_ = (char *)malloc(len + 1);
	new_[len] = '\0';
	(char *)memcpy(new_, old, len);
	return new_;
}

/* Find the first occurrence of C in S.  */
__device__ char *strchr_(const char *s, int c) {
#ifndef OMIT_PTX
	char *r;
	asm(
		".reg .pred p1, p2;\n\t"
		".reg .u32 c1;\n\t"
		".reg " _UX " u2l;\n\t"
#if _WIN64
		".reg " _UX " u2;\n\t"
#endif
		"cvta.const" _UX " u2l, __curtUpperToLower;\n\t"

#if _WIN64
		"cvt.u64.u32	u2, %2;\n\t"
		"add" _UX "		u2, u2, u2l; ld.u8 %2, [u2];\n\t"
#else
		"add" _UX " 	%2, %2, u2l; ld.u8 %2, [%2];\n\t"
#endif
		//
		"_While0:\n\t"
		"ld.u8 			c1, [%1];\n\t"
		"setp.ne.u32	p2, c1, 0;\n\t"
		"@!p2 bra _Ret;\n\t"
#if _WIN64
		"cvt.u64.u32	u2, c1;\n\t"
		"add" _UX " 	u2, u2, u2l; ld.u8 c1, [u2];\n\t"
#else
		"add" _UX " 	c1, c1, u2l; ld.u8 c1, [c1];\n\t"
#endif
		"setp.ne.u32	p1, c1, %2;\n\t"
		"@!p1 bra _Ret;\n\t"
		"add" _UX " 		%1, %1, 1;\n\t"
		"bra.uni _While0;\n\t"

		"_Ret:\n\t"
		"@p2 mov" _UX " %0, %1;\n\t"
		"@!p2 mov" _UX " %0, 0;\n\t"
		: "=" __R(r) : __R(s), __I(c), __R(__curtUpperToLower));
	return r;
#else
	register unsigned char *s1 = (unsigned char *)s;
	register unsigned char l = (unsigned char)__curtUpperToLower[c];
	while (*s1 && __curtUpperToLower[*s1] != l) s1++;
	return (char *)(*s1 ? s1 : nullptr);
#endif
}

/* Find the last occurrence of C in S.  */
__device__ char *strrchr_(const char *s, int c) {
#ifndef OMIT_PTX
	char *r;
	asm(
		".reg .pred p1;\n\t"
		".reg .u32 c1;\n\t"
		"mov" _UX " 	%0, 0;\n\t"

		//
		"_While0:\n\t"
		"ld.u8 			c1, [%1];\n\t"
		"setp.ne.u32	p1, c1, 0;\n\t"
		"@!p1 bra _Ret;\n\t"
		"setp.eq.u32	p1, c1, %2;\n\t"
		"@p1 mov" _UX "	%0, %1;\n\t"
		"add" _UX " 	%1, %1, 1;\n\t"
		"bra.uni _While0;\n\t"

		"_Ret:\n\t"
		: "=" __R(r) : __R(s), __I(c));
	return r;
#else
	char *save; char c1;
	for (save = (char *)0; c1 = *s; s++) if (c1 == c) save = (char *)s;
	return save;
#endif
}

/* Return the length of the initial segment of S which consists entirely of characters not in REJECT.  */
__device__ size_t strcspn_(const char *s, const char *reject) {
	panic("Not Implemented");
	return 0;
}

/* Return the length of the initial segment of S which consists entirely of characters in ACCEPT.  */
__device__ size_t strspn_(const char *s, const char *accept) {
	panic("Not Implemented");
	return 0;
}

/* Find the first occurrence in S of any character in ACCEPT.  */
__device__ char *strpbrk_(const char *s, const char *accept) {
#ifndef OMIT_PTX
	char *r;
	asm(
		".reg .pred p1, p2;\n\t"
		".reg .u32 c1, c2;\n\t"
		".reg " _UX " scanp1;\n\t"
		"mov" _UX "		%0, 0;\n\t"

		// while
		"_While0:\n\t"
		"ld.u8 			c1, [%1];\n\t"
		"setp.ne.u32	p1, c1, 0;\n\t"
		"@!p1 bra _End;\n\t"

		// for
		"mov" _UX " 	scanp1, %2;\n\t"
		"_For0:\n\t"
		"ld.u8 			c2, [scanp1];\n\t"
		"setp.ne.u32	p2, c2, 0;\n\t"
		"@!p2 bra _nWhile0;\n\t"
		"setp.eq.u32	p2, c1, c2;\n\t"
		"@p2 bra _Ret;\n\t"
		"add" _UX " 	scanp1, scanp1, 1;\n\t"
		"bra.uni _For0;\n\t"

		// ^while
		"_nWhile0:\n\t"
		"add" _UX " 	%1, %1, 1;\n\t"
		"bra.uni _While0;\n\t"

		"_Ret:\n\t"
		"mov" _UX " 	%0, %1;\n\t"
		"_End:\n\t"
		: "=" __R(r) : __R(s), __R(accept));
	return r;
#else
	register const char *scanp;
	register int c, c2;
	while (c = *s++) {
		for (scanp = accept; c2 = *scanp++;)
			if (c2 == c)
				return (char *)(s - 1);
	}
	return nullptr;
#endif
}

/* Find the first occurrence of NEEDLE in HAYSTACK.  */
__device__ char *strstr_(const char *haystack, const char *needle) {
	if (!*needle) return (char *)haystack;
	char *p1 = (char *)haystack, *p2 = (char *)needle;
	char *p1Adv = (char *)haystack;
	while (*++p2)
		p1Adv++;
	while (*p1Adv) {
		char *p1Begin = p1;
		p2 = (char *)needle;
		while (*p1 && *p2 && *p1 == *p2) {
			p1++;
			p2++;
		}
		if (!*p2)
			return p1Begin;
		p1 = p1Begin + 1;
		p1Adv++;
	}
	return nullptr;
}

/* Divide S into tokens separated by characters in DELIM.  */
__device__ char *strtok_(char *__restrict s, const char *__restrict delim) {
	panic("Not Implemented");
	return nullptr;
}

/* Return the length of S.  */
__device__ size_t strlen_(const char *s) {
#ifndef OMIT_PTX
	size_t r;
	asm(
		".reg .pred p1;\n\t"
		".reg " _UX " s2;\n\t"
		".reg " _BX " r;\n\t"
		".reg .b16 c;\n\t"
		"mov" _BX "		%0, 0;\n\t"

		"setp.eq" _UX "	p1, %1, 0;\n\t"
		"@p1 bra _End;\n\t"
		"mov" _UX "		s2, %1;\n\t"

		"_While:\n\t"
		"ld.u8			c, [s2];\n\t"
		//"and.b16		c, c, 255;\n\t"
		"setp.ne.u16	p1, c, 0;\n\t"
		"@!p1 bra _Value;\n\t"
		"add" _UX "		s2, s2, 1;\n\t"
		"bra.uni _While;\n\t"

		"_Value:\n\t"
		"sub" _UX "		r, s2, %1;\n\t"
		"and" _BX "		%0, r, 0x3fffffff;\n\t"
		"_End:\n\t"
		: "=" __R(r) : __R(s));
	return r;
#else
	if (!s) return 0;
	register const char *s2 = s;
	while (*s2) { s2++; }
	return 0x3fffffff & (int)(s2 - s);
#endif
}

/* Return the length of S.  */
__device__ size_t strlen16_(const void *s) {
#ifndef OMIT_PTX
	size_t r;
	asm(
		".reg .pred p1;\n\t"
		".reg " _UX " s2;\n\t"
		".reg " _BX " r;\n\t"
		".reg .b16 c;\n\t"
		"mov" _BX "		%0, 0;\n\t"

		"setp.eq" _UX "	p1, %1, 0;\n\t"
		"@p1 bra _End;\n\t"
		"mov" _UX "		s2, %1;\n\t"

		"_While:\n\t"
		"ld.u16			c, [s2];\n\t"
		"setp.ne.u16	p1, c, 0;\n\t"
		"@!p1 bra _Value;\n\t"
		"add" _UX "		s2, s2, 2;\n\t"
		"bra.uni _While;\n\t"

		"_Value:\n\t"
		"sub" _UX "		r, s2, %1;\n\t"
		"shr" _UX "		r, r, 1;\n\t"
		"and" _BX "		%0, r, 0x3fffffff;\n\t"
		"_End:\n\t"
		: "=" __R(r) : __R(s));
	return r;
#else
	if (!s) return 0;
	register const short *s2 = (const short *)s;
	while (*s2) { s2++; }
	return 0x3fffffff & ((int)(s2 - (const short *)s) >> 1);
#endif
}

/* Find the length of STRING, but scan at most MAXLEN characters. If no '\0' terminator is found in that many characters, return MAXLEN.  */
__device__ size_t strnlen_(const char *s, size_t maxlen) {
#ifndef OMIT_PTX
	size_t r;
	asm(
		".reg .pred p1;\n\t"
		".reg " _UX " s2, s2m;\n\t"
		".reg " _BX " r;\n\t"
		".reg .b16 c;\n\t"
		"mov" _BX "		%0, 0;\n\t"

		"setp.eq" _UX "	p1, %1, 0;\n\t"
		"@p1 bra _End;\n\t"
		"mov" _UX "		s2, %1;\n\t"
		"add" _UX "		s2m, %1, %2;\n\t"

		"_While:\n\t"
		"ld.u8 			c, [s2];\n\t"
		//"and.b16  	c, c, 255;\n\t"
		"setp.ne.u16	p1, c, 0;\n\t"
		"setp.lt.and" _UX " p1, s2, s2m, p1;\n\t"
		"@!p1 bra _Value;\n\t"
		"add" _UX " 	s2, s2, 1;\n\t"
		"bra.uni _While;\n\t"

		"_Value:\n\t"
		"sub" _UX "		r, s2, %1;\n\t"
		"and" _BX "		%0, r, 0x3fffffff;\n\t"
		"_End:\n\t"
		: "=" __R(r) : __R(s), __R(maxlen));
	return r;
#else
	if (!s) return 0;
	register const char *s2 = s;
	register const char *s2m = s + maxlen;
	while (*s2 && s2 < s2m) { s2++; }
	return 0x3fffffff & (int)(s2 - s);
#endif
}

__device__ void *mempcpy_(void *__restrict dest, const void *__restrict src, size_t n) {
	panic("Not Implemented");
	return nullptr;
}

/* Return a string describing the meaning of the `errno' code in ERRNUM.  */
__device__ char *strerror_(int errnum) {
	return (char *)"ERROR";
}

#pragma region strbld

#define TYPE_RADIX		0 // non-decimal integer types.  %x %o
#define TYPE_FLOAT		1 // Floating point.  %f
#define TYPE_EXP		2 // Exponentional notation. %e and %E
#define TYPE_GENERIC	3 // Floating or exponential, depending on exponent. %g
#define TYPE_SIZE		4 // Return number of characters processed so far. %n
#define TYPE_STRING	5 // Strings. %s
#define TYPE_DYNSTRING	6 // Dynamically allocated strings. %z
#define TYPE_PERCENT	7 // Percent symbol. %%
#define TYPE_CHARX		8 // Characters. %c
// The rest are extensions, not normally found in printf()
#define TYPE_SQLESCAPE	9 // Strings with '\'' doubled.  %q
#define TYPE_SQLESCAPE2 10 // Strings with '\'' doubled and enclosed in '', NULL pointers replaced by SQL NULL.  %Q
#define TYPE_TOKEN		11 // a pointer to a Token structure
#define TYPE_SRCLIST	12 // a pointer to a SrcList
#define TYPE_POINTER	13 // The %p conversion
#define TYPE_SQLESCAPE3 14 // %w -> Strings with '\"' doubled
#define TYPE_ORDINAL	15 // %r -> 1st, 2nd, 3rd, 4th, etc.  English only
#define TYPE_DECIMAL	16 // %d or %u, but not %x, %o
#define TYPE_INVALID	17 // Any unrecognized conversion type

/* An "etByte" is an 8-bit unsigned value. */
typedef unsigned char etByte;

/* Each builtin conversion character (ex: the 'd' in "%d") is described by an instance of the following structure */
typedef struct info_t { // Information about each format field
	char fmtType;	// The format field code letter
	etByte base;	// The base for radix conversion
	etByte flags;	// One or more of FLAG_ constants below
	etByte type;	// Conversion paradigm
	etByte charset;	// Offset into aDigits[] of the digits string
	etByte prefix;	// Offset into aPrefix[] of the prefix string
} info_t;

/* Allowed values for et_info.flags */
#define FLAG_SIGNED	1 // True if the value to convert is signed
#define FLAG_STRING	4 // Allow infinite precision

// The following table is searched linearly, so it is good to put the most frequently used conversion types first.
static __host_constant__ const char _digits[] = "0123456789ABCDEF0123456789abcdef";
static __host_constant__ const char _prefix[] = "-x0\000X0";
static __host_constant__ const info_t _info[] = {
	{ 'd', 10, 1, TYPE_DECIMAL,    0,  0 },
	{ 's',  0, 4, TYPE_STRING,     0,  0 },
	{ 'g',  0, 1, TYPE_GENERIC,    30, 0 },
	{ 'z',  0, 4, TYPE_DYNSTRING,  0,  0 },
	{ 'q',  0, 4, TYPE_SQLESCAPE,  0,  0 },
	{ 'Q',  0, 4, TYPE_SQLESCAPE2, 0,  0 },
	{ 'w',  0, 4, TYPE_SQLESCAPE3, 0,  0 },
	{ 'c',  0, 0, TYPE_CHARX,      0,  0 },
	{ 'o',  8, 0, TYPE_RADIX,      0,  2 },
	{ 'u', 10, 0, TYPE_DECIMAL,    0,  0 },
	{ 'x', 16, 0, TYPE_RADIX,      16, 1 },
	{ 'X', 16, 0, TYPE_RADIX,      0,  4 },
#ifndef OMIT_FLOATING_POINT
	{ 'f',  0, 1, TYPE_FLOAT,      0,  0 },
	{ 'e',  0, 1, TYPE_EXP,        30, 0 },
	{ 'E',  0, 1, TYPE_EXP,        14, 0 },
	{ 'G',  0, 1, TYPE_GENERIC,    14, 0 },
#endif
	{ 'i', 10, 1, TYPE_DECIMAL,    0,  0 },
	{ 'n',  0, 0, TYPE_SIZE,       0,  0 },
	{ '%',  0, 0, TYPE_PERCENT,    0,  0 },
	{ 'p', 16, 0, TYPE_POINTER,    0,  1 },
	// All the rest are undocumented and are for internal use only
	{ 'T',  0, 0, TYPE_TOKEN,      0,  0 },
	{ 'S',  0, 0, TYPE_SRCLIST,    0,  0 },
	{ 'r', 10, 1, TYPE_ORDINAL,    0,  0 },
};

#ifndef OMIT_FLOATING_POINT
/* "*val" is a double such that 0.1 <= *val < 10.0 Return the ascii code for the leading digit of *val, then
** multiply "*val" by 10.0 to renormalize.
**
** Example:
**     input:     *val = 3.14159
**     output:    *val = 1.4159    function return = '3'
**
** The counter *cnt is incremented each time.  After counter exceeds 16 (the number of significant digits in a 64-bit float) '0' is
** always returned.
*/
static __host_device__ char getDigit(long_double *val, int *cnt) {
	if (*cnt <= 0) return '0';
	(*cnt)--;
	int digit = (int)*val;
	long_double d = digit;
	digit += '0';
	*val = (*val - d)*10.0;
	return (char)digit;
}
#endif

/* Set the StrAccum object to an error mode. */
static __host_device__ void strbldSetError(strbld_t *b, unsigned char error) {
	assert(error == STRACCUM_NOMEM || error == STRACCUM_TOOBIG);
	b->error = error;
	b->size = 0;
}

#define BUFSIZE PRINT_BUF_SIZE  // Size of the output buffer

/* Render a string given by "fmt" into the strbld_t object. */
__host_device__ void strbldAppendFormatv(strbld_t *b, const char *fmt, va_list va) {
	char buf[BUFSIZE]; // Conversion buffer
	char *bufpt = nullptr; // Pointer to the conversion buffer

	bool noArgs; void *args = nullptr; // Arguments for SQLITE_PRINTF_SQLFUNC
	if (b->flags & PRINTF_SQLFUNC) { noArgs = false; args = va_arg(va, void *); }
	else noArgs = true;

	int c; // Next character in the format string
	int width = 0; // Width of the current field
	int length = 0; // Length of the field
	etByte flag_leftjustify;	// True if "-" flag is present
	etByte flag_prefix;			// '+' or ' ' or 0 for prefix
	etByte flag_alternateform;	// True if "#" flag is present
	etByte flag_altform2;		// True if "!" flag is present
	etByte flag_zeropad;		// True if field width constant starts with zero
	etByte flag_long;			// 1 for the "l" flag, 2 for "ll", 0 by default
	etByte done;				// Loop termination flag
	etByte thousand;			// Thousands separator for %d and %u
	etByte type = TYPE_INVALID;// Conversion paradigm
	for (; (c = *fmt); ++fmt) {
		if (c != '%') {
			bufpt = (char *)fmt;
#if HAVE_STRCHRNUL
			fmt = strchrnul(fmt, '%');
#else
			do { fmt++; } while (*fmt && *fmt != '%');
#endif
			strbldAppend(b, bufpt, (int)(fmt - bufpt));
			if (!*fmt) break;
		}
		if (!(c = *++fmt)) {
			strbldAppend(b, "%", 1);
			break;
		}
		// Find out what flags are present
		flag_leftjustify = flag_prefix = thousand = flag_alternateform = flag_altform2 = flag_zeropad = 0;
		done = false; // Loop termination flag
		do {
			switch (c) {
			case '-': flag_leftjustify = true; break;
			case '+': flag_prefix = '+'; break;
			case ' ': flag_prefix = ' '; break;
			case '#': flag_alternateform = true; break;
			case '!': flag_altform2 = true; break;
			case '0': flag_zeropad = true; break;
			case ',': thousand = ','; break;
			default: done = true; break;
			}
		} while (!done && (c = *++fmt));
		// Get the field width
		if (c == '*') {
			width = noArgs ? va_arg(va, int) : (int)__extsystem.getIntegerArg(args);
			if (width < 0) {
				flag_leftjustify = true;
				width = width >= -2147483647 ? -width : 0;
			}
			c = *++fmt;
		}
		else {
			unsigned wx = 0;
			while (c >= '0' && c <= '9') {
				wx = wx * 10 + c - '0';
				c = *++fmt;
			}
			TESTCASE_(wx > 0x7fffffff);
			width = wx & 0x7fffffff;
		}
		assert(width >= 0);
#ifdef LIBCU_PRINTF_PRECISION_LIMIT
		if (width > LIBCU_PRINTF_PRECISION_LIMIT)
			width = LIBCU_PRINTF_PRECISION_LIMIT;
#endif

		// Get the precision
		int precision; // Precision of the current field
		if (c == '.') {
			c = *++fmt;
			if (c == '*') {
				precision = noArgs ? va_arg(va, int) : (int)__extsystem.getIntegerArg(args);
				c = *++fmt;
				if (precision < 0)
					precision = precision >= -2147483647 ? -precision : -1;
			}
			else {
				unsigned px = 0;
				while (c >= '0' && c <= '9') {
					px = px * 10 + c - '0';
					c = *++fmt;
				}
				TESTCASE_(px > 0x7fffffff);
				precision = px & 0x7fffffff;
			}
		}
		else precision = -1;
		assert(precision >= -1);
#ifdef LIBCU_PRINTF_PRECISION_LIMIT
		if (precision > LIBCU_PRINTF_PRECISION_LIMIT)
			precision = LIBCU_PRINTF_PRECISION_LIMIT;
#endif

		// Get the conversion type modifier
		if (c == 'l') {
			flag_long = 1;
			c = *++fmt;
			if (c == 'l') {
				flag_long = 2;
				c = *++fmt;
			}
		}
		else flag_long = 0;
		// Fetch the info entry for the field
		const info_t *info = &_info[0]; // Pointer to the appropriate info structure
		type = TYPE_INVALID; // Conversion paradigm
		int idx; for (idx = 0; idx < ARRAYSIZE_(_info); idx++) {
			if (c == _info[idx].fmtType) {
				info = &_info[idx];
				type = info->type;
				break;
			}
		}

		// At this point, variables are initialized as follows:
		//   flag_alternateform          TRUE if a '#' is present.
		//   flag_altform2               TRUE if a '!' is present.
		//   flag_prefix                 '+' or ' ' or zero
		//   flag_leftjustify            TRUE if a '-' is present or if the field width was negative.
		//   flag_zeropad                TRUE if the width began with 0.
		//   flag_long                   1 for "l", 2 for "ll"
		//   width                       The specified field width.  This is always non-negative.  Zero is the default.
		//   precision                   The specified precision.  The default is -1.
		//   type                        The class of the conversion.
		//   info                        Pointer to the appropriate info struct.
		char prefix; // Prefix character.  "+" or "-" or " " or '\0'.
		unsigned long long longvalue; // Value for integer types
		long_double realvalue; // Value for real types
		char *extra = nullptr; // Malloced memory used by some conversion
		char *out; // Rendering buffer
		int outLength; // Size of the rendering buffer
#ifndef OMIT_FLOATING_POINT
		int exp, e2; // exponent of real numbers
		int nsd; // Number of significant digits returned
		double rounder; // Used for rounding floating point values
		etByte flag_dp; // True if decimal point should be shown
		etByte flag_rtz; // True if trailing zeros should be removed
#endif
		switch (type) {
		case TYPE_POINTER:
			flag_long = sizeof(char *) == sizeof(int64_t) ? 2 : sizeof(char *) == sizeof(long int) ? 1 : 0;
			// Fall through into the next case
		case TYPE_ORDINAL:
		case TYPE_RADIX:
			thousand = 0;
			// Fall through into the next case
		case TYPE_DECIMAL:
			if (info->flags & FLAG_SIGNED) {
				int64_t v = noArgs ? flag_long ? flag_long == 2 ? va_arg(va, int64_t) : va_arg(va, long int) : va_arg(va, int) : __extsystem.getIntegerArg(args);
				if (v < 0) { longvalue = (v == LLONG_MIN ? ((uint64_t)1) << 63 : -v); prefix = '-'; }
				else { longvalue = v; prefix = flag_prefix; }
			}
			else {
				longvalue = noArgs ? flag_long ? flag_long == 2 ? va_arg(va, uint64_t) : va_arg(va, unsigned long int) : va_arg(va, unsigned int) : (uint64_t)__extsystem.getIntegerArg(args);
				prefix = 0;
			}
			if (longvalue == 0) flag_alternateform = false;
			if (flag_zeropad && precision < width - (prefix != 0))
				precision = width - (prefix != 0);
			if (precision < BUFSIZE - 10 - BUFSIZE / 3) {
				outLength = BUFSIZE;
				out = buf;
			}
			else {
				uint64_t n = (uint64_t)precision + 10 + precision / 3;
				out = extra = (char *)malloc(n);
				if (!out) {
					strbldSetError(b, STRACCUM_NOMEM);
					return;
				}
				outLength = (int)n;
			}
			bufpt = &out[outLength - 1];
			if (type == TYPE_ORDINAL) {
				static const char ord[] = "thstndrd";
				int x = (int)(longvalue % 10);
				if (x >= 4 || (longvalue / 10) % 10 == 1) x = 0;
				*(--bufpt) = ord[x * 2 + 1];
				*(--bufpt) = ord[x * 2];
			}
			{
				const char *cset = &_digits[info->charset]; // Use registers for speed
				uint8_t base = info->base;
				do { *(--bufpt) = cset[longvalue%base]; longvalue = longvalue / base; } while (longvalue > 0); // Convert to ascii
			}
			length = (int)(&out[outLength - 1] - bufpt);
			while (precision > length) { *(--bufpt) = '0'; length++; } // Zero pad
			if (thousand) {
				int nn = (length - 1) / 3; // Number of "," to insert
				int ix = (length - 1) % 3 + 1;
				bufpt -= nn;
				for (idx = 0; nn > 0; idx++) {
					bufpt[idx] = bufpt[idx + nn];
					ix--;
					if (!ix) { bufpt[++idx] = thousand; nn--; ix = 3; }
				}
			}
			if (prefix) *(--bufpt) = prefix; // Add sign
			if (flag_alternateform && info->prefix) { // Add "0" or "0x"
				const char *pre = &_prefix[info->prefix];
				char x; for (; (x = *pre); pre++) *(--bufpt) = x;
			}
			length = (int)(&out[outLength - 1] - bufpt);
			break;
		case TYPE_FLOAT:
		case TYPE_EXP:
		case TYPE_GENERIC:
			realvalue = noArgs ? va_arg(va, double) : __extsystem.getDoubleArg(args);
#ifdef OMIT_FLOATING_POINT
			length = 0;
#else
			if (precision < 0) precision = 6; // Set default precision
			if (realvalue < 0.0) { realvalue = -realvalue; prefix = '-'; }
			else prefix = flag_prefix;
			if (type == TYPE_GENERIC && precision > 0) precision--;
			TESTCASE_(precision > 0xfff);
			for (idx = precision & 0xfff, rounder = 0.5; idx > 0; idx--, rounder *= 0.1) {}
			if (type == TYPE_FLOAT) realvalue += rounder;
			// Normalize realvalue to within 10.0 > realvalue >= 1.0
			exp = 0;
			if (isnan((double)realvalue)) {
				bufpt = (char *)"NaN";
				length = 3;
				break;
			}
			if (realvalue > 0.0) {
				long_double scale = 1.0;
				while (realvalue >= 1e100*scale && exp <= 350) { scale *= 1e100; exp += 100; }
				while (realvalue >= 1e10*scale && exp <= 350) { scale *= 1e10; exp += 10; }
				while (realvalue >= 10.0*scale && exp <= 350) { scale *= 10.0; exp++; }
				realvalue /= scale;
				while (realvalue < 1e-8) { realvalue *= 1e8; exp -= 8; }
				while (realvalue < 1.0) { realvalue *= 10.0; exp--; }
				if (exp > 350) {
					bufpt = buf;
					buf[0] = prefix;
					memcpy(buf + (prefix != 0), "Inf", 4);
					length = 3 + (prefix != 0);
					break;
				}
			}
			bufpt = buf;
			// If the field type is etGENERIC, then convert to either etEXP or etFLOAT, as appropriate.
			if (type != TYPE_FLOAT) {
				realvalue += rounder;
				if (realvalue >= 10.0) { realvalue *= 0.1; exp++; }
			}
			if (type == TYPE_GENERIC) {
				flag_rtz = !flag_alternateform;
				if (exp < -4 || exp > precision) type = TYPE_EXP;
				else { precision = precision - exp; type = TYPE_FLOAT; }
			}
			else flag_rtz = flag_altform2;
			e2 = type == TYPE_EXP ? 0 : exp;
			if (MAX_(e2, 0) + (int64_t)precision + (int64_t)width > BUFSIZE - 15) {
				bufpt = extra = (char *)malloc(MAX_(e2, 0) + (int64_t)precision + (int64_t)width + 15);
				if (!bufpt) {
					strbldSetError(b, STRACCUM_NOMEM);
					return;
				}
			}
			out = bufpt;
			nsd = 16 + flag_altform2 * 10;
			flag_dp = (precision > 0 ? 1 : 0) | flag_alternateform | flag_altform2;
			// The sign in front of the number
			if (prefix) *(bufpt++) = prefix;
			// Digits prior to the decimal point
			if (e2 < 0) *(bufpt++) = '0';
			else for (; e2 >= 0; e2--) *(bufpt++) = getDigit(&realvalue, &nsd);
			// The decimal point
			if (flag_dp) *(bufpt++) = '.';
			// "0" digits after the decimal point but before the first significant digit of the number
			for (e2++; e2 < 0; precision--, e2++) { assert(precision > 0); *(bufpt++) = '0'; }
			// Significant digits after the decimal point
			while (precision-- > 0) *(bufpt++) = getDigit(&realvalue, &nsd);
			// Remove trailing zeros and the "." if no digits follow the "."
			if (flag_rtz && flag_dp) {
				while (bufpt[-1] == '0') *(--bufpt) = 0;
				assert(bufpt > out);
				if (bufpt[-1] == '.') {
					if (flag_altform2) *(bufpt++) = '0';
					else *(--bufpt) = 0;
				}
			}
			// Add the "eNNN" suffix
			if (type == TYPE_EXP) {
				*(bufpt++) = _digits[info->charset];
				if (exp < 0) { *(bufpt++) = '-'; exp = -exp; }
				else *(bufpt++) = '+';
				if (exp >= 100) { *(bufpt++) = (char)(exp / 100 + '0'); exp %= 100; } // 100's digit
				*(bufpt++) = (char)(exp / 10 + '0'); // 10's digit
				*(bufpt++) = (char)(exp % 10 + '0'); // 1's digit
			}
			*bufpt = 0;

			// The converted number is in buf[] and zero terminated. Output it. Note that the number is in the usual order, not reversed as with integer conversions.
			length = (int)(bufpt - out);
			bufpt = out;

			// Special case:  Add leading zeros if the flag_zeropad flag is set and we are not left justified
			if (flag_zeropad && !flag_leftjustify && length < width) {
				int pad = width - length;
				for (idx = width; idx >= pad; idx--) bufpt[idx] = bufpt[idx - pad];
				idx = (prefix != 0);
				while (pad--) bufpt[idx++] = '0';
				length = width;
			}
#endif /* !defined(OMIT_FLOATING_POINT) */
			break;
		case TYPE_SIZE:
			if (noArgs) *(va_arg(va, int*)) = (int)b->size;
			length = width = 0;
			break;
		case TYPE_PERCENT:
			buf[0] = '%';
			bufpt = buf;
			length = 1;
			break;
		case TYPE_CHARX:
			if (noArgs) c = va_arg(va, int); else { bufpt = __extsystem.getStringArg(args); c = bufpt ? bufpt[0] : 0; }
			if (precision > 1) {
				width -= precision - 1;
				if (width > 1 && !flag_leftjustify) {
					strbldAppendChar(b, width - 1, ' ');
					width = 0;
				}
				strbldAppendChar(b, precision - 1, c);
			}
			length = 1;
			buf[0] = (char)c;
			bufpt = buf;
			break;
		case TYPE_STRING:
		case TYPE_DYNSTRING:
			if (noArgs) bufpt = va_arg(va, char*); else { bufpt = __extsystem.getStringArg(args); type = TYPE_STRING; }
			if (!bufpt) bufpt = (char *)"";
			else if (type == TYPE_DYNSTRING) extra = bufpt;
			if (precision >= 0) for (length = 0; length < precision && bufpt[length]; length++) {}
			else length = 0x7fffffff & strlen(bufpt);
			break;
		case TYPE_SQLESCAPE:
		case TYPE_SQLESCAPE2:
		case TYPE_SQLESCAPE3: {
			char q = type == TYPE_SQLESCAPE3 ? '"' : '\''; // Quote character
			char *escarg = noArgs ? va_arg(va, char*) : __extsystem.getStringArg(args);
			bool isnull = !escarg;
			if (isnull) escarg = type == TYPE_SQLESCAPE2 ? "NULL" : "(NULL)";
			int k = precision;
			int i, j, n; char ch; for (i = n = 0; k != 0 && (ch = escarg[i]) != 0; i++, k--)
				if (ch == q) n++;
			bool needQuote = !isnull && type == TYPE_SQLESCAPE2;
			n += i + 3;
			if (n > BUFSIZE) {
				bufpt = extra = (char *)malloc(n);
				if (!bufpt) {
					strbldSetError(b, STRACCUM_NOMEM);
					return;
				}
			}
			else bufpt = buf;
			j = 0;
			if (needQuote) bufpt[j++] = q;
			k = i;
			for (i = 0; i < k; i++) {
				bufpt[j++] = ch = escarg[i];
				if (ch == q) bufpt[j++] = ch;
			}
			if (needQuote) bufpt[j++] = q;
			bufpt[j] = 0;
			length = j;
			// The precision in %q and %Q means how many input characters to consume, not the length of the output...
			// if (precision>=0 && precision<length) length = precision;
			break; }
		case TYPE_TOKEN: {
			if (!(b->flags & PRINTF_INTERNAL)) return;
			__extsystem.appendFormat[0](b, &va);
			length = width = 0;
			break; }
		case TYPE_SRCLIST: {
			if (!(b->flags & PRINTF_INTERNAL)) return;
			__extsystem.appendFormat[1](b, &va);
			length = width = 0;
			break; }
		default: {
			assert(type == TYPE_INVALID);
			return; }
		}
		// The text of the conversion is pointed to by "bufpt" and is "length" characters long.  The field width is "width".  Do the output.
		width -= length;
		if (width > 0) {
			if (!flag_leftjustify) strbldAppendChar(b, width, ' ');
			strbldAppend(b, bufpt, length);
			if (flag_leftjustify) strbldAppendChar(b, width, ' ');
		}
		else strbldAppend(b, bufpt, length);
		//
		if (extra) { free(extra); extra = nullptr; }
	}
}

/*
** Enlarge the memory allocation on a StrAccum object so that it is able to accept at least N more bytes of text.
**
** Return the number of bytes of text that StrAccum is able to accept after the attempted enlargement.  The value returned might be zero.
*/
static __host_device__ int strbldEnlarge(strbld_t *b, int n) {
	assert(b->index + (int64_t)n >= b->size); // Only called if really needed
	if (b->error) {
		TESTCASE_(b->error == STRACCUM_TOOBIG);
		TESTCASE_(b->error == STRACCUM_NOMEM);
		return 0;
	}
	if (!b->maxSize) {
		n = b->size - b->index - 1;
		strbldSetError(b, STRACCUM_TOOBIG);
		return n;
	}
	char *oldText = PRINTF_ISMALLOCED(b) ? b->text : nullptr;
	int64_t sizeNew = b->index;
	assert((!b->text || b->text == b->base) == !PRINTF_ISMALLOCED(b));
	sizeNew += n + 1;
	if (sizeNew + b->index <= b->maxSize)
		sizeNew += b->index; // Force exponential buffer size growth as long as it does not overflow, to avoid having to call this routine too often
	if (sizeNew > b->maxSize) {
		strbldReset(b);
		strbldSetError(b, STRACCUM_TOOBIG);
		return 0;
	}
	else b->size = (int)sizeNew;
	char *newText = (char *)(b->tag ? __extsystem.tagrealloc(b->tag, oldText, b->size) : realloc(oldText, b->size));
	if (newText) {
		assert(b->text || !b->index);
		if (!PRINTF_ISMALLOCED(b) && b->index > 0) memcpy(newText, b->text, b->index);
		b->text = newText;
		b->size = b->tag ? (size_t)__extsystem.tagallocSize(b->tag, newText) : _msize(newText);
		b->flags |= PRINTF_MALLOCED;
	}
	else {
		strbldReset(b);
		strbldSetError(b, STRACCUM_NOMEM);
		return 0;
	}
	return n;
}

/* Append N copies of character c to the given string buffer. */
__host_device__ void strbldAppendChar(strbld_t *b, int n, int c) {
	TESTCASE_(b->size + (int64_t)n > 0x7fffffff);
	if (b->index + (int64_t)n >= b->size && (n = strbldEnlarge(b, n)) <= 0)
		return;
	assert((b->text == b->base) == !PRINTF_ISMALLOCED(b));
	while (n-- > 0) b->text[b->index++] = c;
}

/*
** The StrAccum "b" is not large enough to accept N new bytes of str[]. So enlarge if first, then do the append.
**
** This is a helper routine to sqlite3StrAccumAppend() that does special-case work (enlarging the buffer) using tail recursion, so that the
** sqlite3StrAccumAppend() routine can use fast calling semantics.
*/
static __host_device__ void enlargeAndAppend(strbld_t *b, const char *str, int length) {
	length = strbldEnlarge(b, length);
	if (length > 0) {
		memcpy(&b->text[b->index], str, length);
		b->index += length;
	}
}

/* Append N bytes of text from str to the StrAccum object.  Increase the size of the memory allocation for StrAccum if necessary. */
__host_device__ void strbldAppend(strbld_t *b, const char *str, int length) {
	assert(str || !length);
	assert(b->text || !b->index || b->error);
	assert(length >= 0);
	assert(!b->error || !b->size);
	if (b->index + length >= b->size)
		enlargeAndAppend(b, str, length);
	else if (length) {
		assert(b->text);
		b->index += length;
		memcpy(&b->text[b->index - length], str, length);
	}
}

/* Append the complete text of zero-terminated string str[] to the b string. */
__host_device__ void strbldAppendAll(strbld_t *b, const char *str) {
	strbldAppend(b, str, (int)strlen(str));
}

/*
** Finish off a string by making sure it is zero-terminated. Return a pointer to the resulting string.  Return a NULL
** pointer if any kind of error was encountered.
*/
static __host_device__ char *strbldFinishRealloc(strbld_t *b) {
	assert(b->maxSize > 0 && !PRINTF_ISMALLOCED(b));
	b->text = (char *)(b->tag ? __extsystem.tagallocRaw(b->tag, b->index + 1) : malloc(b->index + 1));
	if (b->text) {
		memcpy(b->text, b->base, b->index + 1);
		b->flags |= PRINTF_MALLOCED;
	}
	else strbldSetError(b, STRACCUM_NOMEM);
	return b->text;
}

__host_device__ char *strbldToString(strbld_t *b) {
	if (b->text) {
		assert((b->text == b->base) == !PRINTF_ISMALLOCED(b));
		b->text[b->index] = 0;
		if (b->maxSize > 0 && !PRINTF_ISMALLOCED(b))
			return strbldFinishRealloc(b);
	}
	return b->text;
}

/* Reset an StrAccum string.  Reclaim all malloced memory. */
__host_device__ void strbldReset(strbld_t *b) {
	assert((!b->text || b->text == b->base) == !PRINTF_ISMALLOCED(b));
	if (PRINTF_ISMALLOCED(b)) {
		if (b->tag) __extsystem.tagfree(b->tag, b->text);
		else free(b->text);
		b->flags &= ~PRINTF_MALLOCED;
	}
	b->text = nullptr;
}

/*
** Initialize a string accumulator.
**
** b: The accumulator to be initialized.
** tag: Pointer to a database connection.  May be NULL.  Lookaside memory is used if not NULL. db->mallocFailed is set appropriately when not NULL.
** base: An initial buffer.  May be NULL in which case the initial buffer is malloced.
** capacity: Size of zBase in bytes.  If total space requirements never exceed n then no memory allocations ever occur.
** maxSize: Maximum number of bytes to accumulate.  If mx==0 then no memory allocations will ever occur.
*/
__host_device__ void strbldInit(strbld_t *b, void *tag, char *base, int capacity, int maxSize) {
	b->text = b->base = base;
	b->tag = tag;
	b->index = 0;
	b->size = capacity;
	b->maxSize = maxSize;
	b->error = 0;
	b->flags = 0;
}

///* variable-argument wrapper around sqlite3VXPrintf().  The bFlags argument can contain the bit SQLITE_PRINTF_INTERNAL enable internal formats. */
//__device__ void strbldAppendFormat(strbld_t *b, const char *fmt, ...) { va_list va; va_start(va, fmt); strbldAppendFormatv(b, fmt, va); va_end(va); }

#pragma endregion

__END_DECLS;