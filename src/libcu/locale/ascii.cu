#ifndef LIBCU_LEAN_AND_MEAN
#include <stddefcu.h>
#include <errnocu.h>
#include <wchar.h>
#include "xlocale_private.h"

static __device__ int _ascii_mbsinit(const mbstate_t *ps, localecu_t loc) {
	// Encoding is not state dependent - we are always in the initial state.
	return 1;
}

static __device__ size_t _ascii_mbrtowc(wchar_t *__restrict pwc, const char *__restrict s, size_t n, mbstate_t *__restrict ps, localecu_t loc) {
	if (!s) return 0; // Reset to initial shift state (no-op)
	if (!n) return (size_t)-2; // Incomplete multibyte sequence
	if (*s & 0x80) { errno = EILSEQ; return (size_t)-1; }
	if (pwc) *pwc = (unsigned char)*s;
	return *s == '\0' ? 0 : 1;
}

static __device__ size_t _ascii_wcrtomb(char *__restrict s, wchar_t wc, mbstate_t *__restrict ps, localecu_t loc) {
	if (!s) return 1;// Reset to initial shift state (no-op)
	if (wc < 0 || wc > 127) { errno = EILSEQ; return (size_t)-1; }
	*s = (unsigned char)wc;
	return 1;
}

static __device__ size_t _ascii_mbsnrtowcs(wchar_t *__restrict dst, const char **__restrict src, size_t nms, size_t len, mbstate_t *__restrict ps, localecu_t loc) {
	const char *s;
	if (!dst) {
		for (s = *src; nms > 0 && *s != '\0'; s++, nms--) {
			if (*s & 0x80) { errno = EILSEQ; return (size_t)-1; }
		}
		return s - *src;
	}
	s = *src;
	size_t nchr = 0;
	while (len-- > 0 && nms-- > 0) {
		if (*s & 0x80) { errno = EILSEQ; return (size_t)-1; }
		if ((*dst++ = (unsigned char)*s++) == L'\0') { *src = nullptr; return nchr; }
		nchr++;
	}
	*src = s;
	return nchr;
}

static __device__ size_t _ascii_wcsnrtombs(char *__restrict dst, const wchar_t **__restrict src, size_t nwc, size_t len, mbstate_t *__restrict ps, localecu_t loc) {
	const wchar_t *s;
	if (!dst) {
		for (s = *src; nwc > 0 && *s != L'\0'; s++, nwc--) {
			if (*s < 0 || *s > 127) { errno = EILSEQ; return (size_t)-1; }
		}
		return s - *src;
	}
	s = *src;
	size_t nchr = 0;
	while (len-- > 0 && nwc-- > 0) {
		if (*s < 0 || *s > 127) { errno = EILSEQ; return (size_t)-1; }
		if ((*dst++ = *s++) == '\0') { *src = nullptr; return nchr; }
		nchr++;
	}
	*src = s;
	return nchr;
}

__device__ int _ascii_init(struct __xlocalecu_ctype *xl) {
	xl->__mbrtowc = _ascii_mbrtowc;
	xl->__mbsinit = _ascii_mbsinit;
	xl->__mbsnrtowcs = _ascii_mbsnrtowcs;
	xl->__wcrtomb = _ascii_wcrtomb;
	xl->__wcsnrtombs = _ascii_wcsnrtombs;
	xl->__mb_cur_max = 1;
	xl->__mb_sb_limit = 128;
	return 0;
}
#endif