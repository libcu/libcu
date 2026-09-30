#ifndef LIBCU_LEAN_AND_MEAN
#include <stddefcu.h>
#include <errnocu.h>
#include <stringcu.h>
#include <wchar.h>
#include "xlocale_private.h"

__device__ int _none_mbsinit(const mbstate_t *ps, localecu_t loc) {
	// Encoding is not state dependent - we are always in the initial state.
	return 1;
}

__device__ size_t _none_mbrtowc(wchar_t *__restrict pwc, const char *__restrict s, size_t n, mbstate_t *__restrict ps, localecu_t loc) {
	if (!s) return 0; // Reset to initial shift state (no-op)
	if (!n) return (size_t)-2; // Incomplete multibyte sequence
	if (pwc) *pwc = (unsigned char)*s;
	return *s == '\0' ? 0 : 1;
}

__device__ size_t _none_wcrtomb(char *__restrict s, wchar_t wc, mbstate_t *__restrict ps, localecu_t loc) {
	if (!s) return 1; // Reset to initial shift state (no-op)
	if (wc < 0 || wc > UCHAR_MAX) { errno = EILSEQ; return (size_t)-1; }
	*s = (unsigned char)wc;
	return 1;
}

__device__ size_t _none_mbsnrtowcs(wchar_t *__restrict dst, const char **__restrict src, size_t nms, size_t len, mbstate_t *__restrict ps, localecu_t loc) {
	const char *s;
	if (!dst) {
		s = (const char *)memchr(*src, '\0', nms);
		return s ? s - *src : nms;
	}
	s = *src;
	size_t nchr = 0;
	while (len-- > 0 && nms-- > 0) {
		if ((*dst++ = (unsigned char)*s++) == L'\0') { *src = nullptr; return nchr; }
		nchr++;
	}
	*src = s;
	return nchr;
}

__device__ size_t _none_wcsnrtombs(char *__restrict dst, const wchar_t **__restrict src, size_t nwc, size_t len, mbstate_t * __restrict ps, localecu_t loc) {
	const wchar_t *s;
	if (!dst) {
		for (s = *src; nwc > 0 && *s != L'\0'; s++, nwc--) {
			if (*s < 0 || *s > UCHAR_MAX) { errno = EILSEQ; return (size_t)-1; }
		}
		return s - *src;
	}
	s = *src;
	size_t nchr = 0;
	while (len-- > 0 && nwc-- > 0) {
		if (*s < 0 || *s > UCHAR_MAX) { errno = EILSEQ; return (size_t)-1; }
		if ((*dst++ = *s++) == '\0') { *src = nullptr; return nchr; }
		nchr++;
	}
	*src = s;
	return nchr;
}

static __device__ int _none_init(struct __xlocalecu_ctype *xl) {
	xl->__mbrtowc = _none_mbrtowc;
	xl->__mbsinit = _none_mbsinit;
	xl->__mbsnrtowcs = _none_mbsnrtowcs;
	xl->__wcrtomb = _none_wcrtomb;
	xl->__wcsnrtombs = _none_wcsnrtombs;
	xl->__mb_cur_max = 1;
	xl->__mb_sb_limit = 128;
	return 0;
}
#endif