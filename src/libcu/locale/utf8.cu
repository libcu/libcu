#ifndef LIBCU_LEAN_AND_MEAN
#include <stddefcu.h>
#include <errnocu.h>
#include <wchar.h>
#include "xlocale_private.h"

#define UTF8_MB_CUR_MAX 6

typedef struct {
	wchar_t	ch;
	int	want;
	wchar_t	lbound;
} _UTF8State;

static __device__ int _UTF8_mbsinit(const mbstate_t *ps, localecu_t loc) {
	return !ps || !((const _UTF8State *)ps)->want;
}

static __device__ size_t _UTF8_mbrtowc(wchar_t *__restrict pwc, const char *__restrict s, size_t n, mbstate_t *__restrict ps, localecu_t loc) {
	_UTF8State *us = (_UTF8State *)ps;
	if (us->want < 0 || us->want > 6) { errno = EINVAL; return (size_t)-1; }
	if (!s) { s = ""; n = 1; pwc = nullptr; }
	if (!n) return (size_t)-2; // Incomplete multibyte sequence
	int ch;
	if (!us->want && (!(ch = (unsigned char)*s) & ~0x7f)) { if (pwc) *pwc = ch; return ch != '\0' ? 1 : 0; } // Fast path for plain ASCII characters.
	int mask, want;
	wchar_t lbound;
	if (!us->want) {
		ch = (unsigned char)*s;
		if ((ch & 0x80) == 0) { mask = 0x7f; want = 1; lbound = 0; }
		else if ((ch & 0xe0) == 0xc0) { mask = 0x1f; want = 2; lbound = 0x80; }
		else if ((ch & 0xf0) == 0xe0) { mask = 0x0f; want = 3; lbound = 0x800; }
		else if ((ch & 0xf8) == 0xf0) { mask = 0x07; want = 4; lbound = 0x10000; }
		else if ((ch & 0xfc) == 0xf8) { mask = 0x03; want = 5; lbound = 0x200000; }
		else if ((ch & 0xfe) == 0xfc) { mask = 0x01; want = 6; lbound = 0x4000000; }
		else { errno = EILSEQ; return (size_t)-1; } /* Malformed input; input is not UTF-8. */
	}
	else { want = us->want; lbound = us->lbound; }
	// Decode the octet sequence representing the character in chunks of 6 bits, most significant first.
	wchar_t wch = !us->want ? (unsigned char)*s++ & mask : us->ch;
	int i;
	for (i = (us->want == 0) ? 1 : 0; i < MIN_(want, n); i++) {
		if ((*s & 0xc0) != 0x80) { errno = EILSEQ; return (size_t)-1; } // Malformed input; bad characters in the middle of a character.
		wch <<= 6; wch |= *s++ & 0x3f;
	}
	if (i < want) { us->want = want - i; us->lbound = lbound; us->ch = wch; return (size_t)-2; } // Incomplete multibyte sequence. 
	if (wch < lbound) { errno = EILSEQ; return (size_t)-1; } // Malformed input; redundant encoding.
	if (pwc) *pwc = wch;
	us->want = 0;
	return wch == L'\0' ? 0 : want;
}

static __device__ size_t _UTF8_mbsnrtowcs(wchar_t *__restrict dst, const char **__restrict src, size_t nms, size_t len, mbstate_t *__restrict ps, localecu_t loc) {
	_UTF8State *us = (_UTF8State *)ps;
	wchar_t wc;
	size_t nb;
	const char *s = *src;
	size_t nchr = 0;
	if (!dst) {
		if (nms > 0 && us->want > 0 && (signed char)*s > 0) { errno = EILSEQ; return ((size_t)-1); }
		for (;;) {
			if (nms > 0 && (signed char)*s > 0) nb = 1; // Fast path for plain ASCII characters excluding NUL.
			else if ((nb = _UTF8_mbrtowc(&wc, s, nms, ps, loc)) == (size_t)-1) return (size_t)-1; // Invalid sequence - mbrtowc() sets errno.
			else if (nb == 0 || nb == (size_t)-2) return nchr;
			s += nb; nms -= nb; nchr++;
		}
	}
	if (nms > 0 && len > 0 && us->want > 0 && (signed char)*s > 0) { errno = EILSEQ; return (size_t)-1; }
	while (len-- > 0) {
		if (nms > 0 && (signed char)*s > 0) { *dst = (wchar_t)*s; nb = 1; } // Fast path for plain ASCII characters excluding NUL.
		else if ((nb = _UTF8_mbrtowc(dst, s, nms, ps, loc)) == (size_t)-1) { *src = s; return (size_t)-1; }
		else if (nb == (size_t)-2) { *src = s + nms; return nchr; }
		else if (nb == 0) { *src = nullptr; return nchr; }
		s += nb; nms -= nb; nchr++; dst++;
	}
	*src = s;
	return nchr;
}

static __device__ size_t _UTF8_wcrtomb(char *__restrict s, wchar_t wc, mbstate_t *__restrict ps, localecu_t loc) {
	_UTF8State *us = (_UTF8State *)ps;
	if (us->want != 0) { errno = EINVAL; return (size_t)-1; }
	if (!s) return 1; // Reset to initial shift state (no-op)
	if ((wc & ~0x7f) == 0) { *s = (char)wc; return 1; } // Fast path for plain ASCII characters.
	unsigned char lead; int len;
	if ((wc & ~0x7f) == 0) { lead = 0; len = 1; }
	else if ((wc & ~0x7ff) == 0) { lead = 0xc0; len = 2; }
	else if ((wc & ~0xffff) == 0) { lead = 0xe0; len = 3; }
	else if ((wc & ~0x1fffff) == 0) { lead = 0xf0; len = 4; }
	else if ((wc & ~0x3ffffff) == 0) { lead = 0xf8; len = 5; }
	else if ((wc & ~0x7fffffff) == 0) { lead = 0xfc; len = 6; }
	else { errno = EILSEQ; return (size_t)-1; }
	for (int i = len - 1; i > 0; i--) { s[i] = (wc & 0x3f) | 0x80; wc >>= 6; }
	*s = (wc & 0xff) | lead;
	return len;
}

static __device__ size_t _UTF8_wcsnrtombs(char *__restrict dst, const wchar_t **__restrict src, size_t nwc, size_t len, mbstate_t *__restrict ps, localecu_t loc) {
	_UTF8State *us = (_UTF8State *)ps;
	char buf[MB_LEN_MAX];
	if (us->want != 0) { errno = EINVAL; return (size_t)-1; }
	const wchar_t *s = *src;
	size_t nbytes = 0;
	size_t nb;
	if (!dst) {
		while (nwc-- > 0) {
			if (0 <= *s && *s < 0x80) nb = 1; // Fast path for plain ASCII characters.
			else if ((nb = _UTF8_wcrtomb(buf, *s, ps, loc)) == (size_t)-1) return (size_t)-1; // Invalid character - wcrtomb() sets errno.
			if (*s == L'\0') return nbytes + nb - 1;
			s++; nbytes += nb;
		}
		return nbytes;
	}
	while (len > 0 && nwc-- > 0) {
		if (0 <= *s && *s < 0x80) { nb = 1; *dst = *s; } // Fast path for plain ASCII characters.
		else if (len > (size_t)UTF8_MB_CUR_MAX) { // Enough space to translate in-place.
			if ((nb = _UTF8_wcrtomb(dst, *s, ps, loc)) == (size_t)-1) { *src = s; return ((size_t)-1); }
		}
		else { // May not be enough space; use temp. buffer.
			if ((nb = _UTF8_wcrtomb(buf, *s, ps, loc)) == (size_t)-1) { *src = s; return (size_t)-1; }
			if (nb > (int)len) break; // MB sequence for character won't fit.
			memcpy(dst, buf, nb);
		}
		if (*s == L'\0') { *src = nullptr; return nbytes + nb - 1; }
		s++; dst += nb; len -= nb; nbytes += nb;
	}
	*src = s;
	return nbytes;
}

__device__ int _UTF8_init(struct __xlocalecu_ctype *xl) {
	xl->__mbrtowc = _UTF8_mbrtowc;
	xl->__wcrtomb = _UTF8_wcrtomb;
	xl->__mbsinit = _UTF8_mbsinit;
	xl->__mbsnrtowcs = _UTF8_mbsnrtowcs;
	xl->__wcsnrtombs = _UTF8_wcsnrtombs;
	xl->__mb_cur_max = UTF8_MB_CUR_MAX;
	xl->__mb_sb_limit = 128;
	return 0;
}
#endif