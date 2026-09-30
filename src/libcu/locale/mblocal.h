#ifndef _MBLOCAL_H_
#define	_MBLOCAL_H_
#include <stddefcu.h>
#include "xlocale_private.h"

__device__ int	_none_init(struct __xlocalecu_ctype *);
__device__ int	_ascii_init(struct __xlocalecu_ctype *);
__device__ int	_UTF8_init(struct __xlocalecu_ctype *);

__device__ int _none_mbsinit(const mbstate_t *, localecu_t);
__device__ size_t _none_mbrtowc(wchar_t * __restrict, const char * __restrict, size_t, mbstate_t * __restrict, localecu_t);
__device__ size_t _none_mbsnrtowcs(wchar_t *__restrict dst, const char **__restrict src, size_t nms, size_t len, mbstate_t *__restrict ps, localecu_t);
__device__ size_t _none_wcrtomb(char *__restrict, wchar_t, mbstate_t *__restrict, localecu_t);
__device__ size_t _none_wcsnrtombs(char *__restrict, const wchar_t **__restrict, size_t, size_t, mbstate_t *__restrict, localecu_t);

#endif	/* _MBLOCAL_H_ */
