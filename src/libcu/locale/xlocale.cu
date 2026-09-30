#ifndef LIBCU_LEAN_AND_MEAN
#include <stddefcu.h>
#include "xlocale_private.h"
#include "mblocal.h"

#define C_LOCALE_INITIALIZER { \
	{}, {}, {}, \
	XMAGIC, \
	&_DefaultXLocaleCType \
}

static __device__ struct __xlocalecu_ctype _DefaultXLocaleCType = {
	1, 128,
	_none_mbsinit,
	_none_mbrtowc,
	_none_mbsnrtowcs,
	_none_wcrtomb,
	_none_wcsnrtombs
};
__device__ const struct _xlocalecu __cu_locale = C_LOCALE_INITIALIZER;
#endif