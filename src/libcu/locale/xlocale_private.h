#ifndef _XLOCALE_PRIVATE_H_
#define _XLOCALE_PRIVATE_H_
#include <limits.h>
#include <wchar.h>
#undef __mb_cur_max

#define XMAGIC 0x786c6f63616c6530LL

//#ifdef _MSC_VER
struct _xlocalecu;
typedef struct _xlocalecu *localecu_t;
//#endif

struct __xlocalecu_ctype {
	int __mb_cur_max;
	int __mb_sb_limit;
	int(*__mbsinit)(const mbstate_t *, struct _xlocalecu *);
	size_t(*__mbrtowc)(wchar_t *__restrict, const char *__restrict, size_t, mbstate_t *__restrict, struct _xlocalecu *);
	size_t(*__mbsnrtowcs)(wchar_t *__restrict, const char **__restrict, size_t, size_t, mbstate_t *__restrict, struct _xlocalecu *);
	size_t(*__wcrtomb)(char *__restrict, wchar_t, mbstate_t *__restrict, struct _xlocalecu *);
	size_t(*__wcsnrtombs)(char *__restrict, const wchar_t **__restrict, size_t, size_t, mbstate_t *__restrict, struct _xlocalecu *);
};

struct _xlocalecu {
	/* 10 independent mbstate_t buffers! */
	mbstate_t __mbs_mblen;
	mbstate_t __mbs_mbtowc;
	mbstate_t __mbs_wctomb;
	/* magic (Here up to the end is copied when duplicating locale_t's) */
	int64_t __magic;
	/* ctype */
	struct __xlocalecu_ctype *__ctype;
};

#define NORMALIZE_LOCALE(x)	if (!(x)) { (x) = (localecu_t)&__cu_locale; }

__BEGIN_DECLS;
extern __device__ const struct _xlocalecu __cu_locale;
static __device__ inline localecu_t __current_locale() { return (localecu_t)&__cu_locale; }
__END_DECLS;

#endif /* _XLOCALE_PRIVATE_H_ */