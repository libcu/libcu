#include <stdlibcu.h>
#include <stdiocu.h>
#include <sentinel-stdlibmsg.h>
#include <bits/libcu_fpmax.h>
#include <ext/hash.h>
#include <ctypecu.h>
#include <errnocu.h>
#include <fcntlcu.h>
#include <assert.h>

__BEGIN_DECLS;

#pragma region header

/* Handle _STRTOD_HEXADECIMAL_FLOATS via libcu config now. */
#undef _STRTOD_HEXADECIMAL_FLOATS
#ifdef __LIBCU_HAS_HEXADECIMAL_FLOATS__
#define _STRTOD_HEXADECIMAL_FLOATS 1
#endif

/* Defined if we want to recognize "nan", "inf", and "infinity". (C99) */
#define _STRTOD_NAN_INF_STRINGS  1

/* Defined if we want support hexadecimal floating point notation. (C99) */
/* Note!  Now controlled by uClibc configuration.  See below. */
#define _STRTOD_HEXADECIMAL_FLOATS 1

/* Defined if we want to scale with a O(log2(exp)) multiplications.
* This is generally a good thing to do unless you are really tight
* on space and do not expect to convert values of large magnitude. */
#define _STRTOD_LOG_SCALING	  1

/* Set if we want errno set appropriately. */
/* NOTE: Implies _STRTO_ENDPTR below */
#define _STRTO_ERRNO            1

/* Set if we want support for the endptr arg. */
/* Implied by _STRTO_ERRNO. */
#define _STRTO_ENDPTR           1

/* Defined if we want to prevent overflow in accumulating the exponent. */
/* Implied by _STRTOD_ERRNO. */
#define _STRTOD_RESTRICT_EXP	 1

/* Defined if we want to process mantissa digits more intelligently. */
/* Implied by _STRTOD_ERRNO. */
#define _STRTOD_RESTRICT_DIGITS  1

/* Defined if we want to skip scaling 0 for the exponent. */
/* Implied by _STRTOD_ERRNO. */
#define _STRTOD_ZERO_CHECK	   1

#ifdef _STRTOD_ERRNO
#undef _STRTOD_ENDPTR
#undef _STRTOD_RESTRICT_EXP
#undef _STRTOD_RESTRICT_DIGITS
#undef _STRTOD_ZERO_CHECK
#define _STRTOD_ENDPTR		   1
#define _STRTOD_RESTRICT_EXP	 1
#define _STRTOD_RESTRICT_DIGITS  1
#define _STRTOD_ZERO_CHECK	   1
#endif

#if _STRTO_ERRNO
#undef _STRTO_ENDPTR
#define _STRTO_ENDPTR 1
#define SET_ERRNO(X) _set_errno(X)
#else
#define SET_ERRNO(X) ((void)(X))	/* keep side effects */
#endif

#if defined(WIDE)
#define Wchar wchar_t
#define Wuchar __uwchar_t
#define ISSPACE(C) iswspace((C))
#else
#define Wchar char
#define Wuchar unsigned char
#define ISSPACE(C) isspace((C))
#endif

#pragma endregion

#pragma region stdlib_strtod

#undef _STRTOD_FPMAX
#if FPMAX_TYPE == 3
#define NEED_STRTOLD_WRAPPER
#define NEED_STRTOD_WRAPPER
#define NEED_STRTOF_WRAPPER
#elif FPMAX_TYPE == 2
#define NEED_STRTOD_WRAPPER
#define NEED_STRTOF_WRAPPER
#elif FPMAX_TYPE == 1
#define NEED_STRTOF_WRAPPER
#else
#error ERROR: unknown FPMAX_TYPE!
#endif

static __device__ void __fp_range_check(__fpmax_t y, __fpmax_t x);

#ifdef _STRTOD_RESTRICT_DIGITS
#define EXP_DENORM_ADJUST DECIMAL_DIG
#define MAX_ALLOWED_EXP (DECIMAL_DIG  + EXP_DENORM_ADJUST - FPMAX_MIN_10_EXP)
#if MAX_ALLOWED_EXP > INT_MAX
#error size assumption violated for MAX_ALLOWED_EXP
#endif
#else
/* We want some excess if we're not restricting mantissa digits. */
#define MAX_ALLOWED_EXP ((20 - FPMAX_MIN_10_EXP) * 2)
#endif

#if defined(_STRTOD_RESTRICT_DIGITS) || defined(_STRTOD_ENDPTR) || defined(_STRTOD_HEXADECIMAL_FLOATS)
#undef _STRTOD_NEED_NUM_DIGITS
#define _STRTOD_NEED_NUM_DIGITS 1
#endif

static __device__ __fpmax_t __strtofpmax(const Wchar *str, Wchar **endptr, int exponent_power) {
	__fpmax_t number;
	__fpmax_t p_base = 10;			/* Adjusted to 16 in the hex case. */
	Wchar *pos0;
#ifdef _STRTOD_ENDPTR
	Wchar *pos1;
#endif
	Wchar *pos = (Wchar *)str;
	int exponent_temp;
	int negative; /* A flag for the number, a multiplier for the exponent. */
#ifdef _STRTOD_NEED_NUM_DIGITS
	int num_digits;
#endif

#ifdef _STRTOD_HEXADECIMAL_FLOATS
	Wchar expchar = 'e';
	Wchar *poshex = NULL;
	int is_mask = _DIGIT;
#define EXPCHAR		expchar
#define IS_X_DIGIT(C) isctype((C), is_mask)
#else  /* _STRTOD_HEXADECIMAL_FLOATS */
#define EXPCHAR		'e'
#define IS_X_DIGIT(C) isdigit((C))
#endif /* _STRTOD_HEXADECIMAL_FLOATS */

	while (ISSPACE(*pos)) {		/* Skip leading whitespace. */
		++pos;
	}

	negative = 0;
	switch (*pos) {				/* Handle optional sign. */
	case '-': negative = 1;		/* Fall through to increment position. */
	case '+': ++pos;
	}

#ifdef _STRTOD_HEXADECIMAL_FLOATS
	if ((*pos == '0') && (((pos[1]) | 0x20) == 'x')) {
		poshex = ++pos;			/* Save position of 'x' in case no digits */
		++pos;					/*   and advance past it.  */
		is_mask = _HEX;			/* Used by IS_X_DIGIT. */
		expchar = 'p';			/* Adjust exponent char. */
		p_base = 16;			/* Adjust base multiplier. */
	}
#endif

	number = 0.;
#ifdef _STRTOD_NEED_NUM_DIGITS
	num_digits = -1;
#endif
	/* 	exponent_power = 0; */
	pos0 = NULL;

LOOP:
	while (IS_X_DIGIT(*pos)) {	/* Process string of (hex) digits. */
#ifdef _STRTOD_RESTRICT_DIGITS
		if (num_digits < 0) {	/* First time through? */
			++num_digits;		/* We've now seen a digit. */
		}
		if (num_digits || (*pos != '0')) { /* Had/have nonzero. */
			++num_digits;
			if (num_digits <= DECIMAL_DIG) { /* Is digit significant? */
#ifdef _STRTOD_HEXADECIMAL_FLOATS
				number = number * p_base
					+ (isdigit(*pos)
						? (*pos - '0')
						: (((*pos) | 0x20) - ('a' - 10)));
#else  /* _STRTOD_HEXADECIMAL_FLOATS */
				number = number * p_base + (*pos - '0');
#endif /* _STRTOD_HEXADECIMAL_FLOATS */
			}
		}
#else  /* _STRTOD_RESTRICT_DIGITS */
#ifdef _STRTOD_NEED_NUM_DIGITS
		++num_digits;
#endif
#ifdef _STRTOD_HEXADECIMAL_FLOATS
		number = number * p_base
			+ (isdigit(*pos)
				? (*pos - '0')
				: (((*pos) | 0x20) - ('a' - 10)));
#else  /* _STRTOD_HEXADECIMAL_FLOATS */
		number = number * p_base + (*pos - '0');
#endif /* _STRTOD_HEXADECIMAL_FLOATS */
#endif /* _STRTOD_RESTRICT_DIGITS */
		++pos;
	}

	if ((*pos == '.') && !pos0) { /* First decimal point? */
		pos0 = ++pos;			/* Save position of decimal point */
		goto LOOP;				/*   and process rest of digits. */
	}

#ifdef _STRTOD_NEED_NUM_DIGITS
	if (num_digits < 0) {			/* Must have at least one digit. */
#ifdef _STRTOD_HEXADECIMAL_FLOATS
		if (poshex) {			/* Back up to '0' in '0x' prefix. */
			pos = poshex;
			goto DONE;
		}
#endif /* _STRTOD_HEXADECIMAL_FLOATS */

#ifdef _STRTOD_NAN_INF_STRINGS
		if (!pos0) {			/* No decimal point, so check for inf/nan. */
			/* Note: nan is the first string so 'number = i/0.;' works. */
			static const char nan_inf_str[] = "\05nan\0\012infinity\0\05inf\0";
			int i = 0;

			do {
				/* Unfortunately, we have no memcasecmp(). */
				int j = 0;
				/* |0x20 is a cheap lowercasing (valid for ASCII letters and numbers only) */
				while ((pos[j] | 0x20) == nan_inf_str[i + 1 + j]) {
					++j;
					if (!nan_inf_str[i + 1 + j]) {
						number = i / 0.;
						if (negative) {	/* Correct for sign. */
							number = -number;
						}
						pos += nan_inf_str[i] - 2;
						goto DONE;
					}
				}
				i += nan_inf_str[i];
			} while (nan_inf_str[i]);
		}

#endif /* STRTOD_NAN_INF_STRINGS */
#ifdef _STRTOD_ENDPTR
		pos = (Wchar *)str;
#endif
		goto DONE;
	}
#endif /* _STRTOD_NEED_NUM_DIGITS */

#ifdef _STRTOD_RESTRICT_DIGITS
	if (num_digits > DECIMAL_DIG) { /* Adjust exponent for skipped digits. */
		exponent_power += num_digits - DECIMAL_DIG;
	}
#endif

	if (pos0) {
		exponent_power += pos0 - pos; /* Adjust exponent for decimal point. */
	}

#ifdef _STRTOD_HEXADECIMAL_FLOATS
	if (poshex) {
		exponent_power *= 4;	/* Above is 2**4, but below is 2. */
		p_base = 2;
	}
#endif /* _STRTOD_HEXADECIMAL_FLOATS */

	if (negative) {				/* Correct for sign. */
		number = -number;
	}

	/* process an exponent string */
	if (((*pos) | 0x20) == EXPCHAR) {
#ifdef _STRTOD_ENDPTR
		pos1 = pos;
#endif
		negative = 1;
		switch (*++pos) {		/* Handle optional sign. */
		case '-': negative = -1; /* Fall through to increment pos. */
		case '+': ++pos;
		}

		pos0 = pos;
		exponent_temp = 0;
		while (isdigit(*pos)) {	/* Process string of digits. */
#ifdef _STRTOD_RESTRICT_EXP
			if (exponent_temp < MAX_ALLOWED_EXP) { /* Avoid overflow. */
				exponent_temp = exponent_temp * 10 + (*pos - '0');
			}
#else
			exponent_temp = exponent_temp * 10 + (*pos - '0');
#endif
			++pos;
		}

#ifdef _STRTOD_ENDPTR
		if (pos == pos0) {	/* No digits? */
			pos = pos1;		/* Back up to {e|E}/{p|P}. */
		} /* else */
#endif

		exponent_power += negative * exponent_temp;
	}

#ifdef _STRTOD_ZERO_CHECK
	if (number == 0.) {
		goto DONE;
	}
#endif

	/* scale the result */
#ifdef _STRTOD_LOG_SCALING
	exponent_temp = exponent_power;

	if (exponent_temp < 0) {
		exponent_temp = -exponent_temp;
	}

	while (exponent_temp) {
		if (exponent_temp & 1) {
			if (exponent_power < 0) {
				/* Warning... caluclating a factor for the exponent and
				* then dividing could easily be faster.  But doing so
				* might cause problems when dealing with denormals. */
				number /= p_base;
			}
			else {
				number *= p_base;
			}
		}
		exponent_temp >>= 1;
		p_base *= p_base;
	}

#else  /* _STRTOD_LOG_SCALING */
	while (exponent_power) {
		if (exponent_power < 0) {
			number /= p_base;
			exponent_power++;
		}
		else {
			number *= p_base;
			exponent_power--;
		}
	}
#endif /* _STRTOD_LOG_SCALING */

#ifdef _STRTOD_ERRNO
	if (__FPMAX_ZERO_OR_INF_CHECK(number)) {
		__set_errno(ERANGE);
	}
#endif

DONE:
#ifdef _STRTOD_ENDPTR
	if (endptr) {
		*endptr = pos;
	}
#endif

	return number;
}

static __device__ void __fp_range_check(__fpmax_t y, __fpmax_t x) {
	if (__FPMAX_ZERO_OR_INF_CHECK(y) /* y is 0 or +/- infinity */
		&& (y != 0)	/* y is not 0 (could have x>0, y==0 if underflow) */
		&& !__FPMAX_ZERO_OR_INF_CHECK(x) /* x is not 0 or +/- infinity */
		) {
		SET_ERRNO(ERANGE);
	} /* Then x is not in y's range. */
}

__device__ float strtof_(const Wchar *__restrict str, Wchar **__restrict endptr) {
#if FPMAX_TYPE == 1
	return __XL_NPP(__strtofpmax)(str, endptr, 0);
#else
	__fpmax_t x;
	float y;
	x = __strtofpmax(str, endptr, 0);
	y = (float)x;
	__fp_range_check(y, x);
	return y;
#endif
}

__device__ double strtod_(const Wchar *__restrict str, Wchar **__restrict endptr) {
#if FPMAX_TYPE == 2
	return __strtofpmax(str, endptr, 0);
#else
	__fpmax_t x;
	double y;
	x = __strtofpmax(str, endptr, 0);
	y = (double)x;
	__fp_range_check(y, x);
	return y;
#endif
}

#if 0 // cuda - not supported
__device__ long double strtold_(const Wchar *__restrict str, Wchar **__restrict endptr) {
#if FPMAX_TYPE == 3
	return __strtofpmax(str, endptr, 0);
#else
	__fpmax_t x;
	long double y;
	x = __strtofpmax(str, endptr, 0);
	y = (long double)x;
	__fp_range_check(y, x);
	return y;
#endif
}
#endif

#pragma endregion

#pragma region stdlib_strto

__device__ unsigned long __strtol(register const Wchar *__restrict str, Wchar **__restrict endptr, int base, int sflag) {
	unsigned long number, cutoff;
#if _STRTO_ENDPTR
	const Wchar *fail_char;
#define SET_FAIL(X) fail_char = (X)
#else
#define SET_FAIL(X) ((void)(X)) /* Keep side effects. */
#endif
	unsigned char negative, digit, cutoff_digit;

	SET_FAIL(str);
	while (ISSPACE(*str)) ++str; /* Skip leading whitespace. */

	/* Handle optional sign. */
	negative = 0;
	switch (*str) {
	case '-': negative = 1;	/* Fall through to increment str. */
	case '+': ++str;
	}

	if (!(base & ~0x10)) {		/* Either dynamic (base = 0) or base 16. */
		base += 10;				/* Default is 10 (26). */
		if (*str == '0') {
			++str;
			base -= 2;			/* Now base is 8 or 16 (24). */
			if ((0x20 | (*str)) == 'x') { /* WARNING: assumes ascii. */
				++str;
				base += base;	/* Base is 16 (16 or 48). */
			}
		}

		if (base > 16) {		/* Adjust in case base wasn't dynamic. */
			base = 16;
		}
	}

	number = 0;

	if (((unsigned)(base - 2)) < 35) { /* Legal base. */
		cutoff_digit = ULONG_MAX % base;
		cutoff = ULONG_MAX / base;
		do {
			digit = ((Wuchar)(*str - '0') <= 9)
				? /* 0..9 */ (*str - '0')
				: /* else */ (((Wuchar)(0x20 | *str) >= 'a') /* WARNING: assumes ascii. */
					? /* >= A/a */ ((Wuchar)(0x20 | *str) - ('a' - 10))
					: /* else   */ 40 /* bad value */);

			if (digit >= base) {
				break;
			}

			SET_FAIL(++str);
			if ((number > cutoff)
				|| ((number == cutoff) && (digit > cutoff_digit))) {
				number = ULONG_MAX;
				negative &= sflag;
				SET_ERRNO(ERANGE);
			}
			else {
				number = number * base + digit;
			}
		} while (1);
	}

#if _STRTO_ENDPTR
	if (endptr) {
		*endptr = (Wchar *)fail_char;
	}
#endif

	{
		unsigned long tmp = (negative
			? ((unsigned long)(-(1 + LONG_MIN))) + 1
			: LONG_MAX);
		if (sflag && (number > tmp)) {
			number = tmp;
			SET_ERRNO(ERANGE);
		}
	}

	return negative ? (unsigned long)(-((long)number)) : number;
}

__device__ unsigned long long __strtoll(register const Wchar * __restrict str, Wchar ** __restrict endptr, int base, int sflag) {
	unsigned long long number;
#if _STRTO_ENDPTR
	const Wchar *fail_char;
#define SET_FAIL(X) fail_char = (X)
#else
#define SET_FAIL(X) ((void)(X)) /* Keep side effects. */
#endif
	unsigned int n1;
	unsigned char negative, digit;

	SET_FAIL(str);
	while (ISSPACE(*str)) ++str;	/* Skip leading whitespace. */

	/* Handle optional sign. */
	negative = 0;
	switch (*str) {
	case '-': negative = 1;	/* Fall through to increment str. */
	case '+': ++str;
	}

	if (!(base & ~0x10)) {		/* Either dynamic (base = 0) or base 16. */
		base += 10;				/* Default is 10 (26). */
		if (*str == '0') {
			SET_FAIL(++str);
			base -= 2;			/* Now base is 8 or 16 (24). */
			if ((0x20 | (*str)) == 'x') { /* WARNING: assumes ascii. */
				++str;
				base += base;	/* Base is 16 (16 or 48). */
			}
		}

		if (base > 16) {		/* Adjust in case base wasn't dynamic. */
			base = 16;
		}
	}

	number = 0;

	if (((unsigned)(base - 2)) < 35) { /* Legal base. */
		do {
			digit = ((Wuchar)(*str - '0') <= 9)
				? /* 0..9 */ (*str - '0')
				: /* else */ (((Wuchar)(0x20 | *str) >= 'a') /* WARNING: assumes ascii. */
					? /* >= A/a */ ((Wuchar)(0x20 | *str) - ('a' - 10))
					: /* else   */ 40 /* bad value */);

			if (digit >= base) {
				break;
			}

			SET_FAIL(++str);
#if 1
			/* Optional, but speeds things up in the usual case. */
			if (number <= (ULLONG_MAX >> 6)) {
				number = number * base + digit;
			}
			else
#endif
			{
				n1 = ((unsigned char)number) * base + digit;
				number = (number >> CHAR_BIT) * base;

				if (number + (n1 >> CHAR_BIT) <= (ULLONG_MAX >> CHAR_BIT)) {
					number = (number << CHAR_BIT) + n1;
				}
				else {		/* Overflow. */
					number = ULLONG_MAX;
					negative &= sflag;
					SET_ERRNO(ERANGE);
				}
			}

		} while (1);
	}

#if _STRTO_ENDPTR
	if (endptr) {
		*endptr = (Wchar *)fail_char;
	}
#endif

	{
		unsigned long long tmp = (negative)
			? ((unsigned long long)(-(1 + LLONG_MIN))) + 1
			: LLONG_MAX;
		if (sflag && (number > tmp)) {
			number = tmp;
			SET_ERRNO(ERANGE);
		}
	}

	return negative ? (unsigned long long)(-((long long)number)) : number;
}

#pragma endregion

/* Return a random integer between 0 and RAND_MAX inclusive.  */
__device__ uint32_t _rand_value = 1;
__device__ int rand_() {
#if 0 //#ifndef OMIT_PTX
	int r;
	asm(
		".reg .u32 value;\n\t"
		"ld.u32		value, [_rand_value];\n\t"
		"and.u32	%0, value, "RAND_MAX";\n\t"
		// advance to next random element
		"shr.b32	value, value, 1;\n\t"
		"xor.b32	value, value, 0x00012000;\n\t"
		: "="__R(r));
	return r;
#else
	int x = _rand_value & RAND_MAX;		// X = low 8 bits
	unsigned lsb = _rand_value & 1;		// Get the output bit
	_rand_value >>= 1;					// Shift register
	if (lsb) _rand_value ^= 0x00012000; // If the output is 0, the xor can be skipped
	return x;
#endif
}

/* Seed the random number generator with the given number.  */
__device__ void srand_(unsigned int seed) {
	_rand_value = seed ? seed : 1;
}

/*
#undef malloc
#undef free
#ifndef _WIN64
#define MALLOCSIZETYPE long int
#else
#define MALLOCSIZETYPE long long int
#endif
__device__ void *malloc_(size_t size) {
assert(size > 0);
size = ROUND8_(size);
MALLOCSIZETYPE *p = (MALLOCSIZETYPE *)malloc(sizeof(MALLOCSIZETYPE) + size);
if (p)
p[0] = size;
else panic("failed to allocate %u bytes of memory", size);
return (void *)(p+1);
}

__device__ void *calloc_(size_t nmemb, size_t size) {
return malloc(size);
}

__device__ void free_(void *ptr) {
assert(ptr);
MALLOCSIZETYPE *p = (MALLOCSIZETYPE *)ptr;
free(p-1);
}

__device__ void *realloc_(void *ptr, size_t size) {
assert(size > 0);
size = ROUND8_(size);
MALLOCSIZETYPE *p = (MALLOCSIZETYPE *)malloc(sizeof(MALLOCSIZETYPE) + size);
if (p)
p[0] = size;
else panic("failed to allocate %u bytes of memory", size);
if (ptr)
{
MALLOCSIZETYPE *p2 = (MALLOCSIZETYPE *)ptr;
size_t ptrSize = (size_t)p2[0];
if (ptrSize) memcpy(p+1, p2+1, ptrSize);
free(p2-1);
}
return (void *)(p+1);
}
#define malloc malloc_
#define free free_
*/

/* Call all functions registered with `atexit' and `on_exit', in the reverse of the order in which they were registered, perform stdio cleanup, and terminate program execution with STATUS.  */
__device__ void exit_(int status) {
	stdlib_exit msg(true, status);
}

/* Terminate the program with STATUS without calling any of the functions registered with `atexit' or `on_exit'.  */
__device__ void _Exit_(int status) {
	stdlib_exit msg(false, status);
}

__device__ hash_t __env_dir = HASHINIT;

/* Return the value of envariable NAME, or NULL if it doesn't exist.  */
__device__ char *getenv_(const char *name) {
	if (ISHOSTENV(name)) { stdlib_getenv msg(name); return msg.RC; }
	//if (!strcmp(name, "HOME") || !strcmp(name, "PATH")) return "gpu:\\";
	return (char *)hashFind(&__env_dir, name);
}

/* Set NAME to VALUE in the environment. If REPLACE is nonzero, overwrite an existing value.  */
__device__ int setenv_(const char *name, const char *value, int replace) {
	if (ISHOSTENV(name)) { stdlib_setenv msg(name, value, replace); return msg.RC; }
	if (!replace && hashFind(&__env_dir, name)) return 0;
	if (hashInsert(&__env_dir, name, (void *)value))
		panic("removed environment");
	return 0;
}

/* Remove the variable NAME from the environment.  */
__device__ int unsetenv_(const char *name) {
	if (ISHOSTENV(name)) { stdlib_unsetenv msg(name); return msg.RC; }
	if (hashInsert(&__env_dir, name, nullptr))
		panic("removed environment");
	return 0;
}

/* Generate a unique temporary file name from TEMPLATE. */
__device__ char *mktemp_(char *template_) {
	panic("Not Implemented");
	return nullptr;
}

/* Generate a unique temporary file name from TEMPLATE. */
__device__ int mkstemp_(char *template_) {
	return open(mktemp_(template_), 0);
}

/* Execute the given line as a shell command.  */
__device__ int system_(const char *command) {
	stdlib_system msg(command); return msg.RC;
}

/* Do a binary search for KEY in BASE, which consists of NMEMB elements of SIZE bytes each, using COMPAR to perform the comparisons.  */
__device__ void *bsearch_(const void *key, const void *base, size_t nmemb, size_t size, __compar_fn_t compar) {
	panic("Not Implemented");
	return nullptr;
}

// qsort
#pragma region qsort

#define MIN(a, b) ((a) < (b) ? a : b)
#define SWAPCODE(TYPE, parmi, parmj, n) { \
	long i = (n) / sizeof(TYPE); \
	register TYPE *pi = (TYPE *)(parmi); \
	register TYPE *pj = (TYPE *)(parmj); \
	do { register TYPE t = *pi; *pi++ = *pj; *pj++ = t; } while (--i > 0); \
}
#define SWAPINIT(a, size) swaptype = (((char*)a-(char*)0)%sizeof(long)||size%sizeof(long)?2:(size==sizeof(long)?0:1));
__forceinline__ __device__ void swapfunc(char *a, char *b, int n, int swaptype) {
	if (swaptype <= 1) SWAPCODE(long, a, b, n)
	else SWAPCODE(char, a, b, n)
}
#define SWAP(a, b) \
	if (swaptype == 0) { long t = *(long *)(a); *(long *)(a) = *(long *)(b); *(long *)(b) = t; } \
	else swapfunc(a, b, size, swaptype)
#define VECSWAP(a, b, n) if ((n) > 0) swapfunc(a, b, n, swaptype)

__forceinline__ __device__ char *med3(char *a, char *b, char *c, __compar_fn_t compar) {
	return compar(a, b) < 0 ? (compar(b, c) < 0 ? b : (compar(a, c) < 0 ? c : a)) : (compar(b, c) > 0 ? b : (compar(a, c) < 0 ? a : c));
}

__device__ void qsort_(void *base, size_t nmemb, size_t size, __compar_fn_t compar) {
	char *a = (char *)base;
	char *pa, *pb, *pc, *pd, *pl, *pm, *pn;
	int d, r, swaptype, swap_cnt;
loop:
	SWAPINIT(a, size);
	swap_cnt = 0;
	if (nmemb < 7) {
		for (pm = a + size; pm < (char *)a + nmemb * size; pm += size)
			for (pl = pm; pl > (char *)a && compar(pl - size, pl) > 0; pl -= size)
				SWAP(pl, pl - size);
		return;
	}
	pm = a + (nmemb / 2) * size;
	if (nmemb > 7) {
		pl = a;
		pn = a + (nmemb - 1) * size;
		if (nmemb > 40) {
			d = (nmemb / 8) * size;
			pl = med3(pl, pl + d, pl + 2 * d, compar);
			pm = med3(pm - d, pm, pm + d, compar);
			pn = med3(pn - 2 * d, pn - d, pn, compar);
		}
		pm = med3(pl, pm, pn, compar);
	}
	SWAP(a, pm);
	pa = pb = a + size;
	//
	pc = pd = a + (nmemb - 1) * size;
	for (;;) {
		while (pb <= pc && (r = compar(pb, a)) <= 0) {
			if (r == 0) {
				swap_cnt = 1;
				SWAP(pa, pb);
				pa += size;
			}
			pb += size;
		}
		while (pb <= pc && (r = compar(pc, a)) >= 0) {
			if (r == 0) {
				swap_cnt = 1;
				SWAP(pc, pd);
				pd -= size;
			}
			pc -= size;
		}
		if (pb > pc)
			break;
		SWAP(pb, pc);
		swap_cnt = 1;
		pb += size;
		pc -= size;
	}
	if (swap_cnt == 0) { // Switch to insertion sort
		for (pm = a + size; pm < (char *)a + nmemb * size; pm += size)
			for (pl = pm; pl > (char *)a && compar(pl - size, pl) > 0; pl -= size)
				SWAP(pl, pl - size);
		return;
	}
	//
	pn = a + nmemb * size;
	r = MIN(pa - (char *)a, pb - pa);
	VECSWAP(a, pb - r, r);
	r = MIN(pd - pc, pn - pd - size);
	VECSWAP(pb, pn - r, r);
	if ((r = pb - pa) > size)
		qsort(a, r / size, size, compar);
	if ((r = pd - pc) > size) {
		// Iterate rather than recurse to save stack space
		a = pn - r;
		nmemb = r / size;
		goto loop;
	}
	/*qsort(pn - r, r / size, size, compar);*/
}

#undef SWAP
#undef VECSWAP
#undef MIN
#undef SWAPCODE
#undef SWAPINIT

#pragma endregion

__device__ div_t div_(int numer, int denom) {
	div_t r;
	r.quot = numer / denom;
	r.rem = numer % denom;
	if (numer >= 0 && r.rem < 0) {
		r.quot++;
		r.rem -= denom;
	}
	return r;
}

__device__ ldiv_t ldiv_(long int numer, long int denom) {
	ldiv_t r;
	r.quot = numer / denom;
	r.rem = numer % denom;
	if (numer >= 0 && r.rem < 0) {
		r.quot++;
		r.rem -= denom;
	}
	return r;
}

#if defined(ULLONG_MAX)
__device__ lldiv_t lldiv_(long long int numer, long long int denom) {
	lldiv_t r;
	r.quot = numer / denom;
	r.rem = numer % denom;
	if (numer >= 0 && r.rem < 0) {
		r.quot++;
		r.rem -= denom;
	}
	return r;
}
#endif

/* Return the length of the multibyte character in S, which is no longer than N.  */
__device__ int mblen_(const char *s, size_t n) {
	panic("Not Implemented");
	return 0;
}
/* Return the length of the given multibyte character, putting its `wchar_t' representation in *PWC.  */
__device__ int mbtowc_(wchar_t *__restrict __pwc, const char *__restrict s, size_t n) {
	panic("Not Implemented");
	return 0;
}
/* Put the multibyte character represented by WCHAR in S, returning its length.  */
__device__ int wctomb_(char *s, wchar_t wchar) {
	panic("Not Implemented");
	return 0;
}

/* Convert a multibyte string to a wide char string.  */
__device__ size_t mbstowcs_(wchar_t *__restrict pwcs, const char *__restrict s, size_t n) {
	panic("Not Implemented");
	return 0;
}
/* Convert a wide char string to multibyte string.  */
__device__ size_t wcstombs_(char *__restrict s, const wchar_t *__restrict pwcs, size_t n) {
	panic("Not Implemented");
	return 0;
}

#if defined(__GNUC__)
__device__ uint16_t __builtin_bswap16_(uint16_t x) { char *p = (char *)x; return p[0] << 8 | p[1]; }
__device__ uint32_t __builtin_bswap32_(uint32_t x) { char *p = (char *)x; return p[0] << 24 | p[1] << 16 | p[2] << 8 | p[3]; }
__device__ uint64_t __builtin_bswap64_(uint64_t x) { char *p = (char *)x; return p[0] << 56 | p[1] << 48 | p[2] << 40 | p[3] << 32 | p[0] << 24 | p[1] << 16 | p[2] << 8 | p[3]; }
#elif defined(_MSC_VER)
__device__ unsigned short _byteswap_ushort_(unsigned short x) { char *p = (char *)x; return p[0] << 8 | p[1]; }
__device__ unsigned long _byteswap_ulong_(unsigned long x) { char *p = (char *)x; return p[0] << 24 | p[1] << 16 | p[2] << 8 | p[3]; }
__device__ unsigned __int64 _byteswap_uint64_(unsigned __int64 x) { char *p = (char *)x; return p[0] << 56 | p[1] << 48 | p[2] << 40 | p[3] << 32 | p[0] << 24 | p[1] << 16 | p[2] << 8 | p[3]; }
#endif

__END_DECLS;
