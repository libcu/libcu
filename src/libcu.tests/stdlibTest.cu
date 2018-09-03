//#include <cuda_runtimecu.h>
#include <stdiocu.h>
#include <stdlibcu.h>
#include <assert.h>

static __device__ int compareints(const void *a, const void *b) {
	return (*(int*)a - *(int*)b);
}

__device__ int _values[] = { 50, 20, 60, 40, 10, 30 };

static __global__ void g_stdlib_test1() {
	printf("stdlib_test1\n");

	//// ATOI, ATOL, ATOLL ////
	//__forceinline__ __device__ double atof_(const char *nptr);
	//__forceinline__ __device__ int atoi_(const char *nptr);
	//__forceinline__ __device__ long int atol_(const char *nptr);
	//__forceinline__ __device__ long long int atoll_(const char *nptr);
	double a0a = atof("1.0"); assert(a0a == 1.0);
	int a1a = atoi("1.0"); assert(a1a == 1);
	long int a2a = atol("1.0"); assert(a2a == 1);
	long long int a3a = atoll("1.0"); assert(a3a == 1L);

	//// STRTOD, STRTOF, STROLD, STRTOL, STRTOUL, STRTOLL, STRTOULL ///
	//extern __device__ double strtod_(const char *__restrict nptr, char **__restrict endptr);
	//extern __device__ float strtof_(const char *__restrict nptr, char **__restrict endptr);
	//extern __device__ long double strtold_(const char *__restrict nptr, char **__restrict endptr);
	//__forceinline__ __device__ long int strtol_(const char *__restrict nptr, char **__restrict endptr, int base);
	//__forceinline__ __device__ unsigned long int strtoul_(const char *__restrict nptr, char **__restrict endptr, int base);
	//__forceinline__ __device__ long long int strtoll_(const char *__restrict nptr, char **__restrict endptr, int base):
	//__forceinline__ __device__ unsigned long long int strtoull_(const char *__restrict nptr, char **__restrict endptr, int base):
	double b0a = strtod("1.0", nullptr); assert(b0a == 1.0);
	float b1a = strtof("1.0", nullptr); assert(b1a == 1.0F);
	long_double b2a = strtold("1.0", nullptr); assert(b2a == 1.0);
	long int b3a = strtol("1.0", nullptr, 10); assert(b3a == 1);
	unsigned long int b4a = strtoul("1.0", nullptr, 10); assert(b4a == 1L);
	long long int b5a = strtoll("1.0", nullptr, 10); assert(b5a == 1L);

	//// RAND, SRAND ////
	//extern __device__ int rand_(void);
	//extern __device__ void srand_(unsigned int seed);
	int c0a = rand();
	srand(10);

	//// MALLOC, CALLOC, MSIZE, REALLOC, FREE ////
	//extern __device__ void *malloc_(size_t size);
	//__device__ __forceinline__ void *calloc_(size_t nmemb, size_t size);
	//extern __device__ size_t _msize_(void *ptr);
	//extern __device__ void *realloc_(void *ptr, size_t size);
	//extern __device__ void free_(void *ptr);
	char *d0a = (char *)malloc(10); int d0b = _msize(d0a); char *d0c = (char *)realloc(d0a, 15); int d0d = _msize(d0c); free(d0c); assert(d0a && d0b == 10 && d0c && d0d == 15);
	char *d1a = (char *)calloc(10, 1); free(d1a); assert(d1a);

	//// ABORT, ATEXIT, EXIT, _EXIT ////
	//skipped: __forceinline__ __device__ void abort_(void); #trap
	//skipped: __forceinline__ __device__ int atexit_(void(*func)(void)); #notsupported
	//skipped: __forceinline__ __device__ void exit_(int status); #sentinel
	//skipped: __forceinline__ __device__ void _Exit_(int status); #sentinel

	//// GETENV, SETENV, UNSETENV ////
	//extern __device__ char *getenv_(const char *name);
	//extern __device__ int setenv_(const char *name, const char *value, int replace);
	//extern __device__ int unsetenv_(const char *name);
	/* Host */
	char *f0a = getenv("Test"); assert(f0a);
	int f1a = setenv("Test", "value", true);
	char *f1b = getenv("Test");
	int f1c = unsetenv("Test");
	assert(f1a && f1b && f1c);
	/* Device */
	//char *f0a = getenv("Test"); assert(f0a);
	//int f1a = setenv("Test", "value", true);
	//char *f1b = getenv("Test");
	//int f1c = unsetenv("Test");
	//assert(f1a && f1b && f1c);

	//// MKTEMP, MKSTEMP ////
	//extern __device__ char *mktemp_(char *template_);
	//extern __device__ int mkstemp_(char *template_);
	char *g0a = mktemp("Test"); assert(g0a);
	int g1a = mkstemp("Test"); assert(g1a);

	//// SYSTEM ////
	//extern __device__ int system_(const char *command); #sentinel
	int h0a = system("echo"); assert(h0a);

	//// BSEARCH ////
	//extern __device__ void *bsearch_(const void *key, const void *base, size_t nmemb, size_t size, __compar_fn_t compar);
	qsort(_values, 6, sizeof(int), compareints);
	int i0_in = 4; int i0_out = 41;
	int *i0a = (int *)bsearch(&i0_in, _values, 6, sizeof(int), compareints);
	int *i0b = (int *)bsearch(&i0_out, _values, 6, sizeof(int), compareints);
	assert(i0a && !i0b);

	//// QSORT ////
	//extern __device__ void qsort_(void *base, size_t nmemb, size_t size, __compar_fn_t compar);
	qsort(_values, 6, sizeof(int), compareints);
	int j0a = (int)_values[0];
	assert(j0a == 10);

	//// ABS, LABS, LLABS ////
	//__forceinline__ __device__ int abs_(int x);
	//__forceinline__ __device__ long int labs_(long int x):
	//__forceinline__ __device__ long long int llabs_(long long int x);
	int k0a = abs(0); int k0b = abs(1); int k0c = abs(-1); assert(k0a == 0 && k0b == 1 & k0c == 1);
	long int k1a = abs(0); long int k1b = abs(1); long int k1c = abs(-1); assert(k1a == 0 && k1b == 1 & k1c == 1);
	long long int k2a = abs(0L); long long int k2b = abs(1L); long long int k2c = abs(-1L); assert(k2a == 0L && k2b == 1L & k2c == 1L);

	//// DIV, LDIV, LLDIV ////
	//extern __device__ div_t div_(int numer, int denom);
	//extern __device__ ldiv_t ldiv_(long int numer, long int denom);
	//extern __device__ lldiv_t lldiv_(long long int numer, long long int denom);
	div_t l0a = div(1, 2); assert(l0a.quot == 0 && l0a.rem == 2);
	ldiv_t l1a = ldiv(1, 2); assert(l1a.quot == 0 && l1a.rem == 2);
	lldiv_t l2a = lldiv(1, 2); assert(l2a.quot == 0 && l2a.rem == 2);

	//// MBLEN, MBTOWC, WCTOMB, MBSTOWSCS, WCSTOMBS ////
	//extern __device__ int mblen_(const char *s, size_t n);
	//extern __device__ int mbtowc_(wchar_t *__restrict __pwc, const char *__restrict s, size_t n);
	//extern __device__ int wctomb_(char *s, wchar_t wchar);
	//extern __device__ size_t mbstowcs_(wchar_t *__restrict pwcs, const char *__restrict s, size_t n);
	//extern __device__ size_t wcstombs_(char *__restrict s, const wchar_t *__restrict pwcs, size_t n);
	char buf[10];
	int m0a = mblen("test", 4); assert(m0a == 4);
	int m1a = mbtowc(L"test", buf, sizeof(buf)); assert(m1a == 4);
	int m2a = wctomb(buf, L'a'); bool m2b = (buf[0] == 1 && buf[1] == 0); assert(m2a && m2b);
	size_t m3a = mbstowcs(L"test", buf, sizeof(buf));

	//// STRTOQ, STRTOUQ ////
	//__forceinline__ __device__ quad_t strtoq_(const char *__restrict nptr, char **__restrict endptr, int base);
	//__forceinline__ __device__ u_quad_t strtouq_(const char *__restrict nptr, char **__restrict endptr, int base);
	quad_t n0a = strtoq("1.0", nullptr, 10); assert(n0a == 1);
	u_quad_t n1a = strtouq("1.0", nullptr, 10); assert(n1a == 1);

	//// MALLOCZERO //// ??different than calloc??
	//__forceinline__ __device__ void *mallocZero(size_t size);
	// SKIP
}
cudaError_t stdlib_test1() { g_stdlib_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }

#pragma region qsort

/*
static __device__ int qsortSelectFiles(const struct dirent *dirbuf) {
	return dirbuf->d_name[0] == '.' ? 0 : 1;
}

static __global__ void stdlib_qsort() {
struct dirent **a;
struct dirent *dirbuf;

int i, numdir;

chdir("/");
numdir = scandir(".", &a, qsortSelectFiles, NULL);
printf("\nGot %d entries from scandir().\n", numdir);
for (i = 0; i < numdir; ++i) {
dirbuf = a[i];
printf("[%d] %s\n", i, dirbuf->d_name);
free(a[i]);
}
free(a);
numdir = scandir(".", &a, qsortSelectFiles, alphasort);
printf("\nGot %d entries from scandir() using alphasort().\n", numdir);
for (i = 0; i < numdir; ++i) {
dirbuf = a[i];
printf("[%d] %s\n", i, dirbuf->d_name);
}
printf("\nCalling qsort()\n");
// Even though some manpages say that alphasort should be int alphasort(const void *a, const void *b),
// in reality glibc and uclibc have const struct dirent** instead of const void*.
// Therefore we get a warning here unless we use a cast, which makes people think that alphasort prototype needs to be fixed in uclibc headers.
qsort(a, numdir, sizeof(struct dirent *), (void *)alphasort);
for (i = 0; i < numdir; ++i) {
dirbuf = a[i];
printf("[%d] %s\n", i, dirbuf->d_name);
free(a[i]);
}
free(a);
return 0;
}
*/

#pragma endregion

#pragma region strtol

__constant__ const char *_strtol_strings[] = {
	/* some simple stuff */
	"0", "1", "10",
	"100", "1000", "10000", "100000", "1000000",
	"10000000", "100000000", "1000000000",

	/* negative */
	"-0", "-1", "-10",
	"-100", "-1000", "-10000", "-100000", "-1000000",
	"-10000000", "-100000000", "-1000000000",

	/* test base>10 */
	"a", "b", "f", "g", "z",

	/* test hex */
	"0x0", "0x1", "0xa", "0xf", "0x10",

	/* test octal */
	"00", "01", "07", "08", "0a", "010",

	/* other */
	"0x8000000",

	/* check overflow cases: (for 32 bit) */
	"2147483645",
	"2147483646",
	"2147483647",
	"2147483648",
	"2147483649",
	"-2147483645",
	"-2147483646",
	"-2147483647",
	"-2147483648",
	"-2147483649",
	"4294967293",
	"4294967294",
	"4294967295",
	"4294967296",
	"4294967297",
	"-4294967293",
	"-4294967294",
	"-4294967295",
	"-4294967296",
	"-4294967297",

	/* bad input tests */
	"",
	"00",
	"0x",
	"0x0",
	"-",
	"+",
	" ",
	" -",
	" - 0",
};
__device__ int _strtol_ntests = ARRAYSIZE_(_strtol_strings);

static __device__ void strtol_test(int base) {
	int i;
	long n;
	char *endptr;
	for (i = 0; i < _strtol_ntests; i++) {
		n = strtol(_strtol_strings[i], &endptr, base);
		printf("strtol(\"%s\",%d) len=%lu res=%ld\n", _strtol_strings[i], base, (unsigned long)(endptr - _strtol_strings[i]), n);
	}
}

static __device__ void strtol_utest(int base) {
	int i;
	unsigned long n;
	char *endptr;
	for (i = 0; i < _strtol_ntests; i++) {
		n = strtoul(_strtol_strings[i], &endptr, base);
		printf("strtoul(\"%s\",%d) len=%lu res=%lu\n", _strtol_strings[i], base, (unsigned long)(endptr - _strtol_strings[i]), n);
	}
}

__global__ void g_stdlib_strtol() {
	strtol_test(0); strtol_utest(0);
	strtol_test(8); strtol_utest(8);
	strtol_test(10); strtol_utest(10);
	strtol_test(16); strtol_utest(16);
	strtol_test(36); strtol_utest(36);
}
cudaError_t stdlib_strtol() { g_stdlib_strtol<<<1, 1>>>(); return cudaDeviceSynchronize(); }

#pragma endregion

#pragma region strtoq

__constant__ const char *_strtoq_strings[] = {
	/* some simple stuff */
	"0", "1", "10",
	"100", "1000", "10000", "100000", "1000000",
	"10000000", "100000000", "1000000000",

	/* negative */
	"-0", "-1", "-10",
	"-100", "-1000", "-10000", "-100000", "-1000000",
	"-10000000", "-100000000", "-1000000000",

	/* test base>10 */
	"a", "b", "f", "g", "z",

	/* test hex */
	"0x0", "0x1", "0xa", "0xf", "0x10",

	/* test octal */
	"00", "01", "07", "08", "0a", "010",

	/* other */
	"0x8000000",

	/* check overflow cases: (for 32 bit) */
	"2147483645",
	"2147483646",
	"2147483647",
	"2147483648",
	"2147483649",
	"-2147483645",
	"-2147483646",
	"-2147483647",
	"-2147483648",
	"-2147483649",
	"4294967293",
	"4294967294",
	"4294967295",
	"4294967296",
	"4294967297",
	"-4294967293",
	"-4294967294",
	"-4294967295",
	"-4294967296",
	"-4294967297",

	/* bad input tests */
	"",
	"00",
	"0x",
	"0x0",
	"-",
	"+",
	" ",
	" -",
	" - 0",
};
__device__ int _strtoq_ntests = ARRAYSIZE_(_strtoq_strings);

static __device__ void strtoq_test(int base) {
	int i;
	long long n; //quad_t n;
	char *endptr;
	for (i = 0; i < _strtoq_ntests; i++) {
		n = strtoq(_strtoq_strings[i], &endptr, base);
		printf("strtoq(\"%s\",%d) len=%lu res=%qd\n", _strtoq_strings[i], base, (unsigned long)(endptr - _strtoq_strings[i]), n);
	}
}

__global__ void g_stdlib_strtoq() {
	strtoq_test(0);
	strtoq_test(8);
	strtoq_test(10);
	strtoq_test(16);
	strtoq_test(36);
}
cudaError_t stdlib_strtoq() { g_stdlib_strtoq<<<1, 1>>>(); return cudaDeviceSynchronize(); }

#pragma endregion
