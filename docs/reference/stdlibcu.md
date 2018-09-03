## #include <stdlibcu.h>

Also includes:
```
#include <stdlib.h>
```

## Device Side
Prototype | Description | Tags
--- | --- | :---:
```__device__ double atof(const char *nptr);``` | Convert a string to a floating-point number.
```__device__ int atoi(const char *nptr);``` | Convert a string to an integer.
```__device__ long int atol(const char *nptr);``` | Convert a string to a long integer.
```__device__ long long int atoll(const char *nptr);``` | Convert a string to a long long integer.
```__device__ double strtod(const char *__restrict nptr, char **__restrict endptr);``` | Convert a string to a floating-point number.
```__device__ float strtof(const char *__restrict nptr, char **__restrict endptr);``` | Likewise for 'float' and `long double' sizes of floating-point numbers.
```__device__ long double strtold(const char *__restrict nptr, char **__restrict endptr);``` | Likewise for 'float' and 'long double' sizes of floating-point numbers.
```__device__ long int strtol(const char *__restrict nptr, char **__restrict endptr, int base);``` | Convert a string to a long integer.
```__device__ unsigned long int strtoul(const char *__restrict nptr, char **__restrict endptr, int base);``` | Convert a string to an unsigned long integer.
```__device__ long long int strtoll(const char *__restrict nptr, char **__restrict endptr, int base);``` | Convert a string to a quadword integer.
```__device__ unsigned long long int strtoull(const char *__restrict nptr, char **__restrict endptr, int base);``` | Convert a string to an unsigned quadword integer.
```__device__ int rand(void);``` | Return a random integer between 0 and RAND_MAX inclusive.
```__device__ void srand(unsigned int seed);``` | Seed the random number generator with the given number.
```__device__ void *malloc(size_t size);``` | Allocate SIZE bytes of memory. | #ganging
```__device__ void *calloc(size_t nmemb, size_t size);``` | Allocate NMEMB elements of SIZE bytes each, all initialized to 0. | #ganging
```__device__ size_t _msize(void *ptr);``` | SIZE bytes of memory.
```__device__ void *realloc(void *ptr, size_t size);``` | Re-allocate the previously allocated block in PTR, making the new block SIZE bytes long. | #ganging
```__device__ void free(void *ptr);``` | Free a block allocated by 'malloc', 'realloc' or 'calloc'. | #ganging
```__device__ void abort(void);``` | Abort execution and generate a core-dump. | #trap
```__device__ int atexit(void(*func)(void));``` | Register a function to be called when `exit' is called. | #notsupported
```__device__ void exit(int status);``` | Call all functions registered with 'atexit' and 'on_exit', in the reverse of the order in which they were registered, perform stdio cleanup, and terminate program execution with STATUS. | #sentinel
```__device__ void _Exit(int status);``` | Terminate the program with STATUS without calling any of the functions registered with 'atexit' or 'on_exit'. | #sentinel
```__device__ char *getenv(const char *name);``` | Return the value of envariable NAME, or NULL if it doesn't exist.
```__device__ int setenv(const char *name, const char *value, int replace);``` | Set NAME to VALUE in the environment. If REPLACE is nonzero, overwrite an existing value.
```__device__ int unsetenv(const char *name);``` | Remove the variable NAME from the environment.
```__device__ char *mktemp(char *template_);``` | Generate a unique temporary file name from TEMPLATE.
```__device__ int mkstemp(char *template_);``` | Generate a unique temporary file name from TEMPLATE.
```__device__ int mkstemp64(char *template_);``` | Generate a unique temporary file name from TEMPLATE. | #file64
```__device__ int system(const char *command);``` | Execute the given line as a shell command. | #sentinel
```__device__ void *bsearch(const void *key, const void *base, size_t nmemb, size_t size, __compar_fn_t compar);``` | Do a binary search for KEY in BASE, which consists of NMEMB elements of SIZE bytes each, using COMPAR to perform the comparisons.
```__device__ void qsort(void *base, size_t nmemb, size_t size, __compar_fn_t compar);``` | Sort NMEMB elements of BASE, of SIZE bytes each, using COMPAR to perform the comparisons.
```__device__ int abs(int x);``` | Return the absolute value of X.
```__device__ long int labs(long int x);``` | Return the absolute value of X.
```__device__ long long int llabs(long long int x);``` | Return the absolute value of X.
```__device__ div_t div(int numer, int denom);``` | Return the 'div_t', 'ldiv_t' or 'lldiv_t' representation of the value of NUMER over DENOM.
```__device__ ldiv_t ldiv(long int numer, long int denom);``` | Return the 'div_t', 'ldiv_t' or 'lldiv_t' representation of the value of NUMER over DENOM.
```__device__ lldiv_t lldiv(long long int numer, long long int denom);``` | Return the 'div_t', 'ldiv_t' or 'lldiv_t' representation of the value of NUMER over DENOM.
```__device__ int mblen(const char *s, size_t n);``` | Return the length of the multibyte character in S, which is no longer than N.
```__device__ int mbtowc(wchar_t *__restrict __pwc, const char *__restrict s, size_t n);``` | Return the length of the given multibyte character, putting its `wchar_t' representation in *PWC.
```__device__ int wctomb(char *s, wchar_t wchar);``` | Put the multibyte character represented by WCHAR in S, returning its length.
```__device__ size_t mbstowcs(wchar_t *__restrict pwcs, const char *__restrict s, size_t n);``` | Convert a multibyte string to a wide char string.
```__device__ size_t wcstombs(char *__restrict s, const wchar_t *__restrict pwcs, size_t n);``` | Convert a wide char string to multibyte string.
```__device__ quad_t strtoq(const char *__restrict nptr, char **__restrict endptr, int base);``` | Convert a string to a quadword integer.
```__device__ u_quad_t strtouq(const char *__restrict nptr, char **__restrict endptr, int base);``` | Convert a string to an unsigned quadword integer.
```__device__ void *mallocZero(size_t size);``` | Allocate SIZE bytes of memory.  Then Zero memory.
