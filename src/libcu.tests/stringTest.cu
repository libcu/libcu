#include <stdiocu.h>
#include <stringcu.h>
#include <assert.h>

static __global__ void g_string_test1() {
	printf("string_test1\n");

	char src[50] = "abcdefghijklmnopqrstuvwxyz";
	char dest[50]; memset(dest, 0, 50);

	//// MEMCPY, MEMMOVE, MEMSET, MEMCPY, MEMCHR ////
	//__forceinline__ __device__ void *memcpy_(void *__restrict dest, const void *__restrict src, size_t n);
	//extern __device__ void *memmove_(void *dest, const void *src, size_t n);
	//__forceinline__ __device__ void *memset_(void *s, int c, size_t n);
	//extern __device__ int memcmp_(const void *s1, const void *s2, size_t n);
	//extern __device__ void *memchr_(const void *s, int c, size_t n);
	void *a0a = memcpy(dest, src, 0);
	void *a0b = memcpy(dest + 1, src, 1);
	assert(a0a == &dest[0] && ((char *)a0a)[0] == 0 && a0b == &dest[1] && ((char *)a0b)[0] == 'a');
	void *a1a = memmove(dest, src, 0); void *a1b = memmove(dest, src, 10); void *a1c = memmove(dest + 5, dest, 5);
	assert(a1a == dest && !strncmp((char *)a1b, "abcde", 5) && !strncmp((char *)a1c, "abcde", 5));
	void *a2a = memset(dest, 0, 0); void *a2b = memset(dest + 1, 0, 5);
	assert(a2a == dest && ((char *)a2a)[0] == 'a' && a2b == dest + 1 && ((char *)a2b)[0] == 0);
	int a3a = memcmp(nullptr, nullptr, 0); int a3b = memcmp("abc", "abc", 2); int a3c = memcmp("abc", "abc", 3); int a3d = memcmp("abc", "axc", 3); assert(!a3a && !a3b && !a3c && a3d);
	void *a4a = memchr(src, 0, 0); void *a4b = memchr(src, 'b', 1); void *a4c = memchr(src, 'b', 3); void *a4d = memchr(src, 'z', 3); assert(!a4a && !a4b && a4c && !a4d);

	//// STRCPY, STRNCPY, STRCAT, STRNCAT ////
	//extern __device__ char *strcpy_(char *__restrict dest, const char *__restrict src);
	//extern __device__ char *strncpy_(char *__restrict dest, const char *__restrict src, size_t n);
	//extern __device__ char *strcat_(char *__restrict dest, const char *__restrict src);
	//extern __device__ char *strncat_(char *__restrict dest, const char *__restrict src, size_t n);
	strcpy(src, "abcd");
	char *b0a = strcpy(dest, src); assert(!strcmp(b0a, "abcd"));
	char *b1a = strncpy(dest + 5, src, 10); assert(!strcmp(b1a, "abcd"));
	char *b2a = strcat(dest + 10, "xyz"); assert(!strcmp(b2a, "xyz"));
	char *b3a = strncat(dest + 15, "xyz", 2); assert(!strncmp(b3a, "xy", 2));

	//// STRCMP, STRICMP, STRNCMP, STRNICMP ///2
	//extern __device__ int strcmp_(const char *s1, const char *s2);
	//extern __device__ int stricmp_(const char *s1, const char *s2);
	//extern __device__ int strncmp_(const char *s1, const char *s2, size_t n);
	//extern __device__ int strnicmp_(const char *s1, const char *s2, size_t n);
	int c0a = strcmp("", ""); int c0b = strcmp("", "abc"); int c0c = strcmp("abc", ""); int c0d = strcmp("abc", "xyz"); int c0e = strcmp("abc", "ab"); int c0f = strcmp("abc", "abc"); int c0g = strcmp("Abc", "abc"); assert(!c0a && c0b && c0c && c0d && c0e && !c0f && c0g);
	int c1a = stricmp("", ""); int c1b = stricmp("", "abc"); int c1c = stricmp("abc", ""); int c1d = stricmp("abc", "xyz"); int c1e = stricmp("abc", "ab"); int c1f = stricmp("abc", "abc"); int c1g = stricmp("Abc", "abc"); assert(!c1a && c1b && c1c && c1d && c1e && !c1f && !c1g);
	int c2a = strncmp("", "", 3); int c2b = strncmp("", "abc", 3); int c2c = strncmp("abc", "", 3); int c2d = strncmp("abc", "xyz", 3); int c2e = strncmp("abc", "ab", 3); int c2f = strncmp("abc", "abc", 3); int c2g = strncmp("Abc", "abc", 3); int c2h = strncmp("abx", "aby", 2); assert(!c2a && c2b && c2c && c2d && c2e && !c2f && c2g && !c2h);
	int c3a = strnicmp("", "", 3); int c3b = strnicmp("", "abc", 3); int c3c = strnicmp("abc", "", 3); int c3d = strnicmp("abc", "xyz", 3); int c3e = strnicmp("abc", "ab", 3); int c3f = strnicmp("abc", "abc", 3); int c3g = strnicmp("Abc", "abc", 3); int c3h = strnicmp("Abx", "aby", 2); assert(!c3a && c3b && c3c && c3d && c3e && !c3f && !c3g && !c2h);

	//// STRCOLL ////
	//extern __device__ int strcoll_(const char *s1, const char *s2);
	// d0a = strcoll - not:implemented

	//// STRXFRM ////
	//extern __device__ size_t strxfrm_(char *__restrict dest, const char *__restrict src, size_t n);
	// e0a = strxfrm - not:implemented

	//// STRDUP, STRNDUP ////
	//extern __device__ char *strdup_(const char *s);
	//extern __device__ char *strndup_(const char *s, size_t n);
	char *f0a = strdup("abc"); assert(!strcmp(f0a, "abc")); free(f0a);
	char *f1a = strndup("abc", 2); assert(!strncmp(f1a, "ab", 2)); free(f1a);

	//// STRCHR, STRRCHR, STRCSPN, STRSPN, STRPBRK, STRSTR, STRTOK ////
	//extern __device__ char *strchr_(const char *s, int c);
	//extern __device__ char *strrchr_(const char *s, int c);
	//extern __device__ size_t strcspn_(const char *s, const char *reject);
	//extern __device__ size_t strspn_(const char *s, const char *accept);
	//extern __device__ char *strpbrk_(const char *s, const char *accept);
	//extern __device__ char *strstr_(const char *haystack, const char *needle);
	//extern __device__ char *strtok_(char *__restrict s, const char *__restrict delim);
	const char *g0a = strchr("", 'a'); const char *g0b = strchr("abc", 'b'); const char *g0c = strchr("abc", 'z'); assert(!g0a && g0b[0] == 'b' && !g0c);
	const char *g1a = strrchr("", 'a'); const char *g1b = strrchr("abc", 'b'); const char *g1c = strrchr("abc", 'z'); assert(!g1a && g1b[0] == 'b' && !g1c);
	// g2a = strcspn - not:implemented
	// g3a = strspn - not:implemented
	const char *g4a = strpbrk("", ""); const char *g4b = strpbrk("abc", "by"); const char *g4c = strpbrk("abc", "xy"); assert(!g4a && g4b[0] == 'b' && !g4c);
	const char *g5a = strstr("", ""); const char *g5b = strstr("abc", "by"); const char *g5c = strstr("abc", "bc"); assert(g5a && !g5b && g5c[0] == 'b');
	// g6a = strtok - not:implemented

	//// MEMPCPY ////
	//extern __device__ void *mempcpy_(void *__restrict dest, const void *__restrict src, size_t n);
	// h0a = mempcpy - not:implemented

	//// STRLEN, STRLEN16, STRNLEN ////
	//extern __device__ size_t strlen_(const char *s);
	//__forceinline__ __device__ size_t strlen16(const void *s);
	//extern __device__ size_t strnlen_(const char *s, size_t maxlen);
	size_t i0a = strlen(nullptr); size_t i0b = strlen(""); size_t i0c = strlen("abc"); assert(!i0a && !i0b && i0c == 3);
	size_t i1a = strlen16(nullptr); size_t i1b = strlen16(L""); size_t i1c = strlen16(L"abc"); assert(!i1a && !i1b && i1c == 3);
	size_t i2a = strnlen(nullptr, 3); size_t i2b = strnlen("", 3); size_t i2c = strnlen("abc", 3); assert(!i2a && !i2b && i2c == 3);

	//// STRERROR ////
	//extern __device__ char *strerror_(int errnum);
	char *j6a = strerror(0); assert(!strcmp(j6a, "ERROR"));
}
cudaError_t string_test1() { g_string_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }
