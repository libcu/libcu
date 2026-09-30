#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>
#include <ext/pipeline.cpp>

#define ROUND8_(x) (((x)+7)&~7)
// #define panic(fmt, ...) { printf(fmt, __VA_ARGS__); asm("trap;"); }

#ifdef	__cplusplus
#define __BEGIN_DECLS extern "C" {
#define __END_DECLS }
#else
#define __BEGIN_DECLS
#define __END_DECLS
#endif

__BEGIN_DECLS;

#ifndef _WIN64
#define MALLOCSIZETYPE long long int // must be x64 size for malloc alignment
#else
#define MALLOCSIZETYPE long long int
#endif
__device__ void *malloc_(size_t size) {
	assert(size > 0);
	MALLOCSIZETYPE *p = (MALLOCSIZETYPE *)malloc(ROUND8_(sizeof(MALLOCSIZETYPE) + size));
	if (p)
		p[0] = size;
	else panic("failed to allocate %u bytes of memory", size);
	return (void *)(p + 1);
}

__device__ void free_(void *ptr) {
	if (!ptr) return;
	MALLOCSIZETYPE *p = (MALLOCSIZETYPE *)ptr;
	free(p - 1);
}

__device__ void *realloc_(void *ptr, size_t size) {
	assert(size > 0);
	MALLOCSIZETYPE *p = (MALLOCSIZETYPE *)malloc(ROUND8_(sizeof(MALLOCSIZETYPE) + size));
	if (p)
		p[0] = size;
	else panic("failed to allocate %u bytes of memory", size);
	if (ptr) {
		MALLOCSIZETYPE *p2 = (MALLOCSIZETYPE *)ptr;
		size_t ptrSize = (size_t)*(p2 - 1);
		if (ptrSize) memcpy(p + 1, p2, ptrSize);
		free(p2 - 1);
	}
	return (void *)(p + 1);
}

__device__ size_t _msize_(void *ptr) {
	MALLOCSIZETYPE *p2 = (MALLOCSIZETYPE *)ptr;
	return (size_t)*(p2 - 1);
}

__END_DECLS;
