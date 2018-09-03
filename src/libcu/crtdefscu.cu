#include <crtdefscu.h>

__BEGIN_DECLS;

// HOSTPTRS
#pragma region HOSTPTRS

typedef struct __align__(8) {
	hostptr_t *ptr;			// reference
	unsigned short id;		// ID of author
	unsigned short threadid;// thread ID of author
} hostRef;

__device__ hostRef __iob_hostRefs[LIBCU_MAXHOSTPTR]; // Start of circular buffer (set up by host)
volatile __device__ hostRef *__iob_freeDevicePtr = __iob_hostRefs; // Current atomically-incremented non-wrapped offset
volatile __device__ hostRef *__iob_retnDevicePtr = __iob_hostRefs; // Current atomically-incremented non-wrapped offset
__constant__ hostptr_t __iob_hostptrs[LIBCU_MAXHOSTPTR];

static __forceinline__ __device__ void writeHostRef(hostRef *ref, hostptr_t *p) {
	ref->ptr = p;
	ref->id = gridDim.x*blockIdx.y + blockIdx.x;
	ref->threadid = blockDim.x*blockDim.y*threadIdx.z + blockDim.x*threadIdx.y + threadIdx.x;
}

__device__ hostptr_t *__hostptrGet(void *host) {
	// advance circular buffer
	size_t offset = atomicAdd((_uintptr_t *)&__iob_freeDevicePtr, sizeof(hostRef)) - (size_t)&__iob_hostRefs;
	offset %= (sizeof(hostRef)*LIBCU_MAXHOSTPTR);
	int offsetId = offset / sizeof(hostRef);
	hostRef *ref = (hostRef *)((char *)&__iob_hostRefs + offset);
	hostptr_t *p = ref->ptr;
	if (!p) {
		p = &__iob_hostptrs[offsetId];
		writeHostRef(ref, p);
	}
	p->host = host;
	return p;
}

__device__ void __hostptrFree(hostptr_t *p) {
	if (!p) return;
	// advance circular buffer
	size_t offset = atomicAdd((_uintptr_t *)&__iob_retnDevicePtr, sizeof(hostRef)) - (size_t)&__iob_hostRefs;
	offset %= (sizeof(hostRef)*LIBCU_MAXHOSTPTR);
	hostRef *ref = (hostRef *)((char *)&__iob_hostRefs + offset);
	writeHostRef(ref, p);
}

#pragma endregion

__host_constant__ const int __libcuone = 1;

/* Routine needed to support the testcase() macro. */
#ifdef _COVERAGE_TEST
static __host_device__ unsigned dummy = 0;
__host_device__ void __coverage(int x) { dummy += (unsigned)x; }
#endif

__device__ void fsystemReset();
__device__ void libcuReset() {
	fsystemReset();
}

// EXT-METHODS
#pragma region EXT-METHODS

__hostb_device__ ext_methods __extsystem = { nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr };

#pragma endregion

__END_DECLS;