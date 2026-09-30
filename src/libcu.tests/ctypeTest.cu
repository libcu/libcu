#include <stdiocu.h>
#include <ctypecu.h>
#include <assert.h>

static __global__ void g_ctype_test1() {
	printf("ctype_test1\n");

	//// ISCTYPE ////
	//extern __forceinline__ __device__ int isctype_(int c, int type);
	bool _0 = isctype('a', 0x02); assert(_0);

	//// ISALNUM, ISALPHA, ISCNTRL, ISDIGIT, ISLOWER, ISGRAPH, ISPRINT, ISPUNCT, ISSPACE, ISUPPER, ISXDIGIT ////
	//extern __forceinline__ __device__ int isalnum_(int c);
	//extern __forceinline__ __device__ int isalpha_(int c);
	//extern __forceinline__ __device__ int iscntrl_(int c);
	//extern __forceinline__ __device__ int isdigit_(int c);
	//extern __forceinline__ __device__ int islower_(int c);
	//extern __forceinline__ __device__ int isgraph_(int c);
	//extern __forceinline__ __device__ int isprint_(int c);
	//extern __forceinline__ __device__ int ispunct_(int c);
	//extern __forceinline__ __device__ int isspace_(int c);
	//extern __forceinline__ __device__ int isupper_(int c);
	//extern __forceinline__ __device__ int isxdigit_(int c);
	bool a0 = isalnum('a'); bool a0n = isalnum('1'); assert(a0 && a0n);
	bool a1 = isalpha('a'); bool a1n = isalpha('A'); assert(a1 && a1n);
	bool a2 = iscntrl('a'); bool a2n = iscntrl('A'); assert(!a2 && !a2n);
	bool a3 = isdigit('a'); bool a3n = isdigit('1'); assert(!a3 && a3n);
	bool a4 = islower('a'); bool a4n = islower('A'); assert(a4 && !a4n);
	bool a5 = isgraph('a'); bool a5n = isgraph('A'); assert(!a5 && !a5n);
	bool a6 = isprint('a'); bool a6n = isprint('A'); assert(a6 && a6n);
	bool a7 = ispunct('a'); bool a7n = ispunct('A'); assert(!a7 && !a7n);
	bool a8 = isspace('a'); bool a8n = isspace(' '); assert(!a8 && a8n);
	bool a9 = isupper('a'); bool a9n = isupper('A'); assert(!a9 && a9n);
	bool aA = isxdigit('a'); bool aAn = isxdigit('A'); assert(aA && aAn);

	//// TOLOWER, TOUPPER, _TOLOWER, _TOUPPER ////
	//extern __forceinline__ __device__ int tolower_(int c);
	//extern __forceinline__ __device__ int toupper_(int c);
	////existing: #define _tolower(c)
	////existing: #define _toupper(c)
	char b0 = tolower('a'); char b0n = tolower('A'); assert(b0 == 'a' && b0n == 'a');
	char b1 = toupper('a'); char b1n = toupper('A'); assert(b1 == 'A' && b1n == 'A');
	char b2 = _toupper('a'); char b2n = _toupper('A'); assert(b2 == 'A' && b2n != 'A');
	char b3 = _tolower('a'); char b3n = _tolower('A'); assert(b3 != 'a' && b3n == 'a');

	//// ISBLANK, ISIDCHAR ////
	//extern __forceinline__ __device__ int isblank_(int c);
	//extern __forceinline__ __device__ int isidchar_(int c);
	bool c0 = isblank(' '); bool c0n = isblank('A'); assert(c0 && !c0n);
	bool c1 = isidchar('a'); bool c1n = isidchar('A'); assert(c1 && c1n);

	//bool d0 = ispoweroftwo(2); bool d0n = ispoweroftwo(3); assert(d0 && !d0n);
	//bool d1 = isalpha2('a'); bool d1n = isalpha2('A'); assert(d1 && d1n);
}
cudaError_t ctype_test1() { g_ctype_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }
