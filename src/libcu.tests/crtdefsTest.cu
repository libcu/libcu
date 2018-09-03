#include <stdiocu.h>
#include <crtdefscu.h>
#include <stringcu.h>
#include <assert.h>

static __global__ void g_crtdefs_test1() {
	printf("crtdefs_test1\n");
	/* Memory allocation - rounds to the type in T */
	int a0a = ROUNDT_(3, int); int a0b = ROUNDT_(4, int); int a0c = ROUNDT_(5, int); int a0d = ROUNDT_(-5, int); assert(a0a == 4 && a0b == 4 && a0c == 8 && a0d == -4);
	/* Memory allocation - rounds up to 8 */
	int a1a = ROUND8_(3); int a1b = ROUND8_(8); int a1c = ROUND8_(9); int a1d = ROUND8_(-5); assert(a1a == 8 && a1b == 8 && a1c == 16 && a1d == 0);
	/* Memory allocation - rounds up to 64 */
	int a2a = ROUND64_(3); int a2b = ROUND64_(64); int a2c = ROUND64_(65); int a2d = ROUND64_(-65); assert(a2a == 64 && a2b == 64 && a2c == 128 && a2d == -64);
	/* Memory allocation - rounds up to "size" */
	int a3a = ROUNDN_(3, 4); int a3b = ROUNDN_(4, 4); int a3c = ROUNDN_(5, 4); int a3d = ROUNDN_(-5, 4); assert(a3a == 4 && a3b == 4 && a3c == 8 && a3d == -4);

	/* Memory allocation - rounds down to 8 */
	int b0a = ROUNDDOWN8_(3); int b0b = ROUNDDOWN8_(8); int b0c = ROUNDDOWN8_(9); int b0d = ROUNDDOWN8_(-5); assert(b0a == 0 && b0b == 8 && b0c == 8 && b0d == -8);
	/* Memory allocation - rounds down to "size" */
	int b1a = ROUNDDOWNN_(3, 4); int b1b = ROUNDDOWNN_(4, 4); int b1c = ROUNDDOWNN_(5, 4); int b1d = ROUNDDOWNN_(-5, 4); assert(b1a == 0 && b1b == 4 && b1c == 4 && b1d == -8);

	/* Test to see if you are on aligned boundary, affected by BYTEALIGNED4 */
	int c0a = HASALIGNMENT8_(3); int c0b = HASALIGNMENT8_(8); int c0c = HASALIGNMENT8_(9); int c0d = HASALIGNMENT8_(-3); assert(!c0a && c0b && !c0c && !c0d);
	/* Returns the length of an array at compile time (via math) */
	int integerArrayOfSixElements[6]; int d0a = ARRAYSIZE_(integerArrayOfSixElements); assert(d0a == 6);

	/* Determines where you are based on path */
	bool e0a = ISHOSTPATH("C:\\test"); bool e0b = ISHOSTPATH("C:/test"); bool e0c = ISHOSTPATH(":\\test"); bool e0d = ISHOSTPATH(":/test"); assert(e0a && e0b && !e0c && !e0d);
	strcpy(__cwd, ":\\"); bool e1a = ISHOSTPATH("."); bool e1b = ISHOSTPATH("test"); bool e1c = ISHOSTPATH("\test"); bool e1d = ISHOSTPATH("/test"); assert(!e1a && !e1b && !e1c && !e1d);
	strcpy(__cwd, "\0"); bool e2a = ISHOSTPATH("."); bool e2b = ISHOSTPATH("test"); bool e2c = ISHOSTPATH("\test"); bool e2d = ISHOSTPATH("/test"); assert(e2a && e2b && e2c && e2d);
	/* Determines where you are based on number(handle) */
	bool f0a = ISHOSTHANDLE(1); bool f0b = ISHOSTHANDLE(INT_MAX - LIBCU_MAXFILESTREAM); bool f0c = ISHOSTHANDLE(INT_MAX); assert(f0a && !f0b && !f0c);
}
cudaError_t crtdefs_test1() { g_crtdefs_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }
