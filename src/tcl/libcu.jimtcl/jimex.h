#ifndef __JIMEX__H
#define __JIMEX__H
#include "jim.h"
#ifdef __cplusplus
extern "C" {
#endif

	JIM_EXPORT __device__ int Jim_UtfToLower(const char *objPtr);
	JIM_EXPORT __device__ int Jim_ListGetElements(Jim_Interp *interp, Jim_Obj *obj, int *length, Jim_Obj ***flags);

#ifdef __cplusplus
}
#endif
#endif // __JIMEX__H
