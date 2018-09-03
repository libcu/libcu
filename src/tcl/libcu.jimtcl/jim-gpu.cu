#include "jim.h"

__device__ int Jim_gpuInit(Jim_Interp *interp)
{
	if (Jim_PackageProvide(interp, "gpu", "1.0", JIM_ERRMSG))
		return JIM_ERROR;
	return JIM_OK;
}
